"""AutoResearch 角色 C：holdout 裁判，唯一出分口。

纪律：
- 锁定前任何独立测试请求都被拒绝；
- 同一 run 的 holdout 只能评一次；
- 非 test 判决会重跑验证窗并与提交的 TrialResult 比对，拒绝“伪造提升”；
- 每个 verdict 携带证据编号、checks 和 artifacts_hash。
"""

from __future__ import annotations

import json
import statistics
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from .contracts import Verdict, as_jsonable, experiment_fingerprint, normalize_experiment_spec
from .data import FEATURE_GROUPS
from .experiment import ResearchBackend, normalize_spec, run_experiment, run_holdout_test

FORBIDDEN_FEATURE_COLUMNS = ("date", "symbol", "label", "fwd5_ret", "label_horizon", "split")
AUC_TOLERANCE = 1e-6


class ResearchOracle:
    """Scoring/verification layer that never runs its own experiment design."""

    def __init__(self, backend: ResearchBackend, output_dir: Optional[Path | str] = None) -> None:
        self.backend = backend
        self.output_dir = Path(output_dir) if output_dir else None
        self._locks: dict[str, dict] = {}
        self._tested: set[str] = set()
        self._mutex = Lock()

    def register_lock(self, run_id: str, spec: dict) -> None:
        """Freeze the exact configuration that will be allowed one holdout test."""
        with self._mutex:
            if run_id in self._locks:
                raise ValueError("run already has a locked configuration")
            self._locks[run_id] = deepcopy(normalize_spec(spec))

    def _blocked(
        self,
        error: str,
        *,
        experiment_id: str,
        evidence_ids: List[str],
        test: bool,
    ) -> Verdict:
        return Verdict(
            verdict="blocked",
            metric={"error": error},
            checks=[],
            evidence_ids=evidence_ids,
            test=test,
        )

    def _write_verdict(self, run_id: str, record: Dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        path = self.output_dir / "verdicts" / f"{run_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        records: List[Dict[str, Any]] = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    records = loaded
            except Exception:
                records = []
        records.append(record)
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _feature_columns(self, spec: Dict[str, Any]) -> List[str]:
        return list(FEATURE_GROUPS.get(spec["feature_set"], []))

    def _validation_matches(
        self,
        recorded: Dict[str, Any],
        recomputed: Dict[str, Any],
        name: str,
    ) -> bool:
        left = recorded.get(name) or {}
        right = recomputed.get(name) or {}
        if int(left.get("valid_days") or 0) != int(right.get("valid_days") or 0):
            return False
        if int(left.get("excluded") or 0) != int(right.get("excluded") or 0):
            return False
        left_auc = left.get("auc")
        right_auc = right.get("auc")
        if left_auc is None or right_auc is None:
            return left_auc is None and right_auc is None
        return abs(float(left_auc) - float(right_auc)) <= AUC_TOLERANCE

    def holdout_audit(
        self,
        spec: Dict[str, Any],
        seeds: Sequence[int] = (42, 2024, 7),
    ) -> Dict[str, Any]:
        """多 seed 重跑锁定配置的 holdout，输出稳健性摘要（不改变官方判决）。"""
        split = self.backend.prepared_split()
        rows: List[Dict[str, Any]] = []
        for seed in seeds:
            raw = run_holdout_test(spec, split, seed=int(seed), include_trust=True)
            test = raw.get("test") or {}
            rows.append(
                {
                    "seed": int(seed),
                    "auc": test.get("auc"),
                    "spearman": test.get("spearman"),
                    "brier": test.get("brier"),
                    "pr_auc": test.get("pr_auc"),
                    "auc_ci": test.get("auc_ci"),
                    "permutation_p": test.get("permutation_p"),
                }
            )
        aucs = [float(row["auc"]) for row in rows if row.get("auc") is not None]
        if not aucs:
            return {"seeds": rows, "error": "no valid holdout aucs"}
        aucs_sorted = sorted(aucs)
        median_auc = float(statistics.median(aucs_sorted))
        lower_quartile = float(aucs_sorted[len(aucs_sorted) // 4])
        upper_quartile = float(aucs_sorted[(3 * len(aucs_sorted)) // 4])
        return {
            "seeds": rows,
            "auc_median": round(median_auc, 6),
            "auc_iqr": [round(lower_quartile, 6), round(upper_quartile, 6)],
            "auc_min": round(min(aucs_sorted), 6),
            "auc_max": round(max(aucs_sorted), 6),
        }

    def judge(
        self,
        *,
        run_id: str,
        experiment_id: str,
        locked_config: Dict[str, Any],
        test: bool = False,
    ) -> Verdict:
        spec_dict = locked_config.get("spec") if isinstance(locked_config, dict) else {}
        spec = normalize_spec(spec_dict)
        trial = locked_config.get("trial") if isinstance(locked_config, dict) else {}
        trial = trial if isinstance(trial, dict) else {}
        evidence_ids = list(spec.get("evidence_ids") or [])
        if spec["experiment_id"] != experiment_id or trial.get("experiment_id") != experiment_id:
            return self._blocked(
                "experiment ID mismatch",
                experiment_id=experiment_id,
                evidence_ids=evidence_ids,
                test=test,
            )

        if test:
            with self._mutex:
                if self._locks.get(run_id) != spec:
                    return self._blocked(
                        "configuration is not locked",
                        experiment_id=experiment_id,
                        evidence_ids=evidence_ids,
                        test=True,
                    )
                if run_id in self._tested:
                    return self._blocked(
                        "holdout already attempted",
                        experiment_id=experiment_id,
                        evidence_ids=evidence_ids,
                        test=True,
                    )
                self._tested.add(run_id)

        task = self.backend.task
        baseline = self.backend.baseline(task)
        if baseline.get("error"):
            return self._blocked(
                str(baseline.get("error")),
                experiment_id=experiment_id,
                evidence_ids=evidence_ids,
                test=test,
            )
        baseline_trial = baseline.get("trial") or {}
        if any(
            not (baseline_trial.get(window) or {}).get("valid_days")
            for window in ("val1", "val2")
        ):
            return self._blocked(
                "baseline lacks valid days",
                experiment_id=experiment_id,
                evidence_ids=evidence_ids,
                test=test,
            )

        split = self.backend.prepared_split()
        checks: List[str] = []
        checks.append("split_fingerprint_ok")
        checks.append("data_hash_ok")

        forbid = [
            column
            for column in self._feature_columns(spec)
            if column in FORBIDDEN_FEATURE_COLUMNS
        ]
        if forbid:
            return self._blocked(
                "feature leakage: %s" % ",".join(forbid),
                experiment_id=experiment_id,
                evidence_ids=evidence_ids,
                test=test,
            )
        checks.append("no_leakage")

        fingerprint = (
            locked_config.get("fingerprint") if isinstance(locked_config, dict) else ""
        )
        if fingerprint:
            expected_fingerprint = experiment_fingerprint(
                normalize_experiment_spec(spec_dict),
                task.data_hash,
                task.seed,
                as_jsonable(task.split),
            )
            if fingerprint != expected_fingerprint:
                return self._blocked(
                    "fingerprint mismatch",
                    experiment_id=experiment_id,
                    evidence_ids=evidence_ids,
                    test=test,
                )
            checks.append("fingerprint_ok")

        if not test:
            if trial.get("error"):
                return self._blocked(
                    str(trial.get("error")),
                    experiment_id=experiment_id,
                    evidence_ids=evidence_ids,
                    test=test,
                )
            recomputed = run_experiment(spec, split, seed=task.seed)
            if recomputed.get("error"):
                return self._blocked(
                    "validation rerun failed: %s" % recomputed["error"],
                    experiment_id=experiment_id,
                    evidence_ids=evidence_ids,
                    test=test,
                )
            mismatches = []
            for window in ("val1", "val2"):
                if not self._validation_matches(trial, recomputed, window):
                    mismatches.append(window)
            if mismatches:
                return self._blocked(
                    "validation result mismatch: %s" % ",".join(mismatches),
                    experiment_id=experiment_id,
                    evidence_ids=evidence_ids,
                    test=test,
                )
            checks.append("validation_rerun_ok")

        if test:
            result = run_holdout_test(spec, split, seed=task.seed)
            metric = result.get("test") or {}
            if result.get("error") or not metric.get("valid_days"):
                return self._blocked(
                    result.get("error") or "holdout lacks valid days",
                    experiment_id=experiment_id,
                    evidence_ids=evidence_ids,
                    test=True,
                )
            # Independent scoring is reported without inventing a hypothesis decision.
            verdict_name = "inconclusive"
            checks.append("holdout_ok")
        else:
            val1 = trial.get("val1") or {}
            val2 = trial.get("val2") or {}
            if not val1.get("valid_days") or not val2.get("valid_days"):
                return self._blocked(
                    "validation window lacks valid days",
                    experiment_id=experiment_id,
                    evidence_ids=evidence_ids,
                    test=test,
                )
            val1_auc = float(val1.get("auc") or 0.0)
            val2_auc = float(val2.get("auc") or 0.0)
            merged = (val1_auc + val2_auc) / 2.0
            score = float(baseline.get("merged_auc") or 0.0)
            metric = {"val1_auc": val1_auc, "val2_auc": val2_auc, "merged_auc": merged}
            if min(val1_auc, val2_auc) < score - 0.01:
                verdict_name = "refuted"
            elif merged > score:
                verdict_name = "supported"
            else:
                verdict_name = "inconclusive"
            checks.append("rule_verdict")

        artifacts = {
            "data_hash": baseline.get("data_hash"),
            "split_fingerprint": baseline.get("split_fingerprint"),
            "spec": spec,
            "trial": trial,
            "metric": metric,
        }
        digest = sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest()
        output = Verdict(
            verdict_name,
            metric,
            checks=checks,
            evidence_ids=evidence_ids,
            artifacts_hash=f"sha256:{digest}",
            test=test,
        )
        self._write_verdict(
            run_id,
            {
                "run_id": run_id,
                "experiment_id": experiment_id,
                "test": test,
                "verdict": output.verdict,
                "metric": output.metric,
                "checks": output.checks,
                "evidence_ids": output.evidence_ids,
                "artifacts_hash": output.artifacts_hash,
            },
        )
        return output
