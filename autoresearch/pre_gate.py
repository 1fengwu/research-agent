"""AutoResearch 角色 C：数据前置门（pre_gate）。

固定基线 + 前 3 个候选在两个验证窗各跑一次，输出日均 AUC、有效天数、
排除样本数和配对 bootstrap 95% CI。没有可证伪的研究空间时返回失败，
让 A 在启动主循环前直接拒绝开工。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import TaskSpec, load_task_spec
from .experiment import ResearchBackend, baseline_spec, run_experiment
from .research_pv import ResearchPVPlanner

PRE_GATE_CANDIDATES = 3
BOOTSTRAP_REPS = 2000
WINDOW_DROP_TOLERANCE = 0.01


def paired_bootstrap_ci(
    base_days: List[Dict[str, Any]],
    candidate_days: List[Dict[str, Any]],
    *,
    seed: int = 42,
    reps: int = BOOTSTRAP_REPS,
) -> Dict[str, Any]:
    """按同一交易日配对，bootstrap 日均 AUC 差值的 95% CI。"""
    by_date = {str(item.get("date")): item for item in candidate_days}
    pairs = []
    for item in base_days:
        candidate = by_date.get(str(item.get("date")))
        if candidate is None:
            continue
        base_auc = item.get("auc")
        candidate_auc = candidate.get("auc")
        if base_auc is None or candidate_auc is None:
            continue
        pairs.append((float(base_auc), float(candidate_auc)))
    if not pairs:
        return {
            "delta": None,
            "lower": None,
            "upper": None,
            "pairs": 0,
            "reps": reps,
        }

    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(reps):
        total = 0.0
        for _index in range(n):
            base_auc, candidate_auc = pairs[rng.randrange(n)]
            total += candidate_auc - base_auc
        diffs.append(total / n)
    diffs.sort()
    lower = diffs[max(0, int(round(reps * 0.025)) - 1)]
    upper = diffs[min(reps - 1, int(round(reps * 0.975)) - 1)]
    delta = sum(candidate - base for base, candidate in pairs) / n
    return {
        "delta": round(delta, 6),
        "lower": round(lower, 6),
        "upper": round(upper, 6),
        "pairs": n,
        "reps": reps,
    }


def _window_daily(payload: Dict[str, Any], window: str) -> List[Dict[str, Any]]:
    return list((payload.get(window) or {}).get("daily") or [])


def _window_summary(payload: Dict[str, Any], window: str) -> Dict[str, Any]:
    data = payload.get(window) or {}
    return {
        "auc": data.get("auc"),
        "valid_days": int(data.get("valid_days") or 0),
        "excluded": int(data.get("excluded") or 0),
    }


def _paired_merged_ci(
    base_payload: Dict[str, Any],
    candidate_payload: Dict[str, Any],
    *,
    seed: int,
    reps: int,
) -> Dict[str, Any]:
    base_days = _window_daily(base_payload, "val1") + _window_daily(base_payload, "val2")
    candidate_days = _window_daily(candidate_payload, "val1") + _window_daily(
        candidate_payload, "val2"
    )
    return paired_bootstrap_ci(base_days, candidate_days, seed=seed, reps=reps)


def _candidate_row(
    task: TaskSpec,
    baseline_payload: Dict[str, Any],
    baseline_auc: float,
    candidate_payload: Dict[str, Any],
    *,
    seed: int,
    reps: int,
) -> Dict[str, Any]:
    val1 = candidate_payload.get("val1") or {}
    val2 = candidate_payload.get("val2") or {}
    merged = candidate_payload.get("merged_auc")
    merged = float(merged) if merged is not None else 0.0
    val1_auc = float(val1.get("auc") or 0.0)
    val2_auc = float(val2.get("auc") or 0.0)
    ci_val1 = paired_bootstrap_ci(
        _window_daily(baseline_payload, "val1"),
        _window_daily(candidate_payload, "val1"),
        seed=seed,
        reps=reps,
    )
    ci_val2 = paired_bootstrap_ci(
        _window_daily(baseline_payload, "val2"),
        _window_daily(candidate_payload, "val2"),
        seed=seed,
        reps=reps,
    )
    ci_merged = _paired_merged_ci(
        baseline_payload,
        candidate_payload,
        seed=seed,
        reps=reps,
    )

    reasons: List[str] = []
    if merged <= baseline_auc:
        reasons.append(f"merged_auc_not_above_baseline({merged:.6f}<={baseline_auc:.6f})")
    if ci_merged.get("lower") is not None and float(ci_merged["lower"]) <= 0:
        reasons.append(f"merged_ci_lower_not_positive({ci_merged['lower']:.6f})")
    if baseline_auc > 0:
        if val1_auc < baseline_auc - WINDOW_DROP_TOLERANCE:
            reasons.append(
                f"val1_drop({val1_auc:.6f}<{baseline_auc - WINDOW_DROP_TOLERANCE:.6f})"
            )
        if val2_auc < baseline_auc - WINDOW_DROP_TOLERANCE:
            reasons.append(
                f"val2_drop({val2_auc:.6f}<{baseline_auc - WINDOW_DROP_TOLERANCE:.6f})"
            )

    return {
        "experiment_id": candidate_payload.get("experiment_id"),
        "feature_set": candidate_payload.get("spec", {}).get("feature_set"),
        "model": candidate_payload.get("spec", {}).get("model"),
        "merged_auc": round(merged, 6),
        "val1": _window_summary(candidate_payload, "val1"),
        "val2": _window_summary(candidate_payload, "val2"),
        "delta": round(merged - baseline_auc, 6),
        "paired_ci": {
            "val1": ci_val1,
            "val2": ci_val2,
            "merged": ci_merged,
        },
        "passed": not reasons,
        "reasons": reasons,
    }


def run_pre_gate(
    task: TaskSpec,
    *,
    reps: int = BOOTSTRAP_REPS,
) -> Dict[str, Any]:
    """执行前置门并返回可落盘的 pre_gate.json 结构。"""
    backend = ResearchBackend(task)
    baseline = backend.baseline(task)
    result: Dict[str, Any] = {
        "schema_version": "autoresearch-pre-gate-v1",
        "task_id": task.task_id,
        "data_path": task.data_path,
        "data_hash": baseline.get("data_hash") or task.data_hash or "",
        "split_fingerprint": baseline.get("split_fingerprint") or "",
        "seed": task.seed,
        "reps": reps,
        "conclusion": "fail",
        "passed": False,
        "reasons": [],
        "baseline": None,
        "candidates": [],
    }
    if baseline.get("error") or baseline.get("merged_auc") is None:
        result["reasons"].append(
            "baseline_failed:%s" % (baseline.get("error") or "merged_auc_missing")
        )
        return result

    split = backend.prepared_split()
    baseline_payload = run_experiment(
        baseline_spec("baseline"),
        split,
        seed=task.seed,
        include_daily=True,
    )
    if baseline_payload.get("error"):
        result["reasons"].append("baseline_daily_failed:%s" % baseline_payload["error"])
        return result

    baseline_auc = float(baseline["merged_auc"])
    result["baseline"] = {
        "experiment_id": "baseline",
        "merged_auc": baseline_auc,
        "val1": _window_summary(baseline_payload, "val1"),
        "val2": _window_summary(baseline_payload, "val2"),
    }

    planner = ResearchPVPlanner()
    specs = planner.plan(task, {"attempted": []}, "pre_gate")[:PRE_GATE_CANDIDATES]
    for spec in specs:
        candidate_payload = run_experiment(
            {
                "experiment_id": spec.experiment_id,
                "hypothesis_id": spec.hypothesis_id,
                "evidence_ids": list(spec.evidence_ids),
                "feature_set": spec.feature_set,
                "model": spec.model,
                "params": dict(spec.params),
                "missing": spec.missing,
                "scaling": spec.scaling,
            },
            split,
            seed=task.seed,
            include_daily=True,
        )
        if candidate_payload.get("error"):
            result["candidates"].append(
                {
                    "experiment_id": spec.experiment_id,
                    "feature_set": spec.feature_set,
                    "model": spec.model,
                    "error": candidate_payload["error"],
                    "passed": False,
                    "reasons": ["experiment_failed:%s" % candidate_payload["error"]],
                }
            )
            result["reasons"].append(
                "candidate_failed:%s" % candidate_payload["error"]
            )
            continue
        result["candidates"].append(
            _candidate_row(
                task,
                baseline_payload,
                baseline_auc,
                candidate_payload,
                seed=task.seed,
                reps=reps,
            )
        )

    passed_rows = [row for row in result["candidates"] if row.get("passed")]
    if passed_rows:
        best = max(passed_rows, key=lambda row: float(row.get("merged_auc") or 0.0))
        result["passed"] = True
        result["conclusion"] = "pass"
        result["reasons"].append(
            "best_candidate=%s merged_auc=%.6f" % (
                best.get("experiment_id"),
                best.get("merged_auc") or 0.0,
            )
        )
    if not result["reasons"]:
        result["reasons"].append("no_candidate_passed")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="panel CSV")
    parser.add_argument("--out", type=Path, default=Path(".arc"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args(argv)

    task = load_task_spec(
        {
            "task_id": args.data.stem,
            "data_path": str(args.data.resolve()),
            "seed": args.seed,
            "budget": {"candidates": PRE_GATE_CANDIDATES},
        }
    )
    result = run_pre_gate(task, reps=args.reps)
    output = args.out / "pre_gate.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
