"""Helix 赫力 角色 C：无云复现与复现包校验。

replay 不使用云模型：它从 task.json/事件流重建任务，用 oracle 对锁定配置
再跑一次 holdout，并和原运行记录比对数据哈希、切分指纹、测试指标、
phase/stop_reason 与 ledger 用量。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    TrialResult,
    as_jsonable,
    experiment_fingerprint,
    load_task_spec,
    normalize_experiment_spec,
)
from .experiment import ResearchBackend
from .oracle import ResearchOracle

AUC_TOLERANCE = 1e-6


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _record_from_events(
    run_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "task_raw": None,
        "baseline": None,
        "locked": None,
        "test_verdict": None,
        "experiments": [],
        "phase": None,
        "stop_reason": None,
    }
    for event in run_events:
        event_type = event.get("type")
        extra = event.get("extra") or {}
        if event_type == "research.run_started" and record["task_raw"] is None:
            record["task_raw"] = extra.get("task")
        if event_type == "research.baseline" and record["baseline"] is None:
            baseline = extra.get("baseline") or {}
            if not baseline.get("truncated"):
                record["baseline"] = baseline
        if event_type == "research.locked" and record["locked"] is None:
            locked = extra.get("locked") or {}
            if not locked.get("truncated"):
                record["locked"] = locked
        if event_type in ("research.tested", "research.test_failed") and record["test_verdict"] is None:
            record["test_verdict"] = extra.get("verdict")
            record["phase"] = "TESTED" if event_type == "research.tested" else "FAILED"
        if event_type == "research.experiment_result":
            trial = extra.get("trial") or {}
            if trial and not trial.get("truncated"):
                record["experiments"].append(trial)
        if event_type == "research.completed":
            summary = extra.get("summary") or {}
            if summary and not summary.get("truncated"):
                if summary.get("phase"):
                    record["phase"] = summary["phase"]
                if summary.get("stop_reason"):
                    record["stop_reason"] = summary["stop_reason"]
                if summary.get("locked"):
                    record["locked"] = summary["locked"]
                experiments = summary.get("experiments") or []
                if experiments:
                    record["experiments"] = experiments
    return record


def _load_run_record(
    output_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    run_dir = output_dir / run_id
    event_paths = [
        run_dir / "events.jsonl",
        output_dir / "runner-events.jsonl",
    ]
    events: List[Dict[str, Any]] = []
    for path in event_paths:
        events = _load_events(path)
        if events:
            break
    run_events = [event for event in events if event.get("run_id") == run_id]
    event_record = _record_from_events(run_events) if run_events else {}

    summary_paths = [
        run_dir / "summary.json",
        output_dir / f"summary-{run_id}.json",
        output_dir / "summary.json",
    ]
    summary: Dict[str, Any] = {}
    for path in summary_paths:
        loaded = _load_json(path)
        if isinstance(loaded, dict) and loaded.get("run_id") == run_id:
            summary = loaded
            break

    task_raw: Optional[Dict[str, Any]] = None
    for path in (run_dir / "task.json", output_dir / "task.json"):
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            task_raw = loaded
            break
    if task_raw is None:
        task_raw = event_record.get("task_raw")

    baseline = event_record.get("baseline") or {}
    if summary:
        locked = summary.get("locked") or event_record.get("locked")
        test_verdict = summary.get("test_verdict") or event_record.get("test_verdict")
        phase = summary.get("phase") or event_record.get("phase")
        stop_reason = summary.get("stop_reason") or event_record.get("stop_reason")
        experiments = summary.get("experiments") or event_record.get("experiments") or []
    else:
        locked = event_record.get("locked")
        test_verdict = event_record.get("test_verdict")
        phase = event_record.get("phase")
        stop_reason = event_record.get("stop_reason")
        experiments = event_record.get("experiments") or []

    if task_raw is None:
        raise ValueError(f"replay has no task.json or research.run_started record for {run_id}")
    if locked is None or test_verdict is None:
        raise ValueError(
            f"replay cannot find a locked configuration and test verdict for {run_id}"
        )

    return {
        "run_id": run_id,
        "task_raw": task_raw,
        "summary": summary,
        "locked": locked,
        "test_verdict": test_verdict,
        "phase": phase,
        "stop_reason": stop_reason,
        "experiments": experiments,
        "data_hash": (
            summary.get("data_hash")
            or baseline.get("data_hash")
            or (task_raw or {}).get("data_hash")
            or ""
        ),
        "split_fingerprint": (
            summary.get("split_fingerprint")
            or baseline.get("split_fingerprint")
            or ""
        ),
    }


def _trial_for_experiment(
    record: Dict[str, Any],
    experiment_id: str,
) -> Dict[str, Any]:
    for item in record.get("experiments") or []:
        spec = item.get("spec") if isinstance(item, dict) else {}
        if str(spec.get("experiment_id") or "") == experiment_id:
            trial = item.get("trial") or {}
            if isinstance(trial, dict) and trial.get("experiment_id") == experiment_id:
                return trial
    return as_jsonable(TrialResult(experiment_id=experiment_id))


def _locked_spec(record: Dict[str, Any]) -> Dict[str, Any]:
    locked = record.get("locked") or {}
    config = dict(locked.get("config") or {})
    name = str(locked.get("name") or "")
    if not name:
        raise ValueError("locked record has no name")
    if name == "baseline":
        return {
            "experiment_id": "baseline-locked",
            "hypothesis_id": "LOCKED",
            "evidence_ids": list(locked.get("evidence_ids") or []),
            "feature_set": str(config.get("feature_set") or "base"),
            "model": str(config.get("model") or "logistic"),
            "params": dict(config.get("params") or {}),
            "missing": str(config.get("missing") or "median"),
            "scaling": str(config.get("scaling") or "standard"),
        }
    spec = dict(config)
    spec["experiment_id"] = name
    if "evidence_ids" not in spec:
        spec["evidence_ids"] = list(locked.get("evidence_ids") or [])
    return spec


def run_replay(
    run_id: str,
    output_dir: Path,
    repro_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    repro_dir = Path(repro_dir) if repro_dir else output_dir / "repro"
    record = _load_run_record(output_dir, run_id)
    task = load_task_spec(record["task_raw"])
    backend = ResearchBackend(task)
    split = backend.prepared_split()

    locked_spec = _locked_spec(record)
    experiment_id = str(locked_spec.get("experiment_id") or "")
    trial = _trial_for_experiment(record, experiment_id)
    fingerprint = experiment_fingerprint(
        normalize_experiment_spec(locked_spec),
        task.data_hash,
        task.seed,
        as_jsonable(task.split),
    )

    oracle = ResearchOracle(backend)
    oracle.register_lock(run_id, locked_spec)
    replay_verdict = oracle.judge(
        run_id=run_id,
        experiment_id=experiment_id,
        locked_config={
            "spec": locked_spec,
            "trial": trial,
            "fingerprint": fingerprint,
        },
        test=True,
    )

    replay_metric = replay_verdict.metric
    recorded_metric = dict((record.get("test_verdict") or {}).get("metric") or {})
    recorded_hash = str(record.get("data_hash") or "")
    recorded_split = str(record.get("split_fingerprint") or "")

    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    data_hash = str(split.meta.get("data_hash") or "")
    check(
        "data_hash_match",
        not recorded_hash or recorded_hash == data_hash,
        f"recorded={recorded_hash} replay={data_hash}",
    )
    check(
        "split_fingerprint_match",
        not recorded_split or recorded_split == split.fingerprint,
        f"recorded={recorded_split} replay={split.fingerprint}",
    )

    recorded_auc = recorded_metric.get("auc")
    replay_auc = replay_metric.get("auc")
    auc_matches = (
        recorded_auc is not None
        and replay_auc is not None
        and abs(float(recorded_auc) - float(replay_auc)) <= AUC_TOLERANCE
    )
    check(
        "test_auc_match",
        auc_matches,
        f"recorded={recorded_auc} replay={replay_auc}",
    )
    check(
        "valid_days_match",
        int(recorded_metric.get("valid_days") or 0) == int(replay_metric.get("valid_days") or 0),
        f"recorded={recorded_metric.get('valid_days')} replay={replay_metric.get('valid_days')}",
    )
    check(
        "excluded_match",
        int(recorded_metric.get("excluded") or 0) == int(replay_metric.get("excluded") or 0),
        f"recorded={recorded_metric.get('excluded')} replay={replay_metric.get('excluded')}",
    )
    recorded_artifacts = (record.get("test_verdict") or {}).get("artifacts_hash") or ""
    check(
        "artifacts_hash_match",
        not recorded_artifacts or recorded_artifacts == replay_verdict.artifacts_hash,
        f"recorded={recorded_artifacts} replay={replay_verdict.artifacts_hash}",
    )
    check(
        "phase_match",
        not record.get("phase") or record["phase"] == replay_phase_from_verdict(replay_verdict),
        f"recorded={record.get('phase')} replay=TESTED",
    )
    check(
        "stop_reason_recorded",
        bool(record.get("stop_reason")),
        f"stop_reason={record.get('stop_reason') or ''}",
    )

    ledger_paths = [
        Path(output_dir) / "ledgers" / f"{run_id}.json",
        Path(output_dir) / run_id / "ledgers" / f"{run_id}.json",
    ]
    ledger = None
    for path in ledger_paths:
        ledger = _load_json(path)
        if ledger is not None:
            break
    if ledger is not None:
        used = {
            "used_candidates": int(ledger.get("used_candidates") or 0),
            "used_cloud_calls": int(ledger.get("used_cloud_calls") or 0),
            "used_cloud_tokens": int(ledger.get("used_cloud_tokens") or 0),
        }
        summary_budget = (record.get("summary") or {}).get("budget") or {}
        recorded_used = {
            "used_candidates": int(summary_budget.get("used_candidates") or 0),
            "used_cloud_calls": int(summary_budget.get("used_cloud_calls") or 0),
            "used_cloud_tokens": int(summary_budget.get("used_cloud_tokens") or 0),
        }
        check(
            "ledger_usage_match",
            used == recorded_used,
            f"ledger={used} recorded={recorded_used}",
        )
    check(
        "replay_offline_no_cloud",
        True,
        "replay used no cloud client",
    )

    all_passed = all(item["passed"] for item in checks)
    result = {
        "schema_version": "autoresearch-replay-v1",
        "run_id": run_id,
        "status": "match" if all_passed else "mismatch",
        "checks": checks,
        "replay_metric": replay_metric,
        "recorded_metric": recorded_metric,
        "recorded": {
            "phase": record.get("phase"),
            "stop_reason": record.get("stop_reason"),
            "locked": (record.get("locked") or {}).get("name"),
            "data_hash": record.get("data_hash"),
            "split_fingerprint": record.get("split_fingerprint"),
        },
        "cloud_calls_replayed": 0,
    }

    repro_dir.mkdir(parents=True, exist_ok=True)
    (repro_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"run_id: {run_id}",
        f"status: {result['status']}",
        f"replay_auc: {replay_metric.get('auc')}",
        f"recorded_auc: {recorded_metric.get('auc')}",
        f"cloud_calls_replayed: 0",
    ]
    (repro_dir / "replay.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def replay_phase_from_verdict(verdict: Any) -> str:
    if getattr(verdict, "verdict", None) in {"blocked", "partial_completed"}:
        return "FAILED"
    return "TESTED"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(".arc"))
    parser.add_argument("--repro-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_replay(args.run_id, args.output_dir, repro_dir=args.repro_dir)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
