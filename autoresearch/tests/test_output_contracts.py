"""后端输出契约冻结测试：所有前端/API 消费的产物结构。"""

from __future__ import annotations

import unittest
from typing import Any, Dict

from autoresearch.outputs import (
    validate_event_list,
    validate_ledger,
    validate_metrics,
    validate_pre_gate,
    validate_replay,
    validate_summary,
    validate_verdict_record,
    validate_verdicts,
)


def _window(auc: float = 0.63, valid_days: int = 59, excluded: int = 0) -> Dict[str, Any]:
    return {"auc": auc, "valid_days": valid_days, "excluded": excluded}


def _verdict(
    verdict: str = "inconclusive",
    *,
    metric: Dict[str, Any],
    test: bool = False,
) -> Dict[str, Any]:
    return {
        "verdict": verdict,
        "metric": metric,
        "checks": ["ok"],
        "evidence_ids": ["E1"],
        "tokens": 0,
        "seconds": 0.0,
        "artifacts_hash": "sha256:abc",
        "test": test,
    }


def _ledger() -> Dict[str, Any]:
    return {
        "entries": [
            {
                "kind": "candidate",
                "ts": "2026-01-01T00:00:00+00:00",
                "note": "exp-001",
                "snapshot": {"used_candidates": 1},
            },
            {
                "kind": "cloud_call",
                "ts": "2026-01-01T00:00:01+00:00",
                "estimated_input": 200,
                "actual_input": 200,
                "actual_output": 50,
                "charged": 250,
                "decision_note": "trigger=window_conflict",
            },
        ],
        "used_candidates": 1,
        "used_cloud_calls": 1,
        "used_cloud_tokens": 250,
    }


def _summary() -> Dict[str, Any]:
    return {
        "run_id": "R1",
        "task_id": "task-1",
        "phase": "TESTED",
        "stop_reason": "plan_exhausted",
        "tested": True,
        "locked": {"name": "exp-001", "merged_auc": 0.66, "evidence_ids": [], "reason": "best"},
        "test_verdict": _verdict(metric={"auc": 0.63, "valid_days": 59, "excluded": 0}, test=True),
        "budget": {
            "used_candidates": 1,
            "used_cloud_calls": 1,
            "used_cloud_tokens": 250,
        },
        "ledger": _ledger(),
        "elapsed_seconds": 5.0,
        "event_path": ".arc/runner-events.jsonl",
    }


def _event() -> Dict[str, Any]:
    return {
        "type": "research.run_started",
        "phase": "UNSEEN",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "run_id": "R1",
        "decision_reason": "task=task-1",
        "experiment_id": "",
        "tokens": {},
        "remaining": {},
        "extra": {"task": {"task_id": "task-1", "data_path": "data/demo_panel.csv"}},
    }


def _pre_gate() -> Dict[str, Any]:
    return {
        "schema_version": "autoresearch-pre-gate-v1",
        "task_id": "task-1",
        "data_hash": "sha256:data",
        "split_fingerprint": "sha256:split",
        "seed": 42,
        "reps": 300,
        "conclusion": "pass",
        "passed": True,
        "reasons": ["best_candidate=exp-001"],
        "baseline": {
            "experiment_id": "baseline",
            "merged_auc": 0.66,
            "val1": _window(0.65, 46),
            "val2": _window(0.67, 46),
        },
        "candidates": [
            {
                "experiment_id": "exp-001",
                "merged_auc": 0.67,
                "val1": _window(0.66, 46),
                "val2": _window(0.68, 46),
                "paired_ci": {
                    "merged": {
                        "delta": 0.01,
                        "lower": 0.001,
                        "upper": 0.02,
                        "pairs": 92,
                        "reps": 300,
                    }
                },
                "passed": True,
            }
        ],
    }


def _metrics() -> Dict[str, Any]:
    return {
        "schema_version": "autoresearch-metrics-v2",
        "task_id": "task-1",
        "completion_rate": 1.0,
        "arms": [
            {
                "run_id": "R-A-1",
                "arm": "A",
                "completion": 1.0,
                "phase": "TESTED",
                "test_auc": 0.63,
                "locked": "exp-001",
                "stop_reason": "candidate_budget",
                "candidates": 1,
                "cloud_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "charged_tokens": 0,
                "usage_source": "none",
                "cost": None,
                "wall_seconds": 2.0,
                "tokens_saved": 0,
                "cache_hits": 0,
                "local_rules": 0,
                "cloud_calls_avoided": 0,
                "baseline_auc": 0.62,
                "auc_gain_vs_baseline": 0.01,
                "per_10k_token_gain": None,
                "negative_result_rate": 0.0,
                "escalation_attempts": 0,
                "escalation_hit_rate": 0.0,
                "budget_blocked_rate": 0.0,
                "speedup_vs_greedy": 1.0,
                "speedup_vs_local": 1.0,
            }
        ],
    }


def _replay() -> Dict[str, Any]:
    return {
        "schema_version": "autoresearch-replay-v1",
        "run_id": "R1",
        "status": "match",
        "checks": [{"name": "test_auc_match", "passed": True, "detail": "0.63=0.63"}],
        "replay_metric": _window(),
        "recorded_metric": _window(),
        "cloud_calls_replayed": 0,
    }


class OutputContractTest(unittest.TestCase):
    def test_valid_artifacts_pass(self) -> None:
        validate_summary(_summary())
        validate_ledger(_ledger())
        validate_event_list([_event()])
        record = dict(_verdict(metric={"auc": 0.63}, test=True))
        record["run_id"] = "R1"
        record["experiment_id"] = "exp-001"
        validate_verdict_record(record)
        validate_verdicts([record])
        validate_pre_gate(_pre_gate())
        validate_metrics(_metrics())
        validate_replay(_replay())

    def test_summary_requires_run_id(self) -> None:
        summary = _summary()
        del summary["run_id"]
        with self.assertRaises(ValueError):
            validate_summary(summary)

    def test_event_rejects_missing_run_id(self) -> None:
        event = _event()
        del event["run_id"]
        with self.assertRaises(ValueError):
            validate_event_list([event])

    def test_ledger_rejects_unknown_kind(self) -> None:
        ledger = _ledger()
        ledger["entries"][0]["kind"] = "unknown"
        with self.assertRaises(ValueError):
            validate_ledger(ledger)

    def test_pre_gate_requires_ci_lower(self) -> None:
        pre_gate = _pre_gate()
        del pre_gate["candidates"][0]["paired_ci"]["merged"]["lower"]
        with self.assertRaises(ValueError):
            validate_pre_gate(pre_gate)

    def test_metrics_rejects_partial_completion(self) -> None:
        metrics = _metrics()
        metrics["arms"][0]["completion"] = 0.5
        with self.assertRaises(ValueError):
            validate_metrics(metrics)

    def test_replay_requires_status(self) -> None:
        replay = _replay()
        del replay["status"]
        with self.assertRaises(ValueError):
            validate_replay(replay)


if __name__ == "__main__":
    unittest.main()
