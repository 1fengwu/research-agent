"""Helix 赫力 角色 C：评测与复现模块测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict

from autoresearch.agent_loop import EventWriter, ResearchOrchestrator
from autoresearch.budget_gate import BudgetGate
from autoresearch.contracts import (
    TaskSpec,
    TrialResult,
    TrialWindowResult,
    as_jsonable,
    experiment_fingerprint,
    load_task_spec,
    normalize_experiment_spec,
    normalize_trial_result,
    validate_run_event,
)
from autoresearch.evaluate import OfflineCloudClient, run_arm, run_evaluate
from autoresearch.experiment import ResearchBackend
from autoresearch.llm_router import NullCloudClient, ResearchRouter
from autoresearch.oracle import ResearchOracle
from autoresearch.pre_gate import paired_bootstrap_ci, run_pre_gate
from autoresearch.replay import run_replay
from autoresearch.research_pv import ResearchPVPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = REPO_ROOT / "data" / "demo_panel.csv"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def make_task(
    *,
    candidates: int = 2,
    cloud_calls: int = 2,
    cloud_tokens: int = 6000,
    seed: int = 42,
) -> TaskSpec:
    return load_task_spec(
        {
            "task_id": "task-c",
            "data_path": str(DEMO_CSV),
            "label_horizon": 5,
            "split": {"train": 0.6, "val_windows": 2, "test": 0.2, "gap_days": 5},
            "seed": seed,
            "budget": {
                "candidates": candidates,
                "cloud_calls": cloud_calls,
                "cloud_tokens": cloud_tokens,
            },
        }
    )


def run_local(task: TaskSpec, folder: Path) -> tuple[str, Dict[str, Any]]:
    run_id = f"R-{uuid.uuid4().hex[:12]}"
    backend = ResearchBackend(task)
    ledger = BudgetGate(task.budget)
    oracle = ResearchOracle(backend, output_dir=folder)
    planner = ResearchPVPlanner()
    router = ResearchRouter(ledger, client=NullCloudClient())
    writer = EventWriter(folder / "runner-events.jsonl")
    summary = ResearchOrchestrator(
        task,
        data=backend,
        oracle=oracle,
        planner=planner,
        router=router,
        ledger=ledger,
        writer=writer,
        run_id=run_id,
        run_ledger_dir=folder / "ledgers",
    ).run()
    prepared = backend.prepared_split()
    summary["data_hash"] = prepared.meta.get("data_hash") or ""
    summary["split_fingerprint"] = prepared.fingerprint
    (folder / "task.json").write_text(
        json.dumps(as_jsonable(task), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (folder / f"summary-{run_id}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_id, summary


def _dispatch_invalid(kind: str, raw: Dict[str, Any]) -> None:
    if kind == "task":
        load_task_spec(raw)
    elif kind == "experiment":
        normalize_experiment_spec(raw)
    elif kind == "trial":
        normalize_trial_result(raw)
    elif kind == "event":
        validate_run_event(raw)
    else:
        raise AssertionError(f"unknown invalid contract kind {kind}")


class ContractsTest(unittest.TestCase):
    def test_valid_and_invalid_contracts(self) -> None:
        valid = json.loads((FIXTURES / "valid_contracts.json").read_text(encoding="utf-8"))
        task = load_task_spec(valid["task"])
        self.assertGreater(task.budget.candidates, 0)
        normalize_experiment_spec(valid["experiment"])
        trial = normalize_trial_result(valid["trial"])
        self.assertEqual(trial.experiment_id, "exp-001")
        validate_run_event(valid["event"])

        invalid = json.loads((FIXTURES / "invalid_contracts.json").read_text(encoding="utf-8"))
        for item in invalid:
            with self.subTest(reason=item["reason"]):
                with self.assertRaises(ValueError):
                    _dispatch_invalid(item["kind"], item["raw"])


class PreGateTest(unittest.TestCase):
    def test_pre_gate_shape_and_paired_ci(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as folder:
            result = run_pre_gate(make_task(candidates=3), reps=200)
            self.assertEqual(result["schema_version"], "autoresearch-pre-gate-v1")
            self.assertIsNotNone(result["baseline"])
            self.assertEqual(len(result["candidates"]), 3)
            for row in result["candidates"]:
                ci = (row.get("paired_ci") or {}).get("merged") or {}
                self.assertIn("lower", ci)
                self.assertIn("pairs", ci)

            first = [
                {"date": "2022-01-03", "auc": 0.60},
                {"date": "2022-01-04", "auc": 0.58},
            ]
            second = [
                {"date": "2022-01-03", "auc": 0.62},
                {"date": "2022-01-04", "auc": 0.60},
            ]
            ci_a = paired_bootstrap_ci(first, second, seed=42, reps=200)
            ci_b = paired_bootstrap_ci(first, second, seed=42, reps=200)
            self.assertEqual(ci_a, ci_b)
            self.assertGreater(float(ci_a["delta"]), 0.0)


class OracleAdversarialTest(unittest.TestCase):
    def _locked_config(self, spec: Dict[str, Any], trial: TrialResult) -> Dict[str, Any]:
        task = make_task()
        fingerprint = experiment_fingerprint(
            normalize_experiment_spec(spec),
            task.data_hash,
            task.seed,
            as_jsonable(task.split),
        )
        return {
            "spec": spec,
            "trial": as_jsonable(trial),
            "fingerprint": fingerprint,
        }

    def test_oracle_rejects_fake_improvement_and_guards_holdout(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as folder:
            task = make_task(candidates=1)
            backend = ResearchBackend(task)
            planner = ResearchPVPlanner()
            spec = planner.plan(task, {"attempted": []}, "first")[0]
            spec_dict = as_jsonable(spec)
            trial = backend.run_candidate(spec)
            oracle = ResearchOracle(backend, output_dir=Path(folder))

            verdict = oracle.judge(
                run_id="run-oracle",
                experiment_id=spec.experiment_id,
                locked_config=self._locked_config(spec_dict, trial),
            )
            self.assertIn(verdict.verdict, {"supported", "refuted", "inconclusive"})
            self.assertEqual(verdict.evidence_ids, list(spec.evidence_ids))
            self.assertTrue(verdict.artifacts_hash.startswith("sha256:"))

            fake = TrialResult(
                experiment_id=spec.experiment_id,
                val1=TrialWindowResult(auc=trial.val1.auc + 0.15, valid_days=trial.val1.valid_days, excluded=trial.val1.excluded),
                val2=TrialWindowResult(auc=trial.val2.auc + 0.15, valid_days=trial.val2.valid_days, excluded=trial.val2.excluded),
            )
            fake_verdict = oracle.judge(
                run_id="run-oracle",
                experiment_id=spec.experiment_id,
                locked_config=self._locked_config(spec_dict, fake),
            )
            self.assertEqual(fake_verdict.verdict, "blocked")
            self.assertIn("validation result mismatch", fake_verdict.metric.get("error", ""))

            before_lock = oracle.judge(
                run_id="run-oracle",
                experiment_id=spec.experiment_id,
                locked_config=self._locked_config(spec_dict, trial),
                test=True,
            )
            self.assertEqual(before_lock.verdict, "blocked")
            oracle.register_lock("run-oracle", spec_dict)
            with self.assertRaises(ValueError):
                oracle.register_lock("run-oracle", spec_dict)

            tested = oracle.judge(
                run_id="run-oracle",
                experiment_id=spec.experiment_id,
                locked_config=self._locked_config(spec_dict, trial),
                test=True,
            )
            self.assertGreater(tested.metric.get("valid_days", 0), 0)
            again = oracle.judge(
                run_id="run-oracle",
                experiment_id=spec.experiment_id,
                locked_config=self._locked_config(spec_dict, trial),
                test=True,
            )
            self.assertEqual(again.verdict, "blocked")
            self.assertTrue((Path(folder) / "verdicts" / "run-oracle.json").exists())


class EvaluateTest(unittest.TestCase):
    def test_fixed_candidates_executes_cloud_selection_and_preserves_test(self) -> None:
        client = OfflineCloudClient(reply={
            "decision": "test momentum first",
            "evidence_ids": ["E00"],
            "actions": [{"action": "select_candidate", "target": "exp-007"}],
        })
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as folder:
            task = make_task()
            record = run_arm(task, Path(folder), "B", cloud_client=client, fixed_candidates=True)
            events = [
                json.loads(line)
                for line in (Path(record["run_dir"]) / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            started = [event["experiment_id"] for event in events if event["type"] == "research.experiment_started"]
            self.assertEqual(started, ["exp-007", "exp-001"])
            self.assertTrue(any(event["decision_reason"] == "candidate_selected" for event in events))
            self.assertTrue(any(event["type"] == "research.escalation" for event in events))
            self.assertTrue(record["summary"]["tested"])

    def test_evaluate_runs_three_arms_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as folder:
            metrics = run_evaluate(make_task(), Path(folder), price=0.02)
            self.assertEqual([row["arm"] for row in metrics["arms"]], ["A", "B", "C"])
            self.assertGreaterEqual(float(metrics["completion_rate"]), 0.0)
            a_row = next(row for row in metrics["arms"] if row["arm"] == "A")
            self.assertEqual(a_row["cloud_calls"], 0)
            b_row = next(row for row in metrics["arms"] if row["arm"] == "B")
            self.assertGreater(b_row["cloud_calls"], 0)
            self.assertEqual(b_row["usage_source"], "offline_simulated")
            for name in ("metrics.csv", "metrics.json", "task.json"):
                self.assertTrue((Path(folder) / name).exists())


class ReplayTest(unittest.TestCase):
    def test_replay_matches_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as folder:
            run_id, _summary = run_local(make_task(), Path(folder))
            result = run_replay(
                run_id,
                Path(folder),
                repro_dir=Path(folder) / "repro",
            )
            self.assertEqual(result["status"], "match")
            self.assertEqual(result["cloud_calls_replayed"], 0)
            self.assertEqual(
                result["replay_metric"].get("auc"),
                result["recorded_metric"].get("auc"),
            )

            summary_path = Path(folder) / f"summary-{run_id}.json"
            tampered = json.loads(summary_path.read_text(encoding="utf-8"))
            tampered["test_verdict"]["metric"]["auc"] = float(
                tampered["test_verdict"]["metric"]["auc"]
            ) + 0.05
            summary_path.write_text(
                json.dumps(tampered, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            bad = run_replay(run_id, Path(folder), repro_dir=Path(folder) / "repro-bad")
            self.assertEqual(bad["status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
