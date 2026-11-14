"""AutoResearch 角色 B：数据、实验、规划器与 A↔B 端到端测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

from autoresearch.agent_loop import EventWriter, ResearchOrchestrator
from autoresearch.budget_gate import BudgetGate
from autoresearch.contracts import (
    ExperimentSpec,
    TaskSpec,
    TrialResult,
    Verdict,
    load_task_spec,
)
from autoresearch.data import (
    canonical_panel_hash,
    compute_labels,
    make_split,
    read_panel,
    split_document,
)
from autoresearch.experiment import ResearchBackend
from autoresearch.llm_router import NullCloudClient, ResearchRouter
from autoresearch.oracle import ResearchOracle
from autoresearch.research_pv import ResearchPVPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_CSV = REPO_ROOT / "data" / "demo_panel.csv"


def make_task(*, candidates: int = 1) -> TaskSpec:
    return load_task_spec(
        {
            "task_id": "task-b",
            "data_path": str(DEMO_CSV),
            "label_horizon": 5,
            "split": {"train": 0.6, "val_windows": 2, "test": 0.2, "gap_days": 5},
            "seed": 42,
            "budget": {
                "candidates": candidates,
                "cloud_calls": 3,
                "cloud_tokens": 6000,
                "wall_seconds": 180,
            },
        }
    )


class BTestOracle:
    def __init__(self, baseline_auc: float = 0.62) -> None:
        self.baseline_auc = baseline_auc
        self.calls = 0
        self.test_calls = 0

    def judge(
        self,
        *,
        run_id: str,
        experiment_id: str,
        locked_config: Dict[str, Any],
        test: bool = False,
    ) -> Verdict:
        del run_id, experiment_id
        self.calls += 1
        trial = locked_config.get("trial") or {}
        if test:
            self.test_calls += 1
            return Verdict(
                verdict="supported",
                metric={"auc": 0.60},
                checks=["split_hash_ok", "label_maturity_ok", "no_leakage"],
                test=True,
            )
        if trial.get("error"):
            return Verdict(verdict="blocked", metric={"error": trial["error"]}, test=test)
        v1 = float((trial.get("val1") or {}).get("auc", 0.0))
        v2 = float((trial.get("val2") or {}).get("auc", 0.0))
        if (
            self.baseline_auc > 0
            and (v1 < self.baseline_auc - 0.01 or v2 < self.baseline_auc - 0.01)
        ):
            verdict = "refuted"
        elif v1 > 0 and v2 > 0 and (v1 + v2) / 2.0 > self.baseline_auc + 0.01:
            verdict = "supported"
        else:
            verdict = "inconclusive"
        return Verdict(
            verdict=verdict,
            metric={"val1": v1, "val2": v2},
            checks=["split_hash_ok"],
            test=test,
        )


class TestBData(unittest.TestCase):
    def test_data_profile_and_split_are_fixed(self) -> None:
        panel = read_panel(DEMO_CSV)
        self.assertFalse(panel.duplicated(subset=["date", "symbol"]).any())
        labelled = compute_labels(panel, horizon=5)
        self.assertEqual(int(labelled["label_excluded"].sum()), 5 * 14)

        data_hash = canonical_panel_hash(panel)
        split = make_split(
            labelled,
            train_ratio=0.6,
            val_windows=2,
            test_ratio=0.2,
            gap_days=5,
            seed=42,
            data_hash=data_hash,
        )
        split2 = make_split(
            labelled,
            train_ratio=0.6,
            val_windows=2,
            test_ratio=0.2,
            gap_days=5,
            seed=42,
            data_hash=data_hash,
        )
        self.assertEqual(split.fingerprint, split2.fingerprint)
        doc = split_document(split)
        self.assertEqual(set(doc["segments"]), {"train", "val1", "val2", "test"})
        self.assertTrue(split.meta["excluded_tail_rows"] > 0)


class TestBBackend(unittest.TestCase):
    def test_backend_serves_ports(self) -> None:
        task = make_task(candidates=2)
        backend = ResearchBackend(task)
        profile = backend.describe_data(task)
        self.assertEqual(profile["symbols"], 14)
        self.assertEqual(profile["date_range"], [profile["date_start"], profile["date_end"]])
        self.assertFalse(profile["data_anomaly"])

        baseline = backend.baseline(task)
        self.assertGreater(float(baseline["merged_auc"]), 0.5)
        self.assertEqual(baseline["config"]["missing"], "median")
        self.assertEqual(baseline["config"]["scaling"], "standard")
        self.assertIsNone(baseline["error"])

        spec = ExperimentSpec(
            experiment_id="exp-b001",
            hypothesis_id="volume",
            feature_set="+volume",
            model="logistic",
            missing="median",
            scaling="standard",
        )
        trial = backend.run_candidate(spec)
        self.assertIsInstance(trial, TrialResult)
        self.assertIsNone(trial.error)
        self.assertGreaterEqual(trial.val1.valid_days, 1)
        self.assertGreaterEqual(trial.val2.valid_days, 1)


class TestBPlanner(unittest.TestCase):
    def test_plan_and_interpret(self) -> None:
        task = make_task(candidates=3)
        planner = ResearchPVPlanner()
        view: Dict[str, Any] = {"attempted": []}
        specs = planner.plan(task, view, "first")
        self.assertEqual(len(specs), 3)
        self.assertTrue(all(isinstance(s, ExperimentSpec) for s in specs))
        first = specs[0]
        signals = planner.interpret(
            {
                "trial": {
                    "experiment_id": first.experiment_id,
                    "val1": {"auc": 0.55},
                    "val2": {"auc": 0.66},
                },
                "verdict": {"verdict": "refuted"},
            },
            {"baseline": {"merged_auc": 0.62}},
        )
        self.assertTrue(signals)
        self.assertIn(first.hypothesis_id, signals[0]["refute"])


class TestBEndToEnd(unittest.TestCase):
    def test_real_oracle_only_scores_locked_holdout_once(self) -> None:
        backend = ResearchBackend(make_task())
        oracle = ResearchOracle(backend)
        spec = dict(backend.baseline(backend.task)["config"], experiment_id="baseline")
        request = {
            "run_id": "test-run",
            "experiment_id": "baseline",
            "locked_config": {"spec": spec, "trial": {"experiment_id": "baseline"}},
            "test": True,
        }
        self.assertEqual(oracle.judge(**request).verdict, "blocked")
        oracle.register_lock("test-run", spec)
        spec["scaling"] = "none"
        self.assertEqual(oracle.judge(**request).verdict, "blocked")
        spec["scaling"] = "standard"
        result = oracle.judge(**request)
        self.assertGreater(result.metric["valid_days"], 0)
        self.assertTrue(result.artifacts_hash.startswith("sha256:"))
        self.assertEqual(oracle.judge(**request).verdict, "blocked")

    def test_a_orchestrator_runs_with_real_backend(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(candidates=1)
            backend = ResearchBackend(task)
            planner = ResearchPVPlanner()
            ledger = BudgetGate(task.budget)
            router = ResearchRouter(ledger, client=NullCloudClient())
            writer = EventWriter(os.path.join(folder, "events.jsonl"))
            orchestrator = ResearchOrchestrator(
                task,
                data=backend,
                oracle=BTestOracle(),
                planner=planner,
                router=router,
                ledger=ledger,
                writer=writer,
                run_ledger_dir=folder,
            )
            summary = orchestrator.run()
            self.assertTrue(summary["tested"])
            self.assertEqual(summary["phase"], "TESTED")
            self.assertEqual(len(summary["experiments"]), 1)
            self.assertTrue(summary["budget"]["used_candidates"] >= 1)


if __name__ == "__main__":
    unittest.main()
