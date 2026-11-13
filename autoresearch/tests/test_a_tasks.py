from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch.agent_loop import EventWriter, ResearchOrchestrator
from autoresearch.budget_gate import BudgetGate
from autoresearch.contracts import (
    ExperimentSpec,
    TaskSpec,
    TrialResult,
    TrialWindowResult,
    Verdict,
    load_task_spec,
)
from autoresearch.llm_router import ResearchRouter


class FakeCloudClient:
    def __init__(
        self,
        *,
        reply: Optional[Dict[str, Any]] = None,
        invalid: bool = False,
        error: bool = False,
    ) -> None:
        self.reply = reply or {
            "decision": "switch to volatility feature group",
            "evidence_ids": ["E1"],
            "actions": [{"action": "add_volatility", "target": "features"}],
        }
        self.invalid = invalid
        self.error = error
        self.calls = 0

    def call(
        self,
        summary: Dict[str, Any],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        self.calls += 1
        if self.error:
            raise RuntimeError("cloud down")
        content = "not-json" if self.invalid else json.dumps(self.reply, ensure_ascii=False)
        return {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }


class FakeData:
    def __init__(
        self,
        results: Dict[str, Any],
        *,
        profile: Optional[Dict[str, Any]] = None,
        baseline: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.results = results
        self.profile = profile or {}
        self.baseline_data = baseline or {
            "merged_auc": 0.60,
            "config": {"model": "logistic", "feature_set": "base"},
        }
        self.calls: List[str] = []

    def describe_data(self, task: TaskSpec) -> Dict[str, Any]:
        return dict(self.profile)

    def baseline(self, task: TaskSpec) -> Dict[str, Any]:
        return dict(self.baseline_data)

    def run_candidate(self, spec: ExperimentSpec) -> TrialResult:
        self.calls.append(spec.experiment_id)
        item = self.results.get(spec.experiment_id)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, TrialResult):
            return item
        if item is None:
            return TrialResult(
                experiment_id=spec.experiment_id,
                error=f"unknown:{spec.experiment_id}",
            )
        return TrialResult(
            experiment_id=spec.experiment_id,
            val1=TrialWindowResult(**item.get("val1", {})),
            val2=TrialWindowResult(**item.get("val2", {})),
            missing_rates=item.get("missing_rates", {}),
            seconds=float(item.get("seconds", 1.0)),
            error=item.get("error"),
        )


class FakeOracle:
    def __init__(self, *, baseline_auc: float = 0.60) -> None:
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
        self.calls += 1
        if test:
            self.test_calls += 1
            return Verdict(
                verdict="supported",
                metric={"auc": 0.65, "valid_days": 200, "excluded": 50},
                checks=["split_hash_ok", "label_maturity_ok", "no_leakage"],
                tokens=0,
                test=True,
            )
        if locked_config.get("error"):
            return Verdict(
                verdict="blocked",
                metric={"error": locked_config["error"]},
                test=test,
            )
        v1 = float((locked_config.get("val1") or {}).get("auc", 0.0))
        v2 = float((locked_config.get("val2") or {}).get("auc", 0.0))
        merged = (v1 + v2) / 2.0
        if merged > self.baseline_auc + 0.001:
            verdict = "supported"
        elif merged < self.baseline_auc - 0.01:
            verdict = "refuted"
        else:
            verdict = "inconclusive"
        return Verdict(
            verdict=verdict,
            metric={"val1": v1, "val2": v2, "merged": merged},
            checks=["split_hash_ok"],
            test=test,
        )


class FakePlanner:
    def __init__(
        self,
        specs: List[ExperimentSpec],
        *,
        signals_by_experiment: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> None:
        self.specs = list(specs)
        self.signals_by_experiment = signals_by_experiment or {}
        self.plan_calls = 0
        self.interpret_calls = 0

    def plan(self, task: TaskSpec, view: Dict[str, Any], run_kind: str) -> List[ExperimentSpec]:
        self.plan_calls += 1
        return list(self.specs)

    def interpret(
        self,
        feedback: Dict[str, Any],
        view: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self.interpret_calls += 1
        trial = feedback.get("trial") or {}
        experiment_id = str(trial.get("experiment_id") or "")
        return [dict(x) for x in self.signals_by_experiment.get(experiment_id, [])]


def spec(experiment_id: str, feature_set: str = "+volatility", model: str = "logistic") -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id="H1",
        feature_set=feature_set,
        model=model,
        evidence_ids=["E1"],
        params={"C": 1.0},
    )


def tree_spec(
    experiment_id: str,
    feature_set: str,
    *,
    model: str = "tree_shallow",
    params: Optional[Dict[str, Any]] = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id="H2",
        feature_set=feature_set,
        model=model,
        evidence_ids=["E2"],
        params=params or {"max_depth": 5},
    )


def good_result(experiment_id: str) -> Dict[str, Any]:
    return {
        "val1": {"auc": 0.66, "valid_days": 100, "excluded": 20},
        "val2": {"auc": 0.68, "valid_days": 100, "excluded": 20},
        "missing_rates": {"volatility": 0.01},
        "seconds": 2.0,
    }


def make_task(folder: str, *, budget: Optional[Dict[str, Any]] = None) -> TaskSpec:
    raw = {
        "task_id": "task-a",
        "data_path": "data/demo.csv",
        "data_hash": "sha256:demo",
        "label_horizon": 5,
        "split": {"train": 0.6, "val_windows": 2, "test": 0.2, "gap_days": 5},
        "seed": 42,
        "budget": budget or {},
    }
    return load_task_spec(raw)


def make_orchestrator(
    folder: str,
    task: TaskSpec,
    data: FakeData,
    planner: FakePlanner,
    *,
    client: Optional[FakeCloudClient] = None,
    clock=None,
):
    ledger = BudgetGate(task.budget, ledger_path=os.path.join(folder, "token_ledger.json"), clock=clock)
    router = ResearchRouter(ledger, client=client, event_cb=None)
    oracle = FakeOracle()
    writer = EventWriter(os.path.join(folder, "events.jsonl"))
    orchestrator = ResearchOrchestrator(
        task,
        data=data,
        oracle=oracle,
        planner=planner,
        router=router,
        ledger=ledger,
        writer=writer,
        clock=clock,
    )
    return orchestrator, ledger, writer, oracle


def events(folder: str) -> List[Dict[str, Any]]:
    path = Path(folder, "events.jsonl")
    out: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class AgentATest(unittest.TestCase):
    def test_main_loop_reaches_tested_with_revision(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            planner = FakePlanner(
                [spec("exp-001")],
                signals_by_experiment={
                    "exp-001": [{"id": "E1", "reason": "volatility supported"}]
                },
            )
            data = FakeData({"exp-001": good_result("exp-001")})
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, planner, clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertTrue(summary["tested"])
            self.assertEqual(summary["phase"], "TESTED")
            self.assertEqual(summary["stop_reason"], "plan_exhausted")
            self.assertEqual(summary["locked"]["name"], "exp-001")
            evs = events(folder)
            kinds = [e["type"] for e in evs]
            self.assertIn("research.revised", kinds)
            self.assertIn("research.tested", kinds)
            self.assertEqual(oracle.test_calls, 1)
            self.assertTrue(summary["test_verdict"]["test"])
            for event in evs:
                self.assertTrue(str(event.get("decision_reason") or "").strip())
                self.assertTrue(str(event.get("phase") or "").strip())

    def test_budget_blocks_cloud_call(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder, budget={"cloud_tokens": 100, "reserve_tokens": 1200})
            client = FakeCloudClient()
            data = FakeData(
                {"exp-001": good_result("exp-001")},
                profile={"data_anomaly": True},
            )
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner([spec("exp-001")]), client=client, clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertEqual(client.calls, 0)
            self.assertEqual(ledger.remaining_cloud_calls(), task.budget.cloud_calls)
            self.assertTrue(any(e["type"] == "research.escalation_blocked" for e in events(folder)))
            self.assertEqual(summary["phase"], "PARTIAL_COMPLETED")

    def test_duplicate_fingerprint_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            same = spec("exp-001")
            data = FakeData({"exp-001": good_result("exp-001")})
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner([same, same]), clock=lambda: 0.0
            )
            orchestrator.run()
            self.assertEqual(data.calls, ["exp-001"])
            self.assertTrue(any(e["type"] == "research.experiment_skipped" for e in events(folder)))

    def test_invalid_cloud_reply_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            client = FakeCloudClient(invalid=True)
            data = FakeData(
                {"exp-001": good_result("exp-001")},
                profile={"data_anomaly": True},
            )
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner([spec("exp-001")]), client=client, clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertGreaterEqual(client.calls, 1)
            self.assertTrue(any(e["type"] == "research.escalation_invalid" for e in events(folder)))
            self.assertTrue(summary["tested"])

    def test_repeated_error_signature_triggers_escalation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            specs = [
                spec("exp-001", "+volatility"),
                tree_spec("exp-002", "+volume"),
                spec("exp-003", "+volatility_volume", model="tree_shallow"),
            ]
            err = TrialResult(experiment_id="x", error="boom")
            data = FakeData({s.experiment_id: err for s in specs})
            client = FakeCloudClient()
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner(specs), client=client, clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertGreaterEqual(client.calls, 1)
            self.assertTrue(summary["repeated_error_signature"])
            self.assertTrue(any(e["type"] == "research.escalation" for e in events(folder)))

    def test_candidate_budget_stops_loop(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder, budget={"candidates": 2})
            specs = [
                spec("exp-001", "+volatility"),
                tree_spec("exp-002", "+volume"),
                spec("exp-003", "+volatility_volume", model="tree_shallow"),
            ]
            data = FakeData({s.experiment_id: good_result(s.experiment_id) for s in specs})
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner(specs), clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertEqual(len(data.calls), 2)
            self.assertEqual(summary["stop_reason"], "candidate_budget")


if __name__ == "__main__":
    unittest.main()
