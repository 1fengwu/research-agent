from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch.agent_loop import EventWriter, ResearchOrchestrator
from autoresearch.budget_gate import BudgetBlocked, BudgetGate
from autoresearch.contracts import (
    BudgetSpec,
    ExperimentSpec,
    TaskSpec,
    TrialResult,
    TrialWindowResult,
    Verdict,
    load_task_spec,
)
from autoresearch.llm_router import ResearchRouter, _load_env_file


class CloudEnvTestCase(unittest.TestCase):
    def test_load_env_file_sets_and_does_not_override(self) -> None:
        keys = ("ARC_LLM_BASE_URL", "ARC_LLM_API_KEY", "ARC_LLM_MODEL")
        old = {key: os.environ.get(key) for key in keys}
        try:
            with tempfile.TemporaryDirectory() as folder:
                env_path = Path(folder) / ".env"
                env_path.write_text(
                    "ARC_LLM_BASE_URL=https://api.example.com/v1\n"
                    "ARC_LLM_API_KEY=\"sk-test\"\n"
                    "ARC_LLM_MODEL=model-b\n",
                    encoding="utf-8",
                )
                os.environ.pop("ARC_LLM_BASE_URL", None)
                os.environ.pop("ARC_LLM_API_KEY", None)
                os.environ["ARC_LLM_MODEL"] = "model-a"
                _load_env_file(env_path)
                self.assertEqual(os.environ["ARC_LLM_BASE_URL"], "https://api.example.com/v1")
                self.assertEqual(os.environ["ARC_LLM_API_KEY"], "sk-test")
                self.assertEqual(os.environ["ARC_LLM_MODEL"], "model-a")
        finally:
            for key in keys:
                if old[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old[key]


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
        self.last_locked_config: Optional[Dict[str, Any]] = None

    def judge(
        self,
        *,
        run_id: str,
        experiment_id: str,
        locked_config: Dict[str, Any],
        test: bool = False,
    ) -> Verdict:
        self.calls += 1
        spec = locked_config.get("spec") if isinstance(locked_config, dict) else None
        trial = locked_config.get("trial") if isinstance(locked_config, dict) else locked_config
        trial = trial or locked_config or {}
        if test:
            self.test_calls += 1
            self.last_locked_config = dict(locked_config or {})
            return Verdict(
                verdict="supported",
                metric={"auc": 0.65, "valid_days": 200, "excluded": 50},
                checks=["split_hash_ok", "label_maturity_ok", "no_leakage"],
                tokens=0,
                test=True,
            )
        if trial.get("error"):
            return Verdict(
                verdict="blocked",
                metric={"error": trial["error"]},
                test=test,
            )
        v1 = float((trial.get("val1") or {}).get("auc", 0.0))
        v2 = float((trial.get("val2") or {}).get("auc", 0.0))
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


class RefutingOracle(FakeOracle):
    """Every validation verdict is refuted so A must not lock the candidate."""

    def judge(
        self,
        *,
        run_id: str,
        experiment_id: str,
        locked_config: Dict[str, Any],
        test: bool = False,
    ) -> Verdict:
        if test:
            return super().judge(
                run_id=run_id,
                experiment_id=experiment_id,
                locked_config=locked_config,
                test=True,
            )
        return Verdict(
            verdict="refuted",
            metric={"reason": "ci excludes zero in the wrong direction"},
            checks=["paired_bootstrap_ci"],
            test=test,
        )


class FailingOncePlanner(FakePlanner):
    """First plan call fails; the revised call after escalation returns specs."""

    def plan(self, task: TaskSpec, view: Dict[str, Any], run_kind: str) -> List[ExperimentSpec]:
        self.plan_calls += 1
        if run_kind == "first":
            raise RuntimeError("planner crashed before pre-registration")
        return list(self.specs)


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
    ledger = BudgetGate(task.budget, clock=clock)
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
        run_ledger_dir=folder,
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


    def test_refuted_candidate_not_locked(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            specs = [spec("exp-001", "+volatility")]
            data = FakeData({"exp-001": good_result("exp-001")})
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner(specs), clock=lambda: 0.0
            )
            orchestrator.oracle = RefutingOracle()
            summary = orchestrator.run()
            self.assertEqual(summary["locked"]["name"], "baseline")
            self.assertIn("no viable candidate", summary["locked"]["reason"])
            self.assertEqual(
                [x["verdict"]["verdict"] for x in summary["experiments"]],
                ["refuted"],
            )

    def test_window_drop_rule_rejects_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            drop_result = {
                "val1": {"auc": 0.71, "valid_days": 100, "excluded": 20},
                "val2": {"auc": 0.55, "valid_days": 100, "excluded": 20},
                "missing_rates": {"volume": 0.02},
                "seconds": 2.5,
            }
            data = FakeData({"exp-001": drop_result})
            specs = [spec("exp-001", "+volume")]
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner(specs), clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertEqual(summary["locked"]["name"], "baseline")
            self.assertIn("no viable candidate", summary["locked"]["reason"])

    def test_final_test_receives_real_locked_config(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            specs = [spec("exp-001", "+volatility")]
            data = FakeData({"exp-001": good_result("exp-001")})
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner(specs), clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertIsNotNone(oracle.last_locked_config)
            locked_spec = oracle.last_locked_config["spec"]
            self.assertEqual(locked_spec["experiment_id"], "exp-001")
            self.assertEqual(locked_spec["feature_set"], "+volatility")
            self.assertTrue(oracle.last_locked_config["fingerprint"])
            self.assertEqual(oracle.last_locked_config["trial"]["experiment_id"], "exp-001")
            self.assertTrue(summary["test_verdict"]["test"])

    def test_second_run_uses_fresh_per_run_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder, budget={"candidates": 1})
            specs = [spec("exp-001", "+volatility")]
            data = FakeData({"exp-001": good_result("exp-001")})
            client = FakeCloudClient()
            ledger = BudgetGate(task.budget, clock=lambda: 0.0)

            def build(run_id: str, events_name: str) -> ResearchOrchestrator:
                router = ResearchRouter(ledger, client=client)
                writer = EventWriter(os.path.join(folder, events_name))
                return ResearchOrchestrator(
                    task,
                    data=data,
                    oracle=FakeOracle(),
                    planner=FakePlanner(specs),
                    router=router,
                    ledger=ledger,
                    writer=writer,
                    run_id=run_id,
                    run_ledger_dir=folder,
                    clock=lambda: 0.0,
                )

            o1 = build("R1", "events1.jsonl")
            o1.run()
            o2 = build("R2", "events2.jsonl")
            summary2 = o2.run()
            self.assertEqual(len(summary2["experiments"]), 1)
            self.assertEqual(summary2["budget"]["used_candidates"], 1)
            self.assertTrue(summary2["tested"])
            self.assertEqual(ledger.path.name, "R2.json")

    def test_partial_completed_reason_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(
                folder,
                budget={"cloud_tokens": 100, "reserve_tokens": 1200},
            )
            specs = [spec("exp-001", "+volatility")]
            data = FakeData({"exp-001": good_result("exp-001")})
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder, task, data, FakePlanner(specs), clock=lambda: 0.0
            )
            summary = orchestrator.run()
            self.assertEqual(summary["phase"], "PARTIAL_COMPLETED")
            self.assertEqual(summary["stop_reason"], "partial_completed")
            self.assertFalse(summary["tested"])
            self.assertTrue(
                any(
                    e["type"] == "research.completed"
                    and e["decision_reason"] == "stop_reason:partial_completed"
                    for e in events(folder)
                )
            )

    def test_plan_failure_escalates_and_replans(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task = make_task(folder)
            specs = [spec("exp-001", "+volatility")]
            data = FakeData({"exp-001": good_result("exp-001")})
            client = FakeCloudClient()
            orchestrator, ledger, writer, oracle = make_orchestrator(
                folder,
                task,
                data,
                FailingOncePlanner(specs),
                client=client,
                clock=lambda: 0.0,
            )
            summary = orchestrator.run()
            self.assertGreaterEqual(client.calls, 1)
            self.assertGreaterEqual(summary["escalation_attempts"], 1)
            self.assertGreaterEqual(summary["escalation_valid"], 1)
            self.assertTrue(
                any(
                    e["type"] == "research.revised"
                    and "plan_failure_cloud_decision" in e["decision_reason"]
                    for e in events(folder)
                )
            )
            self.assertEqual(data.calls, ["exp-001"])
            self.assertTrue(summary["tested"])

    def test_preflight_reserves_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            gate = BudgetGate(
                BudgetSpec(cloud_tokens=100, reserve_tokens=80),
                clock=lambda: 0.0,
            )
            with self.assertRaises(BudgetBlocked) as context:
                gate.preflight_cloud(30, 60)
            self.assertIn("reserve", context.exception.reason)


if __name__ == "__main__":
    unittest.main()
