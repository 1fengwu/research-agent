from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .budget_gate import BudgetGate
from .contracts import (
    ExperimentSpec,
    TaskSpec,
    TrialResult,
    TrialWindowResult,
    Verdict,
    as_jsonable,
    build_run_event,
    experiment_fingerprint,
    normalize_experiment_spec,
)
from .llm_router import ResearchRouter, apply_reply
from .ports import DataPort, OraclePort, PlannerPort


PHASE_UNSEEN = "UNSEEN"
PHASE_PLANNING = "PLANNING"
PHASE_PLANNED = "PLANNED"
PHASE_EXPERIMENTING = "EXPERIMENTING"
PHASE_VERIFIED = "VERIFIED"
PHASE_REVISED = "REVISED"
PHASE_LOCKED = "LOCKED"
PHASE_TESTED = "TESTED"
PHASE_PASSED = "PASSED"
PHASE_CONVERGED = "CONVERGED"
PHASE_FAILED = "FAILED"
PHASE_PARTIAL = "PARTIAL_COMPLETED"


class EventWriter:
    """Append-only JSONL under `.arc/`, the single source for the frontend."""

    def __init__(self, path: str = ".arc/runner-events.jsonl") -> None:
        self.path = Path(path)

    def append(
        self,
        event_type: str,
        phase: str,
        *,
        decision_reason: str = "",
        experiment_id: str = "",
        tokens: Optional[Dict[str, Any]] = None,
        remaining: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = build_run_event(
            event_type,
            phase,
            datetime.now(timezone.utc).isoformat(),
            decision_reason=decision_reason,
            experiment_id=experiment_id,
            tokens=tokens,
            remaining=remaining,
            extra=extra,
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(as_jsonable(event), ensure_ascii=False) + "\n")


def _short(value: Dict[str, Any], limit: int = 400) -> Dict[str, Any]:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return {"truncated": True, "preview": text[:limit]}


class ResearchOrchestrator:
    """A owns the serial research loop; B/C/D are injected through ports."""

    def __init__(
        self,
        task: TaskSpec,
        data: DataPort,
        oracle: OraclePort,
        planner: PlannerPort,
        router: ResearchRouter,
        ledger: BudgetGate,
        writer: Optional[EventWriter] = None,
        *,
        run_id: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.task = task
        self.data = data
        self.oracle = oracle
        self.planner = planner
        self.router = router
        self.ledger = ledger
        self.writer = writer or EventWriter()
        self.run_id = run_id or f"R{int(time.time() * 1000)}"
        self._clock = clock or time.monotonic
        self._started_wall = self._clock()
        self.router.event_cb = self._router_event

    # ---------------------------------------------------------------- events
    def _emit(
        self,
        event_type: str,
        phase: str,
        reason: str,
        *,
        experiment_id: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.writer.append(
            event_type,
            phase,
            decision_reason=reason,
            experiment_id=experiment_id,
            remaining=self.ledger.snapshot(),
            extra=extra,
        )

    def _router_event(
        self,
        kind: str,
        phase: str,
        reason: str,
        tokens: Optional[Dict[str, Any]],
        extra: Optional[Dict[str, Any]],
    ) -> None:
        self.writer.append(
            kind,
            phase,
            decision_reason=reason,
            tokens=tokens,
            remaining=self.ledger.snapshot(),
            extra=extra,
        )

    # ------------------------------------------------------------- helpers
    def _profile_reason(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return "profile empty"
        parts = []
        if profile.get("date_range"):
            parts.append(f"date={profile['date_range']}")
        if profile.get("symbols"):
            parts.append(f"symbols={profile['symbols']}")
        if profile.get("missing_rate") is not None:
            parts.append(f"missing={profile['missing_rate']}")
        if profile.get("anomalies"):
            parts.append("anomalies=" + ",".join(str(x) for x in profile["anomalies"]))
        return "; ".join(parts) or "profile summary"

    def _baseline_reason(self, baseline: Dict[str, Any]) -> str:
        if not baseline:
            return "baseline returned empty"
        merged = baseline.get("merged_auc")
        if merged is None:
            return "baseline executed"
        return f"baseline merged_auc={merged}"

    def _view(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return dict(state)

    def _normalize_specs(self, items: List[Any]) -> List[ExperimentSpec]:
        out: List[ExperimentSpec] = []
        for item in items or []:
            if isinstance(item, ExperimentSpec):
                out.append(item)
            elif isinstance(item, dict):
                out.append(normalize_experiment_spec(item))
            else:
                raise ValueError("planner returned a non-ExperimentSpec item")
        return out

    def _judge(
        self,
        spec: ExperimentSpec,
        trial: TrialResult,
        *,
        test: bool = False,
    ) -> Verdict:
        try:
            return self.oracle.judge(
                run_id=self.run_id,
                experiment_id=spec.experiment_id,
                locked_config=as_jsonable(trial),
                test=test,
            )
        except Exception as exc:
            return Verdict(
                verdict="blocked",
                metric={"error": str(exc)},
                checks=[],
                test=test,
            )

    def _initial_state(
        self,
        profile: Dict[str, Any],
        baseline: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "phase": PHASE_UNSEEN,
            "active_config": baseline.get("config") or {},
            "baseline": baseline,
            "evidence_ids": [],
            "stall_rounds": 0,
            "window_conflict": False,
            "critical_decision": False,
            "data_anomaly": bool(profile.get("data_anomaly") or profile.get("anomalies")),
            "repeated_error_signature": False,
            "error_signatures": {},
            "experiments": [],
            "attempted": [],
            "plan_hint": "",
            "actions": [],
            "last_trial": None,
            "escalation_attempts": 0,
            "escalation_valid": 0,
            "budget_snapshot": self.ledger.snapshot(),
        }

    def _trial_reason(self, trial: TrialResult) -> str:
        if trial.error:
            return f"error:{trial.error}"
        return (
            f"val1={trial.val1.auc:.4f}/{trial.val1.valid_days}d "
            f"val2={trial.val2.auc:.4f}/{trial.val2.valid_days}d"
        )

    def _window_conflict(self, trial: TrialResult, baseline: Dict[str, Any]) -> bool:
        if trial.error:
            return False
        base = float(baseline.get("merged_auc") or 0.0)
        if base <= 0:
            return False
        above1 = trial.val1.auc > base
        above2 = trial.val2.auc > base
        return above1 != above2

    def _lock_best(self, state: Dict[str, Any]) -> Dict[str, Any]:
        records = [
            item
            for item in state.get("experiments", [])
            if item.get("trial", {}).get("error") is None
        ]
        if not records:
            return {
                "name": "baseline",
                "config": state.get("baseline", {}).get("config") or {},
                "merged_auc": state.get("baseline", {}).get("merged_auc"),
                "evidence_ids": list(state.get("evidence_ids", [])),
                "reason": "no valid candidate; keep baseline",
            }

        def merged(item: Dict[str, Any]) -> float:
            trial = item.get("trial", {})
            v1 = trial.get("val1", {})
            v2 = trial.get("val2", {})
            return (float(v1.get("auc", 0.0)) + float(v2.get("auc", 0.0))) / 2.0

        best = max(records, key=merged)
        best_merged = merged(best)
        baseline_auc = float(state.get("baseline", {}).get("merged_auc") or 0.0)
        if best_merged <= baseline_auc:
            return {
                "name": "baseline",
                "config": state.get("baseline", {}).get("config") or {},
                "merged_auc": baseline_auc,
                "evidence_ids": list(state.get("evidence_ids", [])),
                "reason": "candidate did not beat baseline",
            }
        return {
            "name": str(best.get("spec", {}).get("experiment_id", "candidate")),
            "config": best.get("spec", {}),
            "merged_auc": best_merged,
            "evidence_ids": list(state.get("evidence_ids", [])),
            "reason": f"best merged candidate auc={best_merged:.4f}",
        }

    # --------------------------------------------------------------- main run
    def run(self) -> Dict[str, Any]:
        self._emit(
            "research.run_started",
            PHASE_UNSEEN,
            f"task={self.task.task_id}",
            extra={"task": _short(as_jsonable(self.task))},
        )

        try:
            profile = self.data.describe_data(self.task) or {}
        except Exception as exc:
            profile = {"error": str(exc), "data_anomaly": True}
        self._emit(
            "research.data_profile",
            PHASE_PLANNING,
            self._profile_reason(profile),
            extra={"profile": _short(profile)},
        )

        try:
            baseline = self.data.baseline(self.task) or {}
        except Exception as exc:
            baseline = {"error": str(exc), "merged_auc": 0.0, "config": {}}
        self._emit(
            "research.baseline",
            PHASE_PLANNED,
            self._baseline_reason(baseline),
            extra={"baseline": _short(baseline)},
        )

        state = self._initial_state(profile, baseline)
        try:
            planned = self._normalize_specs(self.planner.plan(self.task, self._view(state), "first"))
        except Exception as exc:
            planned = []
            state["data_anomaly"] = True
            state["error_signatures"][f"plan_failed:{exc}"] = state["error_signatures"].get("plan_failed", 0) + 1
        for spec in planned:
            self._emit(
                "research.hypothesis_registered",
                PHASE_PLANNED,
                f"hypothesis={spec.hypothesis_id}",
                experiment_id=spec.experiment_id,
                extra={"spec": _short(as_jsonable(spec))},
            )

        pending = list(planned)
        stop: Optional[str] = None
        while pending and self.ledger.can_run_experiment() and stop is None:
            spec = pending.pop(0)
            fingerprint = spec.fingerprint or experiment_fingerprint(
                spec,
                self.task.data_hash,
                self.task.seed,
                as_jsonable(self.task.split),
            )
            if fingerprint in state["attempted"]:
                self._emit(
                    "research.experiment_skipped",
                    PHASE_PLANNED,
                    f"duplicate_fingerprint:{fingerprint}",
                    experiment_id=spec.experiment_id,
                )
                continue

            state["budget_snapshot"] = self.ledger.snapshot()
            outcome = self.router.escalate_if_needed(state)
            if outcome.get("escalated"):
                state["escalation_attempts"] = int(state.get("escalation_attempts", 0)) + 1
                if outcome.get("valid"):
                    state = apply_reply(state, outcome["reply"])
                    state["escalation_valid"] = int(state.get("escalation_valid", 0)) + 1
                    self._emit(
                        "research.revised",
                        PHASE_REVISED,
                        f"cloud_decision:{outcome['reply']['decision']}",
                        experiment_id=spec.experiment_id,
                        extra={"reply": _short(outcome["reply"])},
                    )
                    try:
                        more = self._normalize_specs(
                            self.planner.plan(self.task, self._view(state), "revised")
                        )
                        pending = more + pending
                    except Exception as exc:
                        self._emit(
                            "research.escalation_failed",
                            PHASE_REVISED,
                            f"replan_failed:{exc}",
                            experiment_id=spec.experiment_id,
                        )

            self.ledger.record_experiment(note=spec.experiment_id)
            self._emit(
                "research.experiment_started",
                PHASE_EXPERIMENTING,
                f"candidate={spec.experiment_id}",
                experiment_id=spec.experiment_id,
            )
            try:
                trial = self.data.run_candidate(spec)
            except Exception as exc:
                trial = TrialResult(
                    experiment_id=spec.experiment_id,
                    error=f"run_failed:{exc}",
                )
            state["attempted"].append(fingerprint)
            state["last_trial"] = as_jsonable(trial)
            state["budget_snapshot"] = self.ledger.snapshot()
            self._emit(
                "research.experiment_result",
                PHASE_VERIFIED,
                self._trial_reason(trial),
                experiment_id=spec.experiment_id,
                extra={"trial": as_jsonable(trial)},
            )

            verdict = self._judge(spec, trial)
            state["experiments"].append(
                {
                    "spec": as_jsonable(spec),
                    "trial": as_jsonable(trial),
                    "verdict": as_jsonable(verdict),
                }
            )
            self._emit(
                "research.experiment_verdict",
                PHASE_VERIFIED,
                f"verdict:{verdict.verdict}",
                experiment_id=spec.experiment_id,
                extra={"verdict": as_jsonable(verdict)},
            )

            signals: List[Dict[str, Any]] = []
            if trial.error is None:
                try:
                    signals = list(
                        self.planner.interpret(
                            {
                                "trial": as_jsonable(trial),
                                "verdict": as_jsonable(verdict),
                                "previous_evidence": list(state.get("evidence_ids", [])),
                            },
                            self._view(state),
                        )
                    )
                except Exception as exc:
                    self._emit(
                        "research.escalation_failed",
                        PHASE_VERIFIED,
                        f"interpret_failed:{exc}",
                        experiment_id=spec.experiment_id,
                    )
            if signals:
                existing = list(state.get("evidence_ids", []))
                for signal in signals:
                    sid = str(signal.get("id") or signal.get("reason") or f"E{len(existing) + 1}")
                    if sid not in existing:
                        existing.append(sid)
                state["evidence_ids"] = existing
                state["stall_rounds"] = 0
                state["phase"] = PHASE_REVISED
                self._emit(
                    "research.revised",
                    PHASE_REVISED,
                    f"evidence_signals:{len(signals)}",
                    experiment_id=spec.experiment_id,
                )
            else:
                state["stall_rounds"] = int(state.get("stall_rounds", 0)) + 1

            if self._window_conflict(trial, baseline):
                state["window_conflict"] = True
            signature = str(trial.error or verdict.verdict or "unknown")
            count = int(state["error_signatures"].get(signature, 0)) + 1
            state["error_signatures"][signature] = count
            if count >= 2:
                state["repeated_error_signature"] = True

            state["budget_snapshot"] = self.ledger.snapshot()
            stop = self.ledger.stop_reason()

        if stop is None:
            if not pending:
                stop = "no_experiments" if not state["experiments"] else "plan_exhausted"
            else:
                stop = self.ledger.stop_reason() or "budget_exhausted"

        locked = self._lock_best(state)
        self._emit(
            "research.locked",
            PHASE_LOCKED,
            f"locked:{locked['name']}",
            extra={"locked": _short(locked)},
        )

        tested = False
        test_verdict: Optional[Verdict] = None
        if self.ledger.can_finalize():
            test_verdict = self._judge(
                ExperimentSpec(
                    experiment_id="locked-candidate",
                    hypothesis_id="LOCKED",
                    feature_set=str(locked.get("config", {}).get("feature_set", "locked")),
                    model=str(locked.get("config", {}).get("model", "baseline")),
                    evidence_ids=list(locked.get("evidence_ids", [])),
                ),
                TrialResult(
                    experiment_id="locked-candidate",
                    val1=TrialWindowResult(),
                    val2=TrialWindowResult(),
                ),
                test=True,
            )
            tested = True
            self._emit(
                "research.tested",
                PHASE_TESTED,
                f"verdict:{test_verdict.verdict}",
                experiment_id="locked-candidate",
                extra={"verdict": as_jsonable(test_verdict)},
            )
            final_phase = PHASE_TESTED
        else:
            final_phase = PHASE_PARTIAL
            if stop is None:
                stop = "partial_completed"

        summary = {
            "run_id": self.run_id,
            "task_id": self.task.task_id,
            "phase": final_phase,
            "stop_reason": stop or "completed",
            "tested": tested,
            "locked": locked,
            "test_verdict": as_jsonable(test_verdict) if test_verdict else None,
            "experiments": list(state.get("experiments", [])),
            "evidence_ids": list(state.get("evidence_ids", [])),
            "stall_rounds": state.get("stall_rounds"),
            "window_conflict": state.get("window_conflict"),
            "repeated_error_signature": state.get("repeated_error_signature"),
            "error_signatures": dict(state.get("error_signatures", {})),
            "escalation_attempts": state.get("escalation_attempts"),
            "escalation_valid": state.get("escalation_valid"),
            "budget": self.ledger.snapshot(),
            "ledger": self.ledger.usage_ledger(),
            "event_path": str(self.writer.path),
            "elapsed_seconds": round(self._clock() - self._started_wall, 3),
        }
        self._emit(
            "research.completed",
            final_phase,
            f"stop_reason:{stop or 'completed'}",
            extra={"summary": _short(summary)},
        )
        return summary
