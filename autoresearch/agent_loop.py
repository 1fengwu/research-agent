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
        run_id: str,
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
            run_id,
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
        run_ledger_dir: Optional[str] = None,
        candidate_limit: Optional[int] = None,
    ) -> None:
        self.task = task
        self.data = data
        self.oracle = oracle
        self.planner = planner
        self.router = router
        self.ledger = ledger
        self.writer = writer or EventWriter()
        self.run_id = run_id or f"R{int(time.time() * 1000)}"
        self.run_ledger_dir = run_ledger_dir
        self.candidate_limit = candidate_limit
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
            self.run_id,
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
            self.run_id,
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
        fingerprint = spec.fingerprint or experiment_fingerprint(
            spec,
            self.task.data_hash,
            self.task.seed,
            as_jsonable(self.task.split),
        )
        locked_config = {
            "spec": as_jsonable(spec),
            "trial": as_jsonable(trial),
            "fingerprint": fingerprint,
        }
        try:
            return self.oracle.judge(
                run_id=self.run_id,
                experiment_id=spec.experiment_id,
                locked_config=locked_config,
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
            "local_rule_decisions": 0,
            "cache_hits": 0,
            "budget_snapshot": self.ledger.snapshot(),
            "budget_warnings": [],
        }

    def _trial_reason(self, trial: TrialResult) -> str:
        if trial.error:
            return f"error:{trial.error}"
        return (
            f"val1={trial.val1.auc:.4f}/{trial.val1.valid_days}d "
            f"val2={trial.val2.auc:.4f}/{trial.val2.valid_days}d"
        )

    def _budget_warning(
        self,
        state: Dict[str, Any],
        phase: str,
        warning: str,
        *,
        mode: Optional[str] = None,
    ) -> None:
        warnings = state.setdefault("budget_warnings", [])
        if warning not in warnings:
            warnings.append(warning)
        self._emit(
            "research.budget_warning",
            phase,
            warning,
            extra={
                "mode": mode or self.ledger.budget_mode(),
                "overrun": self.ledger.overrun(),
            },
        )

    def _adaptive_plan_cap(self, specs: List[ExperimentSpec]) -> List[ExperimentSpec]:
        """Plan smaller as budget pressure rises so later plans cost fewer tokens."""
        if self.candidate_limit is not None:
            return specs
        cap = {"economy": 2, "critical": 1, "overrun": 1}.get(
            self.ledger.budget_mode()
        )
        if cap is None:
            return specs
        return list(specs[:cap])

    def _apply_escalation_outcome(
        self,
        state: Dict[str, Any],
        outcome: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Count cloud/local/cache decisions separately for cost accounting."""
        source = str(outcome.get("source") or "cloud")
        if source == "local":
            state["local_rule_decisions"] = int(state.get("local_rule_decisions", 0)) + 1
        elif source == "cache":
            state["cache_hits"] = int(state.get("cache_hits", 0)) + 1
        else:
            state["escalation_attempts"] = int(state.get("escalation_attempts", 0)) + 1
            if outcome.get("valid"):
                state["escalation_valid"] = int(state.get("escalation_valid", 0)) + 1
        return state

    def _window_conflict(self, trial: TrialResult, baseline: Dict[str, Any]) -> bool:
        if trial.error:
            return False
        base = float(baseline.get("merged_auc") or 0.0)
        if base <= 0:
            return False
        above1 = trial.val1.auc > base
        above2 = trial.val2.auc > base
        return above1 != above2

    def _candidate_viable(
        self,
        item: Dict[str, Any],
        baseline_auc: float,
    ) -> bool:
        """PRD 0.01 rule: reject refuted/blocked and any window drop > 0.01."""
        trial = item.get("trial") or {}
        if trial.get("error"):
            return False
        verdict = str((item.get("verdict") or {}).get("verdict") or "inconclusive")
        if verdict in {"refuted", "blocked"}:
            return False
        v1 = float((trial.get("val1") or {}).get("auc", 0.0))
        v2 = float((trial.get("val2") or {}).get("auc", 0.0))
        if baseline_auc > 0 and (v1 < baseline_auc - 0.01 or v2 < baseline_auc - 0.01):
            return False
        return True

    def _lock_best(self, state: Dict[str, Any]) -> Dict[str, Any]:
        baseline_auc = float(state.get("baseline", {}).get("merged_auc") or 0.0)
        records = [
            item
            for item in state.get("experiments", [])
            if self._candidate_viable(item, baseline_auc)
        ]
        if not records:
            return {
                "name": "baseline",
                "config": state.get("baseline", {}).get("config") or {},
                "merged_auc": baseline_auc,
                "evidence_ids": list(state.get("evidence_ids", [])),
                "reason": "no viable candidate (refuted/blocked/window drop); keep baseline",
            }

        def merged(item: Dict[str, Any]) -> float:
            trial = item.get("trial", {})
            v1 = trial.get("val1", {})
            v2 = trial.get("val2", {})
            return (float(v1.get("auc", 0.0)) + float(v2.get("auc", 0.0))) / 2.0

        best = max(records, key=merged)
        best_merged = merged(best)
        if best_merged <= baseline_auc:
            return {
                "name": "baseline",
                "config": state.get("baseline", {}).get("config") or {},
                "merged_auc": baseline_auc,
                "evidence_ids": list(state.get("evidence_ids", [])),
                "reason": "viable candidate did not beat baseline",
            }
        return {
            "name": str(best.get("spec", {}).get("experiment_id", "candidate")),
            "config": best.get("spec", {}),
            "merged_auc": best_merged,
            "evidence_ids": list(state.get("evidence_ids", [])),
            "reason": f"best merged candidate auc={best_merged:.4f}",
        }

    def _prepare_run_ledger(self) -> None:
        """Give every run its own ledger so R2 does not inherit R1's spend."""
        if self.run_ledger_dir is None:
            return
        path = Path(self.run_ledger_dir) / f"{self.run_id}.json"
        self.ledger.reset(str(path))

    def _locked_record(
        self,
        state: Dict[str, Any],
        locked: Dict[str, Any],
    ) -> tuple[ExperimentSpec, TrialResult]:
        """Return the real locked spec and its trial for the final holdout test."""
        if locked.get("name") == "baseline":
            baseline = state.get("baseline", {})
            config = baseline.get("config") or {}
            locked_spec = ExperimentSpec(
                experiment_id="baseline-locked",
                hypothesis_id="LOCKED",
                feature_set=str(config.get("feature_set") or "base"),
                model=str(config.get("model") or "logistic"),
                evidence_ids=list(state.get("evidence_ids", [])),
                params=dict(config.get("params") or {}),
                missing=str(config.get("missing") or "median"),
                scaling=str(config.get("scaling") or "standard"),
            )
            return locked_spec, TrialResult(experiment_id=locked_spec.experiment_id)

        locked_name = str(locked.get("name") or "")
        for item in state.get("experiments", []):
            spec_payload = item.get("spec") or {}
            if str(spec_payload.get("experiment_id") or "") != locked_name:
                continue
            locked_spec = (
                normalize_experiment_spec(spec_payload)
                if isinstance(spec_payload, dict)
                else spec_payload
            )
            trial_payload = item.get("trial") or {}
            if isinstance(trial_payload, TrialResult):
                locked_trial = trial_payload
            elif isinstance(trial_payload, dict):
                val1_data = trial_payload.get("val1") or {}
                val2_data = trial_payload.get("val2") or {}
                locked_trial = TrialResult(
                    experiment_id=locked_spec.experiment_id,
                    val1=TrialWindowResult(**val1_data),
                    val2=TrialWindowResult(**val2_data),
                    missing_rates=dict(trial_payload.get("missing_rates") or {}),
                    seconds=float(trial_payload.get("seconds") or 0.0),
                    error=trial_payload.get("error"),
                )
            else:
                locked_trial = TrialResult(experiment_id=locked_spec.experiment_id)
            return locked_spec, locked_trial

        # The lock must always point at a recorded experiment; fail loudly if not.
        raise ValueError(f"locked candidate {locked_name} has no experiment record")

    # --------------------------------------------------------------- main run
    def run(self) -> Dict[str, Any]:
        self._prepare_run_ledger()
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
        mode = self.ledger.budget_mode()
        if mode != "normal":
            self._budget_warning(state, PHASE_PLANNED, f"budget_mode:{mode}")
        try:
            planned = self._adaptive_plan_cap(
                self._normalize_specs(
                    self.planner.plan(self.task, self._view(state), "first")
                )
            )
        except Exception as exc:
            planned = []
            state["data_anomaly"] = True
            state["error_signatures"][f"plan_failed:{exc}"] = state["error_signatures"].get("plan_failed", 0) + 1
            state["budget_snapshot"] = self.ledger.snapshot()
            outcome = self.router.escalate_if_needed(state)
            if outcome.get("escalated"):
                state = self._apply_escalation_outcome(state, outcome)
                if outcome.get("valid"):
                    state = apply_reply(state, outcome["reply"])
                    state["data_anomaly"] = False
                    self._emit(
                        "research.revised",
                        PHASE_REVISED,
                        f"plan_failure_cloud_decision:{outcome['reply']['decision']}",
                        extra={"reply": _short(outcome["reply"])},
                    )
                    try:
                        planned = self._adaptive_plan_cap(
                            self._normalize_specs(
                                self.planner.plan(self.task, self._view(state), "revised")
                            )
                        )
                    except Exception as replan_exc:
                        self._emit(
                            "research.escalation_failed",
                            PHASE_REVISED,
                            f"replan_failed:{replan_exc}",
                        )
                        planned = []
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
            if self.candidate_limit is not None and len(state["attempted"]) >= self.candidate_limit:
                stop = "candidate_limit"
                break
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
            options = [spec, *pending]
            outcome = self.router.escalate_if_needed({
                **state,
                "pending_candidates": [
                    {"id": item.experiment_id, "feature_set": item.feature_set, "model": item.model}
                    for item in options
                ],
            })
            if outcome.get("escalated"):
                state = self._apply_escalation_outcome(state, outcome)
                if outcome.get("valid"):
                    state = apply_reply(state, outcome["reply"])
                    self._emit(
                        "research.revised",
                        PHASE_REVISED,
                        f"cloud_decision:{outcome['reply']['decision']}",
                        experiment_id=spec.experiment_id,
                        extra={"reply": _short(outcome["reply"])},
                    )
                    actions = outcome["reply"].get("actions") or [{}]
                    action = actions[0]
                    target = action.get("target")
                    if action.get("action") == "select_candidate":
                        selected = next((item for item in options if item.experiment_id == target), None)
                        if selected is not None:
                            pending = [item for item in options if item is not selected]
                            fingerprint = selected.fingerprint or experiment_fingerprint(
                                selected, self.task.data_hash, self.task.seed, as_jsonable(self.task.split)
                            )
                            self._emit(
                                "research.revised", PHASE_REVISED, "candidate_selected",
                                experiment_id=selected.experiment_id,
                                extra={"previous_next": spec.experiment_id, "source": outcome.get("source", "cloud")},
                            )
                            spec = selected
                        else:
                            self._emit(
                                "research.revised", PHASE_REVISED, "candidate_selection_rejected",
                                experiment_id=spec.experiment_id,
                            )
                    elif action.get("action") == "replan" and self.candidate_limit is None:
                        try:
                            pending = self._adaptive_plan_cap(
                                self._normalize_specs(
                                    self.planner.plan(self.task, self._view(state), "revised")
                                )
                            ) + pending
                        except Exception as exc:
                            self._emit("research.escalation_failed", PHASE_REVISED, f"replan_failed:{exc}")

            if fingerprint in state["attempted"]:
                continue

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

            state["window_conflict"] = self._window_conflict(trial, baseline)
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
            elif not self.ledger.can_run_experiment():
                stop = "wall_time_guard"
            else:
                stop = self.ledger.stop_reason() or "plan_interrupted"

        locked = self._lock_best(state)
        self._emit(
            "research.locked",
            PHASE_LOCKED,
            f"locked:{locked['name']}",
            extra={"locked": _short(locked)},
        )

        tested = False
        test_verdict: Optional[Verdict] = None
        finalize = self.ledger.finalize_status()
        if finalize.get("warning"):
            self._budget_warning(
                state,
                PHASE_LOCKED,
                finalize["warning"],
                mode=finalize["mode"],
            )
        if finalize["allow"]:
            locked_spec, locked_trial = self._locked_record(state, locked)
            register_lock = getattr(self.oracle, "register_lock", None)
            try:
                if register_lock is not None:
                    register_lock(self.run_id, as_jsonable(locked_spec))
                test_verdict = self._judge(locked_spec, locked_trial, test=True)
            except Exception as exc:
                test_verdict = Verdict("blocked", {"error": str(exc)}, test=True)
            tested = test_verdict.verdict not in {"blocked", "partial_completed"}
            final_phase = PHASE_TESTED if tested else PHASE_FAILED
            if not tested:
                stop = "holdout_failed"
            self._emit(
                "research.tested" if tested else "research.test_failed",
                final_phase,
                f"verdict:{test_verdict.verdict}",
                experiment_id=locked_spec.experiment_id,
                extra={
                    "verdict": as_jsonable(test_verdict),
                    "locked_spec": _short(as_jsonable(locked_spec)),
                },
            )
        else:
            final_phase = PHASE_PARTIAL
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
            "budget_mode": self.ledger.budget_mode(),
            "budget_overrun": self.ledger.overrun(),
            "budget_warnings": list(state.get("budget_warnings", [])),
            "local_rule_decisions": int(state.get("local_rule_decisions", 0)),
            "cache_hits": int(state.get("cache_hits", 0)),
            "token_savings": self.router.savings_summary(),
            "event_path": str(self.writer.path),
            "elapsed_seconds": round(self._clock() - self._started_wall, 3),
        }
        self._emit(
            "research.token_savings",
            final_phase,
            "token_savings_report",
            extra={"token_savings": self.router.savings_summary()},
        )
        self._emit(
            "research.completed",
            final_phase,
            f"stop_reason:{stop or 'completed'}",
            extra={"summary": _short(summary)},
        )
        return summary
