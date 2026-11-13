from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from .contracts import ExperimentSpec, TaskSpec, TrialResult, Verdict


@runtime_checkable
class DataPort(Protocol):
    """B implements this port: data intake, profile, baseline, and candidates."""

    def describe_data(self, task: TaskSpec) -> Dict[str, Any]:
        """Return a JSON-able data profile: date range, symbols, missing, balance."""
        ...

    def baseline(self, task: TaskSpec) -> Dict[str, Any]:
        """Run the fixed PRD baseline `LogisticRegression[r5, r20]`."""
        ...

    def run_candidate(self, spec: ExperimentSpec) -> TrialResult:
        """Run one validated candidate experiment on train/validation only."""
        ...


@runtime_checkable
class OraclePort(Protocol):
    """C implements this port: the only scorer and holdout guard."""

    def judge(
        self,
        *,
        run_id: str,
        experiment_id: str,
        locked_config: Dict[str, Any],
        test: bool = False,
    ) -> Verdict:
        """Return supported/refuted/inconclusive/blocked/partial_completed.

        `test=True` is allowed only once after the run is locked.
        """
        ...


@runtime_checkable
class PlannerPort(Protocol):
    """A calls this port; the research domain implements plan and interpret."""

    def plan(self, task: TaskSpec, view: Dict[str, Any], run_kind: str) -> List[ExperimentSpec]:
        """Pre-register hypotheses and return the next batch of experiments."""
        ...

    def interpret(
        self,
        feedback: Dict[str, Any],
        view: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Translate a TrialResult + Verdict into evidence signals."""
        ...
