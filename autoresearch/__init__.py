"""AutoResearch: A owns orchestration and reasoning."""

from .contracts import (
    BudgetSpec,
    ExperimentSpec,
    RunEvent,
    SplitSpec,
    TaskSpec,
    TrialResult,
    TrialWindowResult,
    Verdict,
    as_jsonable,
    experiment_fingerprint,
    load_task_spec,
    normalize_experiment_spec,
)

__all__ = [
    "BudgetSpec",
    "ExperimentSpec",
    "RunEvent",
    "SplitSpec",
    "TaskSpec",
    "TrialResult",
    "TrialWindowResult",
    "Verdict",
    "as_jsonable",
    "experiment_fingerprint",
    "load_task_spec",
    "normalize_experiment_spec",
]
