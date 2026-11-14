from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SplitSpec:
    """Day-grouped split: train 60%, two consecutive validation windows, test 20%."""

    train: float = 0.6
    val_windows: int = 2
    test: float = 0.2
    gap_days: int = 5


@dataclass(frozen=True)
class BudgetSpec:
    """Default PRD budget plus reserves for final test and report."""

    candidates: int = 4
    cloud_calls: int = 3
    cloud_tokens: int = 6000
    wall_seconds: int = 180
    reserve_seconds: int = 60
    reserve_tokens: int = 1200


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    data_path: str
    data_hash: str = ""
    label_horizon: int = 5
    split: SplitSpec = field(default_factory=SplitSpec)
    seed: int = 42
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    primary_metric: str = "daily_auc_mean"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    hypothesis_id: str
    feature_set: str
    model: str
    evidence_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    missing: str = "median"
    scaling: str = "standard"
    fingerprint: str = ""


@dataclass(frozen=True)
class TrialWindowResult:
    auc: float = 0.0
    valid_days: int = 0
    excluded: int = 0


@dataclass(frozen=True)
class TrialResult:
    experiment_id: str
    val1: TrialWindowResult = field(default_factory=TrialWindowResult)
    val2: TrialWindowResult = field(default_factory=TrialWindowResult)
    missing_rates: Dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    error: Optional[str] = None


@dataclass(frozen=True)
class Verdict:
    verdict: str = "inconclusive"
    metric: Dict[str, Any] = field(default_factory=dict)
    checks: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    tokens: int = 0
    seconds: float = 0.0
    artifacts_hash: str = ""
    test: bool = False


@dataclass(frozen=True)
class RunEvent:
    type: str
    phase: str
    timestamp: str
    run_id: str
    decision_reason: str = ""
    experiment_id: str = ""
    tokens: Dict[str, Any] = field(default_factory=dict)
    remaining: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


def as_jsonable(value: Any) -> Any:
    """Convert dataclasses and simple containers into JSON-able values."""
    if is_dataclass(value):
        return {k: as_jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    return value


def _int(raw: Dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(raw.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _float(raw: Dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(raw.get(key, default))
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    return value


def load_task_spec(raw: Dict[str, Any]) -> TaskSpec:
    """Validate and normalize TaskSpec input before a run starts."""
    if not isinstance(raw, dict):
        raise ValueError("TaskSpec must be an object")
    task_id = str(raw.get("task_id") or "").strip()
    data_path = str(raw.get("data_path") or "").strip()
    if not task_id:
        raise ValueError("TaskSpec.task_id is required")
    if not data_path:
        raise ValueError("TaskSpec.data_path is required")

    split_raw = raw.get("split") or {}
    if not isinstance(split_raw, dict):
        raise ValueError("TaskSpec.split must be an object")
    train = _float(split_raw, "train", 0.6)
    test = _float(split_raw, "test", 0.2)
    if not (train > 0 and test > 0 and train + test < 1):
        raise ValueError("TaskSpec.split must keep train+test < 1")

    budget_raw = raw.get("budget") or {}
    if not isinstance(budget_raw, dict):
        raise ValueError("TaskSpec.budget must be an object")

    return TaskSpec(
        task_id=task_id,
        data_path=data_path,
        data_hash=str(raw.get("data_hash") or "").strip(),
        label_horizon=_int(raw, "label_horizon", 5, 1),
        split=SplitSpec(
            train=train,
            val_windows=_int(split_raw, "val_windows", 2, 1),
            test=test,
            gap_days=_int(split_raw, "gap_days", 5, 0),
        ),
        seed=_int(raw, "seed", 42),
        budget=BudgetSpec(
            candidates=_int(budget_raw, "candidates", 4, 1),
            cloud_calls=_int(budget_raw, "cloud_calls", 3, 0),
            cloud_tokens=_int(budget_raw, "cloud_tokens", 6000, 1),
            wall_seconds=_int(budget_raw, "wall_seconds", 180, 1),
            reserve_seconds=_int(budget_raw, "reserve_seconds", 60, 0),
            reserve_tokens=_int(budget_raw, "reserve_tokens", 1200, 0),
        ),
        primary_metric=str(raw.get("primary_metric") or "daily_auc_mean").strip(),
    )


def normalize_experiment_spec(raw: Dict[str, Any]) -> ExperimentSpec:
    """Validate an ExperimentSpec emitted by the planner."""
    if not isinstance(raw, dict):
        raise ValueError("ExperimentSpec must be an object")
    experiment_id = str(raw.get("experiment_id") or "").strip()
    hypothesis_id = str(raw.get("hypothesis_id") or "").strip()
    feature_set = str(raw.get("feature_set") or "").strip()
    model = str(raw.get("model") or "").strip()
    if not experiment_id or not hypothesis_id or not feature_set or not model:
        raise ValueError("experiment_id/hypothesis_id/feature_set/model are required")
    params = raw.get("params")
    if not isinstance(params, dict):
        raise ValueError("ExperimentSpec.params must be an object")
    evidence = [str(x) for x in raw.get("evidence_ids", []) if str(x).strip()]
    return ExperimentSpec(
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        feature_set=feature_set,
        model=model,
        evidence_ids=evidence,
        params=params,
        missing=str(raw.get("missing") or "median").strip(),
        scaling=str(raw.get("scaling") or "standard").strip(),
        fingerprint=str(raw.get("fingerprint") or "").strip(),
    )


def _normalized_window(value: Any) -> TrialWindowResult:
    if value is None:
        return TrialWindowResult()
    if not isinstance(value, dict):
        raise ValueError("TrialResult window must be an object")
    auc_raw = value.get("auc")
    auc = 0.0 if auc_raw is None else float(auc_raw)
    if not 0.0 <= auc <= 1.0:
        raise ValueError("TrialResult window auc must be in [0, 1]")
    valid_days = int(value.get("valid_days") or 0)
    excluded = int(value.get("excluded") or 0)
    if valid_days < 0 or excluded < 0:
        raise ValueError("TrialResult window counts must be non-negative")
    return TrialWindowResult(auc=auc, valid_days=valid_days, excluded=excluded)


def normalize_trial_result(raw: Dict[str, Any]) -> TrialResult:
    """Validate a persisted TrialResult payload before it is trusted."""
    if not isinstance(raw, dict):
        raise ValueError("TrialResult must be an object")
    experiment_id = str(raw.get("experiment_id") or "").strip()
    if not experiment_id:
        raise ValueError("TrialResult.experiment_id is required")
    missing_rates_raw = raw.get("missing_rates") or {}
    if not isinstance(missing_rates_raw, dict):
        raise ValueError("TrialResult.missing_rates must be an object")
    missing_rates = {}
    for key, value in missing_rates_raw.items():
        try:
            missing_rates[str(key)] = float(value)
        except (TypeError, ValueError):
            raise ValueError("TrialResult.missing_rates values must be numbers") from None
    try:
        seconds = float(raw.get("seconds") or 0.0)
    except (TypeError, ValueError):
        raise ValueError("TrialResult.seconds must be a number") from None
    if seconds < 0:
        raise ValueError("TrialResult.seconds must be non-negative")
    error = raw.get("error")
    if error is not None and not isinstance(error, str):
        error = str(error)
    return TrialResult(
        experiment_id=experiment_id,
        val1=_normalized_window(raw.get("val1")),
        val2=_normalized_window(raw.get("val2")),
        missing_rates=missing_rates,
        seconds=seconds,
        error=error,
    )


def validate_run_event(raw: Dict[str, Any]) -> RunEvent:
    """Validate a RunEvent payload; required contract for replay and D."""
    if not isinstance(raw, dict):
        raise ValueError("RunEvent must be an object")
    for key in ("type", "phase", "timestamp", "run_id"):
        if not str(raw.get(key) or "").strip():
            raise ValueError(f"RunEvent.{key} is required")
    tokens = raw.get("tokens")
    remaining = raw.get("remaining")
    extra = raw.get("extra")
    if tokens is None:
        tokens = {}
    if remaining is None:
        remaining = {}
    if extra is None:
        extra = {}
    if not all(isinstance(value, dict) for value in (tokens, remaining, extra)):
        raise ValueError("RunEvent.tokens/remaining/extra must be objects")
    return build_run_event(
        str(raw["type"]).strip(),
        str(raw["phase"]).strip(),
        str(raw["timestamp"]).strip(),
        str(raw["run_id"]).strip(),
        decision_reason=str(raw.get("decision_reason") or ""),
        experiment_id=str(raw.get("experiment_id") or ""),
        tokens=tokens,
        remaining=remaining,
        extra=extra,
    )


def experiment_fingerprint(
    spec: ExperimentSpec,
    data_hash: str,
    seed: int,
    split: Any,
) -> str:
    """Deterministic fingerprint used to skip duplicate experiments."""
    body = {
        "data_hash": data_hash,
        "seed": seed,
        "split": as_jsonable(split),
        "feature_set": spec.feature_set,
        "model": spec.model,
        "params": spec.params,
        "missing": spec.missing,
        "scaling": spec.scaling,
    }
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_run_event(
    event_type: str,
    phase: str,
    timestamp: str,
    run_id: str,
    *,
    decision_reason: str = "",
    experiment_id: str = "",
    tokens: Optional[Dict[str, Any]] = None,
    remaining: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RunEvent:
    return RunEvent(
        type=event_type,
        phase=phase,
        timestamp=timestamp,
        run_id=run_id,
        decision_reason=decision_reason,
        experiment_id=experiment_id,
        tokens=dict(tokens or {}),
        remaining=dict(remaining or {}),
        extra=dict(extra or {}),
    )
