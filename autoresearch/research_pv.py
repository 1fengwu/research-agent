"""Helix 赫力 角色 B：量价研究规划器（A 的 PlannerPort 实现）。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    ExperimentSpec,
    TaskSpec,
    as_jsonable,
    experiment_fingerprint,
)

THRESHOLD = 0.01

PLAN_ORDER: List[Tuple[str, str, str, Dict[str, Any], str, str]] = [
    ("volatility", "+volatility", "logistic", {}, "median", "standard"),
    ("volume", "+volume", "logistic", {}, "median", "standard"),
    ("volatility_volume", "+volatility_volume", "logistic", {}, "median", "standard"),
    ("tree_shallow", "base", "tree_shallow", {"max_depth": 5, "min_samples_leaf": 20}, "median", "standard"),
    ("constant", "base", "logistic", {}, "constant", "standard"),
    ("robust", "base", "logistic", {}, "median", "robust"),
    ("momentum", "+momentum", "logistic", {}, "median", "standard"),
    ("cross", "+cross", "logistic", {}, "median", "standard"),
    ("gradient_boost", "base", "gradient_boost", {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "min_samples_leaf": 20}, "median", "standard"),
    ("random_forest", "base", "random_forest", {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 20}, "median", "standard"),
]


class ResearchPVPlanner:
    """预注册假设，并把两窗 TrialResult 转成 A 可读的证据信号。"""

    def __init__(
        self,
        order: Optional[List[str]] = None,
        evolution: Optional[Any] = None,
    ) -> None:
        self._order = list(order) if order else [item[0] for item in PLAN_ORDER]
        self._specs: Dict[str, ExperimentSpec] = {}
        self._evolution = evolution

    def _evolved_order(self) -> List[str]:
        if self._evolution is not None:
            return list(self._evolution.active_plan_order(self._order))
        return list(self._order)

    def _record_evolution(self, signals: List[Dict[str, Any]]) -> None:
        if self._evolution is not None:
            self._evolution.record_signals(signals)

    def _make_spec(self, task: TaskSpec, index: int, hid: str) -> ExperimentSpec:
        feature_set, model, params, missing, scaling = None, None, None, None, None
        for candidate_hid, fs, md, prm, mis, sca in PLAN_ORDER:
            if candidate_hid == hid:
                feature_set, model, params, missing, scaling = fs, md, prm, mis, sca
                break
        if feature_set is None:
            raise ValueError("未知候选假设: %s" % hid)
        return ExperimentSpec(
            experiment_id="exp-%03d" % index,
            hypothesis_id=hid,
            feature_set=feature_set,
            model=model,
            evidence_ids=["hypothesis:" + hid],
            params=dict(params),
            missing=missing,
            scaling=scaling,
        )

    def plan(
        self,
        task: TaskSpec,
        view: Dict[str, Any],
        run_kind: str,
    ) -> List[ExperimentSpec]:
        del run_kind
        attempted = set(str(x) for x in view.get("attempted", []))
        limit = max(1, int(task.budget.candidates))
        out: List[ExperimentSpec] = []
        index = 1
        for hid in self._evolved_order():
            if len(out) >= limit:
                break
            spec = self._make_spec(task, index, hid)
            index += 1
            fingerprint = experiment_fingerprint(
                spec,
                task.data_hash,
                task.seed,
                as_jsonable(task.split),
            )
            if fingerprint in attempted:
                continue
            self._specs[spec.experiment_id] = spec
            out.append(spec)
        return out

    def _hypothesis(self, experiment_id: str) -> Tuple[str, str]:
        spec = self._specs.get(experiment_id)
        if spec is None:
            return "features", ""
        axis = "features"
        if spec.model != "logistic" or spec.params:
            axis = "model"
        elif spec.missing != "median":
            axis = "missing"
        elif spec.scaling != "standard":
            axis = "scaling"
        return axis, spec.hypothesis_id

    def interpret(
        self,
        feedback: Dict[str, Any],
        view: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        trial = feedback.get("trial") or {}
        if not isinstance(trial, dict):
            return []
        experiment_id = str(trial.get("experiment_id") or "")
        axis, hypothesis = self._hypothesis(experiment_id)
        baseline = view.get("baseline") or {}
        baseline_auc = float(baseline.get("merged_auc") or 0.0)

        if trial.get("error"):
            return [
                {
                    "id": "%s:experiment_error" % experiment_id,
                    "reason": "experiment_error",
                    "axis": axis,
                    "hypothesis_id": hypothesis,
                    "refute": [],
                    "support": [],
                    "detail": str(trial.get("error")),
                }
            ]

        val1 = trial.get("val1") or {}
        val2 = trial.get("val2") or {}
        v1 = float(val1.get("auc") or 0.0)
        v2 = float(val2.get("auc") or 0.0)
        merged = (v1 + v2) / 2.0

        dropped = []
        improved = []
        for name, current in (("val1", v1), ("val2", v2)):
            if baseline_auc > 0:
                delta = current - baseline_auc
                if delta >= THRESHOLD:
                    improved.append(name)
                if delta <= -THRESHOLD:
                    dropped.append(name)

        support: List[str] = []
        refute: List[str] = []
        if dropped:
            reason = "window_conflict" if improved else "window_drop"
            refute = [hypothesis] if hypothesis else []
        elif baseline_auc > 0 and merged - baseline_auc >= THRESHOLD:
            reason = "merged_improve"
            support = [hypothesis] if hypothesis else []
        elif baseline_auc > 0 and baseline_auc - merged >= THRESHOLD:
            reason = "merged_drop"
            refute = [hypothesis] if hypothesis else []
        else:
            verdict = (feedback.get("verdict") or {}).get("verdict")
            if verdict == "refuted":
                reason = "oracle_refuted"
                refute = [hypothesis] if hypothesis else []
            elif verdict == "blocked":
                reason = "oracle_blocked"
            else:
                return []

        out = [
            {
                "id": "%s:%s" % (experiment_id, reason),
                "reason": reason,
                "axis": axis,
                "hypothesis_id": hypothesis,
                "refute": refute,
                "support": support,
                "detail": "merged=%.6f baseline=%.6f dropped=%s improved=%s"
                % (merged, baseline_auc, dropped, improved),
            }
        ]
        self._record_evolution(out)
        return out


def plan(task: TaskSpec, view: Dict[str, Any], run_kind: str) -> List[ExperimentSpec]:
    return _DEFAULT.plan(task, view, run_kind)


def interpret(
    feedback: Dict[str, Any],
    view: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return _DEFAULT.interpret(feedback, view)


_DEFAULT = ResearchPVPlanner()
