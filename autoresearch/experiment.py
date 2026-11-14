"""AutoResearch 角色 B：固定基线、候选实验与锁定配置的独立测试入口。

纪律：
- 插补、缩放、模型只在 train 段 fit；
- 验证窗只读 val1/val2，holdout 只通过 run_holdout_test 提供给 C；
- 单轮实验只能改一个特征组或一个模型/预处理设置。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .contracts import ExperimentSpec, TaskSpec, TrialResult, TrialWindowResult
from .data import (
    FEATURE_GROUPS,
    SplitResult,
    add_features,
    build_profile,
    canonical_panel_hash,
    compute_labels,
    feature_fingerprint,
    file_sha256,
    make_split,
    read_panel,
)

BASELINE_SPEC: Dict[str, Any] = {
    "experiment_id": "baseline",
    "hypothesis_id": None,
    "evidence_ids": [],
    "feature_set": "base",
    "model": "logistic",
    "params": {},
    "missing": "median",
    "scaling": "standard",
}

ALLOWED_MODELS = ("logistic", "tree_shallow")
ALLOWED_MISSING = ("drop", "median", "constant")
ALLOWED_SCALING = ("none", "standard", "robust")


def baseline_spec(experiment_id: str = "baseline") -> Dict[str, Any]:
    return dict(BASELINE_SPEC, experiment_id=experiment_id)


def normalize_spec(
    spec: Dict[str, Any],
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """把 ExperimentSpec 字典归一化；缺字段给基线默认值。"""
    if not isinstance(spec, dict):
        raise TypeError("ExperimentSpec 必须是 dict")
    out = {
        "experiment_id": experiment_id or spec.get("experiment_id") or "exp-unknown",
        "hypothesis_id": spec.get("hypothesis_id"),
        "evidence_ids": list(spec.get("evidence_ids") or []),
        "feature_set": spec.get("feature_set", "base"),
        "model": spec.get("model", "logistic"),
        "params": dict(spec.get("params") or {}),
        "missing": spec.get("missing", "median"),
        "scaling": spec.get("scaling", "standard"),
    }
    if out["feature_set"] not in FEATURE_GROUPS:
        raise ValueError(
            "未知特征组 %s，可选: %s" % (out["feature_set"], ", ".join(sorted(FEATURE_GROUPS)))
        )
    if out["model"] not in ALLOWED_MODELS:
        raise ValueError("未知模型 %s，可选: %s" % (out["model"], ", ".join(ALLOWED_MODELS)))
    if out["missing"] not in ALLOWED_MISSING:
        raise ValueError("未知缺失处理 %s，可选: %s" % (out["missing"], ", ".join(ALLOWED_MISSING)))
    if out["scaling"] not in ALLOWED_SCALING:
        raise ValueError("未知缩放 %s，可选: %s" % (out["scaling"], ", ".join(ALLOWED_SCALING)))
    return out


def experiment_fingerprint(spec: Dict[str, Any]) -> str:
    """特征/模型/预处理配置的稳定指纹，重复实验据此跳过。"""
    normalized = normalize_spec(spec)
    body = json.dumps(
        {
            "schema": "autoresearch-experiment-v1",
            "feature_set": normalized["feature_set"],
            "model": normalized["model"],
            "params": normalized["params"],
            "missing": normalized["missing"],
            "scaling": normalized["scaling"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_single_change(
    spec: Dict[str, Any],
    base: Optional[Dict[str, Any]] = None,
) -> None:
    """只允许一个特征组变化，或一个模型/预处理设置变化。"""
    current = normalize_spec(spec)
    reference = normalize_spec(base or BASELINE_SPEC)
    feature_changed = current["feature_set"] != reference["feature_set"]
    model_changed = (
        current["model"] != reference["model"] or current["params"] != reference["params"]
    )
    preprocess_changed = (
        current["missing"] != reference["missing"] or current["scaling"] != reference["scaling"]
    )
    setting_changes = int(model_changed) + int(preprocess_changed)
    if feature_changed and setting_changes:
        raise ValueError(
            "候选 exp=%s 同时改了特征组与模型/预处理设置；单轮只允许一个变化轴"
            % current["experiment_id"]
        )
    if not feature_changed and setting_changes > 1:
        raise ValueError(
            "候选 exp=%s 改了 %d 个模型/预处理设置；单轮最多改一个"
            % (current["experiment_id"], setting_changes)
        )


def feature_columns(spec: Dict[str, Any]) -> List[str]:
    normalized = normalize_spec(spec)
    return list(FEATURE_GROUPS[normalized["feature_set"]])


def daily_auc(
    labels: Sequence[float],
    scores: Sequence[float],
    dates: Sequence[Any],
    *,
    include_daily: bool = False,
) -> Dict[str, Any]:
    """逐交易日 AUC，再对两类都出现的有效日等权平均。"""
    frame = pd.DataFrame(
        {"date": pd.to_datetime(list(dates)), "label": list(labels), "score": list(scores)}
    ).dropna()
    aucs: List[float] = []
    daily_aucs: List[Dict[str, Any]] = []
    invalid_days = 0
    excluded_rows = 0
    for _, day in frame.groupby("date", sort=True):
        if day["label"].nunique() < 2 or len(day) < 2:
            invalid_days += 1
            excluded_rows += int(len(day))
            continue
        value = float(roc_auc_score(day["label"], day["score"]))
        if math.isfinite(value):
            aucs.append(value)
            daily_aucs.append(
                {
                    "date": str(day["date"].iloc[0].date()),
                    "auc": round(float(value), 6),
                }
            )
        else:
            invalid_days += 1
            excluded_rows += int(len(day))
    result = {
        "auc": round(float(sum(aucs) / len(aucs)), 6) if aucs else None,
        "valid_days": len(aucs),
        "excluded_days": invalid_days,
        "excluded": excluded_rows,
    }
    if include_daily:
        result["daily"] = daily_aucs
    return result


def _fit_model(
    x_train: Any,
    y_train: Any,
    spec: Dict[str, Any],
    seed: int,
):
    if len(set(y_train)) < 2:
        raise ValueError("训练段只有一个类别，无法训练二分类模型")
    params = dict(spec.get("params") or {})
    model_name = spec["model"]
    if model_name == "logistic":
        model_params = {
            "C": 1.0,
            "max_iter": 500,
            "random_state": seed,
        }
        model_params.update(params)
        model = LogisticRegression(**model_params)
    elif model_name == "tree_shallow":
        tree_params = {
            "max_depth": 5,
            "min_samples_leaf": 20,
            "random_state": seed,
        }
        tree_params.update(params)
        model = DecisionTreeClassifier(**tree_params)
    else:  # pragma: no cover - normalize_spec 已拦截
        raise ValueError("不支持模型 %s" % model_name)
    model.fit(x_train, y_train)
    return model


def _preprocess(
    train: pd.DataFrame,
    eval_frames: List[pd.DataFrame],
    feature_cols: List[str],
    missing: str,
    scaling: str,
) -> Tuple[Any, Any, List[Tuple[Any, pd.DataFrame]]]:
    if missing == "drop":
        train_clean = train.dropna(subset=feature_cols).copy()
        eval_clean = [frame.dropna(subset=feature_cols).copy() for frame in eval_frames]
    elif missing == "median":
        medians = train[feature_cols].median()
        train_clean = train.fillna(medians).copy()
        eval_clean = [frame.fillna(medians).copy() for frame in eval_frames]
    elif missing == "constant":
        train_clean = train.fillna(0.0).copy()
        eval_clean = [frame.fillna(0.0).copy() for frame in eval_frames]
    else:  # pragma: no cover - normalize_spec 已拦截
        raise ValueError("未知缺失处理 %s" % missing)

    if scaling == "standard":
        scaler = StandardScaler().fit(train_clean[feature_cols].to_numpy(dtype="float64"))
    elif scaling == "robust":
        scaler = RobustScaler().fit(train_clean[feature_cols].to_numpy(dtype="float64"))
    else:
        scaler = None

    def transform(frame: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
        x = frame[feature_cols].to_numpy(dtype="float64")
        if scaler is not None:
            x = scaler.transform(x)
        return x, frame

    x_train, _ = transform(train_clean)
    y_train = train_clean["label"].to_numpy(dtype="float64")
    transformed_evals = [transform(frame) for frame in eval_clean]
    return x_train, y_train, transformed_evals


def _score_window(
    model,
    x_eval: Any,
    frame: pd.DataFrame,
    *,
    include_daily: bool = False,
) -> Dict[str, Any]:
    proba = model.predict_proba(x_eval)[:, 1]
    return daily_auc(
        frame["label"],
        proba,
        frame["date"],
        include_daily=include_daily,
    )


def _run_segments(
    spec: Dict[str, Any],
    labelled: pd.DataFrame,
    eval_names: Sequence[str],
    seed: int,
    *,
    include_daily: bool = False,
):
    train = labelled[labelled["split"] == "train"].dropna(subset=["label"]).copy()
    eval_frames = [
        labelled[labelled["split"] == name].dropna(subset=["label"]).copy()
        for name in eval_names
    ]
    if len(train) == 0:
        raise ValueError("训练段没有可用样本")
    if any(len(frame) == 0 for frame in eval_frames):
        raise ValueError("验证/测试段没有可用样本")

    feature_cols = feature_columns(spec)
    x_train, y_train, transformed = _preprocess(
        train, eval_frames, feature_cols, spec["missing"], spec["scaling"]
    )
    model = _fit_model(x_train, y_train, spec, seed)
    results = {
        name: _score_window(
            model,
            x_eval,
            frame,
            include_daily=include_daily,
        )
        for (name, (x_eval, frame)) in zip(eval_names, transformed)
    }
    return results, model


def _base_payload(spec: Dict[str, Any], split: SplitResult, started: float) -> Dict[str, Any]:
    return {
        "experiment_id": spec["experiment_id"],
        "hypothesis_id": spec.get("hypothesis_id"),
        "evidence_ids": list(spec.get("evidence_ids") or []),
        "spec": {
            "feature_set": spec["feature_set"],
            "model": spec["model"],
            "params": dict(spec["params"]),
            "missing": spec["missing"],
            "scaling": spec["scaling"],
        },
        "fingerprint": experiment_fingerprint(spec),
        "data_hash": split.meta.get("data_hash"),
        "split_fingerprint": split.fingerprint,
        "seconds": round(time.perf_counter() - started, 3),
    }


def run_experiment(
    spec: Dict[str, Any],
    split: SplitResult,
    *,
    seed: int = 42,
    include_daily: bool = False,
) -> Dict[str, Any]:
    """执行一个候选实验并输出两窗 TrialResult。"""
    started = time.perf_counter()
    normalized = normalize_spec(spec)
    if normalized["experiment_id"].lower() not in ("baseline",):
        validate_single_change(normalized)
    out = _base_payload(normalized, split, started)
    out["val1"] = {"auc": None, "valid_days": 0, "excluded_days": 0, "excluded": 0}
    out["val2"] = {"auc": None, "valid_days": 0, "excluded_days": 0, "excluded": 0}
    out["merged_auc"] = None
    out["missing_rates"] = {}
    out["error"] = None

    try:
        labelled = add_features(split.frame, normalized["feature_set"])
        results, _model = _run_segments(
            normalized,
            labelled,
            ("val1", "val2"),
            seed,
            include_daily=include_daily,
        )
        out["val1"] = results["val1"]
        out["val2"] = results["val2"]
        aucs = [out["val1"]["auc"], out["val2"]["auc"]]
        if all(auc is not None for auc in aucs):
            out["merged_auc"] = round(float(sum(aucs) / len(aucs)), 6)
        working = labelled[labelled["split"].isin(["train", "val1", "val2"])]
        out["missing_rates"] = {
            col: round(float(working[col].isna().mean()), 6) for col in feature_columns(normalized)
        }
    except Exception as exc:  # noqa: BLE001 - 实验结果必须带 error 而不是中断编排
        out["error"] = "%s: %s" % (type(exc).__name__, exc)

    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def baseline(split: SplitResult, *, seed: int = 42) -> Dict[str, Any]:
    """PRD 固定基线：LogisticRegression[r5, r20]。"""
    return run_experiment(baseline_spec(), split, seed=seed)


def run_candidate(
    spec: Dict[str, Any],
    split: SplitResult,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """A 批准的候选实验入口。"""
    return run_experiment(spec, split, seed=seed)


def run_holdout_test(
    spec: Dict[str, Any],
    split: SplitResult,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """锁定配置的独立测试入口；只提供 C 的 oracle 调用，B 不自行判决。"""
    started = time.perf_counter()
    normalized = normalize_spec(spec)
    out = _base_payload(normalized, split, started)
    out["test"] = {"auc": None, "valid_days": 0, "excluded_days": 0, "excluded": 0}
    out["error"] = None
    try:
        labelled = add_features(split.frame, normalized["feature_set"])
        results, _model = _run_segments(normalized, labelled, ("test",), seed)
        out["test"] = results["test"]
    except Exception as exc:  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out


def write_trial_result(trial: Dict[str, Any], path) -> None:
    """把 TrialResult 写成 experiments/*.json。"""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(trial, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


def _as_window(value: Any) -> TrialWindowResult:
    data = value or {}
    return TrialWindowResult(
        auc=float(data.get("auc") or 0.0),
        valid_days=int(data.get("valid_days") or 0),
        excluded=int(data.get("excluded") or 0),
    )


class ResearchBackend:
    """A 的 DataPort 实现：数据画像、固定基线和候选实验。"""

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self._prepared_task: Optional[TaskSpec] = None
        self._panel = None
        self._split: Optional[SplitResult] = None
        self._baseline: Optional[Dict[str, Any]] = None

    def _prepare(self, task: TaskSpec):
        if task != self._prepared_task or self._split is None:
            data_path = Path(task.data_path)
            if not data_path.is_absolute():
                data_path = Path(__file__).resolve().parents[1] / data_path
            panel = read_panel(data_path)
            data_hash = canonical_panel_hash(panel)
            if task.data_hash and task.data_hash not in (data_hash, file_sha256(data_path)):
                raise ValueError("TaskSpec.data_hash does not match the input data")
            labelled = compute_labels(panel, horizon=task.label_horizon)
            split = make_split(
                labelled,
                train_ratio=task.split.train,
                val_windows=task.split.val_windows,
                test_ratio=task.split.test,
                gap_days=task.split.gap_days,
                seed=task.seed,
                data_hash=task.data_hash or data_hash,
                split_id=task.task_id,
            )
            self._prepared_task = task
            self._panel = panel
            self._split = split
            self._baseline = None
        return self._panel, self._split

    def prepared_split(self) -> SplitResult:
        """Share the run's frozen split with the scoring component."""
        return self._prepare(self.task)[1]

    def describe_data(self, task: TaskSpec) -> Dict[str, Any]:
        panel, _split = self._prepare(task)
        profile = build_profile(panel, horizon=task.label_horizon)
        profile["date_range"] = [profile["date_start"], profile["date_end"]]
        rates = profile.get("missing_rates") or {}
        profile["missing_rate"] = max(rates.values()) if rates else 0.0
        anomalies = []
        if profile.get("label_computable_ratio", 1.0) < 0.5:
            anomalies.append("label_maturity")
        if not profile.get("symbols"):
            anomalies.append("empty_panel")
        profile["anomalies"] = anomalies
        profile["data_anomaly"] = bool(anomalies)
        return profile

    def baseline(self, task: TaskSpec) -> Dict[str, Any]:
        _panel, split = self._prepare(task)
        if self._baseline is not None:
            return self._baseline
        trial = run_experiment(baseline_spec("baseline"), split, seed=task.seed)
        result = {
            "merged_auc": float(trial["merged_auc"] if trial["merged_auc"] is not None else 0.0),
            "config": dict(trial["spec"]),
            "trial": trial,
            "error": trial["error"],
            "data_hash": split.meta.get("data_hash"),
            "split_fingerprint": split.fingerprint,
        }
        self._baseline = result
        return result

    def run_candidate(self, spec: ExperimentSpec) -> TrialResult:
        _panel, split = self._prepare(self.task)
        raw: Dict[str, Any] = {
            "experiment_id": spec.experiment_id,
            "hypothesis_id": spec.hypothesis_id,
            "evidence_ids": list(spec.evidence_ids),
            "feature_set": spec.feature_set,
            "model": spec.model,
            "params": dict(spec.params),
            "missing": spec.missing or "median",
            "scaling": spec.scaling or "standard",
        }
        trial = run_experiment(raw, split, seed=self.task.seed)
        return TrialResult(
            experiment_id=trial["experiment_id"],
            val1=_as_window(trial.get("val1")),
            val2=_as_window(trial.get("val2")),
            missing_rates=dict(trial.get("missing_rates") or {}),
            seconds=float(trial.get("seconds") or 0.0),
            error=trial.get("error"),
        )
