"""Backend output contract validators.

Version: ``autoresearch-outputs-v1``.

前端、API 队友和评审/复现工具都应只消费这些结构；任何改动需要在
docs/后端数据契约.md 中同步更新。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from .contracts import validate_run_event

REQUIRED_SUMMARY_FIELDS = (
    "run_id",
    "task_id",
    "phase",
    "stop_reason",
    "tested",
    "locked",
    "test_verdict",
    "budget",
    "ledger",
    "elapsed_seconds",
    "event_path",
)

REQUIRED_LEDGER_FIELDS = (
    "entries",
    "used_candidates",
    "used_cloud_calls",
    "used_cloud_tokens",
)

REQUIRED_VERDICT_FIELDS = (
    "run_id",
    "experiment_id",
    "verdict",
    "metric",
    "checks",
    "evidence_ids",
    "artifacts_hash",
    "test",
)

REQUIRED_PRE_GATE_FIELDS = (
    "schema_version",
    "task_id",
    "data_hash",
    "split_fingerprint",
    "seed",
    "conclusion",
    "passed",
    "reasons",
    "baseline",
    "candidates",
)

REQUIRED_METRICS_FIELDS = (
    "schema_version",
    "task_id",
    "completion_rate",
    "arms",
)

REQUIRED_REPLAY_FIELDS = (
    "schema_version",
    "run_id",
    "status",
    "checks",
    "replay_metric",
    "recorded_metric",
    "cloud_calls_replayed",
)


def _require_keys(value: Any, keys: tuple, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 JSON 对象")
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{label} 缺少契约字段: {', '.join(missing)}")


def _require_list(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{label} 必须是数组")


def _require_dict(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 JSON 对象")


def _require_str(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{label} 必须是字符串")


def _require_int(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} 必须是整数")


def _require_bool(value: Any, label: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{label} 必须是布尔值")


def _window(value: Any, label: str) -> None:
    _require_dict(value, label)
    _require_keys(value, ("auc", "valid_days", "excluded"), label)


def validate_summary(summary: Dict[str, Any]) -> None:
    """校验 `python -m autoresearch` 的 summary 结构。"""
    _require_keys(summary, REQUIRED_SUMMARY_FIELDS, "summary")
    _require_bool(summary["tested"], "summary.tested")
    _require_dict(summary["locked"], "summary.locked")
    if summary["tested"] or summary["test_verdict"] is not None:
        _require_dict(summary["test_verdict"], "summary.test_verdict")
        _require_keys(
            summary["test_verdict"],
            ("verdict", "metric", "checks", "evidence_ids", "artifacts_hash", "test"),
            "summary.test_verdict",
        )
    _require_dict(summary["budget"], "summary.budget")
    _require_dict(summary["ledger"], "summary.ledger")
    _require_str(summary["run_id"], "summary.run_id")
    _require_str(summary["task_id"], "summary.task_id")
    _require_str(summary["phase"], "summary.phase")
    _require_str(summary["stop_reason"], "summary.stop_reason")
    _require_keys(summary["locked"], ("name",), "summary.locked")
    _require_keys(
        summary["budget"],
        ("used_candidates", "used_cloud_calls", "used_cloud_tokens"),
        "summary.budget",
    )
    _require_keys(summary["ledger"], REQUIRED_LEDGER_FIELDS, "summary.ledger")


def validate_event_list(events: List[Dict[str, Any]]) -> None:
    """校验 runner-events.jsonl 的每一行。"""
    _require_list(events, "events")
    for index, event in enumerate(events):
        try:
            validate_run_event(event)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"events[{index}] 不符合 RunEvent 契约: {exc}") from exc


def validate_ledger(ledger: Dict[str, Any]) -> None:
    """校验 per-run ledger JSON。"""
    _require_keys(ledger, REQUIRED_LEDGER_FIELDS, "ledger")
    _require_list(ledger["entries"], "ledger.entries")
    for index, entry in enumerate(ledger["entries"]):
        _require_dict(entry, f"ledger.entries[{index}]")
        _require_keys(entry, ("kind", "ts"), f"ledger.entries[{index}]")
        kind = entry["kind"]
        if kind == "candidate":
            _require_keys(entry, ("note", "snapshot"), f"ledger.entries[{index}]")
            _require_dict(entry["snapshot"], f"ledger.entries[{index}].snapshot")
        elif kind == "cloud_call":
            _require_keys(
                entry,
                (
                    "estimated_input",
                    "actual_input",
                    "actual_output",
                    "charged",
                    "decision_note",
                ),
                f"ledger.entries[{index}]",
            )
            if "meta" in entry:
                _require_dict(entry["meta"], f"ledger.entries[{index}].meta")
            if "failure_signature" in entry and not isinstance(entry["failure_signature"], str):
                raise TypeError(
                    f"ledger.entries[{index}].failure_signature 必须是字符串"
                )
        else:
            raise ValueError(
                f"ledger.entries[{index}].kind 未知: {kind}（允许 candidate/cloud_call）"
            )


def validate_verdict_record(record: Dict[str, Any]) -> None:
    """校验 verdicts/<run_id>.json 中的一条裁决。"""
    _require_keys(record, REQUIRED_VERDICT_FIELDS, "verdict_record")
    _require_str(record["run_id"], "verdict_record.run_id")
    _require_str(record["experiment_id"], "verdict_record.experiment_id")
    _require_str(record["verdict"], "verdict_record.verdict")
    _require_dict(record["metric"], "verdict_record.metric")
    _require_list(record["checks"], "verdict_record.checks")
    _require_list(record["evidence_ids"], "verdict_record.evidence_ids")
    _require_str(record["artifacts_hash"], "verdict_record.artifacts_hash")
    _require_bool(record["test"], "verdict_record.test")


def validate_verdicts(records: List[Dict[str, Any]]) -> None:
    _require_list(records, "verdicts")
    for record in records:
        validate_verdict_record(record)


def validate_pre_gate(payload: Dict[str, Any]) -> None:
    """校验 pre_gate.json。"""
    _require_keys(payload, REQUIRED_PRE_GATE_FIELDS, "pre_gate")
    _require_bool(payload["passed"], "pre_gate.passed")
    _require_str(payload["conclusion"], "pre_gate.conclusion")
    _require_list(payload["reasons"], "pre_gate.reasons")
    _require_dict(payload["baseline"], "pre_gate.baseline")
    _require_keys(
        payload["baseline"],
        ("merged_auc", "val1", "val2"),
        "pre_gate.baseline",
    )
    _window(payload["baseline"]["val1"], "pre_gate.baseline.val1")
    _window(payload["baseline"]["val2"], "pre_gate.baseline.val2")
    _require_list(payload["candidates"], "pre_gate.candidates")
    for index, candidate in enumerate(payload["candidates"]):
        label = f"pre_gate.candidates[{index}]"
        _require_dict(candidate, label)
        _require_keys(
            candidate,
            ("experiment_id", "merged_auc", "val1", "val2", "paired_ci", "passed"),
            label,
        )
        _window(candidate["val1"], f"{label}.val1")
        _window(candidate["val2"], f"{label}.val2")
        _require_dict(candidate["paired_ci"], f"{label}.paired_ci")
        _require_dict(
            candidate["paired_ci"].get("merged") or {},
            f"{label}.paired_ci.merged",
        )
        _require_keys(
            candidate["paired_ci"]["merged"],
            ("delta", "lower", "upper", "pairs"),
            f"{label}.paired_ci.merged",
        )
        _require_bool(candidate["passed"], f"{label}.passed")


def validate_metrics(metrics: Dict[str, Any]) -> None:
    """校验 evaluate.py 输出的 metrics.json。"""
    _require_keys(metrics, REQUIRED_METRICS_FIELDS, "metrics")
    _require_list(metrics["arms"], "metrics.arms")
    for index, arm in enumerate(metrics["arms"]):
        label = f"metrics.arms[{index}]"
        _require_dict(arm, label)
        _require_keys(
            arm,
            ("run_id", "arm", "completion", "phase"),
            label,
        )
        _require_str(arm["run_id"], f"{label}.run_id")
        _require_str(arm["arm"], f"{label}.arm")
        completion = arm["completion"]
        if not isinstance(completion, (int, float)) or isinstance(completion, bool):
            raise TypeError(f"{label}.completion 必须是 0/1 数值")
        if completion not in (0, 1):
            raise ValueError(f"{label}.completion 必须是 0 或 1")
        _require_str(arm["phase"], f"{label}.phase")
        if "usage_source" in arm:
            _require_str(arm["usage_source"], f"{label}.usage_source")


def validate_replay(replay: Dict[str, Any]) -> None:
    """校验 replay 输出 summary.json。"""
    _require_keys(replay, REQUIRED_REPLAY_FIELDS, "replay")
    _require_str(replay["run_id"], "replay.run_id")
    _require_str(replay["status"], "replay.status")
    _require_list(replay["checks"], "replay.checks")
    for index, check in enumerate(replay["checks"]):
        label = f"replay.checks[{index}]"
        _require_keys(check, ("name", "passed", "detail"), label)
        _require_bool(check["passed"], f"{label}.passed")
    _require_dict(replay["replay_metric"], "replay.replay_metric")
    _require_dict(replay["recorded_metric"], "replay.recorded_metric")
    _require_int(replay["cloud_calls_replayed"], "replay.cloud_calls_replayed")


def validate_artifact(path: Union[str, Path], kind: str) -> Any:
    """读取一个产物文件并校验，返回解析后的内容。"""
    load_path = Path(path)
    data = json.loads(load_path.read_text(encoding="utf-8"))
    if kind == "event_list":
        if isinstance(data, list):
            validate_event_list(data)
        else:
            raise ValueError("event_list 必须是数组")
    elif kind == "verdicts":
        validate_verdicts(data)
    elif kind == "summary":
        validate_summary(data)
    elif kind == "ledger":
        validate_ledger(data)
    elif kind == "pre_gate":
        validate_pre_gate(data)
    elif kind == "metrics":
        validate_metrics(data)
    elif kind == "replay":
        validate_replay(data)
    else:
        raise ValueError(f"未知产物类型: {kind}")
    return data
