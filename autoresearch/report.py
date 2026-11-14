"""AutoResearch 角色 D：事件展示、报告生成与复现包打包。

D 只消费 A 的事件流、B 的数据摘要和 C 的指标/verdict 文件；
不生成研究结论、不写 holdout、不调用 LLM。
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_events(path: str | Path) -> List[Dict[str, Any]]:
    """读取 `.json`（数组）或 `.jsonl` 事件文件。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("事件 .json 必须是数组")
        return [event for event in data if isinstance(event, dict)]

    events: List[Dict[str, Any]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"事件文件第 {index} 行不是合法 JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise ValueError(f"事件文件第 {index} 行不是事件对象")
        events.append(event)
    return events


def group_runs(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 run_id 分组并保持首次出现顺序。"""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for event in events:
        run_id = str(event.get("run_id") or "default")
        if run_id not in grouped:
            grouped[run_id] = []
            order.append(run_id)
        grouped[run_id].append(event)
    return [{"run_id": run_id, "events": grouped[run_id]} for run_id in order]


def _first(events: List[Dict[str, Any]], event_type: str) -> Optional[Dict[str, Any]]:
    for event in events or []:
        if event.get("type") == event_type:
            return event
    return None


def _last(events: List[Dict[str, Any]], event_type: str) -> Optional[Dict[str, Any]]:
    for event in reversed(events or []):
        if event.get("type") == event_type:
            return event
    return None


def _count(events: List[Dict[str, Any]], event_type: str) -> int:
    return sum(1 for event in events or [] if event.get("type") == event_type)


def _extra(event: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if not event:
        return default
    extra = event.get("extra")
    if not isinstance(extra, dict):
        return default
    value = extra.get(key)
    return value if value is not None else default


def _window(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "auc": value.get("auc"),
        "valid_days": value.get("valid_days", 0),
        "excluded": value.get("excluded", 0),
        "excluded_days": value.get("excluded_days", 0),
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _window_row(value: Any) -> str:
    window = _window(value)
    return f"{_fmt(window.get('auc'))} ({int(window.get('valid_days') or 0)}d)"


def _summary_from_completed(event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not event:
        return {}
    summary = _extra(event, "summary")
    return summary if isinstance(summary, dict) else {}


def summarize_run(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把一次运行的事件聚合成 UI/报告可读的事实快照。"""
    run_id = str((events or [{}])[0].get("run_id") or "default")
    completed = _last(events, "research.completed")
    completed_summary = _summary_from_completed(completed)
    last_remaining = None
    for event in reversed(events or []):
        remaining = event.get("remaining")
        if isinstance(remaining, dict):
            last_remaining = remaining
            break

    task = _extra(_first(events, "research.run_started"), "task", {})
    if isinstance(task, dict) and task.get("truncated"):
        task = {}
    profile = _extra(_first(events, "research.data_profile"), "profile", {})
    if isinstance(profile, dict) and profile.get("truncated"):
        profile = {}
    baseline = _extra(_first(events, "research.baseline"), "baseline", {})
    if isinstance(baseline, dict) and baseline.get("truncated"):
        baseline = {}

    hypotheses: List[Dict[str, Any]] = []
    for event in events or []:
        if event.get("type") != "research.hypothesis_registered":
            continue
        spec = event.get("extra", {}).get("spec") if isinstance(event.get("extra"), dict) else None
        hypotheses.append(
            {
                "experiment_id": str(event.get("experiment_id") or ""),
                "spec": spec if isinstance(spec, dict) else {},
            }
        )

    experiments: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for event in events or []:
        event_type = event.get("type")
        if event_type not in {
            "research.experiment_started",
            "research.experiment_result",
            "research.experiment_verdict",
            "research.experiment_skipped",
        }:
            continue
        experiment_id = str(event.get("experiment_id") or "")
        if experiment_id not in seen:
            seen[experiment_id] = len(experiments)
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "hypothesis_id": "",
                    "trial": None,
                    "verdict": None,
                    "skipped": False,
                    "started": False,
                }
            )
        index = seen[experiment_id]
        if event_type == "research.experiment_skipped":
            experiments[index]["skipped"] = True
        if event_type == "research.experiment_started":
            experiments[index]["started"] = True
        if event_type == "research.experiment_result":
            experiments[index]["trial"] = _extra(event, "trial", {})
        if event_type == "research.experiment_verdict":
            experiments[index]["verdict"] = _extra(event, "verdict", {})

    for item in experiments:
        for hypothesis in hypotheses:
            if hypothesis["experiment_id"] == item["experiment_id"]:
                item["hypothesis_id"] = str(hypothesis["spec"].get("hypothesis_id") or "")
                item["spec"] = hypothesis["spec"]
                break

    locked = _extra(_last(events, "research.locked"), "locked", {})
    if isinstance(locked, dict) and locked.get("truncated"):
        locked = {}
    tested_event = _last(events, "research.tested")
    test_verdict = _extra(tested_event, "verdict", {})
    if isinstance(test_verdict, dict) and test_verdict.get("truncated"):
        test_verdict = {}

    stop_reason = str(completed.get("decision_reason") or completed_summary.get("stop_reason") or "")
    phase = _last(events, "research.completed").get("phase") if _last(events, "research.completed") else ""
    if not phase and events:
        phase = str(events[-1].get("phase") or "")

    return {
        "run_id": run_id,
        "phase": phase,
        "stop_reason": stop_reason,
        "task": task,
        "data_profile": profile,
        "baseline": baseline,
        "budget": last_remaining or completed_summary.get("budget", {}),
        "elapsed_seconds": completed_summary.get("elapsed_seconds"),
        "hypotheses": hypotheses,
        "experiments": experiments,
        "evidence_ids": list(completed_summary.get("evidence_ids") or []),
        "revised_count": _count(events, "research.revised"),
        "escalation_blocked": [
            event
            for event in events or []
            if event.get("type") == "research.escalation_blocked"
        ],
        "locked": locked,
        "test_verdict": test_verdict,
        "tested": tested_event is not None,
        "events": list(events or []),
    }


def _spec_label(spec: Dict[str, Any]) -> str:
    if not spec:
        return ""
    model = str(spec.get("model") or "")
    params = spec.get("params") or {}
    parts = [str(spec.get("feature_set") or "base"), model]
    if params:
        parts.append("params=" + json.dumps(params, ensure_ascii=False, sort_keys=True))
    if spec.get("missing") and spec.get("missing") != "median":
        parts.append("missing=" + str(spec["missing"]))
    if spec.get("scaling") and spec.get("scaling") != "standard":
        parts.append("scaling=" + str(spec["scaling"]))
    return " / ".join(parts)


def render_markdown(summary: Dict[str, Any]) -> str:
    """从事实快照生成 Markdown 评审报告。"""
    run_id = summary.get("run_id", "unknown")
    task = summary.get("task") or {}
    profile = summary.get("data_profile") or {}
    baseline = summary.get("baseline") or {}
    budget = summary.get("budget") or {}
    lines: List[str] = []
    lines.append(f"# AutoResearch 研究运行报告 — {run_id}")
    lines.append("")

    lines.append("## 1. 运行信息")
    lines.append("")
    lines.append(f"- 任务 ID：`{task.get('task_id') or '—'}`")
    lines.append(f"- 数据路径：`{task.get('data_path') or '—'}`")
    lines.append(f"- 最终阶段：`{summary.get('phase') or '—'}`")
    lines.append(f"- 停止原因：`{summary.get('stop_reason') or '—'}`")
    elapsed = summary.get("elapsed_seconds")
    lines.append(f"- 运行耗时：{_fmt(elapsed, 3) if elapsed is not None else '—'} 秒")
    lines.append("")

    lines.append("## 2. 数据画像")
    lines.append("")
    lines.append(f"- 日期范围：`{profile.get('date_range') or '—'}`")
    lines.append(f"- 标的数：`{profile.get('symbols') or '—'}`")
    lines.append(f"- 最大缺失率：`{_fmt(profile.get('missing_rate'), 4)}`")
    lines.append(f"- 标签可计算比例：`{_fmt(profile.get('label_computable_ratio'), 4)}`")
    lines.append(f"- 异常：`{profile.get('anomalies') or []}`")
    lines.append(f"- 数据哈希：`{profile.get('data_hash') or task.get('data_hash') or '—'}`")
    lines.append("")

    lines.append("## 3. 基线与预算")
    lines.append("")
    lines.append(f"- 固定基线 `LogisticRegression[r5, r20]` merged_auc：`{_fmt(baseline.get('merged_auc'), 6)}`")
    lines.append(f"- 剩余实验数：`{budget.get('remaining_candidates') or 0}`")
    lines.append(f"- 剩余云调用：`{budget.get('remaining_cloud_calls') or 0}`")
    lines.append(f"- 剩余云 token：`{budget.get('remaining_cloud_tokens') or 0}`")
    lines.append(f"- 剩余时间：`{_fmt(budget.get('remaining_seconds'), 3)}` 秒")
    lines.append("")

    lines.append("## 4. 假设与实验")
    lines.append("")
    if not summary.get("experiments"):
        lines.append("（本次运行没有执行候选实验）")
    else:
        lines.append("| 实验 | 假设 | 配置 | 结果 | ver1 | ver2 | 判决 |")
        lines.append("|---|---|---|---|---|---|---|")
        for item in summary.get("experiments", []):
            trial = item.get("trial") or {}
            verdict = item.get("verdict") or {}
            result = "跳过" if item.get("skipped") else (trial.get("error") and "错误" or "完成")
            lines.append(
                "| `%s` | `%s` | %s | %s | %s | %s | `%s` |"
                % (
                    item.get("experiment_id"),
                    item.get("hypothesis_id") or "—",
                    _spec_label(item.get("spec") or {}),
                    result,
                    _window_row(trial.get("val1")),
                    _window_row(trial.get("val2")),
                    verdict.get("verdict") or "—",
                )
            )
    lines.append("")

    lines.append("## 5. 证据与修正")
    lines.append("")
    lines.append(f"- 证据编号：`{summary.get('evidence_ids') or []}`")
    lines.append(f"- 证据驱动修正次数：`{summary.get('revised_count') or 0}`")
    blocked = summary.get("escalation_blocked") or []
    lines.append(f"- 预算/升级拦截次数：`{len(blocked)}`")
    for event in blocked:
        lines.append(
            f"  - `{event.get('decision_reason') or ''}`（剩余 token `{((event.get('remaining') or {}).get('remaining_cloud_tokens') or 0)}`）"
        )
    lines.append("")

    lines.append("## 6. 审查结论")
    lines.append("")
    locked = summary.get("locked") or {}
    if locked:
        lines.append(f"- 锁定对象：`{locked.get('name') or '—'}`")
        lines.append(f"- 锁定 merged_auc：`{_fmt(locked.get('merged_auc'), 6)}`")
        lines.append(f"- 锁定理由：`{locked.get('reason') or '—'}`")
    else:
        lines.append("- 未找到锁定事件（可能仍在运行或为部分完成）。")
    if summary.get("tested"):
        test_verdict = summary.get("test_verdict") or {}
        metric = test_verdict.get("metric") or {}
        lines.append(f"- 独立测试判决：`{test_verdict.get('verdict') or '—'}`")
        lines.append(f"- 独立测试 AUC：`{_fmt(metric.get('auc'))}`")
        if metric.get("valid_days") is not None:
            lines.append(f"- 独立测试有效天数：`{metric.get('valid_days')}`")
        for name in ("spearman", "brier", "pr_auc"):
            if metric.get(name) is not None:
                lines.append(f"- 独立测试 {name}：`{_fmt(metric.get(name))}`")
    else:
        lines.append("- 未执行独立测试（预算不足或部分完成）。")
    lines.append("")

    lines.append("## 7. 复现包")
    lines.append("")
    lines.append("- `runner-events.jsonl`：A 的完整事件流")
    lines.append("- `task.json` / `ledgers/`：任务与预算账本")
    lines.append("- `data/`：数据哈希、切分与画像")
    lines.append("- `metrics.json|csv`：C 的三臂指标（可用时）")
    lines.append("- `run-command.txt`：一条可执行的复现命令")
    return "\n".join(lines)


def _copy_into_zip(
    archive: zipfile.ZipFile,
    archive_name: str,
    source: Optional[Path],
) -> None:
    if source is None or not source.exists():
        return
    archive.write(source, archive_name)


def _jsonl_bytes(events: List[Dict[str, Any]]) -> bytes:
    return "\n".join(
        json.dumps(event, ensure_ascii=False, default=str) for event in events or []
    ).encode("utf-8")


def build_reproduce_package(
    summary: Dict[str, Any],
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    command: str,
    report_md: str,
) -> Path:
    """打包复现目录：事件、报告、任务、账本、数据哈希/切分、指标与命令。"""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(summary.get("run_id") or "run")
    zip_path = out_dir / f"reproduce-{run_id}.zip"
    prefix = f"reproduce-{run_id}"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{prefix}/runner-events.jsonl", _jsonl_bytes(summary.get("events") or []))
        archive.writestr(f"{prefix}/report.md", report_md)
        archive.writestr(f"{prefix}/run-command.txt", command)

        _copy_into_zip(archive, f"{prefix}/requirements.txt", run_dir / "requirements.txt")
        _copy_into_zip(archive, f"{prefix}/requirements-app.txt", run_dir / "requirements-app.txt")
        _copy_into_zip(archive, f"{prefix}/task.json", run_dir / ".arc" / "task.json")
        ledgers = run_dir / ".arc" / "ledgers"
        if ledgers.exists():
            for ledger in sorted(ledgers.glob("*.json")):
                archive.write(ledger, f"{prefix}/ledgers/{ledger.name}")
        data_dir = run_dir / "data"
        for name in (
            "demo_panel.csv.sha256",
            "demo_split.json",
            "demo_profile.json",
        ):
            _copy_into_zip(archive, f"{prefix}/data/{name}", data_dir / name)
        for name in ("metrics.json", "metrics.csv"):
            _copy_into_zip(archive, f"{prefix}/metrics/{name}", run_dir / name)
    return zip_path


def default_run_command(events_path: str | Path, *, out_dir: str | Path | None = None) -> str:
    events_path = Path(events_path)
    parts = [
        "# 一步复现（只读事件，不重新调用云模型）",
        f"python -m autoresearch.cli replay --events {events_path}",
    ]
    if out_dir is not None:
        parts.append(f"  --out {out_dir}")
    return "\n".join(parts) + "\n"
