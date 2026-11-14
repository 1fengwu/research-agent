"""AutoResearch 角色 D：单页 Streamlit 评审工作台。

消费 A 的事件流、B 的数据摘要和 C 的指标；本文件不生成研究结论、
不写 holdout、不调用 LLM。运行方式：

    python -m streamlit run autoresearch/app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from . import report as report_mod
from .contracts import as_jsonable, load_task_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
DEMO_CSV = REPO_ROOT / "data" / "demo_panel.csv"
MOCK_EVENTS = APP_DIR / "mock_events.json"
ARC_EVENTS = REPO_ROOT / ".arc" / "runner-events.jsonl"
DEMO_DIR = REPO_ROOT / ".arc" / "demo"
DEMO_EVENTS = DEMO_DIR / "runner-events.jsonl"
UPLOAD_DIR = REPO_ROOT / ".arc" / "uploads"


def _source_options() -> List[str]:
    options = ["离线演示（mock_events.json）"]
    if ARC_EVENTS.exists():
        options.append(".arc/runner-events.jsonl")
    if DEMO_EVENTS.exists():
        options.append(".arc/demo/runner-events.jsonl")
    return options


def _events_path(source: str) -> Path:
    if source.endswith("runner-events.jsonl"):
        return REPO_ROOT / ".arc" / (source.split(".arc/")[-1])
    return MOCK_EVENTS


def _safe_load_events(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    try:
        return report_mod.load_events(path), ""
    except Exception as exc:
        return [], f"{exc}"


def _task_payload_from_form() -> Dict[str, Any]:
    return {
        "task_id": st.session_state.get("task_id", "task-demo-001"),
        "data_path": str(st.session_state.get("data_path", DEMO_CSV)),
        "data_hash": "",
        "label_horizon": int(st.session_state.get("label_horizon", 5)),
        "split": {"train": 0.6, "val_windows": 2, "test": 0.2, "gap_days": 5},
        "seed": int(st.session_state.get("seed", 42)),
        "budget": {
            "candidates": int(st.session_state.get("candidates", 4)),
            "cloud_calls": int(st.session_state.get("cloud_calls", 3)),
            "cloud_tokens": int(st.session_state.get("cloud_tokens", 6000)),
            "wall_seconds": int(st.session_state.get("wall_seconds", 180)),
        },
        "primary_metric": "daily_auc_mean",
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_mock_to_demo() -> Path:
    events, _ = _safe_load_events(MOCK_EVENTS)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEMO_EVENTS, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return DEMO_EVENTS


def _launch_real_run(task_payload: Dict[str, Any]) -> Tuple[bool, str]:
    task_path = REPO_ROOT / ".arc" / "task.json"
    _write_json(task_path, task_payload)
    ARC_EVENTS.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "autoresearch.cli",
        "run",
        "--task",
        str(task_path),
        "--events",
        str(ARC_EVENTS),
        "--ledger-dir",
        str(REPO_ROOT / ".arc" / "ledgers"),
    ]
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as exc:
        return False, f"启动失败: {exc}"
    st.session_state["process_pid"] = process.pid
    return True, f"后台进程已启动 (pid={process.pid})"


def _preview_upload(file: Any) -> Tuple[Dict[str, Any], str]:
    if file is None:
        return {}, ""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"upload-{int(time.time())}-{file.name}"
    path.write_bytes(file.getvalue())
    try:
        frame = pd.read_csv(path)
        info = {
            "file_name": str(file.name),
            "path": str(path),
            "rows": len(frame),
            "columns": list(frame.columns),
            "date_min": str(frame.iloc[:, 0].min()) if len(frame) else "",
            "date_max": str(frame.iloc[:, 0].max()) if len(frame) else "",
        }
        return info, ""
    except Exception as exc:
        return {}, f"预览失败: {exc}"


def _timeline_markdown(events: List[Dict[str, Any]]) -> str:
    lines = ["| 时间 | 阶段 | 事件 | 实验 | 决策原因 | 剩余token |"]
    lines.append("|---|---|---|---|---|---|")
    for event in events:
        timestamp = str(event.get("timestamp") or "")
        if timestamp:
            timestamp = timestamp[11:19]
        remaining = event.get("remaining") or {}
        lines.append(
            "| %s | %s | `%s` | `%s` | %s | %s |"
            % (
                timestamp,
                event.get("phase") or "",
                event.get("type") or "",
                event.get("experiment_id") or "",
                str(event.get("decision_reason") or "").replace("|", "\\|"),
                remaining.get("remaining_cloud_tokens", ""),
            )
        )
    return "\n".join(lines)


def _experiments_markdown(summary: Dict[str, Any]) -> str:
    items = summary.get("experiments") or []
    if not items:
        return "（没有候选实验事件）"
    lines = ["| 实验 | 假设 | 配置 | verdict | val1 | val2 |"]
    lines.append("|---|---|---|---|---|---|")
    for item in items:
        spec = item.get("spec") or {}
        trial = item.get("trial") or {}
        verdict = item.get("verdict") or {}
        val1 = trial.get("val1") or {}
        val2 = trial.get("val2") or {}
        lines.append(
            "| `%s` | `%s` | %s | `%s` | %s | %s |"
            % (
                item.get("experiment_id"),
                item.get("hypothesis_id") or "",
                report_mod._spec_label(spec),
                verdict.get("verdict") or ("skipped" if item.get("skipped") else ""),
                report_mod._window_row(val1),
                report_mod._window_row(val2),
            )
        )
    return "\n".join(lines)


def _render_create_task() -> None:
    st.subheader("创建任务")
    with st.form("task-form"):
        col_a, col_b, col_c = st.columns(3)
        task_id = col_a.text_input("任务 ID", "task-demo-001")
        label_horizon = col_b.number_input("标签窗口（交易日）", min_value=1, value=5)
        seed = col_c.number_input("seed", min_value=0, value=42)

        data_mode = st.radio(
            "数据来源",
            ["内置演示数据 data/demo_panel.csv", "上传合规 CSV"],
            horizontal=True,
        )
        uploaded = None
        if data_mode == "上传合规 CSV":
            uploaded = st.file_uploader("导入 CSV（必须含 date/symbol/close/volume）")

        st.markdown("**预算**")
        budget_a, budget_b, budget_c, budget_d = st.columns(4)
        candidates = budget_a.number_input("候选实验数", min_value=1, value=4)
        cloud_calls = budget_b.number_input("云调用次数", min_value=0, value=3)
        cloud_tokens = budget_c.number_input("云 token 上限", min_value=1, value=6000)
        wall_seconds = budget_d.number_input("墙钟时间（秒）", min_value=1, value=180)

        st.session_state["task_id"] = task_id
        st.session_state["label_horizon"] = int(label_horizon)
        st.session_state["seed"] = int(seed)
        st.session_state["candidates"] = int(candidates)
        st.session_state["cloud_calls"] = int(cloud_calls)
        st.session_state["cloud_tokens"] = int(cloud_tokens)
        st.session_state["wall_seconds"] = int(wall_seconds)

        if data_mode.startswith("内置"):
            st.session_state["data_path"] = str(DEMO_CSV)
            profile, _ = _safe_load_events(MOCK_EVENTS)
            runs = report_mod.group_runs(profile)
            if runs:
                summary = report_mod.summarize_run(runs[0]["events"])
                info = summary.get("data_profile") or {}
                st.caption(
                    "预置面板：14 标的 / 320 交易日；字段 date, symbol, close, volume；"
                    f"日期 {info.get('date_range')}"
                )
        else:
            upload_info, upload_error = _preview_upload(uploaded)
            if upload_error:
                st.warning(upload_error)
            elif upload_info:
                st.session_state["data_path"] = upload_info["path"]
                st.markdown(
                    "**上传预览**（字段：%s，行数：%s，日期：%s ~ %s）"
                    % (
                        ", ".join(upload_info["columns"]),
                        upload_info["rows"],
                        upload_info["date_min"],
                        upload_info["date_max"],
                    )
                )

        submitted = st.form_submit_button("启动研究")

    if submitted:
        task_payload = _task_payload_from_form()
        try:
            task = load_task_spec(task_payload)
        except Exception as exc:
            st.error(f"任务配置不合法：{exc}")
            return
        if st.session_state.get("source") == "离线演示（mock_events.json）":
            _write_json(DEMO_DIR / "task.json", as_jsonable(task))
            demo_events = _copy_mock_to_demo()
            st.session_state["events_path"] = str(demo_events)
            st.success("离线联调任务已保存到 .arc/demo，事件已就绪。")
        else:
            ok, message = _launch_real_run(as_jsonable(task))
            if ok:
                st.session_state["events_path"] = str(ARC_EVENTS)
                st.success(message)
                st.info("后台研究已启动；若事件文件暂时为空，请稍候在“跟踪研究”页查看。")
            else:
                st.error(message)


def _render_track(summary: Dict[str, Any]) -> None:
    st.subheader("跟踪研究")
    budget = summary.get("budget") or {}
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**研究时间线**")
        st.markdown(_timeline_markdown(summary.get("events") or []))
    with right:
        st.markdown("**预算与状态**")
        st.metric("阶段", summary.get("phase") or "—")
        st.metric("停止原因", str(summary.get("stop_reason") or "—"))
        st.metric("剩余实验", budget.get("remaining_candidates", "—"))
        st.metric("剩余云调用", budget.get("remaining_cloud_calls", "—"))
        st.metric("剩余 token", budget.get("remaining_cloud_tokens", "—"))
        st.metric("剩余时间（秒）", budget.get("remaining_seconds", "—"))

    if summary.get("hypotheses"):
        st.markdown("**假设与证据**")
        for hypothesis in summary["hypotheses"]:
            spec = hypothesis.get("spec") or {}
            st.markdown(
                "- `%s` `%s` — %s"
                % (
                    hypothesis.get("experiment_id"),
                    spec.get("hypothesis_id"),
                    report_mod._spec_label(spec),
                )
            )
        if summary.get("evidence_ids"):
            st.markdown("**已登记证据**：`%s`" % ", ".join(summary["evidence_ids"]))
    st.markdown("**实验结果**")
    st.markdown(_experiments_markdown(summary))


def _render_review(summary: Dict[str, Any]) -> None:
    st.subheader("审查结论")
    baseline = summary.get("baseline") or {}
    locked = summary.get("locked") or {}
    test_verdict = summary.get("test_verdict") or {}
    test_metric = test_verdict.get("metric") or {}

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("固定基线 merged_auc", report_mod._fmt(baseline.get("merged_auc")))
    col_b.metric("锁定对象", locked.get("name") or "—")
    col_c.metric("锁定 merged_auc", report_mod._fmt(locked.get("merged_auc")))

    if summary.get("tested"):
        st.markdown(
            "**独立测试判决**：`%s`，AUC `%s`，有效天数 `%s`"
            % (
                test_verdict.get("verdict") or "—",
                report_mod._fmt(test_metric.get("auc")),
                test_metric.get("valid_days", "—"),
            )
        )
    else:
        st.warning("本次运行未执行独立测试（预算不足或部分完成）。")
    st.markdown(
        "**保留基线/结束研究理由**：`%s`"
        % (locked.get("reason") or "—")
    )

    report_md = report_mod.render_markdown(summary)
    st.download_button(
        "下载 Markdown 报告",
        data=report_md.encode("utf-8"),
        file_name=f"report-{summary.get('run_id', 'run')}.md",
        mime="text/markdown",
    )
    try:
        command = report_mod.default_run_command(
            st.session_state.get("events_path", str(MOCK_EVENTS))
        )
        package = report_mod.build_reproduce_package(
            summary,
            run_dir=REPO_ROOT,
            out_dir=DEMO_DIR,
            command=command,
            report_md=report_md,
        )
        st.download_button(
            "下载复现包 zip",
            data=package.read_bytes(),
            file_name=package.name,
            mime="application/zip",
        )
    except Exception as exc:
        st.warning(f"复现包生成失败：{exc}")


def main() -> None:
    st.set_page_config(page_title="AutoResearch 评审工作台", layout="wide")
    st.title("AutoResearch 研究验证工作台")
    st.caption("D 薄壳界面：只读 A 事件、C 指标与 B 数据摘要，不做研究决策。")

    source = st.sidebar.radio("数据源", _source_options())
    st.session_state["source"] = source
    events_path = _events_path(source)
    st.session_state["events_path"] = str(events_path)
    events, load_error = _safe_load_events(events_path)
    runs = report_mod.group_runs(events)
    if runs:
        run_options = [str(run["run_id"]) for run in runs]
        selected = st.sidebar.selectbox("运行", run_options)
        summary = report_mod.summarize_run(
            next(run["events"] for run in runs if run["run_id"] == selected)
        )
        st.sidebar.metric("阶段", summary.get("phase") or "—")
        st.sidebar.metric("停止原因", str(summary.get("stop_reason") or "—"))
    else:
        summary = {}
        if load_error:
            st.sidebar.error(load_error)

    tab_create, tab_track, tab_review = st.tabs(["创建任务", "跟踪研究", "审查结论"])
    with tab_create:
        _render_create_task()
    with tab_track:
        if summary:
            _render_track(summary)
        else:
            st.info("暂无事件。请先在“创建任务”启动离线联调或真实研究。")
        if st.checkbox("自动刷新（2 秒）") and events:
            time.sleep(2.0)
            st.rerun()
    with tab_review:
        if summary:
            _render_review(summary)
        else:
            st.info("完成研究后，这里会显示基线/锁定候选/独立测试对照与下载。")


if __name__ == "__main__":
    main()
