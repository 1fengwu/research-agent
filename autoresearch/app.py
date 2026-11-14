"""Helix 赫力 角色 D：单页 Streamlit 评审工作台。

消费 A 的事件流、B 的数据摘要和 C 的指标；本文件不生成研究结论、
不写 holdout、不调用 LLM。运行方式：

    python -m streamlit run autoresearch/app.py
"""

from __future__ import annotations

import json
import html
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
import streamlit as st
import altair as alt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoresearch import ui as ui_mod  # noqa: E402
from autoresearch import report as report_mod  # noqa: E402
from autoresearch.contracts import as_jsonable, load_task_spec  # noqa: E402
from autoresearch.data import build_profile, read_panel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
DEMO_CSV = REPO_ROOT / "data" / "demo_panel.csv"
MOCK_EVENTS = APP_DIR / "mock_events.json"
ARC_EVENTS = REPO_ROOT / ".arc" / "runner-events.jsonl"
DEMO_DIR = REPO_ROOT / ".arc" / "demo"
DEMO_EVENTS = DEMO_DIR / "runner-events.jsonl"
UPLOAD_DIR = REPO_ROOT / ".arc" / "uploads"
METRICS_JSON = REPO_ROOT / ".arc" / "eval" / "metrics.json"
ROOT_METRICS_JSON = REPO_ROOT / "metrics.json"
PRE_GATE_JSON = REPO_ROOT / ".arc" / "pre_gate.json"
ROOT_PRE_GATE_JSON = REPO_ROOT / "pre_gate.json"
MOCK_METRICS = APP_DIR / "mock_metrics.json"
MOCK_PRE_GATE = APP_DIR / "mock_pre_gate.json"

TECH_GRID_SVG = quote(
    "<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56'>"
    "<path d='M56 0H0V56' fill='none' stroke='%23a855f7' stroke-opacity='.10'/>"
    "<circle cx='0' cy='0' r='1.2' fill='%23a855f7' fill-opacity='.20'/>"
    "<circle cx='56' cy='56' r='1.2' fill='%233b82f6' fill-opacity='.20'/>"
    "</svg>"
)

TECH_SIGNAL_SVG = quote(
    "<svg xmlns='http://www.w3.org/2000/svg' width='720' height='420' viewBox='0 0 720 420' fill='none'>"
    "<path d='M0 330 L70 285 L140 305 L210 220 L280 250 L350 150 L420 190 L490 105 L560 140 L640 70 L720 100' "
    "stroke='%23a855f7' stroke-opacity='.22' stroke-width='2'/>"
    "<path d='M0 370 L70 355 L140 335 L210 345 L280 310 L350 285 L420 295 L490 260 L560 275 L640 235 L720 250' "
    "stroke='%233b82f6' stroke-opacity='.16' stroke-width='2'/>"
    "<rect x='70' y='330' width='12' height='60' fill='%23a855f7' fill-opacity='.14'/>"
    "<rect x='180' y='300' width='12' height='90' fill='%233b82f6' fill-opacity='.12'/>"
    "<rect x='300' y='350' width='12' height='40' fill='%23a855f7' fill-opacity='.12'/>"
    "<rect x='430' y='280' width='12' height='110' fill='%233b82f6' fill-opacity='.16'/>"
    "<rect x='560' y='320' width='12' height='70' fill='%23a855f7' fill-opacity='.12'/>"
    "<circle cx='350' cy='150' r='3.5' fill='%23a855f7' fill-opacity='.32'/>"
    "<circle cx='490' cy='105' r='3.5' fill='%233b82f6' fill-opacity='.32'/>"
    "<circle cx='640' cy='70' r='3.5' fill='%23a855f7' fill-opacity='.32'/>"
    "<path d='M350 150 L390 260' stroke='%23a855f7' stroke-opacity='.10' stroke-dasharray='4 6'/>"
    "<path d='M490 105 L520 240' stroke='%233b82f6' stroke-opacity='.10' stroke-dasharray='4 6'/>"
    "</svg>"
)


def _source_options() -> Dict[str, Path]:
    options = {}
    for metrics in sorted((REPO_ROOT / ".arc").glob("*/metrics.json"), reverse=True):
        for events in sorted(metrics.parent.glob("*/*/events.jsonl"), reverse=True):
            relative = events.relative_to(metrics.parent)
            options[
                "已保存评测 · %s · %s · %s"
                % (metrics.parent.name, relative.parts[0], events.parent.name)
            ] = events
    options["离线演示（mock_events.json）"] = MOCK_EVENTS
    if ARC_EVENTS.exists():
        options[".arc/runner-events.jsonl"] = ARC_EVENTS
    if DEMO_EVENTS.exists():
        options[".arc/demo/runner-events.jsonl"] = DEMO_EVENTS
    return options


def _safe_load_events(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    try:
        return report_mod.load_events(path), ""
    except Exception as exc:
        return [], f"{exc}"


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_metrics() -> Optional[Dict[str, Any]]:
    for path in (METRICS_JSON, ROOT_METRICS_JSON, MOCK_METRICS):
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            return loaded
    return None


def _load_pre_gate() -> Tuple[Dict[str, Any], str]:
    for path in (PRE_GATE_JSON, ROOT_PRE_GATE_JSON, MOCK_PRE_GATE):
        loaded = _load_json(path)
        if isinstance(loaded, dict):
            return loaded, ""
    return {}, ".arc/pre_gate.json 尚未生成"


def _task_payload_from_form() -> Dict[str, Any]:
    return {
        "task_id": st.session_state.get("task_id", "task-demo-001"),
        "data_path": str(st.session_state.get("data_path", DEMO_CSV)),
        "data_hash": "",
        "label_horizon": int(st.session_state.get("label_horizon", 5)),
        "split": {"train": 0.6, "val_windows": 2, "test": 0.2, "gap_days": 5},
        "seed": int(st.session_state.get("seed", 42)),
        "budget": {
            "candidates": 4,
            "cloud_calls": 3,
            "cloud_tokens": 6000,
            "wall_seconds": 180,
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
        panel = read_panel(path)
        profile = build_profile(panel, horizon=int(st.session_state.get("label_horizon", 5)))
        missing_rates = profile.get("missing_rates") or {}
        info = {
            "file_name": str(file.name),
            "path": str(path),
            "rows": len(frame),
            "columns": list(frame.columns),
            "date_range": [profile["date_start"], profile["date_end"]],
            "symbols": profile.get("symbols"),
            "missing_rate": max(missing_rates.values()) if missing_rates else 0.0,
            "label_computable_ratio": profile.get("label_computable_ratio"),
            "profile": profile,
            "valid": True,
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
        raw_type = str(event.get("type") or "")
        label = ui_mod.event_label(raw_type)
        if raw_type == "research.revised":
            label = "**%s**" % label
        event_cell = "%s `%s`" % (label, raw_type) if label != raw_type else "`%s`" % raw_type
        lines.append(
            "| %s | %s | %s | `%s` | %s | %s |"
            % (
                timestamp,
                ui_mod.phase_label(event.get("phase")),
                event_cell,
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
        raw_verdict = str(verdict.get("verdict") or "")
        skipped = bool(item.get("skipped"))
        verdict_text = "跳过" if skipped else ui_mod.verdict_label(raw_verdict)
        if not skipped and raw_verdict and verdict_text != raw_verdict:
            verdict_text = "%s `%s`" % (verdict_text, raw_verdict)
        lines.append(
            "| `%s` | `%s` | %s | `%s` | %s | %s |"
            % (
                item.get("experiment_id"),
                item.get("hypothesis_id") or "",
                report_mod._spec_label(spec),
                verdict_text,
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

        st.session_state["task_id"] = task_id
        st.session_state["label_horizon"] = int(label_horizon)
        st.session_state["seed"] = int(seed)

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
                    "**上传预览**：%s 行 / %s 个标的 / 字段 %s / 日期 %s ~ %s / 最大缺失率 %s"
                    % (
                        ", ".join(upload_info["columns"]),
                        upload_info["rows"],
                        upload_info.get("symbols", "—"),
                        upload_info["date_range"][0],
                        upload_info["date_range"][1],
                        report_mod._fmt(upload_info.get("missing_rate"), 3),
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

    pre_gate, _pre_gate_note = _load_pre_gate()
    if pre_gate:
        with st.expander("前置门摘要（pre_gate.json）"):
            pre_baseline = pre_gate.get("baseline") or {}
            pre_candidates = pre_gate.get("candidates") or []
            passed = sum(1 for row in pre_candidates if row.get("passed"))
            source_note = "内置 mock 数据" if pre_gate.get("source") == "mock" else "真实运行数据"
            st.caption(
                "结论：%s ｜ 基线 merged_auc %s ｜ 通过候选 %d/%d ｜ 数据来源：%s"
                % (
                    pre_gate.get("conclusion") or "—",
                    report_mod._fmt(pre_baseline.get("merged_auc"), 6),
                    passed,
                    len(pre_candidates),
                    source_note,
                )
            )
            for row in pre_candidates:
                status = "通过" if row.get("passed") else "不通过"
                st.markdown(
                    "- `%s` `%s` merged_auc=%s %s" % (
                        row.get("experiment_id"),
                        row.get("feature_set"),
                        report_mod._fmt(row.get("merged_auc"), 6),
                        status,
                    )
                )


def _render_innovation() -> None:
    points = ui_mod.innovation_points()
    st.markdown("**核心创新点**")
    cols = st.columns(len(points))
    for col, point in zip(cols, points):
        with col:
            st.markdown("**%s** · %s" % (point["title"], point["tag"]))
            st.caption(point["detail"])


def _render_roadshow() -> None:
    if not st.session_state.get("roadshow_mode"):
        return
    if "roadshow_started" not in st.session_state:
        st.session_state["roadshow_started"] = time.monotonic()
    elapsed = max(0.0, min(240.0, time.monotonic() - float(st.session_state["roadshow_started"])))
    info = ui_mod.roadshow_step(elapsed)
    step = info["step"]
    st.markdown("**路演模式 · 4 分钟引导**")
    st.progress(elapsed / 240.0)
    st.caption(
        "当前环节：%s ｜ 建议页面：%s ｜ 已用 %d 秒"
        % (step["title"], step["tab"], int(elapsed))
    )
    st.caption(step["detail"])


def _render_material() -> None:
    st.subheader("评审材料")
    st.markdown("**项目定位**")
    st.markdown(
        "AutoResearch 不是投资建议系统，而是量价信号预测力验证工作台："
        "验证 t 日及之前的量价信息，对未来 5 个交易日收益是否高于同日中位数是否具有预测力。"
    )
    st.markdown("**技术架构与核心技术**")
    st.markdown(
        "- A 编排：研究状态机、云路由、预算账本、事件流；\n"
        "- B 数据：CSV 接入、fwd5 标签、时间切分、滚动特征、候选实验；\n"
        "- C 裁判：pre_gate / oracle / evaluate / replay，唯一出分口；\n"
        "- D 展示：事件时间线、预算面板、报告与复现包。"
    )
    st.markdown("**代码质量与可复现性**")
    st.markdown(
        "- 契约冻结：TaskSpec / ExperimentSpec / TrialResult / Verdict / RunEvent；\n"
        "- 自动化测试：A/B/C/D 全角色测试覆盖；\n"
        "- 复现：无云 replay、数据哈希、切分指纹、复现包 zip；\n"
        "- 降级：Streamlit 异常时 `python -m autoresearch.cli demo` 可继续路演。"
    )
    st.markdown("**商业价值**")
    st.markdown(
        "- 目标用户：量化研究员、研究团队、需要信号验证与审计的机构；\n"
        "- 应用场景：信号是否可复现、预算是否可审计、结论是否可归因；\n"
        "- 实际价值：减少无效实验、避免手工误判、支持低预算降级与合规复现。"
    )
    st.markdown("**Built for Better**")
    st.markdown(
        "- 诚实报告负结果：窗口下滑即 refuted，不隐藏失败；\n"
        "- 预算透明：budget_blocked 可见，每次运行独立 ledger；\n"
        "- 可复现可审计：事件、账本、数据哈希、切分指纹全部随包交付。"
    )


def _render_header() -> None:
    st.markdown(
        """
<div class="ar-hero">
  <div class="ar-eyebrow">HELIX · QUANT SIGNAL VALIDATION WORKBENCH</div>
  <h1>Helix 赫力 · 量价信号验证工作台</h1>
  <p>不是投资建议系统，而是验证 t 日及之前的量价信息能否预测未来 5 日相对强弱，并让每一轮假设、预算、证据与复现都可审计。</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def _inject_styles() -> None:
    st.markdown('<div class="ar-bg" aria-hidden="true"></div>', unsafe_allow_html=True)
    css = """
<style>
:root {
  --ar-bg: #F7F8FA;
  --ar-panel: rgba(255, 255, 255, 0.86);
  --ar-panel-2: #F0F2F7;
  --ar-ink: #1F2328;
  --ar-muted: #4A5160;
  --ar-faint: #4A5160;
  --ar-gold: #a855f7;
  --ar-patina: #3b82f6;
  --ar-vermil: #f87171;
  --ar-line: rgba(139, 92, 246, 0.18);
  --ar-radius: 6px;
}

html, body {
  background: var(--ar-bg);
  color: var(--ar-ink);
  font-family: "Microsoft YaHei UI", "PingFang SC", "HarmonyOS Sans SC", "Noto Sans SC", "Segoe UI", sans-serif;
  font-size: 15.5px;
  line-height: 1.7;
  letter-spacing: 0.01em;
}

[data-testid="stAppViewContainer"] {
  background: transparent;
  color: var(--ar-ink);
  font-family: "Microsoft YaHei UI", "PingFang SC", "HarmonyOS Sans SC", "Noto Sans SC", "Segoe UI", sans-serif;
  font-size: 15.5px;
  line-height: 1.7;
  letter-spacing: 0.01em;
}

.ar-bg {
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  background-color: var(--ar-bg);
  background-image:
    linear-gradient(180deg, rgba(247, 248, 250, 0.00), rgba(255, 255, 255, 0.92)),
    url("data:image/svg+xml,__TECH_GRID__"),
    url("data:image/svg+xml,__TECH_SIGNAL__");
  background-size: cover, 56px 56px, 720px 420px;
  background-position: center, center, right top;
  background-repeat: no-repeat, repeat, no-repeat;
}

[data-testid="stHeader"] {
  background: transparent;
}

[data-testid="stAppViewContainer"] > .main {
  padding-top: 1.5rem;
}

h1, h2, h3 {
  color: #1F2328;
  font-weight: 600;
  letter-spacing: 0.01em;
}

p, li, label, [data-testid="stMarkdownContainer"] {
  color: #1F2328;
  font-weight: 450;
  letter-spacing: 0.01em;
  line-height: 1.7;
}

[data-testid="stSidebar"] {
  background: rgba(255, 255, 255, 0.72);
  border-right: 1px solid var(--ar-line);
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.28);
  font-size: 15px;
  line-height: 1.6;
}

[data-testid="stSidebar"] * {
  color: var(--ar-ink);
}

code, pre, [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stCodeBlock"], [data-testid="stCaptionContainer"] {
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 14px;
}

.ar-hero {
  padding: 0.25rem 0 1.1rem;
  border-bottom: 1px solid var(--ar-line);
  margin-bottom: 1rem;
}

.ar-eyebrow {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  color: #a78bfa;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}

.ar-hero h1 {
  font-size: 2.1rem;
  font-weight: 600;
  line-height: 1.15;
  margin: 0 0 0.6rem;
  background: linear-gradient(90deg, #a855f7, #3b82f6);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.ar-hero p {
  margin: 0;
  max-width: 78ch;
  color: var(--ar-muted);
  line-height: 1.7;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  border-bottom: 1px solid var(--ar-line);
  gap: 0.25rem;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
  color: var(--ar-muted);
  border-radius: 0;
  padding: 0.55rem 0.9rem;
}

[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--ar-gold);
  border-bottom: 2px solid var(--ar-gold);
}

[data-testid="stMetric"] {
  background: var(--ar-panel);
  border: 1px solid var(--ar-line);
  border-radius: var(--ar-radius);
  padding: 0.7rem 0.85rem;
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.35);
}

[data-testid="stMetricValue"] {
  color: #1F2328;
  font-weight: 600;
}

[data-testid="stMetricLabel"] {
  color: #4A5160;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--ar-line);
  border-radius: var(--ar-radius);
  overflow: hidden;
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.28);
}

[data-testid="stDataFrame"] {
  background: var(--ar-panel);
}

[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
  background: linear-gradient(135deg, #a855f7, #3b82f6);
  color: #ffffff;
  border: 1px solid rgba(168, 85, 247, 0.65);
  border-radius: 4px;
  font-weight: 600;
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.5);
}

[data-testid="stButton"] button:hover,
[data-testid="stDownloadButton"] button:hover {
  background: linear-gradient(135deg, #c084fc, #60a5fa);
  color: #ffffff;
  border-color: rgba(192, 132, 252, 0.8);
  box-shadow: 0 0 20px rgba(139, 92, 246, 0.7);
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] [data-baseweb="select"] {
  background: var(--ar-panel);
  border-color: var(--ar-line);
  color: var(--ar-ink);
  border-radius: 4px;
}

[data-testid="stProgress"] > div > div > div {
  background: linear-gradient(90deg, #a855f7, #3b82f6);
}

[data-testid="stCaptionContainer"] {
  color: var(--ar-faint);
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.75rem;
}

[data-testid="stSuccess"] {
  background: rgba(59, 130, 246, 0.14);
  border: 1px solid rgba(59, 130, 246, 0.45);
  color: var(--ar-ink);
  border-radius: 4px;
}

[data-testid="stWarning"],
[data-testid="stError"] {
  background: rgba(248, 113, 113, 0.12);
  border: 1px solid rgba(248, 113, 113, 0.45);
  color: var(--ar-ink);
  border-radius: 4px;
}

[data-testid="stInfo"] {
  background: rgba(139, 92, 246, 0.12);
  border: 1px solid var(--ar-line);
  color: var(--ar-ink);
  border-radius: 4px;
}

[data-testid="stExpander"] details {
  border: 1px solid var(--ar-line);
  border-radius: var(--ar-radius);
  background: var(--ar-panel);
  box-shadow: 0 0 12px rgba(138, 43, 226, 0.22);
}

.ar-timeline {
  border: 1px solid var(--ar-line);
  border-radius: var(--ar-radius);
  background: var(--ar-panel-2);
  padding: 0.5rem 0.8rem;
  max-height: 560px;
  overflow-y: auto;
  box-shadow: 0 0 15px rgba(138, 43, 226, 0.25);
}

.ar-tl-row {
  display: flex;
  gap: 0.65rem;
  padding: 0.55rem 0;
  border-bottom: 1px solid rgba(139, 92, 246, 0.12);
}

.ar-tl-row:last-child {
  border-bottom: 0;
}

.ar-tl-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  margin-top: 0.4rem;
  flex: 0 0 auto;
}

.ar-tl-body {
  min-width: 0;
}

.ar-tl-head {
  display: flex;
  gap: 0.5rem;
  align-items: baseline;
}

.ar-tl-head strong {
  color: var(--ar-ink);
  font-weight: 600;
}

.ar-tl-phase {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  color: var(--ar-muted);
}

.ar-tl-meta {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.68rem;
  color: var(--ar-faint);
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 0.8rem;
  margin: 0.2rem 0;
}

.ar-tl-reason {
  color: var(--ar-muted);
  font-size: 0.8rem;
  line-height: 1.45;
}

.ar-stop-reason {
  font-size: 12px;
  line-height: 1.5;
  word-break: break-word;
  white-space: normal;
  color: var(--ar-muted);
}

.ar-exp-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 0.75rem;
}

.ar-exp-card {
  background: var(--ar-panel);
  border: 1px solid var(--ar-line);
  border-left-width: 3px;
  border-radius: var(--ar-radius);
  padding: 0.8rem 0.9rem;
  box-shadow: 0 0 12px rgba(138, 43, 226, 0.28);
}

.ar-exp-head {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 0.75rem;
  align-items: baseline;
  font-weight: 600;
}

.ar-exp-id {
  color: #a78bfa;
}

.ar-verdict {
  font-weight: 700;
  font-size: 0.85rem;
}

.ar-exp-spec {
  color: var(--ar-muted);
  font-size: 0.75rem;
  margin: 0.25rem 0 0.65rem;
}

.ar-exp-bars {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.ar-bar-row {
  display: grid;
  grid-template-columns: 2.5rem 1fr 6.5rem;
  align-items: center;
  gap: 0.5rem;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.7rem;
  color: var(--ar-muted);
}

.ar-bar-track {
  background: rgba(15, 23, 42, 0.72);
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
}

.ar-bar {
  background: linear-gradient(90deg, #a855f7, #3b82f6);
  height: 100%;
  border-radius: 4px;
}

.ar-bar-2 {
  background: linear-gradient(90deg, #3b82f6, #22d3ee);
}

.ar-exp-meta {
  color: var(--ar-faint);
  font-size: 0.7rem;
  margin-top: 0.55rem;
}

.ar-hyp-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.6rem;
}

.ar-hyp-item {
  background: var(--ar-panel);
  border: 1px solid var(--ar-line);
  border-left-width: 3px;
  border-radius: var(--ar-radius);
  padding: 0.7rem 0.8rem;
  box-shadow: 0 0 10px rgba(138, 43, 226, 0.18);
}

.ar-hyp-title {
  font-weight: 600;
  color: #1F2328;
  margin-bottom: 0.2rem;
}

.ar-hyp-auc {
  color: #4A5160;
  font-size: 0.8rem;
  margin: 0.2rem 0;
}

.ar-hyp-conclusion {
  color: #4A5160;
  font-size: 0.75rem;
}

.ar-hyp-round-empty {
  color: #4A5160;
  font-size: 0.8rem;
  border: 1px dashed var(--ar-line);
  border-radius: var(--ar-radius);
  padding: 0.7rem;
}

.ar-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.35rem 0;
}

.ar-chip {
  background: var(--ar-panel);
  border: 1px solid var(--ar-line);
  color: var(--ar-ink);
  border-radius: 4px;
  padding: 0.3rem 0.55rem;
  font-size: 0.74rem;
  box-shadow: 0 0 8px rgba(138, 43, 226, 0.18);
}

.ar-empty {
  color: var(--ar-muted);
  font-size: 0.8rem;
}

@media (max-width: 768px) {
  .ar-hero h1 {
    font-size: 1.5rem;
  }
  .ar-exp-grid {
    grid-template-columns: 1fr;
  }
  .ar-bar-row {
    grid-template-columns: 2rem 1fr 5rem;
  }
  [data-testid="stMainBlockContainer"] {
    padding: 0.75rem;
  }
}
</style>
        """
    css = css.replace("__TECH_GRID__", TECH_GRID_SVG).replace(
        "__TECH_SIGNAL__", TECH_SIGNAL_SVG
    )
    st.markdown(css, unsafe_allow_html=True)


def _token_line_chart(token_frame: pd.DataFrame):
    melted = token_frame.melt(
        id_vars=["seq"],
        value_vars=["used_cloud_tokens", "remaining_cloud_tokens"],
        var_name="指标",
        value_name="token",
    )
    return (
        alt.Chart(melted)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=55), strokeWidth=2)
        .encode(
            x=alt.X("seq:Q", title="事件序号"),
            y=alt.Y("token:Q", title="云 token"),
            color=alt.Color(
                "指标:N",
                scale=alt.Scale(
                    domain=["used_cloud_tokens", "remaining_cloud_tokens"],
                    range=["#a855f7", "#3b82f6"],
                ),
                legend=alt.Legend(orient="bottom", title=""),
            ),
            tooltip=["seq", "指标", "token"],
        )
        .properties(height=280)
        .configure_view(stroke=None)
        .configure_axis(
            grid=False,
            labelColor="#4A5160",
            titleColor="#1F2328",
            domainColor="rgba(139,92,246,.24)",
        )
        .configure_axisY(
            grid=True,
            gridColor="rgba(139,92,246,.10)",
            gridDash=[2, 2],
        )
        .configure_legend(labelColor="#4A5160", titleColor="#1F2328")
    )


def _token_event_bar(cloud_only: pd.DataFrame):
    melted = cloud_only.melt(
        id_vars=["event"],
        value_vars=["input_tokens", "output_tokens"],
        var_name="类型",
        value_name="token",
    )
    return (
        alt.Chart(melted)
        .mark_bar(size=12)
        .encode(
            x=alt.X("token:Q", title="token"),
            y=alt.Y("event:N", title=""),
            color=alt.Color(
                "类型:N",
                scale=alt.Scale(
                    domain=["input_tokens", "output_tokens"],
                    range=["#a855f7", "#3b82f6"],
                ),
                legend=alt.Legend(orient="bottom", title=""),
            ),
            tooltip=["event", "类型", "token"],
        )
        .properties(height=220)
        .configure_view(stroke=None)
        .configure_axis(
            grid=False,
            labelColor="#4A5160",
            titleColor="#1F2328",
            domainColor="rgba(139,92,246,.24)",
        )
        .configure_axisX(
            grid=True,
            gridColor="rgba(139,92,246,.10)",
            gridDash=[2, 2],
        )
        .configure_legend(labelColor="#4A5160", titleColor="#1F2328")
    )


def _comparison_charts(comparison: pd.DataFrame):
    token_chart = (
        alt.Chart(comparison)
        .mark_bar(size=22)
        .encode(
            x=alt.X("used_cloud_tokens:Q", title="使用云 token"),
            y=alt.Y("label:N", title="", sort=None),
            color=alt.condition(
                alt.datum.label == "进化前（第 1 轮）",
                alt.value("#64748b"),
                alt.value("#a855f7"),
            ),
            tooltip=["label", "run_id", "used_cloud_tokens", "cloud_calls"],
        )
        .properties(height=220)
    )
    calls_chart = (
        alt.Chart(comparison)
        .mark_bar(size=22)
        .encode(
            x=alt.X("cloud_calls:Q", title="云调用次数"),
            y=alt.Y("label:N", title="", sort=None),
            color=alt.value("#94a7c7"),
            tooltip=["label", "run_id", "cloud_calls"],
        )
        .properties(height=220)
    )
    time_chart = (
        alt.Chart(comparison)
        .mark_bar(size=22)
        .encode(
            x=alt.X("elapsed_seconds:Q", title="运行时间（秒）"),
            y=alt.Y("label:N", title="", sort=None),
            color=alt.condition(
                alt.datum.label == "进化前（第 1 轮）",
                alt.value("#64748b"),
                alt.value("#3b82f6"),
            ),
            tooltip=["label", "run_id", "elapsed_seconds"],
        )
        .properties(height=220)
    )
    auc_chart = (
        alt.Chart(comparison)
        .mark_bar(size=22)
        .encode(
            x=alt.X("test_auc:Q", title="独立测试 AUC"),
            y=alt.Y("label:N", title="", sort=None),
            color=alt.condition(
                alt.datum.label == "进化前（第 1 轮）",
                alt.value("#64748b"),
                alt.value("#22d3ee"),
            ),
            tooltip=["label", "run_id", "test_auc"],
        )
        .properties(height=220)
    )
    return (
        token_chart.configure_view(stroke=None)
        .configure_axis(
            grid=False,
            labelColor="#4A5160",
            titleColor="#1F2328",
            domainColor="rgba(139,92,246,.24)",
        )
        .configure_axisX(
            grid=True,
            gridColor="rgba(139,92,246,.10)",
            gridDash=[2, 2],
        ),
        calls_chart.configure_view(stroke=None)
        .configure_axis(
            grid=False,
            labelColor="#4A5160",
            titleColor="#1F2328",
            domainColor="rgba(139,92,246,.24)",
        )
        .configure_axisX(
            grid=True,
            gridColor="rgba(139,92,246,.10)",
            gridDash=[2, 2],
        ),
        time_chart.configure_view(stroke=None)
        .configure_axis(
            grid=False,
            labelColor="#4A5160",
            titleColor="#1F2328",
            domainColor="rgba(139,92,246,.24)",
        )
        .configure_axisX(
            grid=True,
            gridColor="rgba(139,92,246,.10)",
            gridDash=[2, 2],
        ),
        auc_chart.configure_view(stroke=None)
        .configure_axis(
            grid=False,
            labelColor="#4A5160",
            titleColor="#1F2328",
            domainColor="rgba(139,92,246,.24)",
        )
        .configure_axisX(
            grid=True,
            gridColor="rgba(139,92,246,.10)",
            gridDash=[2, 2],
        ),
    )


def _render_track(summary: Dict[str, Any]) -> None:
    st.subheader("跟踪研究")
    events = summary.get("events") or []
    escalations = [event for event in events if event.get("type") == "research.escalation"]
    for index, event in enumerate(escalations, start=1):
        extra = event.get("extra") or {}
        actions = extra.get("actions") or []
        selected = next(
            (action.get("target") for action in actions if action.get("action") == "select_candidate"),
            None,
        )
        with st.expander("云决策 %d" % index + (" · 建议 %s" % selected if selected else "")):
            st.write(ui_mod.reason_text(event.get("decision_reason")))
            st.write(
                "触发依据",
                ", ".join(ui_mod.reason_text(item) for item in extra.get("trigger") or []),
            )
            st.write("证据", ", ".join(extra.get("evidence_ids") or []))
            charged = (event.get("tokens") or {}).get("charged")
            if charged is not None:
                st.write("本次调用计入账本 %s Token" % charged)
            if selected:
                experiment = next(
                    (item for item in summary.get("experiments", []) if item.get("experiment_id") == selected),
                    None,
                )
                if experiment:
                    verdict = experiment.get("verdict") or {}
                    st.write("对应实验已执行" if experiment.get("started") else "对应实验尚未执行")
                    if verdict.get("verdict"):
                        st.write("验证裁决", ui_mod.verdict_label(verdict["verdict"]))
    budget = summary.get("budget") or {}
    left, right = st.columns([2, 1])
    with left:
        st.markdown("**研究时间线**")
        st.markdown(
            ui_mod.timeline_html(summary.get("events") or []),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("**预算与状态**")
        st.markdown(
            "**阶段**：%s  \n**停止原因**：<span class=\"ar-stop-reason\">%s</span>"
            % (
                ui_mod.phase_label(summary.get("phase")),
                html.escape(ui_mod.reason_text(summary.get("stop_reason"))),
            ),
            unsafe_allow_html=True,
        )
        task_budget = (summary.get("task") or {}).get("budget") or {}
        stats = ui_mod.experiment_summary(summary)
        rows = [
            ("候选实验", budget.get("remaining_candidates"), task_budget.get("candidates")),
            ("云调用", budget.get("remaining_cloud_calls"), task_budget.get("cloud_calls")),
            ("云 token", budget.get("remaining_cloud_tokens"), task_budget.get("cloud_tokens")),
            ("剩余时间", budget.get("remaining_seconds"), task_budget.get("wall_seconds")),
        ]
        for label, remaining, limit in rows:
            st.caption("%s：%s / %s" % (label, remaining if remaining is not None else "—", limit if limit is not None else "—"))
            st.progress(ui_mod.progress_fraction(remaining, limit))
        st.caption(
            "实验 %d · 修正 %d · 拦截 %d · 支持 %d · 反驳 %d"
            % (
                stats["experiments"],
                stats["revised"],
                stats["blocked"],
                stats["supported"],
                stats["refuted"],
            )
        )
        st.caption(
            "预算模式：%s"
            % ui_mod.budget_mode_text(budget.get("mode") or summary.get("budget_mode"))
        )

    token_rows = report_mod.token_usage_series(summary)
    if token_rows:
        token_frame = pd.DataFrame(token_rows)
        st.markdown("**Token 消耗动态**")
        st.altair_chart(_token_line_chart(token_frame), use_container_width=True)
        cloud_only = token_frame[
            token_frame[["input_tokens", "output_tokens"]].sum(axis=1) > 0
        ]
        if not cloud_only.empty:
            st.caption("云调用事件 token 明细（input / output）")
            st.altair_chart(_token_event_bar(cloud_only), use_container_width=True)
    else:
        st.caption("当前运行暂无 token 事件。")

    savings = summary.get("token_savings") or {}
    if savings:
        st.markdown("**Token 节省（决策缓存 + 本地规则）**")
        saving_cols = st.columns(4)
        saving_cols[0].metric("估算节省 token", savings.get("total_estimated_saved", 0))
        saving_cols[1].metric("避免云调用", savings.get("cloud_calls_avoided", 0))
        saving_cols[2].metric("缓存命中", savings.get("cache_hits", 0))
        saving_cols[3].metric("本地规则决策", savings.get("local_rule_decisions", 0))
        st.caption(
            "云端实际入账：%s token · 上下文压缩节省：%s token"
            % (
                report_mod._fmt(savings.get("cloud_tokens_charged"), 0),
                report_mod._fmt(savings.get("prompt_compression_saved"), 0),
            )
        )

    if summary.get("hypotheses"):
        st.markdown("**假设与证据**")
        st.markdown(ui_mod.hypotheses_html(summary), unsafe_allow_html=True)
        if summary.get("evidence_ids"):
            st.markdown("**已登记证据**")
            st.markdown(
                ui_mod.hyphen_html(summary["evidence_ids"]),
                unsafe_allow_html=True,
            )
    st.markdown("**实验结果**")
    st.markdown(ui_mod.experiments_html(summary), unsafe_allow_html=True)


def _render_review(
    summary: Dict[str, Any],
    runs: Optional[List[Dict[str, Any]]] = None,
    evaluation: Optional[Path] = None,
) -> None:
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
        raw_verdict = str(test_verdict.get("verdict") or "—")
        st.markdown(
            "**独立测试判决**：`%s`（%s），AUC `%s`，有效天数 `%s`"
            % (
                ui_mod.verdict_label(raw_verdict),
                raw_verdict,
                report_mod._fmt(test_metric.get("auc")),
                test_metric.get("valid_days", "—"),
            )
        )
    else:
        st.warning("本次运行未执行独立测试（数据/裁判不可用或用户取消）。")
    st.markdown(
        "**保留基线/结束研究理由**：`%s`"
        % (locked.get("reason") or "—")
    )

    metrics = _load_metrics()
    arms = ui_mod.metrics_arms(metrics)
    if arms:
        st.markdown("**三臂对照指标**")
        source_note = "内置 mock 数据" if (metrics or {}).get("source") == "mock" else "真实运行数据"
        st.caption("数据来源：%s" % source_note)
        arms_df = pd.DataFrame(arms)
        if "usage_source" in arms_df:
            arms_df["usage_source"] = arms_df["usage_source"].map(ui_mod.usage_source_text)
        if "phase" in arms_df:
            arms_df["phase"] = arms_df["phase"].map(ui_mod.phase_label)
        st.dataframe(arms_df, use_container_width=True)
    else:
        st.caption("未找到 metrics.json，三臂对照暂不可用。")

    if runs:
        comparison = pd.DataFrame(report_mod.run_token_comparison(runs))
        if not comparison.empty:
            st.markdown("**智能体进化前后对比（直观版）**")
            if len(comparison) >= 2:
                first = comparison.iloc[0]
                savings_run = comparison.loc[comparison["used_cloud_tokens"].idxmin()]
                completion_run = comparison.loc[comparison["candidates"].idxmax()]
                token_saved = max(0, int(first["used_cloud_tokens"]) - int(savings_run["used_cloud_tokens"]))
                candidates_gain = max(0, int(completion_run["candidates"]) - int(first["candidates"]))
                cards = st.columns(4)
                cards[0].metric(
                    "Token 对比",
                    "%s → %s" % (first["used_cloud_tokens"], savings_run["used_cloud_tokens"]),
                    "减少 %s" % token_saved,
                )
                cards[1].metric(
                    "已验证假设",
                    "%s → %s" % (int(first["candidates"]), int(completion_run["candidates"])),
                    "增加 %s 个" % candidates_gain,
                )
                cards[2].metric(
                    "运行时间（秒）",
                    "%s → %s" % (first["elapsed_seconds"], completion_run["elapsed_seconds"]),
                )
                cards[3].metric(
                    "独立测试 AUC",
                    "%s → %s" % (first["test_auc"], completion_run["test_auc"]),
                )
                if token_saved > 0 and candidates_gain > 0:
                    st.success(
                        "系统同时做到两点：Token 成本减少 %s，已验证假设从 %s 个提高到 %s 个，测试质量保持。"
                        % (token_saved, int(first["candidates"]), int(completion_run["candidates"]))
                    )
                elif token_saved > 0:
                    st.success("本轮主要体现 Token 节省，完成度保持。")
                elif candidates_gain > 0:
                    st.success(
                        "本轮主要体现完成度提升：已验证假设从 %s 个提高到 %s 个。"
                        % (int(first["candidates"]), int(completion_run["candidates"]))
                    )
            token_chart, calls_chart, time_chart, auc_chart = _comparison_charts(comparison)
            chart_cols = st.columns(4)
            for col, chart in zip(
                chart_cols,
                [token_chart, calls_chart, time_chart, auc_chart],
            ):
                with col:
                    st.altair_chart(chart, use_container_width=True)
            comparison_view = comparison[
                [
                    "label",
                    "run_id",
                    "phase",
                    "used_cloud_tokens",
                    "cloud_calls",
                    "candidates",
                    "elapsed_seconds",
                    "tested",
                    "stop_reason",
                    "test_auc",
                ]
            ].copy()
            comparison_view["phase"] = comparison_view["phase"].map(ui_mod.phase_label)
            comparison_view["stop_reason"] = comparison_view["stop_reason"].map(ui_mod.reason_text)
            comparison_view.columns = [
                "轮次",
                "运行 ID",
                "阶段",
                "使用 Token",
                "云调用",
                "候选数",
                "运行时间（秒）",
                "已完成测试",
                "停止原因",
                "独立测试 AUC",
            ]
            st.dataframe(comparison_view, use_container_width=True)

            if len(runs) >= 2:
                st.markdown("**假设结果对比（第一轮 vs 最新一轮）**")
                round_1 = report_mod.summarize_run(runs[0]["events"])
                round_latest = report_mod.summarize_run(runs[-1]["events"])
                left_round, right_round = st.columns(2)
                with left_round:
                    st.markdown("**第一轮**")
                    st.markdown(ui_mod.hypothesis_round_html(round_1), unsafe_allow_html=True)
                with right_round:
                    st.markdown("**最新一轮**")
                    st.markdown(ui_mod.hypothesis_round_html(round_latest), unsafe_allow_html=True)
                stats_1 = ui_mod.experiment_summary(round_1)
                stats_latest = ui_mod.experiment_summary(round_latest)
                if stats_latest["experiments"] > stats_1["experiments"]:
                    st.success(
                        "完成度提升：已验证假设从 %d 个提高到 %d 个。"
                        % (stats_1["experiments"], stats_latest["experiments"])
                    )
                elif stats_latest["experiments"] == stats_1["experiments"]:
                    st.info(
                        "最新一轮完成度保持，节省主要来自决策缓存和本地规则。"
                    )
                else:
                    st.warning("最新一轮已验证假设数量减少。")

    report_md = report_mod.render_markdown(summary)
    st.download_button(
        "下载 Markdown 报告",
        data=report_md.encode("utf-8"),
        file_name=f"report-{summary.get('run_id', 'run')}.md",
        mime="text/markdown",
    )
    if evaluation is not None:
        archive = evaluation / "evaluation-evidence.zip"
        if archive.is_file():
            st.download_button(
                "下载本轮评测证据包",
                archive.read_bytes(),
                file_name=archive.name,
            )
            return
    try:
        command = report_mod.default_run_command(
            st.session_state.get("events_path", str(MOCK_EVENTS))
        )
        git_commit = report_mod.repository_revision(REPO_ROOT)
        manifest = report_mod.build_delivery_manifest(summary, git_commit=git_commit)
        package = report_mod.build_reproduce_package(
            summary,
            run_dir=REPO_ROOT,
            out_dir=DEMO_DIR,
            command=command,
            report_md=report_md,
            delivery_manifest=manifest,
        )
        snapshot = report_mod.freeze_delivery_snapshot(
            summary,
            package,
            DEMO_DIR,
            delivery_manifest=manifest,
        )
        st.download_button(
            "下载复现包 zip",
            data=package.read_bytes(),
            file_name=package.name,
            mime="application/zip",
        )
        st.download_button(
            "下载交付封存快照",
            data=snapshot.read_bytes(),
            file_name=snapshot.name,
            mime="application/json",
        )
    except Exception as exc:
        st.warning(f"复现包生成失败：{exc}")


def _run_replay(evaluation: Path, run_id: str) -> Tuple[bool, str]:
    command = [
        sys.executable,
        "-m",
        "autoresearch.replay",
        "--run-id",
        run_id,
        "--output-dir",
        str(evaluation),
    ]
    try:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return False, "复现执行失败：%s" % exc
    if proc.returncode == 0:
        return True, "离线复现完成，结果已写入 repro/summary.json"
    return False, (proc.stderr or proc.stdout or "复现执行失败").strip()[-600:]


def _render_evaluation(evaluation: Optional[Path], events_path: Path, run_id: str) -> None:
    if evaluation is None:
        st.info("请在侧栏选择一轮已保存评测。")
        return
    if st.button("运行离线复现（replay）"):
        ok, message = _run_replay(evaluation, run_id)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)
    try:
        metrics = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        rows = pd.DataFrame(metrics["arms"]).reindex(
            columns=[
                "arm",
                "test_auc",
                "candidates",
                "cloud_calls",
                "charged_tokens",
                "usage_source",
                "wall_seconds",
            ]
        )
        rows.columns = [
            "策略",
            "独立测试 AUC",
            "候选数",
            "云调用",
            "Token",
            "用量来源",
            "耗时（秒）",
        ]
        rows["用量来源"] = rows["用量来源"].map(ui_mod.usage_source_text)
        data_name = Path(str(metrics.get("data_path") or "")).name
        st.caption("内置合成面板" if data_name == DEMO_CSV.name else "数据：" + data_name)
        st.dataframe(rows, hide_index=True)
        replay_path = events_path.parent.parent / "repro" / "summary.json"
        if replay_path.is_file():
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            if replay.get("run_id") == run_id:
                st.markdown("**离线复现对账**")
                st.write(ui_mod.status_text(replay.get("status")))
                st.caption("复现云调用：%s 次" % replay.get("cloud_calls_replayed", "—"))
                checks = pd.DataFrame(replay.get("checks", []))
                if not checks.empty:
                    checks["name"] = checks["name"].map(ui_mod.replay_check_text)
                    checks["passed"] = checks["passed"].map(
                        lambda value: "通过" if value else "未通过"
                    )
                    checks["detail"] = checks["detail"].map(ui_mod.replay_detail_text)
                    checks.columns = ["检查项", "是否通过", "对账详情"]
                    st.dataframe(checks, hide_index=True)
                return
        st.info("当前运行暂无复现记录。")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error("读取评测产物失败：%s" % exc)


def main() -> None:
    st.set_page_config(page_title="Helix 赫力 评审工作台", layout="wide")
    _inject_styles()
    _render_header()
    _render_innovation()

    if st.sidebar.button("重置页面"):
        st.session_state.clear()
        st.rerun()

    sources = _source_options()
    source = st.sidebar.radio("数据源", list(sources))
    roadshow_mode = st.sidebar.checkbox(
        "路演模式（4 分钟引导）",
        value=bool(st.session_state.get("roadshow_mode", False)),
    )
    st.session_state["roadshow_mode"] = roadshow_mode
    if roadshow_mode:
        source = "离线演示（mock_events.json）"
    roadshow_auto = st.sidebar.checkbox(
        "路演倒计时自动刷新（2 秒）",
        value=roadshow_mode,
    )
    st.session_state["source"] = source
    events_path = sources[source]
    evaluation = events_path.parents[2] if source.startswith("已保存评测") else None
    mode, mode_style = ui_mod.source_mode(source)
    if mode_style == "success":
        st.sidebar.success("数据模式：%s" % mode)
    else:
        st.sidebar.info("数据模式：%s" % mode)
    st.session_state["events_path"] = str(events_path)
    events, load_error = _safe_load_events(events_path)
    runs = report_mod.group_runs(events)
    if runs:
        run_options = [str(run["run_id"]) for run in runs]
        selected = st.sidebar.selectbox("运行", run_options)
        summary = report_mod.summarize_run(
            next(run["events"] for run in runs if run["run_id"] == selected)
        )
        if evaluation is not None:
            try:
                saved = json.loads((events_path.parent / "summary.json").read_text(encoding="utf-8"))
                if saved.get("run_id") == selected:
                    for field in ("baseline", "locked", "test_verdict", "tested"):
                        if field in saved:
                            summary[field] = saved[field]
            except (OSError, ValueError, AttributeError) as exc:
                st.warning("读取运行摘要失败：%s" % exc)
        st.sidebar.metric("阶段", ui_mod.phase_label(summary.get("phase")))
        st.sidebar.markdown("**停止原因**")
        st.sidebar.markdown(
            '<span class="ar-stop-reason">%s</span>'
            % html.escape(ui_mod.reason_text(summary.get("stop_reason"))),
            unsafe_allow_html=True,
        )
    else:
        summary = {}
        if load_error:
            st.sidebar.error(load_error)
    with st.sidebar.expander("CLI 降级路径"):
        st.code("python -m autoresearch.cli demo", language="bash")
        if events_path != MOCK_EVENTS:
            st.code("python -m autoresearch.cli replay --events %s" % events_path, language="bash")

    if roadshow_mode:
        _render_roadshow()

    tab_create, tab_track, tab_review, tab_eval, tab_material = st.tabs(
        ["创建任务", "跟踪研究", "审查结论", "评测与复现", "评审材料"]
    )
    with tab_create:
        if evaluation is None:
            _render_create_task()
        else:
            st.info("如需创建任务，请在侧栏切换到实时运行或离线演示。")
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
            _render_review(summary, runs, evaluation)
        else:
            st.info("完成研究后，这里会显示基线/锁定候选/独立测试对照与下载。")
    with tab_eval:
        _render_evaluation(evaluation, events_path, str(summary.get("run_id", "")))
    with tab_material:
        _render_material()

    if roadshow_mode and roadshow_auto:
        time.sleep(2.0)
        st.rerun()


if __name__ == "__main__":
    main()
