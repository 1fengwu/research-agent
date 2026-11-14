"""AutoResearch 角色 D：前端展示辅助，保持 UI 文案与状态映射集中可测。"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Tuple

PHASE_LABELS = {
    "UNSEEN": "未开始",
    "PLANNING": "规划中",
    "PLANNED": "已规划",
    "EXPERIMENTING": "实验中",
    "VERIFIED": "已验证",
    "REVISED": "已修正",
    "LOCKED": "已锁定",
    "TESTED": "已测试",
    "PASSED": "通过",
    "CONVERGED": "收敛",
    "FAILED": "失败",
    "PARTIAL_COMPLETED": "部分完成",
}

VERDICT_LABELS = {
    "supported": "支持",
    "refuted": "反驳",
    "inconclusive": "证据不足",
    "blocked": "拦截",
    "partial_completed": "部分完成",
}

EVENT_LABELS = {
    "research.run_started": "运行开始",
    "research.data_profile": "数据画像",
    "research.baseline": "固定基线",
    "research.hypothesis_registered": "假设预注册",
    "research.experiment_started": "实验开始",
    "research.experiment_result": "实验完成",
    "research.experiment_verdict": "实验判决",
    "research.experiment_skipped": "实验跳过",
    "research.revised": "证据修正",
    "research.escalation": "云升级",
    "research.escalation_blocked": "预算拦截",
    "research.escalation_failed": "升级失败",
    "research.escalation_invalid": "无效回包",
    "research.local_rule": "本地规则决策",
    "research.escalation_cached": "缓存命中",
    "research.locked": "锁定候选",
    "research.tested": "独立测试",
    "research.completed": "运行完成",
}

REASON_REPLACEMENTS = [
    ("budget_blocked", "预算不足"),
    ("cloud_token_reserve_exhausted", "预留云 token 不足"),
    ("cloud_token_exhausted", "云 token 已用尽"),
    ("cloud_call_need_", "云调用所需 token 超过剩余预算"),
    ("time_reserve_reached", "时间预留已触发"),
    ("partial_completed", "预算或时间不足，研究部分完成"),
    ("plan_exhausted", "候选规划已穷尽"),
    ("no_experiments", "没有可执行实验"),
    ("candidate_budget", "候选实验预算已用完"),
    ("cloud_call_budget", "云调用次数预算已用完"),
    ("cloud_tokens_budget", "云 token 预算已用完"),
    ("time_budget", "墙钟时间已用完"),
    ("duplicate_fingerprint", "重复实验已跳过"),
    ("window_conflict", "两个验证窗方向相反"),
    ("planning_stall", "连续多轮没有进展"),
    ("repeated_error_signature", "同一错误重复出现"),
    ("data_anomaly", "数据异常"),
    ("escalation_invalid", "云回包无效"),
    ("local_rule:", "本地规则："),
    ("cached:", "缓存命中："),
    ("continue_with_next_hypothesis", "继续下一个假设"),
    ("candidate=", "候选实验 "),
    ("hypothesis=", "假设 "),
    ("locked:", "锁定 "),
    ("verdict:", "判决 "),
    ("evidence_signals:", "证据信号 "),
    ("stop_reason:", "停止原因 "),
]

INNOVATION_POINTS = [
    {
        "title": "唯一裁判",
        "detail": "Oracle 是唯一出分口，Agent 可以执行实验，但不能自评结果。",
        "tag": "裁判隔离",
    },
    {
        "title": "证据驱动修正",
        "detail": "任一验证窗下滑超过 0.01 即标记 refuted，修正必须引用证据编号。",
        "tag": "可证伪",
    },
    {
        "title": "预算透明",
        "detail": "预算不足时显示 budget_blocked，每次运行使用独立 ledger，不偷跑。",
        "tag": "可审计",
    },
    {
        "title": "无云复现",
        "detail": "replay 只读事件流与账本重建流程，不重新调用云模型。",
        "tag": "可复现",
    },
]

ROADSHOW_STEPS = [
    {
        "seconds": 40,
        "tab": "创建任务",
        "title": "问题定义与数据",
        "detail": "一句话定义：验证量价信号对未来 5 日相对强弱是否有预测力，不是给出投资建议。",
    },
    {
        "seconds": 100,
        "tab": "跟踪研究",
        "title": "完整研究闭环",
        "detail": "展示基线→候选→证据修正→锁定→独立测试，让评委看到闭环已经跑通。",
    },
    {
        "seconds": 140,
        "tab": "跟踪研究",
        "title": "预算与证据",
        "detail": "切换低预算运行，展示 budget_blocked、PARTIAL_COMPLETED 和窗口下滑 refuted。",
    },
    {
        "seconds": 190,
        "tab": "审查结论",
        "title": "三臂与前置门",
        "detail": "展示前置门通过、三臂对照表，说明系统会自我评估。",
    },
    {
        "seconds": 230,
        "tab": "评审材料",
        "title": "商业价值与创新",
        "detail": "讲目标用户、应用场景，并指到顶部四个创新点。",
    },
    {
        "seconds": 240,
        "tab": "审查结论",
        "title": "复现包与收尾",
        "detail": "下载报告与复现包，展示 CLI 降级命令，一句话收尾。",
    },
]


def innovation_points():
    return [dict(item) for item in INNOVATION_POINTS]


def roadshow_step(elapsed_seconds: Any) -> Dict[str, Any]:
    try:
        elapsed = float(elapsed_seconds)
    except (TypeError, ValueError):
        elapsed = 0.0
    current = ROADSHOW_STEPS[0]
    index = 0
    for i, step in enumerate(ROADSHOW_STEPS):
        if elapsed >= float(step["seconds"]):
            current = step
            index = i
    return {"index": index, "step": dict(current), "elapsed_seconds": max(0.0, elapsed)}


def phase_label(phase: Any) -> str:
    key = str(phase or "").upper()
    return PHASE_LABELS.get(key, str(phase or "—"))


def verdict_label(verdict: Any) -> str:
    key = str(verdict or "").lower()
    return VERDICT_LABELS.get(key, str(verdict or "—"))


def event_label(event_type: Any) -> str:
    key = str(event_type or "")
    return EVENT_LABELS.get(key, key or "—")


def reason_text(reason: Any) -> str:
    text = str(reason or "")
    for token, replacement in REASON_REPLACEMENTS:
        text = text.replace(token, replacement)
    return text


def source_mode(source: str) -> Tuple[str, str]:
    """返回 (模式名, streamlit 风格)。"""
    text = str(source or "")
    if (
        "mock_events" in text
        or "离线演示" in text
        or "demo/runner-events.jsonl" in text
        or "/demo/runner-events" in text
    ):
        return "离线回放", "info"
    if "已保存评测" in text:
        return "评测回放", "success"
    if ".arc" in text or "runner-events" in text:
        return "真实研究", "success"
    return "自定义数据源", "info"


def progress_fraction(remaining: Any, limit: Any) -> float:
    try:
        remaining = float(remaining)
        limit = float(limit)
    except (TypeError, ValueError):
        return 0.0
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, remaining / limit))


def experiment_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    experiments = summary.get("experiments") or []
    verdicts = [
        str((item.get("verdict") or {}).get("verdict") or "skipped")
        for item in experiments
    ]
    return {
        "experiments": len(experiments),
        "revised": int(summary.get("revised_count") or 0),
        "blocked": len(summary.get("escalation_blocked") or []),
        "refuted": verdicts.count("refuted"),
        "supported": verdicts.count("supported"),
        "inconclusive": verdicts.count("inconclusive"),
        "evidence_ids": list(summary.get("evidence_ids") or []),
    }


def metrics_arms(metrics: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(metrics, dict):
        return []
    arms = metrics.get("arms")
    if not isinstance(arms, list):
        return []
    return [row for row in arms if isinstance(row, dict)]


def phase_color(phase: Any) -> str:
    colors = {
        "UNSEEN": "#64748b",
        "PLANNING": "#94a7c7",
        "PLANNED": "#94a7c7",
        "EXPERIMENTING": "#a855f7",
        "VERIFIED": "#3b82f6",
        "REVISED": "#a855f7",
        "LOCKED": "#3b82f6",
        "TESTED": "#3b82f6",
        "PASSED": "#3b82f6",
        "CONVERGED": "#3b82f6",
        "FAILED": "#f87171",
        "PARTIAL_COMPLETED": "#f87171",
    }
    return colors.get(str(phase or "").upper(), "#64748b")


def verdict_color(verdict: Any) -> str:
    colors = {
        "supported": "#3b82f6",
        "refuted": "#f87171",
        "blocked": "#f87171",
        "inconclusive": "#94a7c7",
        "partial_completed": "#f87171",
    }
    return colors.get(str(verdict or "").lower(), "#94a7c7")


def _spec_text(spec: Optional[Dict[str, Any]]) -> str:
    spec = spec or {}
    parts = [str(spec.get("feature_set") or "base"), str(spec.get("model") or "")]
    params = spec.get("params") or {}
    if params:
        parts.append("params=" + json.dumps(params, ensure_ascii=False, sort_keys=True))
    if spec.get("missing") and spec.get("missing") != "median":
        parts.append("missing=" + str(spec["missing"]))
    if spec.get("scaling") and spec.get("scaling") != "standard":
        parts.append("scaling=" + str(spec["scaling"]))
    return " · ".join(parts)


def timeline_html(events: List[Dict[str, Any]]) -> str:
    rows: List[str] = []
    for event in events or []:
        phase = str(event.get("phase") or "—")
        color = phase_color(phase)
        timestamp = str(event.get("timestamp") or "")
        if timestamp:
            timestamp = timestamp[11:19]
        label = html.escape(event_label(event.get("type")))
        phase_text = html.escape(phase_label(phase))
        experiment_id = html.escape(str(event.get("experiment_id") or ""))
        reason = html.escape(reason_text(event.get("decision_reason")))
        remaining = event.get("remaining") or {}
        tokens = str(remaining.get("remaining_cloud_tokens") or "")
        rows.append(
            '<div class="ar-tl-row">'
            '<span class="ar-tl-dot" style="background:%s"></span>'
            '<div class="ar-tl-body">'
            '<div class="ar-tl-head"><strong>%s</strong>'
            '<span class="ar-tl-phase">%s</span></div>'
            '<div class="ar-tl-meta"><span>%s</span><span>实验 %s</span>'
            '<span>剩余 %s token</span></div>'
            '<div class="ar-tl-reason">%s</div>'
            "</div></div>"
            % (
                color,
                label,
                phase_text,
                timestamp,
                experiment_id or "未指定",
                tokens,
                reason,
            )
        )
    return '<div class="ar-timeline">' + "".join(rows) + "</div>"


def experiments_html(summary: Dict[str, Any]) -> str:
    items = summary.get("experiments") or []
    baseline = summary.get("baseline") or {}
    try:
        baseline_auc = float(baseline.get("merged_auc") or 0.0)
    except (TypeError, ValueError):
        baseline_auc = 0.0
    if not items:
        return '<div class="ar-empty">没有候选实验事件</div>'

    cards: List[str] = []
    for item in items:
        spec = item.get("spec") or {}
        trial = item.get("trial") or {}
        verdict = item.get("verdict") or {}
        raw_verdict = str(verdict.get("verdict") or "")
        skipped = bool(item.get("skipped"))
        if skipped:
            raw_verdict = "skipped"
        color = verdict_color(raw_verdict)
        verdict_text = "跳过" if skipped else verdict_label(raw_verdict)
        experiment_id = html.escape(str(item.get("experiment_id") or ""))
        hypothesis_id = html.escape(str(item.get("hypothesis_id") or ""))
        spec_text = html.escape(_spec_text(spec))

        val1 = trial.get("val1") or {}
        val2 = trial.get("val2") or {}
        try:
            auc1 = float(val1.get("auc") or 0.0)
        except (TypeError, ValueError):
            auc1 = 0.0
        try:
            auc2 = float(val2.get("auc") or 0.0)
        except (TypeError, ValueError):
            auc2 = 0.0
        valid1 = int(val1.get("valid_days") or 0)
        valid2 = int(val2.get("valid_days") or 0)
        width1 = max(2.0, min(100.0, auc1 * 100.0))
        width2 = max(2.0, min(100.0, auc2 * 100.0))

        cards.append(
            '<div class="ar-exp-card" style="border-left-color:%s">'
            '<div class="ar-exp-head">'
            '<span class="ar-exp-id">%s</span>'
            '<span>%s</span>'
            '<span class="ar-verdict" style="color:%s">%s</span>'
            "</div>"
            '<div class="ar-exp-spec">%s</div>'
            '<div class="ar-exp-bars">'
            '<div class="ar-bar-row"><span>val1</span>'
            '<div class="ar-bar-track"><div class="ar-bar" style="width:%s%%"></div></div>'
            '<span>%.4f · %dd</span></div>'
            '<div class="ar-bar-row"><span>val2</span>'
            '<div class="ar-bar-track"><div class="ar-bar ar-bar-2" style="width:%s%%"></div></div>'
            '<span>%.4f · %dd</span></div>'
            "</div>"
            '<div class="ar-exp-meta">基线 %.4f · 缺失 %s</div>'
            "</div>"
            % (
                color,
                experiment_id,
                hypothesis_id or "—",
                color,
                verdict_text,
                spec_text,
                width1,
                auc1,
                valid1,
                width2,
                auc2,
                valid2,
                baseline_auc,
                html.escape(str(trial.get("missing_rates") or "")),
            )
        )
    return '<div class="ar-exp-grid">' + "".join(cards) + "</div>"


def hyphen_html(values: List[str]) -> str:
    chips = [
        '<span class="ar-chip">%s</span>' % html.escape(str(value))
        for value in values
        if str(value).strip()
    ]
    return '<div class="ar-chip-row">' + "".join(chips) + "</div>"


def hypotheses_html(summary: Dict[str, Any]) -> str:
    hypotheses = summary.get("hypotheses") or []
    chips: List[str] = []
    for hypothesis in hypotheses:
        spec = hypothesis.get("spec") or {}
        text = "%s · %s · %s" % (
            hypothesis.get("experiment_id"),
            spec.get("hypothesis_id"),
            _spec_text(spec),
        )
        chips.append('<span class="ar-chip">%s</span>' % html.escape(text))
    return '<div class="ar-chip-row">' + "".join(chips) + "</div>"
