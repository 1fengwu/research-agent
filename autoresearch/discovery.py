"""Generate a testable research question from training-period observations."""

from dataclasses import replace
from time import monotonic
from urllib.error import URLError
from uuid import uuid4

from .data import build_profile, read_panel
from .llm_router import estimate_tokens, parse_json_content

HORIZONS = (1, 3, 5, 10)
PROMPT = (
    "你是自主研究 Agent。场景仅限量化投资。根据提供的训练期量价数据画像，"
    "自主提出一个可检验的研究问题，选择合适的预测区间和有理由的候选线索。"
    "当前实验引擎支持每日横截面收益排序：标签为未来收益是否高于当天同组中位数。"
    "问题必须准确反映该目标和所选交易日数，不承诺收益。只引用输入中确实存在的数据观察，"
    "把待验证的推测和观测区分开。只返回 JSON 对象，字段 question（中文问题）、"
    "rationale（中文数据依据）、horizon（允许的整数）、hypotheses（按优先级排列的候选 ID 数组）。"
    "问题用一句白话，最多45字。依据用一句话，最多65字。把样本规模和专业标签留在数据画像里。"
)


def discover(task, client, ledger, allowed_hypotheses, on_profile=None):
    panel = read_panel(task.data_path)
    dates = sorted(panel["date"].unique())
    non_test_days = len(dates) - round(len(dates) * task.split.test)
    train_days = round(non_test_days * task.split.train)
    training = panel[panel["date"].isin(dates[:train_days])]
    if training.empty:
        raise ValueError("训练期没有可用数据")
    profile = build_profile(training, horizon=1)
    returns = training.sort_values(["symbol", "date"]).groupby("symbol")["close"].pct_change(fill_method=None)
    facts = {
        "training_rows": profile["rows"],
        "symbols": profile["symbols"],
        "trading_days": int(training["date"].nunique()),
        "missing_rates": profile["missing_rates"],
        "daily_return_median_abs": round(float(returns.abs().median()), 6) if returns.notna().any() else None,
        "horizons": HORIZONS,
        "hypotheses": allowed_hypotheses,
    }
    if on_profile is not None:
        on_profile(facts)
    estimated = estimate_tokens(facts) + estimate_tokens(PROMPT)
    max_tokens = 700
    if ledger.remaining_cloud_calls() <= 0:
        raise ValueError("没有剩余云调用额度")
    ledger.preflight_cloud(estimated, max_tokens)
    started = monotonic()
    call = {"call_id": uuid4().hex, "outcome": "responded"}
    try:
        reply = client.call(facts, max_tokens=max_tokens, timeout_s=min(30, ledger.remaining_seconds()), system_prompt=PROMPT)
    except (RuntimeError, TimeoutError) as exc:
        cause = exc.__cause__ or exc
        if isinstance(cause, URLError):
            cause = cause.reason
        call.update(outcome="timeout" if isinstance(cause, TimeoutError) else "failed",
                    elapsed_seconds=monotonic() - started)
        return task, {"status": "failed", "facts": facts}, [], call
    call["elapsed_seconds"] = monotonic() - started
    usage = reply.get("usage") or {}
    ledger.record_cloud(estimated_input=estimated,
                        actual_input=usage.get("prompt_tokens") or estimated,
                        actual_output=usage.get("completion_tokens") or estimate_tokens(reply["content"]),
                        decision_note="research discovery")
    parsed = parse_json_content(reply["content"])
    if not isinstance(parsed, dict):
        return task, {"status": "invalid", "facts": facts}, [], call
    horizon = parsed.get("horizon")
    question, rationale = parsed.get("question"), parsed.get("rationale")
    nominated = parsed.get("hypotheses")
    if (type(horizon) is not int or horizon not in HORIZONS or not isinstance(question, str)
            or not question.strip() or not isinstance(rationale, str) or not rationale.strip()
            or not isinstance(nominated, list)):
        return task, {"status": "invalid", "facts": facts}, [], call
    order = list(dict.fromkeys(item for item in nominated if isinstance(item, str) and item in allowed_hypotheses))
    if not order:
        return task, {"status": "invalid", "facts": facts}, [], call
    discovery = {"status": "generated", "question": question.strip()[:350], "rationale": rationale.strip()[:550],
                 "horizon": horizon, "facts": facts, "hypotheses": order}
    return apply_question(task, discovery), discovery, order, call


def apply_question(task, discovery):
    horizon = discovery["horizon"]
    return replace(task, label_horizon=horizon, split=replace(task.split, gap_days=max(task.split.gap_days, horizon)))
