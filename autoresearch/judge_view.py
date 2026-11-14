"""Single-page competition demo backed by the normal research CLI."""

import json
import csv
from base64 import b64encode
from functools import lru_cache
from math import atan2, degrees, hypot
from html import escape
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

from autoresearch import report, ui
from autoresearch.data import file_sha256

ASSETS = Path(__file__).resolve().parent
STYLES = (ASSETS / "judge.css").read_text(encoding="utf-8")
HERO_STYLES = (ASSETS / "hero-tailwind.css").read_text(encoding="utf-8")
HERO_SCRIPT = (ASSETS / "hero.bundle.js").read_text(encoding="utf-8")
LANDING_HERO = st.components.v2.component(
    "helix_landing", html='<div class="helix-react-hero"></div>',
    js=HERO_SCRIPT, css=f"{HERO_STYLES}\n{STYLES}", isolate_styles=False,
)


def _stage(key, active=False, seen=None):
    container = st.container(key=f"helix-stage-{key}")
    fresh = seen is not None and key not in seen
    if seen is not None:
        seen.add(key)
    marker = "active" if active else ""
    animation = "fresh" if fresh else ""
    container.html(f'<span aria-hidden="true" class="helix-stage-marker {marker}"></span>'
                   f'<span aria-hidden="true" class="helix-stage-content {animation}"></span>')
    return container


def _table(rows, comparison=False):
    columns = list(rows[0])
    headers = "".join(f'<th scope="col">{escape(column)}</th>' for column in columns)
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = str(row[column])
            tone = ""
            if comparison and column == "变化":
                if value.startswith(("提高", "下降", "增加")):
                    tone = "gain" if value.startswith(("提高", "下降")) else "cost"
                    if row["指标"] == "独立测试 AUC" and value.startswith("下降"):
                        tone = "cost"
            cells.append(f'<td class="{tone}">{escape(value)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.html('<div class="helix-table" role="region" aria-label="研究指标表格" tabindex="0">'
            f'<table><thead><tr>{headers}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')


def launch(session_dir, task, previous=None):
    attempt = session_dir / uuid4().hex
    attempt.mkdir(parents=True)
    policy = attempt / "strategy.db"
    if previous is not None:
        # The previous worker has closed SQLite; its policy is the retry checkpoint.
        shutil.copyfile(previous / "strategy.db", policy)
    task_file = attempt / "task.json"
    task_file.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
    command = [
        sys.executable, "-m", "autoresearch.cli", "run",
        "--task", str(task_file),
        "--events", str(attempt / "events.jsonl"),
        "--output-dir", str(attempt),
        "--ledger-dir", str(attempt / "ledgers"),
        "--evolution", str(policy),
        "--strategy-context", str(attempt / "strategy.json"),
        "--summary-file", str(attempt / "summary.json"),
    ]
    if previous is not None:
        question_file = previous / "research-question.json"
        if question_file.exists():
            command.extend(["--research-question", str(question_file)])
        for experiment in _read_json(previous / "summary.json").get("experiments", []):
            hypothesis = (experiment.get("spec") or {}).get("hypothesis_id")
            if hypothesis and (experiment.get("verdict") or {}).get("verdict") == "refuted":
                command.extend(["--defer-hypothesis", hypothesis])
    else:
        command.append("--discover")
    with (attempt / "worker.log").open("w", encoding="utf-8") as log:
        worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    return attempt, worker


def _read_live_events(path):
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    # The writer may still be finishing the last line during a poll.
    return [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _seconds(value):
    return "—" if value is None else f"{value:.1f} 秒"


def _progress(events):
    kinds = {event["type"] for event in events}
    current = 2 if "research.locked" in kinds else 1 if "research.experiment_started" in kinds else 0
    stages = ("分析数据并提出问题", "实验验证", "独立测试")
    st.caption(" → ".join(stages))
    st.progress(current / len(stages), text=f"研究中 · {stages[current]}")
    summary = report.summarize_run(events)
    hypotheses = summary["hypotheses"]
    if current == 0 and hypotheses:
        label = ui.hypothesis_label(hypotheses[-1]["spec"].get("hypothesis_id"))
        st.caption(f"最近提出的假设 · {label}")
    elif current == 1:
        count = sum(bool(item.get("verdict")) for item in summary["experiments"])
        st.caption(f"已验证 {count} 项实验")
    elif current == 2:
        st.caption("已锁定候选，正在读取独立测试结果。")
    _usage(summary, events)


def _hypotheses(summary):
    labels = list(dict.fromkeys(
        ui.hypothesis_label(item["spec"]["hypothesis_id"])
        for item in summary.get("hypotheses", [])
        if (item.get("spec") or {}).get("hypothesis_id")
    ))
    if labels:
        st.subheader("待验证的假设")
        st.write("这些线索能改善排序吗？" + "、".join(labels[:4]) + "。")


def _usage(summary, events):
    tokens, latency = st.container(key="helix-usage").columns(2)
    tokens.metric("累计云端 Token", (summary.get("budget") or {}).get("used_cloud_tokens", 0))
    call = next((event["extra"] for event in reversed(events)
                 if event["type"] == "research.model_call"), None)
    if call is None:
        value = "未记录" if any(event.get("tokens", {}).get("charged") for event in events) else "本轮暂无云调用"
        latency.metric("最近一次模型响应", value)
        return
    outcome = {"failed": "调用失败", "timeout": "调用超时"}.get(call["outcome"])
    latency.metric("最近一次模型响应", _seconds(call["elapsed_seconds"]))
    if outcome:
        latency.caption(f"{outcome} · 等待时长")


def _auc(summary):
    verdict = summary.get("test_verdict") or {}
    if summary.get("tested") and verdict.get("test") and verdict.get("verdict") not in ("blocked", "partial_completed"):
        return (verdict.get("metric") or {}).get("auc")
    return None


def _verdict(summary):
    verdict = (summary.get("test_verdict") or {}).get("verdict")
    return ui.verdict_label(verdict) if verdict else "未完成独立测试"


def _signal(summary):
    locked = summary.get("locked") or {}
    hypothesis = (locked.get("config") or {}).get("hypothesis_id")
    if hypothesis:
        return ui.hypothesis_label(hypothesis)
    name = locked.get("name")
    return "基线" if name == "baseline" else name or "未选定"


def _validation_steps(summary):
    experiments = [
        item for item in summary.get("experiments", [])
        if not item.get("skipped") and (item.get("verdict") or {}).get("verdict")
    ]
    if not experiments:
        return
    st.subheader("Agent 怎样验证假设")
    st.write("先用原有方法作参照，再把候选方案放到两段不同时间的数据里验证，看看表现能否保持。")
    explanations = {
        "volatility": "价格过去波动的大小，能帮助把表现较好的标的排在前面吗？",
        "volume": "成交量的变化能帮助判断哪些标的相对表现更好吗？",
        "volatility_volume": "把价格波动和成交量一起看，能改善排序吗？",
        "tree_shallow": "用一组简单的决策规则，能改善排序吗？",
    }
    outcomes = {
        "supported": "验证结果支持保留这条线索",
        "refuted": "验证结果反驳了这条假设",
        "inconclusive": "两段数据的结果不足以支持这条假设",
        "blocked": "实验未通过检查",
    }
    # Show one retained lead and one counterexample when the run has both.
    selected = [next((item for item in experiments if item["verdict"]["verdict"] == verdict), None)
                for verdict in ("supported", "refuted")]
    examples = [item for item in selected if item is not None] or experiments[:2]
    for item in examples:
        spec = item.get("spec") or {}
        hypothesis = spec.get("hypothesis_id") or item.get("hypothesis_id")
        name = ui.hypothesis_label(hypothesis or spec.get("experiment_id") or item.get("experiment_id"))
        question = explanations.get(hypothesis, f"「{name}」能改善排序吗？")
        verdict = item["verdict"]["verdict"]
        st.write(f"**{name}** · {question} {outcomes.get(verdict, ui.verdict_label(verdict))}。")


def _result(summary, events=None):
    discovery = summary.get("research_question")
    if discovery:
        if discovery.get("facts"):
            with _stage("observation"):
                _data_observation(discovery["facts"])
        with _stage("question"):
            _research_question(discovery)
    timeline = report.summarize_run(events) if events else summary
    if timeline.get("hypotheses"):
        with _stage("hypotheses"):
            _hypotheses(timeline)
    if _has_verdict(summary):
        with _stage("validation"):
            _validation_steps(summary)
    with _stage("result"):
        _conclusion(summary)


def _has_verdict(summary):
    return any(not item.get("skipped") and (item.get("verdict") or {}).get("verdict")
               for item in summary.get("experiments", []))


def _conclusion(summary):
    st.subheader("研究结论")
    signal = _signal(summary)
    if signal == "历史波动率":
        st.write("**这次最值得保留的线索，是价格过去波动有多大。**")
    elif signal == "基线":
        st.write("**这次研究保留了原来的方法。**")
    elif signal != "未选定":
        st.write(f"**这次最值得保留的线索是「{signal}」。**")
    else:
        st.write("**本轮尚未选出可用的线索。**")
    auc = _auc(summary)
    if auc is not None:
        st.write(f"**独立测试排序评分约为 {auc * 100:.0f} 分**，随机排序平均 50 分，理想排序 100 分。")
        horizon = summary.get("label_horizon") or (summary.get("task") or {}).get("label_horizon", 5)
        st.caption(f"同一天选两个标的，一个未来 {horizon} 个交易日的收益高于当天同组标的的中位数，另一个不高于。评分衡量模型把前者排在后者前面的能力，按有效测试日取平均，同分算半次。")
        st.caption("这个分数用于比较排序能力。投资收益需要另做交易回测。")
    else:
        st.write("本轮还没有可用的独立测试评分。")
    with st.expander("专业指标与验证证据"):
        _technical_result(summary, auc)


def _research_question(discovery):
    if not discovery:
        return
    if discovery["status"] == "generated":
        st.subheader("Agent 提出的研究问题")
        question = discovery["question"]
        if len(question) > 60:
            question = f"哪些量价线索能帮我们排出未来 {discovery['horizon']} 个交易日表现更好的标的？"
        st.write(question)
        with st.expander("查看问题依据"):
            st.html(
                '<div class="helix-question-evidence">'
                f'<p>{escape(discovery["question"])}</p>'
                f'<p>{escape(discovery["rationale"])}</p></div>'
            )
        st.caption(f"实际实验口径 · 未来 {discovery['horizon']} 个交易日的同组收益排序")
    else:
        st.caption("本轮沿用已有的量价排序问题。")


def _data_observation(facts):
    st.caption(f"数据观察 · 训练期有 {facts['symbols']} 个标的、{facts['trading_days']} 个交易日")


def _technical_result(summary, auc):
    st.caption(f"独立测试 AUC {_format_auc(auc)} · 裁决 {_verdict(summary)}")
    test = summary.get("test_verdict") or {}
    holdout = test.get("metric") or {}
    if auc is not None and holdout.get("valid_days") is not None:
        details = [f"覆盖 {holdout['valid_days']} 个有效交易日"]
        if holdout.get("pr_auc") is not None:
            details.append(f"PR-AUC {holdout['pr_auc']:.4f}")
        if holdout.get("spearman") is not None:
            details.append(f"Spearman {holdout['spearman']:.4f}")
        st.caption("独立测试 · " + " · ".join(details))
        if test.get("verdict") == "inconclusive" and "holdout_ok" in (test.get("checks") or []):
            st.caption("独立测试已完成并给出评分。假设的支持或反驳来自下方两窗验证。")

    baseline_auc = (summary.get("baseline") or {}).get("merged_auc")
    selected_auc = (summary.get("locked") or {}).get("merged_auc")
    if baseline_auc is not None and selected_auc is not None:
        st.write(
            f"两窗验证平均 AUC：基线 **{baseline_auc:.4f}**，"
            f"最终信号 **{selected_auc:.4f}**（{selected_auc - baseline_auc:+.4f}）。"
        )

    rows = []
    for experiment in summary.get("experiments") or []:
        spec = experiment.get("spec") or {}
        trial = experiment.get("trial") or {}
        verdict = (experiment.get("verdict") or {}).get("verdict")
        rows.append({
            "假设": ui.hypothesis_label(spec.get("hypothesis_id") or experiment.get("hypothesis_id") or spec.get("experiment_id") or experiment.get("experiment_id")),
            "验证窗 1 AUC": _format_auc((trial.get("val1") or {}).get("auc")),
            "验证窗 2 AUC": _format_auc((trial.get("val2") or {}).get("auc")),
            "裁决": "已跳过" if experiment.get("skipped") else ui.verdict_label(verdict),
        })
    if rows:
        _table(rows)
    if summary.get("stop_reason"):
        st.caption(f"停止原因 · {ui.reason_text(summary['stop_reason'])}")


def _format_auc(auc):
    return "—" if auc is None else f"{auc:.4f}"


def _compare(first, second):
    st.subheader("两轮研究对比")
    if (first / "research-question.json").exists():
        st.caption("两轮沿用首轮提出的同一个问题和测试口径，比较研究策略的变化。")
    initial, evolved = [_read_json(path / "strategy.json") for path in (first, second)]
    deferred = evolved.get("deferred_hypotheses") or []
    if deferred:
        st.info("第二轮把首轮验证未通过的方案移到队尾，优先尝试其他线索。延后：" +
                "、".join(map(ui.hypothesis_label, deferred)) + "。")
    elif initial["version"] == evolved["version"]:
        st.info("首轮未形成可采用的策略更新。")
    elif initial["candidate_order"] == evolved["candidate_order"]:
        st.info("首轮更新了策略，但这一更新没有改变第二轮的候选优先级或排除名单。")
    else:
        st.info("第二轮按首轮通过闸门的证据调整了候选验证顺序。")
        for index, context in enumerate((initial, evolved), 1):
            order = " → ".join(map(ui.hypothesis_label, context["candidate_order"]))
            st.caption(f"第 {index} 轮验证顺序 · {order or '无候选'}")

    summaries = [_read_json(path / "summary.json") for path in (first, second)]
    before, after = summaries
    def completed_hypotheses(summary):
        return [item["spec"]["hypothesis_id"] for item in summary["experiments"]
                if (item.get("spec") or {}).get("hypothesis_id") and item.get("trial")
                and not item["trial"].get("error") and not item.get("skipped")]

    first_completed, second_completed = map(completed_hypotheses, summaries)
    newly_tested = list(dict.fromkeys(h for h in second_completed if h not in first_completed))
    if newly_tested:
        st.write(f"**第二轮新增验证了 {len(newly_tested)} 条线索** · " + "、".join(map(ui.hypothesis_label, newly_tested)))
    executed = [
        [ui.hypothesis_label((item.get("spec") or {}).get("hypothesis_id") or item.get("experiment_id")) for item in summary["experiments"]
         if item.get("trial") and not item.get("skipped")]
        for summary in summaries
    ]
    if executed[0] == executed[1] and executed[0]:
        st.caption("两轮实际运行了相同的假设，顺序一致 · " + " → ".join(executed[0]))
    else:
        for index, order in enumerate(executed, 1):
            st.caption(f"第 {index} 轮实际实验 · " + (" → ".join(order) or "无已完成实验"))
    first_auc, second_auc = _auc(before), _auc(after)
    if first_auc is not None and second_auc is not None and round(second_auc - first_auc, 4) == 0:
        if all(after["budget"][key] >= before["budget"][key] for key in ("used_candidates", "used_cloud_tokens")) and after["elapsed_seconds"] >= before["elapsed_seconds"]:
            st.warning("两轮在独立数据上的评分相同，第二轮的实验、用量和时间均未减少。")
        else:
            st.caption("两轮独立测试 AUC 持平。资源与耗时变化见下表，单次模型用量和耗时会随实际调用变化。")
    rows = [
        {"指标": "最终信号", "第 1 轮": _signal(summaries[0]), "第 2 轮": _signal(summaries[1]), "变化": "—"},
        {"指标": "独立测试裁决", "第 1 轮": _verdict(summaries[0]), "第 2 轮": _verdict(summaries[1]), "变化": "—"},
    ]
    numeric = [
        ("独立测试 AUC", [_auc(item) for item in summaries], 4),
        ("实验次数", [item["budget"]["used_candidates"] for item in summaries], 0),
        ("累计云端 Token", [item["budget"]["used_cloud_tokens"] for item in summaries], 0),
        ("研究总耗时（秒）", [item["elapsed_seconds"] for item in summaries], 1),
    ]
    for label, (before, after), precision in numeric:
        delta = "缺少可比较结果"
        if before is not None and after is not None:
            change = round(after - before, precision)
            if change == 0:
                delta = "持平"
            else:
                direction = ("提高" if label == "独立测试 AUC" else "增加") if change > 0 else "下降"
                delta = f"{direction} {abs(change):.{precision}f}"
        rows.append({
            "指标": label,
            "第 1 轮": "—" if before is None else f"{before:.{precision}f}",
            "第 2 轮": "—" if after is None else f"{after:.{precision}f}",
            "变化": delta,
        })
    _table(rows, comparison=True)


@st.fragment(run_every=2)
def _watch():
    attempt, worker = st.session_state.judge_worker
    status = worker.poll()
    if status is not None:
        st.session_state.pop("judge_worker")
        if status == 0 and (attempt / "summary.json").is_file():
            st.session_state.judge_results.append(attempt)
        else:
            st.session_state.judge_failure = f"研究进程退出（代码 {status}）。"
            st.session_state.judge_next = bool(st.session_state.judge_results)
        st.rerun()
    path = attempt / "research-question.json"
    profile = attempt / "discovery-profile.json"
    events = _read_live_events(attempt / "events.jsonl")
    summary = report.summarize_run(events)
    seen = st.session_state.setdefault(f"helix-seen-{attempt.name}", set())
    has_hypotheses = bool(summary.get("hypotheses"))
    validating = any(event["type"] == "research.experiment_started" for event in events)
    try:
        if path.exists():
            discovery = _read_json(path)
            if discovery.get("facts"):
                with _stage("observation", seen=seen):
                    _data_observation(discovery["facts"])
            with _stage("question", not has_hypotheses and not validating, seen):
                _research_question(discovery)
        elif profile.exists():
            facts = _read_json(profile)
            with _stage("observation", True, seen):
                _data_observation(facts)
                st.caption("Agent 正在根据数据观察提出可验证的问题。")
        elif not events:
            with _stage("observation", True, seen):
                st.html('<div class="helix-preparing" role="status"><span aria-hidden="true"></span>正在准备研究数据。</div>')
    except json.JSONDecodeError:
        pass
    if has_hypotheses:
        with _stage("hypotheses", not validating, seen):
            _hypotheses(summary)
    if validating:
        with _stage("validation", True, seen):
            if _has_verdict(summary):
                _validation_steps(summary)
            else:
                st.subheader("正在验证假设")
    _progress(events)


@lru_cache(maxsize=4)
def _market_preview(path, revision):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    symbol = rows[0]["symbol"]
    series = sorted((row["date"], float(row["close"]), float(row["volume"]))
                    for row in rows if row["symbol"] == symbol)[-30:]
    closes = [close for _, close, _ in series]
    low, high = min(closes), max(closes)
    points = [(i * 100 / (len(closes) - 1), 95 - 90 * (close - low) / (high - low or 1))
              for i, close in enumerate(closes)]
    bars = []
    for (x, y), (nx, ny) in zip(points, points[1:]):
        # The plot is twice as wide as it is tall.
        dx, dy = nx - x, (ny - y) / 2
        bars.append(f'<i style="left:{x:.2f}%;top:{y:.2f}%;width:{hypot(dx,dy):.2f}%;'
                    f'transform:rotate({degrees(atan2(dy,dx)):.2f}deg)"></i>')
    delta = closes[-1] - closes[-2]
    tone = "up" if delta > 0 else "down" if delta < 0 else ""
    quote = {"symbol": symbol, "close": f"{closes[-1]:.2f}", "tone": tone,
             "change": f"{delta:+.2f} · {delta / closes[-2]:+.2%}", "points": points}
    volume_max = max(volume for _, _, volume in series) or 1
    candles = []
    for i, ((x, y), (_, close, volume)) in enumerate(zip(points, series)):
        previous = points[max(0, i - 1)][1]
        color = "up" if y <= previous else "down"
        candles.append(f'<b class="{color}" style="left:{x:.2f}%;top:{min(y, previous):.2f}%;'
                       f'height:{max(abs(y-previous), 2):.2f}%"></b>'
                       f'<em style="left:{x:.2f}%;height:{volume / volume_max * 24:.2f}%"></em>')
    chart = "".join(bars + candles)
    return quote, chart


def _research_preview(quote, summary):
    baseline = (summary.get("baseline") or {}).get("merged_auc")
    selected = (summary.get("locked") or {}).get("merged_auc")
    lift = "—" if baseline is None or selected is None else f"{(selected - baseline) * 100:+.2f} pp"
    stats = (("验证基线 AUC", _format_auc(baseline)), ("方案验证 AUC", _format_auc(selected)), ("验证提升", lift))
    auc = _auc(summary)
    score = None if auc is None else auc * 100
    days = ((summary.get("test_verdict") or {}).get("metric") or {}).get("valid_days")
    horizon = (summary.get("task") or {}).get("label_horizon")
    return dict(market=quote, stats=stats, score=score, days=days, horizon=horizon)


def render(root: Path, default_task, mock_events: Path):
    state = st.session_state
    results = state.setdefault("judge_results", [])
    panel = root / "data" / "demo_panel.csv"
    quote, chart = _market_preview(panel, panel.stat().st_mtime_ns)
    example_events = report.group_runs(report.load_events(mock_events))[0]["events"]
    example_summary = report.summarize_run(example_events)
    st.html('<div class="helix-backdrop" aria-hidden="true"><div class="helix-aurora"></div>'
            f'<div class="helix-wall"><div>{chart}</div><div>{chart}</div></div>'
            '<div class="helix-particles"></div></div>')
    running = "judge_worker" in state
    ready = not running and (not results or state.get("judge_next"))
    failure = state.get("judge_failure")
    start = replay = False
    logo = b64encode((ASSETS / "helix-logo-light.png").read_bytes()).decode("ascii")
    with st.container(key="helix-header", horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center"):
        st.html(f'<img class="helix-corner-logo" src="data:image/png;base64,{logo}" alt="HELIX">')
        if st.button("完整工作台", key="helix-workbench", type="tertiary"):
            st.query_params["view"] = "workbench"
            st.rerun()
    with st.container(key="helix-brand"):
        LANDING_HERO(data=_research_preview(quote, example_summary), key="helix-react")
    with st.container(key="helix-actions", horizontal=True, horizontal_alignment="center", vertical_alignment="center"):
        if ready or running:
            start_label = "再次研究" if results else "开始研究"
            start = st.button("重试" if failure and ready else start_label,
                              key="helix-start", type="primary", disabled=running)
        if not running:
            replay = st.button("查看分析", key="helix-replay")

    if running:
        _watch()
    elif results and not state.get("judge_next"):
        attempt = results[-1]
        summary = _read_json(attempt / "summary.json")
        events = _read_live_events(attempt / "events.jsonl")
        _usage(summary, events)
        _result(summary, events)
        if len(results) == 2:
            _compare(*results)
            if st.button("开始一组新研究"):
                for key in ("judge_results", "judge_dir", "judge_task", "judge_declined"):
                    state.pop(key, None)
                st.rerun()
        elif not state.get("judge_declined"):
            st.write("你是否要再次研究？")
            yes, done = st.columns(2)
            if yes.button("是，再次研究", type="primary"):
                state.judge_next = True
                st.rerun()
            if done.button("查看本轮结论"):
                state.judge_declined = True
                st.rerun()
        elif st.button("再次研究"):
            state.judge_next = True
            st.rerun()
    else:
        if failure:
            st.error(failure)
        if start:
            state.pop("judge_failure", None)
            try:
                if "judge_dir" not in state:
                    session_dir = root / ".arc" / "judge" / uuid4().hex
                    session_dir.mkdir(parents=True)
                    task = dict(default_task)
                    data = session_dir / "demo_panel.csv"
                    shutil.copyfile(root / "data" / "demo_panel.csv", data)
                    task.update(data_path=str(data), data_hash=file_sha256(data))
                    state.judge_dir = session_dir
                    state.judge_task = task
                task = {**state.judge_task, "task_id": f"judge-{state.judge_dir.name}-r{len(results) + 1}"}
                state.judge_worker = launch(state.judge_dir, task, results[0] if results else None)
                state.pop("judge_replay", None)
                state.pop("judge_next", None)
            except OSError:
                state.judge_failure = "无法启动研究进程，请重试。"
            st.rerun()

    if replay:
        state.judge_replay = not state.get("judge_replay", False)
    if state.get("judge_replay"):
        st.info("离线回放 · 内置样例")
        _usage(example_summary, example_events)
        _result(example_summary, example_events)
