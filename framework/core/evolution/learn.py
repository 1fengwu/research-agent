"""学习器：把证据变成新策略，并在落库前过闸门。

两种算法，同一个入口 `propose()`：

- **elimination（默认）**：假设表 + 证据计数，`refute >= 阈值 且 support == 0` 才淘汰。
  适合假设可枚举的领域（时段、时长、呈现方式）。这就是 PRD v2.1 里那套机制，
  也是回答"换个失败原因还会不会学"的答案：换的是轴，不是代码。
- **ucb**：置信上界，适合假设不可穷举、只能试的领域（SelfTaught 方案里的用法）。
  分数 = 成功率 + c·sqrt(lnN/n)，冷启动阶段会主动探索。

闸门（`gate`）做三件事，任何一条不过就不许落库：

1. 结构：不许删假设、不许把分数改小、淘汰必须真的满足阈值；
2. 幅度：一次最多动 `max_axis_changes` 个轴（可解释、可回退）；
3. 回放：拿历史真实结果重算命中率，新快照**不得低于**旧快照。
   —— 这条拦住"过度泛化"：把有支持的假设也淘汰掉，回放会掉分。
"""

from __future__ import annotations

import math
from typing import Any

from . import metrics, policy
from .spec import DomainSpec, Signal


class LearnResult(dict):
    """可当 dict 用，也可属性访问；CLI/控制台直接打印。"""

    @property
    def accepted(self) -> bool:
        return bool(self.get("ok"))


def _weight(spec: DomainSpec, source: str) -> tuple[int, int]:
    """(反驳权重, 支持权重)。人明说的比系统推断的重。"""
    if source == "stated":
        return int(spec.learn["stated_refute"]), int(spec.learn["stated_support"])
    return int(spec.learn["observed_refute"]), int(spec.learn["observed_support"])


def validate_signals(spec: DomainSpec, signals: list[Signal]) -> tuple[list[Signal], list[str]]:
    """非法信号丢弃并记录原因（不许因为一条脏数据污染策略）。"""
    good: list[Signal] = []
    bad: list[str] = []
    for s in signals:
        try:
            axis = spec.axis(s.axis)
        except KeyError:
            bad.append(f"未知轴 {s.axis}")
            continue
        unknown = [h for h in (s.refute + s.support) if h not in axis.ids]
        if unknown:
            bad.append(f"{s.axis} 上未知假设 {unknown}")
            continue
        if not s.refute and not s.support:
            bad.append(f"{s.axis} 信号没有指向任何假设")
            continue
        good.append(s)
    return good, bad


def apply_evidence(snapshot: dict, signals: list[Signal], spec: DomainSpec) -> dict:
    s = policy.clone(snapshot)
    for sig in signals:
        rw, sw = _weight(spec, sig.source)
        for hid in sig.refute:
            st = s["axes"][sig.axis]["hypotheses"][hid]
            st["refute"] += rw
            if sig.day and sig.day not in st["days"]:
                st["days"].append(sig.day)
        for hid in sig.support:
            st = s["axes"][sig.axis]["hypotheses"][hid]
            st["support"] += sw
            if sig.day and sig.day not in st["days"]:
                st["days"].append(sig.day)
    return s


def evaluate_exclusions(snapshot: dict, spec: DomainSpec) -> dict:
    """按阈值淘汰假设，并在必要时切换 active（全淘汰 → unknown，等人说话）。"""
    s = policy.clone(snapshot)
    for axis in spec.axes:
        node = s["axes"][axis.id]
        for hid, st in node["hypotheses"].items():
            if st["excluded"]:
                continue
            if st["refute"] >= axis.eliminate_refutes and st["support"] == 0:
                st["excluded"] = True
                st["excluded_at"] = s.get("seq", 0)
        alive = [h for h in axis.priority if not node["hypotheses"][h]["excluded"]]
        if node["active"] in alive:
            node["unknown"] = False
            continue
        if alive:
            node["active"] = alive[0]
            node["unknown"] = False
        else:
            node["unknown"] = True
    return s


def propose_elimination(snapshot: dict, signals: list[Signal], spec: DomainSpec) -> LearnResult:
    s = apply_evidence(snapshot, signals, spec)
    s = evaluate_exclusions(s, spec)
    changes = policy.diff(snapshot, s)
    axes_changed = sorted({c["axis"] for c in changes if c["field"] in ("active", "unknown", "excluded")})
    s["meta"] = {
        "origin": "learn",
        "note": "、".join(sorted({x.reason for x in signals if x.reason}))[:160],
        "changed_axes": axes_changed,
    }
    return LearnResult(
        ok=True,
        snapshot=s,
        changes=changes,
        axes_changed=axes_changed,
        signal_count=len(signals),
        note="",
    )


def propose_ucb(snapshot: dict, signals: list[Signal], spec: DomainSpec, *, c: float = 1.2) -> LearnResult:
    """置信上界：把 support 当成功、refute 当失败，主动探索没试过的假设。"""
    s = apply_evidence(snapshot, signals, spec)
    for axis in spec.axes:
        node = s["axes"][axis.id]
        total = sum(st["support"] + st["refute"] for st in node["hypotheses"].values())
        best, best_score = node["active"], -1.0
        for hid, st in node["hypotheses"].items():
            if st["excluded"]:
                continue
            n = st["support"] + st["refute"]
            if n == 0:
                score = float("inf")  # 没试过 → 先试
            else:
                score = st["support"] / n + c * math.sqrt(math.log(total + 2) / n)
            if score > best_score:
                best, best_score = hid, score
        if best and not node["hypotheses"][best]["excluded"]:
            node["active"] = best
            node["unknown"] = False
    changes = policy.diff(snapshot, s)
    axes_changed = sorted({c_["axis"] for c_ in changes if c_["field"] in ("active", "unknown", "excluded")})
    s["meta"] = {"origin": "learn", "note": "ucb", "changed_axes": axes_changed}
    return LearnResult(
        ok=True, snapshot=s, changes=changes, axes_changed=axes_changed,
        signal_count=len(signals), note="",
    )


def propose(snapshot: dict, signals: list[Signal], spec: DomainSpec) -> LearnResult:
    good, bad = validate_signals(spec, signals)
    if not good:
        return LearnResult(
            ok=False,
            snapshot=snapshot,
            changes=[],
            axes_changed=[],
            signal_count=0,
            note="没有可用证据：" + "；".join(bad[:3]),
        )
    if spec.learner == "ucb":
        res = propose_ucb(snapshot, good, spec)
    else:
        res = propose_elimination(snapshot, good, spec)
    res["dropped"] = bad
    return res


def gate(
    candidate: dict,
    current: dict,
    spec: DomainSpec,
    signals: list[Signal] | None = None,
    history: list[dict] | None = None,
    domain: Any = None,
) -> tuple[bool, list[str], dict]:
    """落库前的最后一道闸。返回 (是否放行, 拒绝原因, 参考数据)。"""
    reasons: list[str] = []
    info: dict[str, Any] = {}
    evidence_axes = {s.axis for s in (signals or [])}

    # 1) 结构：只增不减
    for aid, cur in current["axes"].items():
        cand = candidate["axes"].get(aid)
        if cand is None:
            reasons.append(f"{aid} 被整轴删除")
            continue
        for hid, cst in cur["hypotheses"].items():
            nst = cand["hypotheses"].get(hid)
            if nst is None:
                reasons.append(f"{aid}/{hid} 假设被删除")
                continue
            if nst["support"] < cst["support"] or nst["refute"] < cst["refute"]:
                reasons.append(f"{aid}/{hid} 证据被改小（{cst['support']}/{cst['refute']} → {nst['support']}/{nst['refute']}）")
            if nst["excluded"] and not cst["excluded"]:
                axis_def = spec.axis(aid)
                if not (nst["refute"] >= axis_def.eliminate_refutes and nst["support"] == 0):
                    reasons.append(f"{aid}/{hid} 淘汰不满足阈值（refute={nst['refute']} support={nst['support']}）")
            if cst["excluded"] and not nst["excluded"]:
                reasons.append(f"{aid}/{hid} 已淘汰假设被复活")

    # 2) 每个改动都必须有对应证据；并有总量上限
    changes = policy.diff(current, candidate)
    axes_changed = sorted({c["axis"] for c in changes if c["field"] in ("active", "unknown")})
    info["axes_changed"] = axes_changed
    for aid in axes_changed:
        if signals is not None and aid not in evidence_axes:
            reasons.append(f"{aid} 被改动，但这批反馈里没有指向它的证据")
    limit = int(spec.learn["max_axis_changes"])
    if len(axes_changed) > limit:
        reasons.append(f"一次改动 {len(axes_changed)} 个轴（上限 {limit}），不可解释")

    # 3) 回放
    if domain is not None and history:
        min_n = int(spec.learn["min_replay_samples"])
        old_hit = metrics.replay_hit(domain, current, history)
        new_hit = metrics.replay_hit(domain, candidate, history)
        info["replay"] = {"old": old_hit, "new": new_hit, "samples": len([h for h in history if h.get("done") is not None])}
        if old_hit["total"] >= min_n and new_hit["rate"] + 1e-9 < old_hit["rate"]:
            reasons.append(
                f"回放命中率下降（{old_hit['hits']}/{old_hit['total']} → {new_hit['hits']}/{new_hit['total']}）"
            )

    return (not reasons), reasons, info
