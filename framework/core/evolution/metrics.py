"""指标：全部由程序从数据库算，不接受手填。

| 指标 | 为什么是它 |
|---|---|
| 预测命中率 | 预测在结果之前落库，**不可事后编造**；这是主指标 |
| 冲突数 | 策略与环境不合的次数；辅助指标，**由评委现场给时段表也算得出** |
| 重复犯错 | 第一次预测为 high 却失败、第二次同类再失败 → 真没学到 |
| 新题率 | 第二次验证用的题第一次没出现过 → 不是背题 |
| 回放命中率 | 用历史结果重算任意快照的表现 → 闸门用 |
"""

from __future__ import annotations

from typing import Any, Iterable

from . import policy


def _rows(tasks: Iterable[dict], outcomes: dict[str, Any]) -> list[dict]:
    out = []
    for t in tasks:
        o = outcomes.get(t["id"]) if isinstance(outcomes, dict) else outcomes
        if o is None:
            continue
        done = bool(o["done"]) if not isinstance(o, dict) else bool(o["done"])
        reason = o["reason"] if not isinstance(o, dict) else o["reason"]
        out.append({"task": t, "done": done, "reason": reason})
    return out


def prediction_stats(tasks: list, outcomes: dict[str, Any]) -> dict:
    rows = _rows(tasks, outcomes)
    hits, misses = 0, []
    for r in rows:
        pred = r["task"]["prediction"]
        hit = (pred == "high" and r["done"]) or (pred == "low" and not r["done"])
        if hit:
            hits += 1
        else:
            misses.append(
                {
                    "key": r["task"]["key"],
                    "prediction": pred,
                    "done": r["done"],
                    "reason": r["reason"],
                    "why": r["task"]["prediction_reason"],
                }
            )
    total = len(rows)
    return {
        "hits": hits,
        "total": total,
        "rate": (hits / total) if total else 0.0,
        "text": f"{hits}/{total}",
        "misses": misses,
        "unjudged": len(tasks) - total,
    }


def conflicts(tasks: list, outcomes: dict[str, Any], reasons: Iterable[str]) -> dict:
    rs = set(reasons)
    rows = _rows(tasks, outcomes)
    hit = [r for r in rows if r["reason"] in rs]
    return {"count": len(hit), "keys": [r["task"]["key"] for r in hit], "reasons": sorted(rs)}


def repeat_errors(prev: list[dict], cur: list[dict]) -> dict:
    """按「轴 + 失败原因」归并；只算**预测为 high 却失败**的（预测 low 的失败不算犯错）。"""

    def sig(rows: list[dict]) -> set[tuple[str, str]]:
        s = set()
        for r in rows:
            if r["done"]:
                continue
            if r["task"]["prediction"] != "high":
                continue
            s.add((r["task"]["axis"], r["reason"]))
        return s

    a, b = sig(prev), sig(cur)
    still = sorted(a & b)
    return {
        "first": len(a),
        "second": len(b),
        "still": [f"{k[0]}:{k[1]}" for k in still],
        "repeat": len(still),
        "text": f"{len(a)} 类 → {len(b)} 类（仍在犯 {len(still)} 类）",
    }


def new_item_rate(tasks: list[dict], exposed_before: set[str]) -> dict:
    """验证类任务里，有多少题是这个人此前没见过的。"""
    items = []
    for t in tasks:
        if t["kind"] != "verify":
            continue
        item = t["payload"].get("item")
        if not item:
            continue
        items.append(item)
    if not items:
        return {"total": 0, "fresh": 0, "rate": 0.0, "text": "n/a（该领域无题目概念）"}
    fresh = [i for i in items if i not in exposed_before]
    return {
        "total": len(items),
        "fresh": len(fresh),
        "rate": len(fresh) / len(items),
        "text": f"{len(fresh)}/{len(items)}",
        "items": items,
    }


def replay_hit(domain: Any, snapshot: dict, history: list[dict]) -> dict:
    """用历史真实结果，重算某个快照会拿到多少命中。"""
    view = policy.View(snapshot)
    hits = total = 0
    for h in history:
        if h.get("done") is None:
            continue
        pred = "high"
        if hasattr(domain, "predict"):
            try:
                pred, _reason = domain.predict(view, h["payload"])
            except Exception:
                pred = "high"
        done = bool(h["done"])
        hit = (pred == "high" and done) or (pred == "low" and not done)
        total += 1
        hits += 1 if hit else 0
    return {"hits": hits, "total": total, "rate": (hits / total) if total else 0.0}


def summarize(snapshot: dict) -> str:
    v = policy.View(snapshot)
    parts = []
    for aid, s in v.summary().items():
        state = "未知" if s["unknown"] else s["active"]
        excl = f"，已否 {','.join(s['excluded'])}" if s["excluded"] else ""
        parts.append(f"{aid}={state}{excl}")
    return "；".join(parts)
