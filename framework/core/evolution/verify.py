"""可验证的最小子集：程序判定，不由模型自评。

这套东西来自上一个比赛的教训：脚本打印 "All succeeded"、build 通过、实际功能是空的。
所以验收只看数据库里的事实：预测命中了几个、冲突几个、重复犯错几类、新题几道。

    from evolve.verify import check
    check(rt, run_id, [
        {"type": "prediction_min", "name": "预测命中率 ≥ 80%", "min": 0.8},
        {"type": "conflict_max",    "name": "冲突 ≤ 1",         "max": 1},
        {"type": "repeat_max",      "name": "重复犯错 0 类",     "max": 0},
    ], prev_run_id=r1)
"""

from __future__ import annotations

from typing import Any


def check(rt: Any, run_id: str, checks: list[dict], prev_run_id: str | None = None) -> list[dict]:
    m = rt.metrics(run_id, prev_run_id)
    results: list[dict] = []
    for c in checks:
        typ = c.get("type")
        name = c.get("name") or typ
        ok, detail = False, ""
        if typ == "prediction_min":
            rate = m["prediction"]["rate"]
            ok = rate >= float(c.get("min", 0.8))
            detail = f"实际 {m['prediction']['text']}（{rate:.0%}）"
        elif typ == "conflict_max":
            n = m["conflict"]["count"]
            ok = n <= int(c.get("max", 0))
            detail = f"实际 {n} 次"
        elif typ == "repeat_max":
            if "repeat" not in m:
                ok, detail = False, "缺少对照运行，无法计算重复犯错"
            else:
                n = m["repeat"]["repeat"]
                ok = n <= int(c.get("max", 0))
                detail = f"实际 {n} 类 {m['repeat']['still']}"
        elif typ == "new_item_min":
            if "new_item" not in m:
                ok, detail = False, "缺少对照运行"
            elif m["new_item"]["total"] == 0:
                ok, detail = True, "该领域无题目概念，跳过"
            else:
                rate = m["new_item"]["rate"]
                ok = rate >= float(c.get("min", 1.0))
                detail = f"实际 {m['new_item']['text']}（{rate:.0%}）"
        elif typ == "all_judged":
            miss = m["prediction"]["unjudged"]
            ok = miss == 0
            detail = f"未判定 {miss} 条"
        else:
            ok, detail = False, f"未知检查类型 {typ}"
        results.append({"name": name, "ok": ok, "detail": detail})
    return results
