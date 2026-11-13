"""端到端：两个领域各跑两次运行 + 回退，数字必须和 PRD 对得上。"""

from __future__ import annotations

from _util import Checker, run

from core.evolution import domains, policy
from core.evolution.runtime import Runtime
from core.evolution.store import Store


def simulate(rt: Runtime, domain, run_id: str) -> None:
    for t in rt.normalized_tasks(run_id):
        out = domain.simulate(t["payload"], t["hypothesis"], t["kind"])
        rt.report(run_id, t["key"], out.done, out.reason, out.note)


def feedback_payload(rt: Runtime, domain, run_id: str) -> dict:
    return {
        "stated": domain.stated_feedback(rt._rows(run_id)),
        "observed": domain.observations_from(rt.normalized_tasks(run_id), rt._rows(run_id)),
    }


def test_habitfit() -> int:
    c = Checker("HabitFit：1/7 → 7/7，冲突 3 → 1，重复犯错 2 类 → 0")
    store = Store(":memory:")
    domain = domains.get("habitfit")
    rt = Runtime(store, domain)
    rt.bootstrap("u")

    r1 = rt.start_run("u", "first")
    simulate(rt, domain, r1["run"]["id"])
    m1 = rt.finish(r1["run"]["id"])
    c.eq("第一次命中 1/7", m1["prediction"]["text"], "1/7")
    c.eq("第一次冲突 3", m1["conflict"]["count"], 3)
    c.eq("第一次全部预测 high", {t["prediction"] for t in rt.normalized_tasks(r1["run"]["id"])}, {"high"})

    fb = rt.feedback(r1["run"]["id"], feedback_payload(rt, domain, r1["run"]["id"]), submission_key="k1")
    c.eq("进化为 applied", fb["disposition"], "applied")
    c.eq("新版本 seq=1", fb["new_seq"], 1)

    r2 = rt.start_run("u", "second")
    c.eq("第二次锁定 V1", rt.store.version(r2["version"]["id"])["seq"], 1)
    simulate(rt, domain, r2["run"]["id"])
    m2 = rt.metrics(r2["run"]["id"], prev_run_id=r1["run"]["id"])
    c.eq("第二次命中 7/7", m2["prediction"]["text"], "7/7")
    c.eq("第二次冲突 1", m2["conflict"]["count"], 1)
    c.eq("重复犯错 0 类", m2["repeat"]["repeat"], 0)

    wed = [t for t in rt.normalized_tasks(r2["run"]["id"]) if t["key"] == "wed-ex"][0]
    c.eq("周三承认不知道（预测 low）", wed["prediction"], "low")
    c.check("周三仍然冲突（但算命中）", m2["conflict"]["keys"] == ["wed-ex"], str(m2["conflict"]["keys"]))

    rb = rt.rollback("u")
    c.check("回退内容等于 V0", rb["content_equals"] is True)
    c.eq("回退产生 V2", rb["new_seq"], 2)
    c.eq("版本链 3 条", len(rt.status("u")["versions"]), 3)

    r3 = rt.start_run("u", "third")
    c.eq("回退后新运行读 V2", rt.store.version(r3["version"]["id"])["seq"], 2)
    c.eq("V2 内容与 V0 一致", rt.store.version(r3["version"]["id"])["snapshot_hash"],
         rt.store.version(rt.status("u")["versions"][0]["id"])["snapshot_hash"])
    return c.done()


def test_zhijie() -> int:
    c = Checker("知阶：0/6 → 6/6，验证题 100% 是新题")
    store = Store(":memory:")
    domain = domains.get("zhijie")
    rt = Runtime(store, domain)
    rt.bootstrap("stu")

    r1 = rt.start_run("stu", "first")
    simulate(rt, domain, r1["run"]["id"])
    m1 = rt.finish(r1["run"]["id"])
    c.eq("第一次 0/6", m1["prediction"]["text"], "0/6")

    fb = rt.feedback(r1["run"]["id"], feedback_payload(rt, domain, r1["run"]["id"]))
    c.eq("三条轴同时学到（同一批证据）", len(fb["gate"]["axes_changed"]), 3)

    r2 = rt.start_run("stu", "second")
    simulate(rt, domain, r2["run"]["id"])
    m2 = rt.metrics(r2["run"]["id"], prev_run_id=r1["run"]["id"])
    c.eq("第二次 6/6", m2["prediction"]["text"], "6/6")
    c.eq("新题率 100%", m2["new_item"]["text"], "3/3")
    items = m2["new_item"]["items"]
    c.check("第二次的题不是第一次那批", all(i.endswith("-v2") for i in items), str(items))
    return c.done()


def test_two_feedback_paths() -> int:
    c = Checker("反馈语义：幂等 / 无变化 / 基线已变")
    store = Store(":memory:")
    domain = domains.get("habitfit")
    rt = Runtime(store, domain)
    rt.bootstrap("u")
    # 先开一条"旧运行"，它锁 V0；之后版本推进，再拿它提交反馈 → 应被拒
    r_old = rt.start_run("u", "first")

    r1 = rt.start_run("u", "first")
    simulate(rt, domain, r1["run"]["id"])
    rid = r1["run"]["id"]

    payload = feedback_payload(rt, domain, rid)
    a = rt.feedback(rid, payload, submission_key="same")
    n_after = len(rt.status("u")["versions"])
    b = rt.feedback(rid, payload, submission_key="same")
    c.check("同键重发 → 幂等", b["idempotent"] is True)
    c.eq("版本数不变", len(rt.status("u")["versions"]), n_after)
    c.eq("返回同一新版本", b["new_version_id"], a["new_version_id"])

    # 基线已变：另一条运行锁的是 V0，此时活跃已是 V1
    late = rt.feedback(
        r_old["run"]["id"],
        {"stated": [{"axis": "exercise_time", "refute": ["H1"], "reason": "x"}]},
        submission_key="late",
    )
    c.eq("旧运行反馈 → baseline_changed", late["disposition"], "baseline_changed")
    c.check("策略没有被旧反馈改写", len(rt.status("u")["versions"]) == n_after)

    r2 = rt.start_run("u", "second")
    empty = rt.feedback(r2["run"]["id"], {"text": "还好"}, submission_key="empty")
    c.eq("没有证据 → no_change", empty["disposition"], "no_change")
    return c.done()


def main() -> int:
    codes = [run(f) for f in (test_habitfit, test_zhijie, test_two_feedback_paths)]
    return 0 if all(x == 0 for x in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
