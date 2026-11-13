"""存储层：事务、幂等、基线冲突、回退语义。"""

from __future__ import annotations

from _util import Checker, run

from core.evolution import policy
from core.evolution.store import BaselineChanged, Store
from core.evolution.spec import Axis, DomainSpec, Hypothesis


def spec() -> DomainSpec:
    return DomainSpec(
        name="t",
        title="t",
        version="1",
        axes=[Axis("x", "?", [Hypothesis("A", "a"), Hypothesis("B", "b")], default="A")],
    )


def test_init_idempotent() -> int:
    c = Checker("初始化：重复调用不产生第二个 V0")
    s = Store(":memory:")
    sub = s.subject("u1")
    v1 = s.init_policy(sub["id"], policy.build_initial(spec()))
    v2 = s.init_policy(sub["id"], policy.build_initial(spec()))
    c.eq("版本 id 相同", v1["id"], v2["id"])
    c.eq("只有一个版本", len(s.versions(sub["id"], "t")), 1)
    c.eq("seq 从 0 开始", v1["seq"], 0)
    return c.done()


def test_baseline_conflict() -> int:
    c = Checker("基线冲突：旧运行不能覆盖新经验")
    s = Store(":memory:")
    sub = s.subject("u1")
    v0 = s.init_policy(sub["id"], policy.build_initial(spec()))
    snap = s.snapshot(v0["id"])
    snap["axes"]["x"]["active"] = "B"
    try:
        s.commit_evolution(
            subject_id=sub["id"], domain="t", snapshot=snap, reason="r",
            expected_active="pv_不存在的版本",
        )
        raised = False
    except BaselineChanged:
        raised = True
    c.check("基线不符 → BaselineChanged", raised)
    c.eq("没有多出版本", len(s.versions(sub["id"], "t")), 1)
    c.eq("指针没动", s.active_version(sub["id"], "t")["id"], v0["id"])
    return c.done()


def test_atomic_rollback() -> int:
    c = Checker("事务：写指针时崩了 → 版本表也要回滚")
    s = Store(":memory:")
    sub = s.subject("u1")
    v0 = s.init_policy(sub["id"], policy.build_initial(spec()))
    snap = s.snapshot(v0["id"])
    snap["axes"]["x"]["active"] = "B"

    original = s._point

    def boom(*a, **k):
        raise RuntimeError("模拟进程在写指针前被杀")

    s._point = boom
    try:
        try:
            s.commit_evolution(subject_id=sub["id"], domain="t", snapshot=snap, reason="r", expected_active=v0["id"])
        except RuntimeError:
            pass
    finally:
        s._point = original

    c.eq("版本数没有增加", len(s.versions(sub["id"], "t")), 1)
    c.eq("活跃版本仍是 V0", s.active_version(sub["id"], "t")["id"], v0["id"])
    return c.done()


def test_outcome_idempotent() -> int:
    c = Checker("结果上报：同一任务重复上报返回第一次结果")
    s = Store(":memory:")
    sub = s.subject("u1")
    v0 = s.init_policy(sub["id"], policy.build_initial(spec()))
    r = s.create_run(subject_id=sub["id"], domain="t", version_id=v0["id"], kind="first")

    from core.evolution.spec import Task

    s.add_tasks(r["id"], [Task("k1", "任务", "x", "A", "high", "理由")])
    task = s.task_by_key(r["id"], "k1")
    a = s.set_outcome(task["id"], False, "time_conflict")
    b = s.set_outcome(task["id"], True, "ok")
    c.check("第二次不改写", b["done"] == 0 and b["reason"] == "time_conflict")
    c.eq("只留一条结果", len(s.outcomes(r["id"])), 1)
    return c.done()


def test_rollback_semantics() -> int:
    c = Checker("回退：新版本内容等同旧版；已开的运行仍锁旧版本")
    s = Store(":memory:")
    sub = s.subject("u1")
    v0 = s.init_policy(sub["id"], policy.build_initial(spec()))
    v0snap = s.snapshot(v0["id"])

    r = s.create_run(subject_id=sub["id"], domain="t", version_id=v0["id"], kind="first")

    snap1 = policy.clone(v0snap)
    snap1["axes"]["x"]["active"] = "B"
    v1 = s.commit_evolution(subject_id=sub["id"], domain="t", snapshot=snap1, reason="learn", expected_active=v0["id"])

    rb_snap = policy.restore(s.snapshot(v0["id"]), restored_from=v0["id"])
    v2 = s.commit_evolution(
        subject_id=sub["id"], domain="t", snapshot=rb_snap, reason="rollback",
        expected_active=v1, restored_from_id=v0["id"], event_type="rollback",
    )
    c.eq("回退后是 V2", s.version(v2)["seq"], 2)
    c.eq("V2 内容 hash 等于 V0", s.version(v2)["snapshot_hash"], s.version(v0["id"])["snapshot_hash"])
    c.eq("V1 仍然可查", s.version(v1)["seq"], 1)
    c.eq("已有运行仍锁 V0", s.run(r["id"])["policy_version_id"], v0["id"])
    c.eq("链式 parent 正确", s.version(v2)["parent_id"], v1)
    events = [e["type"] for e in s.events(sub["id"])]
    c.check("有 rollback 事件", "rollback" in events, str(events))
    return c.done()


def test_missing_version() -> int:
    c = Checker("错误处理：不存在的版本要炸得清楚")
    s = Store(":memory:")
    try:
        s.snapshot("pv_nope")
        ok = False
    except KeyError:
        ok = True
    c.check("KeyError", ok)
    return c.done()


def main() -> int:
    codes = [run(f) for f in (test_init_idempotent, test_baseline_conflict, test_atomic_rollback,
                              test_outcome_idempotent, test_rollback_semantics, test_missing_version)]
    return 0 if all(x == 0 for x in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
