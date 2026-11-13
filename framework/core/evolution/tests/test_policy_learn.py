"""策略快照 / 学习器 / 闸门的纯函数测试。"""

from __future__ import annotations

from _util import Checker, run

from core.evolution import learn, policy
from core.evolution.spec import Axis, DomainSpec, Hypothesis, Signal


def fake_spec(learner: str = "elimination") -> DomainSpec:
    return DomainSpec(
        name="fake",
        title="测试领域",
        version="1",
        axes=[
            Axis(
                "time",
                "什么时候？",
                [Hypothesis("A", "早上"), Hypothesis("B", "晚上")],
                default="A",
                eliminate_refutes=2,
            )
        ],
        learn={"min_replay_samples": 3},
        learner=learner,
    )


def test_snapshot() -> int:
    c = Checker("快照：构建 / 差异 / 回退内容")
    spec = fake_spec()
    snap = policy.build_initial(spec)
    c.eq("默认假设生效", policy.View(snap).active("time"), "A")
    c.check("初始没有证据", policy.View(snap).hyp_state("time", "A")["refute"] == 0)

    s2 = policy.clone(snap)
    s2["axes"]["time"]["active"] = "B"
    d = policy.diff(snap, s2)
    c.eq("diff 能报出 active 变更", [(x["field"], x["from"], x["to"]) for x in d], [("active", "A", "B")])

    r = policy.restore(snap, restored_from="pv_x")
    c.eq("回退内容 hash 相等", policy.snapshot_hash(r), policy.snapshot_hash(snap))
    c.eq("回退标记 origin", r["meta"]["origin"], "rollback")
    return c.done()


def test_elimination() -> int:
    c = Checker("学习器：人明说一次即弃 / 推断要两次 / 有支持不弃")
    spec = fake_spec()
    base = policy.build_initial(spec)

    one_stated = learn.propose(base, [Signal("time", "r", refute=["A"], source="stated")], spec)
    v = policy.View(one_stated["snapshot"])
    c.check("人明说 1 条 → A 被淘汰", v.hyp_state("time", "A")["excluded"] is True)
    c.eq("淘汰后切到 B", v.active("time"), "B")

    one_observed = learn.propose(base, [Signal("time", "r", refute=["A"], source="observed")], spec)
    c.check("推断 1 次 → 还不淘汰", policy.View(one_observed["snapshot"]).hyp_state("time", "A")["excluded"] is False)

    two_observed = learn.propose(
        base,
        [Signal("time", "r", refute=["A"], day="mon"), Signal("time", "r", refute=["A"], day="tue")],
        spec,
    )
    c.check("推断 2 次 → 淘汰", policy.View(two_observed["snapshot"]).hyp_state("time", "A")["excluded"] is True)

    sup = learn.propose(base, [Signal("time", "ok", support=["A"])], spec)
    c.check("有支持证据 → 不淘汰", policy.View(sup["snapshot"]).hyp_state("time", "A")["excluded"] is False)

    bad = learn.propose(base, [Signal("nope", "x", refute=["A"])], spec)
    c.check("未知轴的信号被丢弃且不改策略", bad["ok"] is False and not bad["changes"])
    return c.done()


def test_all_excluded() -> int:
    c = Checker("学习器：全被否掉 → unknown（主动问人，不硬猜）")
    spec = DomainSpec(
        name="fake2",
        title="单假设领域",
        version="1",
        axes=[Axis("x", "?", [Hypothesis("only", "唯一方案")], default="only")],
    )
    base = policy.build_initial(spec)
    res = learn.propose(base, [Signal("x", "r", refute=["only"], source="stated")], spec)
    v = policy.View(res["snapshot"])
    c.check("唯一假设被否 → unknown", v.unknown("x") is True)
    c.eq("active 保留原值（供页面展示）", v.active("x"), "only")
    ok, reasons, _ = learn.gate(res["snapshot"], base, spec, signals=[Signal("x", "r", refute=["only"], source="stated")])
    c.check("这种状态仍允许落库（由运行时去问人）", ok, str(reasons))
    return c.done()


def test_gate_struct() -> int:
    c = Checker("闸门：删假设 / 改小证据 / 无证据改动 一律拒绝")
    spec = fake_spec()
    cur = policy.build_initial(spec)
    cand = policy.clone(cur)
    del cand["axes"]["time"]["hypotheses"]["B"]
    ok, reasons, _ = learn.gate(cand, cur, spec, signals=[])
    c.check("删掉假设 → 拒绝", not ok and any("删除" in r for r in reasons), str(reasons))

    cand2 = policy.clone(cur)
    cand2["axes"]["time"]["hypotheses"]["A"]["support"] = -1
    ok2, reasons2, _ = learn.gate(cand2, cur, spec, signals=[])
    c.check("证据被改小 → 拒绝", not ok2 and any("改小" in r for r in reasons2), str(reasons2))

    cand3 = policy.clone(cur)
    cand3["axes"]["time"]["active"] = "B"
    ok3, reasons3, _ = learn.gate(cand3, cur, spec, signals=[Signal("other", "x", refute=["A"])])
    c.check("没有指向该轴的证据 → 拒绝", not ok3 and any("没有指向" in r for r in reasons3), str(reasons3))

    cand4 = policy.clone(cur)
    cand4["axes"]["time"]["hypotheses"]["B"]["excluded"] = True
    ok4, reasons4, _ = learn.gate(cand4, cur, spec, signals=[Signal("time", "x", refute=["B"])])
    c.check("淘汰不满足阈值 → 拒绝", not ok4 and any("阈值" in r for r in reasons4), str(reasons4))
    return c.done()


def test_gate_replay() -> int:
    c = Checker("闸门：回放命中率下降 → 拒绝（拦住过度泛化）")
    spec = fake_spec()

    class Domain:
        spec_ = spec

        def predict(self, view, payload):
            return ("high" if view.active("time") == "A" else "low", "")

    d = Domain()
    cur = policy.build_initial(spec)
    history = [{"payload": {}, "done": 1} for _ in range(6)]

    cand = policy.clone(cur)
    cand["axes"]["time"]["active"] = "B"
    cand["axes"]["time"]["hypotheses"]["A"]["refute"] = 2
    cand["axes"]["time"]["hypotheses"]["A"]["excluded"] = True
    sig = [Signal("time", "r", refute=["A"], support=["B"], source="stated")]
    ok, reasons, info = learn.gate(cand, cur, spec, signals=sig, history=history, domain=d)
    c.check("回放 6/6 → 0/6 被拒", not ok and any("回放" in r for r in reasons), str(reasons))
    c.eq("回放数据可见", (info["replay"]["old"]["hits"], info["replay"]["new"]["hits"]), (6, 0))

    class DomainOK(Domain):
        def predict(self, view, payload):
            return ("high", "")

    ok2, _, _ = learn.gate(cand, cur, spec, signals=sig, history=history, domain=DomainOK())
    c.check("回放不掉分 → 放行", ok2, "")
    return c.done()


def test_ucb() -> int:
    c = Checker("学习器：UCB 会先探索没试过的假设")
    spec = fake_spec(learner="ucb")
    base = policy.build_initial(spec)
    base["axes"]["time"]["hypotheses"]["A"]["support"] = 5
    base["axes"]["time"]["hypotheses"]["A"]["refute"] = 0
    res = learn.propose(base, [Signal("time", "ok", support=["A"])], spec)
    c.eq("冷启动的 B 被选中", policy.View(res["snapshot"]).active("time"), "B")
    return c.done()


def main() -> int:
    codes = [run(f) for f in (test_snapshot, test_elimination, test_all_excluded, test_gate_struct, test_gate_replay, test_ucb)]
    return 0 if all(x == 0 for x in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
