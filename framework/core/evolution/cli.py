"""命令行：一条命令演完「第一次 → 学到 → 第二次 → 回退」。

    python -m core.evolution.cli demo --domain habitfit --db data/habitfit.db --fresh
    python -m evolve.cli status --domain habitfit --db data/habitfit.db
    python -m evolve.cli audit  --db data/habitfit.db
    python -m evolve.cli serve  --db data/habitfit.db --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import domains, policy
from .runtime import RunConfig, Runtime
from .store import Store
from .verify import check as run_checks

W = 26


def _p(*parts: str) -> None:
    print(*parts)


def _rule(ch: str = "=", n: int = 78) -> None:
    _p(ch * n)


def _table(run: dict, m: dict) -> None:
    _p(f"{'任务'.ljust(W)} {'预测':<6} {'结果':<8} 原因")
    _rule("-")
    for t in run["tasks"]:
        o = m["_outcomes"].get(t["id"])
        done = "完成" if o and o["done"] else ("未完成" if o else "未上报")
        reason = o["reason"] if o else "-"
        _p(f"{t['title'][:W].ljust(W)} {t['prediction']:<6} {done:<8} {reason}")


def _simulate(rt: Runtime, domain, run: dict) -> None:
    rid = run["run"]["id"]
    for t in rt.normalized_tasks(rid):
        out = domain.simulate(t["payload"], t["hypothesis"], t["kind"])
        rt.report(rid, t["key"], out.done, out.reason, out.note)


def _print_metrics(label: str, m: dict) -> None:
    parts = [f"预测命中率 {m['prediction']['text']}", f"冲突 {m['conflict']['count']}"]
    if "repeat" in m:
        parts.append(f"重复犯错 {m['repeat']['repeat']} 类")
    if "new_item" in m:
        parts.append(f"新题率 {m['new_item']['text']}")
    _p(f"{label}：" + " ｜ ".join(parts))


def _describe_changes(changes: list[dict], domain) -> list[str]:
    lines = []
    for c in changes:
        if c["field"] == "active":
            lines.append(f"{c['axis']}: {c['from']} → {c['to']}（换假设）")
        elif c["field"] == "excluded":
            lines.append(f"{c['axis']}/{c['hypothesis']}: 淘汰={c['to']}")
        elif c["field"] in ("support", "refute"):
            lines.append(f"{c['axis']}/{c['hypothesis']}: {c['field']} {c['from']} → {c['to']}")
        elif c["field"] == "unknown":
            lines.append(f"{c['axis']}: unknown={c['to']}（需要人给方案）")
    return lines


def _reset_db(db_path: Path) -> None:
    """清库：优先删文件；删不掉（沙箱 / 权限 / 回收站不可用）就清表，效果等价。"""
    if not db_path.exists():
        return
    try:
        db_path.unlink()
        for suffix in ("-wal", "-shm"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        return
    except OSError:
        pass

    from . import db as _db

    conn = _db.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=OFF")
    names = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        if not r[0].startswith("sqlite_")
    ]
    for name in names:
        conn.execute(f'DROP TABLE IF EXISTS "{name}"')
    conn.commit()
    _db.init_schema(conn)
    conn.commit()
    conn.close()


def cmd_demo(args) -> int:
    db_path = Path(args.db)
    if args.fresh:
        _reset_db(db_path)

    store = Store(str(db_path))
    domain = domains.get(args.domain)
    rt = Runtime(store, domain, RunConfig(model=args.model))
    subject = args.subject

    _rule()
    _p(f"领域 {domain.spec.title}（{domain.spec.name} v{domain.spec.version}）  主体 {subject}")
    _p(f"库 {db_path}  模型配置 {args.model}")
    _rule()

    boot = rt.bootstrap(subject)
    _p(f"初始策略 V{boot['version']['seq']}：{policy.snapshot_hash(boot['snapshot'])}")
    _p("")

    # ---------------- 第一次运行
    r1 = rt.start_run(subject, "first")
    rid1 = r1["run"]["id"]
    _p(f"[1] 第一次运行 R1  锁定 V{r1['version']['seq']}  {len(r1['tasks'])} 条任务")
    _simulate(rt, domain, r1)
    m1 = rt.finish(rid1)
    _print_metrics("    统计", m1)
    _p("")

    # ---------------- 反馈 → 进化
    tasks1 = rt.normalized_tasks(rid1)
    rows1 = rt._rows(rid1)
    payload = {
        "stated": domain.stated_feedback(rows1),
        "observed": domain.observations_from(tasks1, rows1),
    }
    fb = rt.feedback(rid1, payload)
    _p(f"[2] 反馈提交  处置={fb['disposition']}")
    for line in _describe_changes(fb.get("changes", []), domain):
        _p(f"    - {line}")
    if fb.get("reasons"):
        for r in fb["reasons"]:
            _p(f"    ! {r}")
    if fb.get("gate", {}).get("replay"):
        g = fb["gate"]["replay"]
        _p(f"    闸门回放：{g['old']['hits']}/{g['old']['total']} → {g['new']['hits']}/{g['new']['total']}（{g['samples']} 条历史）")
    _p(f"    新版本 V{fb.get('new_seq')}  {policy.snapshot_hash(fb['snapshot']) if fb.get('snapshot') else '-'}")
    _p("")

    # ---------------- 第二次运行
    r2 = rt.start_run(subject, "second")
    rid2 = r2["run"]["id"]
    same = r2["version"]["id"] == rt.store.active_version(r1["run"]["subject_id"], domain.spec.name)["id"]
    _p(f"[3] 第二次运行 R2  锁定 V{r2['version']['seq']}（自动读到新快照：{same}）  {len(r2['tasks'])} 条任务")
    _simulate(rt, domain, r2)
    m2 = rt.finish(rid2)
    m2 = rt.metrics(rid2, prev_run_id=rid1)
    _print_metrics("    统计", m2)
    if m2.get("prediction", {}).get("misses"):
        for miss in m2["prediction"]["misses"]:
            _p(f"    未命中 {miss['key']}：预测 {miss['prediction']} 实际{'完成' if miss['done'] else '未完成'}")
    if m2.get("new_item", {}).get("total"):
        _p(f"    验证用新题：{m2['new_item'].get('items')}")
    _p("")

    # ---------------- 回退
    rb = rt.rollback(subject)
    _p(f"[4] 回退  新版本 V{rb['new_seq']} 内容等同 V{rb['restored_from_seq']}：{rb['content_equals']}")
    _p("")

    # ---------------- 对照表
    _rule()
    _p("评委要看的三行")
    _rule("-")
    _p(f"指标两次各是多少  预测命中率 {m1['prediction']['text']} → {m2['prediction']['text']}"
       f"；冲突 {m1['conflict']['count']} → {m2['conflict']['count']}")
    if "repeat" in m2:
        _p(f"学到了什么        {len(fb.get('changes', []))} 条证据改动，重复犯错 {m2['repeat']['text']}")
    _p(f"怎么退回           python -m core.evolution.cli rollback --db {db_path} --domain {domain.spec.name}")
    _p("")
    checks = [
        {"type": "prediction_min", "name": "第二次预测命中率 ≥ 80%", "min": 0.8},
        {"type": "conflict_max", "name": "第二次冲突数不高于第一次", "max": m1["conflict"]["count"]},
        {"type": "repeat_max", "name": "重复犯错 0 类", "max": 0},
        {"type": "all_judged", "name": "所有任务都有结果"},
    ]
    _rule("-")
    _p("程序化验收")
    for r in run_checks(rt, rid2, checks, prev_run_id=rid1):
        _p(f"  [{'PASS' if r['ok'] else 'FAIL'}] {r['name']} — {r['detail']}")
    _rule()
    store.close()
    return 0


def cmd_status(args) -> int:
    store = Store(args.db)
    rt = Runtime(store, domains.get(args.domain))
    st = rt.status(args.subject)
    if not st.get("active"):
        _p("还没有策略，先跑 demo")
        return 1
    a = st["active"]
    _p(f"主体 {args.subject}  活跃版本 V{a['seq']}  {a['snapshot_hash']}  事件 {st['events']} 条")
    _p("版本链：")
    for v in st["versions"]:
        flag = "*" if v["id"] == a["id"] else " "
        _p(f" {flag} V{v['seq']}  {v['snapshot_hash']}  {v['reason'][:40]}  {'(回退自 ' + str(v['restored_from_id'])[:12] + ')' if v['restored_from_id'] else ''}")
    _p("当前策略：")
    for axis, s in st["summary"].items():
        state = "未知" if s["unknown"] else s["active"]
        _p(f"  {axis:<18} {state:<14} 证据 {s['evidence']}")
    _p("运行：")
    for r in st["runs"]:
        _p(f"  {r['kind']:<7} V{r['policy_version_id'][-6:]}  {r['status']:<9} {r['created_at']}")
    store.close()
    return 0


def cmd_audit(args) -> int:
    store = Store(args.db)
    rt = Runtime(store, domains.get(args.domain))
    for e in rt.audit(args.subject):
        extra = ""
        if e["prev_version_id"] or e["new_version_id"]:
            extra = f" {(e['prev_version_id'] or '-')[-6:]} → {(e['new_version_id'] or '-')[-6:]}"
        payload = e["payload_json"]
        if args.verbose and payload and payload != "{}":
            extra += f"  {payload[:120]}"
        _p(f"{e['at']}  {e['type']:<16}{extra}")
    store.close()
    return 0


def cmd_rollback(args) -> int:
    store = Store(args.db)
    rt = Runtime(store, domains.get(args.domain))
    rb = rt.rollback(args.subject, target=args.to)
    _p(f"已回退：新版本 V{rb['new_seq']}，内容等同 V{rb['restored_from_seq']}（{rb['content_equals']}）")
    store.close()
    return 0


def cmd_verify(args) -> int:
    store = Store(args.db)
    rt = Runtime(store, domains.get(args.domain))
    checks = [
        {"type": "prediction_min", "name": f"预测命中率 ≥ {args.min_rate:.0%}", "min": args.min_rate},
        {"type": "repeat_max", "name": "重复犯错 0 类", "max": 0},
        {"type": "new_item_min", "name": "验证题全是新题", "min": 1.0},
        {"type": "all_judged", "name": "所有任务都有结果"},
    ]
    results = run_checks(rt, args.run, checks, prev_run_id=args.prev)
    bad = 0
    for r in results:
        _p(f"[{'PASS' if r['ok'] else 'FAIL'}] {r['name']} — {r['detail']}")
        bad += 0 if r["ok"] else 1
    store.close()
    return 0 if bad == 0 else 2


def cmd_export(args) -> int:
    """把事件镜像进 ARC 的 .arc/runner-events.jsonl（同一套日志，不必学两套）。"""
    from . import arc_bridge

    store = Store(args.db)
    sub = store.subject(args.subject)
    out = args.out or arc_bridge.paths()["runner_events"]
    n = arc_bridge.mirror_events(store, sub["id"], out)
    _p(f"已写入 {n} 条事件 → {out}")
    store.close()
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.db, args.domain, args.subject, args.host, args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evolve", description="自进化 agent 基座")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, db_required: bool = False):
        sp.add_argument("--db", required=db_required, default="core/evolution/_data/evolve.db")
        sp.add_argument("--domain", default="habitfit", choices=domains.names())
        sp.add_argument("--subject", default="demo-user")

    d = sub.add_parser("demo", help="一条命令演完两次运行 + 回退")
    common(d)
    d.add_argument("--fresh", action="store_true", help="清空库重来")
    d.add_argument("--model", default="offline-demo")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("status", help="看版本链、当前策略、运行列表")
    common(s)
    s.set_defaults(func=cmd_status)

    a = sub.add_parser("audit", help="看事件流")
    common(a)
    a.add_argument("--verbose", action="store_true")
    a.set_defaults(func=cmd_audit)

    r = sub.add_parser("rollback", help="回退到上一版（或指定版本）")
    common(r)
    r.add_argument("--to", default=None, help="目标版本 id")
    r.set_defaults(func=cmd_rollback)

    v = sub.add_parser("verify", help="对某次运行做程序化验收")
    common(v)
    v.add_argument("--run", required=True)
    v.add_argument("--prev", default=None)
    v.add_argument("--min-rate", type=float, default=0.8)
    v.set_defaults(func=cmd_verify)

    ex = sub.add_parser("export", help="把事件镜像到 ARC 的 runner-events.jsonl")
    common(ex)
    ex.add_argument("--out", default=None)
    ex.set_defaults(func=cmd_export)

    sv = sub.add_parser("serve", help="起 HTTP 服务（给前端联调）")
    common(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
