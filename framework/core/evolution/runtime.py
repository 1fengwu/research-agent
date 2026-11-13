"""运行时：一次运行的生命周期，全部状态落在 SQLite。

    start_run   锁定当前活跃版本（之后版本推进不影响这条运行）
      ↓
    plan        生成任务 + 每条任务的**事前预测**（写库）
      ↓
    report      上报真实结果（幂等）
      ↓
    feedback    人的反馈 → 证据 → 候选快照 → 闸门 → 新版本（幂等、同事务）
      ↓
    rollback    复制旧快照生成新版本，已有运行不受影响

三个不变量，测试里逐条断言：

1. **运行锁定版本**：R2 创建时锁 V1，之后回退到 V2，R2 仍按 V1 解释；
2. **反馈幂等**：同一 submission_key 重发，版本数不变、返回同一结果；
3. **基线不符拒绝**：拿旧运行去改新策略，返回 `baseline_changed`，策略不动。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from . import learn, metrics, policy
from .spec import DomainSpec, check_domain
from .store import BaselineChanged, Store


@dataclass
class RunConfig:
    """现场必须固定、并写进每次运行的三样东西（答辩证据）。"""

    model: str = "offline-demo"
    prompt_version: str = "p1"
    learn_enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def hashes(self, spec: DomainSpec) -> dict[str, str]:
        def h(*parts: str) -> str:
            return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]

        return {
            "build_hash": h("evolve-harness", "1.0", spec.name, spec.version),
            "prompt_hash": h(self.prompt_version, spec.name),
            "model_config_hash": h(self.model, str(sorted(self.extra.items()))),
        }


class Runtime:
    def __init__(self, store: Store, domain: Any, config: RunConfig | None = None) -> None:
        check_domain(domain)
        self.store = store
        self.domain = domain
        self.spec: DomainSpec = domain.spec
        self.config = config or RunConfig()

    # ------------------------------------------------------------ 初始化
    def bootstrap(self, subject_name: str) -> dict:
        sub = self.store.subject(subject_name)
        v = self.store.init_policy(sub["id"], policy.build_initial(self.spec))
        return {"subject": sub, "version": v, "snapshot": self.store.snapshot(v["id"])}

    # ------------------------------------------------------------ 运行
    def start_run(self, subject_name: str, kind: str = "first") -> dict:
        sub = self.store.subject(subject_name)
        ver = self.store.init_policy(sub["id"], policy.build_initial(self.spec))
        run = self.store.create_run(
            subject_id=sub["id"],
            domain=self.spec.name,
            version_id=ver["id"],
            kind=kind,
            hashes=self.config.hashes(self.spec),
        )
        view = policy.View(self.store.snapshot(ver["id"]))
        tasks = self.domain.plan(view, kind)
        self.store.add_tasks(run["id"], tasks)
        # 题目曝光在发给用户之前登记，才能算"新题率"
        for t in tasks:
            item = t.payload.get("item")
            if item:
                self.store.expose(sub["id"], item, run["id"])
        return {
            "run": run,
            "version": ver,
            "view": view,
            "tasks": self.store.tasks(run["id"]),
            "asks": [t.axis for t in tasks if view.unknown(t.axis)],
        }

    def report(self, run_id: str, task_key: str, done: bool, reason: str = "ok", note: str = "") -> dict:
        task = self.store.task_by_key(run_id, task_key)
        if task is None:
            raise KeyError(f"运行 {run_id} 里没有任务 {task_key}")
        row = self.store.set_outcome(task["id"], done, reason, note)
        return dict(row)

    def judge(self, run_id: str, task_key: str, attempt: dict) -> dict:
        """真实业务用领域判分；返回 {'done','reason'}。"""
        task = self.store.task_by_key(run_id, task_key)
        if task is None:
            raise KeyError(f"运行 {run_id} 里没有任务 {task_key}")
        payload = {"task": dict(task), "attempt": attempt}
        if hasattr(self.domain, "judge"):
            out = self.domain.judge(payload, attempt)
            return self.report(run_id, task_key, out.done, out.reason, out.note)
        return self.report(run_id, task_key, bool(attempt.get("done")), attempt.get("reason", "ok"))

    def finish(self, run_id: str) -> dict:
        self.store.finish_run(run_id)
        return self.metrics(run_id)

    # ------------------------------------------------------------ 反馈 / 进化
    def feedback(self, run_id: str, payload: dict, submission_key: str | None = None) -> dict:
        run = self.store.run(run_id)
        if run is None:
            raise KeyError(f"运行不存在 {run_id}")
        key = submission_key or self._auto_key(run_id, payload)

        dup = self.store.feedback_by_key(run_id, key)
        if dup:
            return {
                "disposition": dup["disposition"],
                "new_version_id": dup["new_version_id"],
                "idempotent": True,
                "changes": [],
                "reasons": ["同一 submission_key 已处理，返回原结果"],
            }

        # 基线核对放在最前：拿过期运行提交反馈，不该先跑学习再被闸门拒
        active_now = self.store.active_version(run["subject_id"], self.spec.name)
        if active_now and active_now["id"] != run["policy_version_id"]:
            msg = f"基线已变化：运行锁定 V{self.store.version(run['policy_version_id'])['seq']}，当前活跃 V{active_now['seq']}"
            self.store.add_feedback(
                run_id=run_id, subject_id=run["subject_id"], submission_key=key,
                raw=payload, normalized={"signals": [], "note": msg},
                disposition="baseline_changed", base_version_id=run["policy_version_id"], new_version_id=None,
            )
            return {"disposition": "baseline_changed", "new_version_id": None, "idempotent": False,
                    "changes": [], "reasons": [msg]}

        locked = self.store.snapshot(run["policy_version_id"])
        view = policy.View(locked)
        signals = self.domain.interpret(payload, view)
        res = learn.propose(locked, signals, self.spec)

        if not res["ok"]:
            self.store.add_feedback(
                run_id=run_id, subject_id=run["subject_id"], submission_key=key,
                raw=payload, normalized={"signals": [], "note": res["note"]},
                disposition="no_change", base_version_id=run["policy_version_id"], new_version_id=None,
            )
            return {"disposition": "no_change", "new_version_id": None, "idempotent": False,
                    "changes": [], "reasons": [res["note"]]}

        active = self.store.active_version(run["subject_id"], self.spec.name)
        current = self.store.snapshot(active["id"])
        history = self.store.history(run["subject_id"], self.spec.name)
        ok, reasons, info = learn.gate(
            res["snapshot"], current, self.spec, signals=signals, history=history, domain=self.domain
        )

        normalized = {
            "signals": [
                {"axis": s.axis, "reason": s.reason, "refute": s.refute, "support": s.support,
                 "source": s.source, "day": s.day}
                for s in signals
            ],
            "dropped": res.get("dropped", []),
            "changes": res["changes"],
            "gate": info,
        }

        if not ok:
            self.store.add_feedback(
                run_id=run_id, subject_id=run["subject_id"], submission_key=key,
                raw=payload, normalized=normalized, disposition="gate_rejected",
                base_version_id=run["policy_version_id"], new_version_id=None,
            )
            return {"disposition": "gate_rejected", "new_version_id": None, "idempotent": False,
                    "changes": [], "reasons": reasons, "gate": info}

        if not res["changes"]:
            self.store.add_feedback(
                run_id=run_id, subject_id=run["subject_id"], submission_key=key,
                raw=payload, normalized=normalized, disposition="no_change",
                base_version_id=run["policy_version_id"], new_version_id=None,
            )
            return {"disposition": "no_change", "new_version_id": None, "idempotent": False,
                    "changes": [], "reasons": ["证据没有改变任何假设"], "gate": info}

        try:
            vid = self.store.commit_evolution(
                subject_id=run["subject_id"],
                domain=self.spec.name,
                snapshot=res["snapshot"],
                reason="、".join(sorted({s.reason for s in signals}))[:120] or "feedback",
                expected_active=run["policy_version_id"],
                run_id=run_id,
                payload={"changes": res["changes"], "gate": info},
            )
        except BaselineChanged as exc:
            self.store.add_feedback(
                run_id=run_id, subject_id=run["subject_id"], submission_key=key,
                raw=payload, normalized=normalized, disposition="baseline_changed",
                base_version_id=run["policy_version_id"], new_version_id=None,
            )
            return {"disposition": "baseline_changed", "new_version_id": None, "idempotent": False,
                    "changes": [], "reasons": [str(exc)], "gate": info}

        self.store.add_feedback(
            run_id=run_id, subject_id=run["subject_id"], submission_key=key,
            raw=payload, normalized=normalized, disposition="applied",
            base_version_id=run["policy_version_id"], new_version_id=vid,
        )
        new_ver = self.store.version(vid)
        return {
            "disposition": "applied",
            "new_version_id": vid,
            "new_seq": new_ver["seq"],
            "idempotent": False,
            "changes": res["changes"],
            "reasons": [],
            "gate": info,
            "snapshot": res["snapshot"],
        }

    # ------------------------------------------------------------ 回退
    def rollback(self, subject_name: str, expected_active: str | None = None, target: str | None = None) -> dict:
        """回退 = 复制目标快照生成新版本。已有运行继续用自己锁定的版本。"""
        sub = self.store.subject(subject_name)
        active = self.store.active_version(sub["id"], self.spec.name)
        if active is None:
            raise KeyError("还没有策略，无法回退")
        if expected_active and active["id"] != expected_active:
            raise BaselineChanged(f"活跃版本 {active['id']} ≠ 期望 {expected_active}")

        if target is None:
            if active["parent_id"] is None:
                raise ValueError("已经是初始版本，没有可回退的上一版")
            target = active["parent_id"]
        tgt = self.store.version(target)
        if tgt is None:
            raise KeyError(f"目标版本不存在 {target}")

        snap = policy.restore(self.store.snapshot(target), restored_from=target)
        vid = self.store.commit_evolution(
            subject_id=sub["id"],
            domain=self.spec.name,
            snapshot=snap,
            reason=f"rollback from {active['seq']} to {tgt['seq']}",
            expected_active=active["id"],
            restored_from_id=target,
            event_type="rollback",
            payload={"from": active["seq"], "to": tgt["seq"], "restored_from": target},
        )
        new_ver = self.store.version(vid)
        return {
            "new_version_id": vid,
            "new_seq": new_ver["seq"],
            "restored_from_seq": tgt["seq"],
            "content_equals": policy.snapshot_hash(snap) == policy.snapshot_hash(self.store.snapshot(target)),
        }

    # ------------------------------------------------------------ 查询
    def normalized_tasks(self, run_id: str) -> list[dict]:
        """把库里的行转成领域包/指标能直接用的 dict（payload 已反序列化）。"""
        import json

        out = []
        for t in self.store.tasks(run_id):
            d = dict(t)
            d["payload"] = json.loads(d.pop("payload_json") or "{}")
            out.append(d)
        return out

    def metrics(self, run_id: str, prev_run_id: str | None = None) -> dict:
        run = self.store.run(run_id)
        tasks = self.store.tasks(run_id)
        outs = self.store.outcomes(run_id)
        stats = metrics.prediction_stats(tasks, outs)
        conflict = metrics.conflicts(tasks, outs, self.spec.conflict_reasons)
        result = {"run": dict(run), "prediction": stats, "conflict": conflict}

        if prev_run_id:
            prev_rows = self._rows(prev_run_id)
            cur_rows = self._rows(run_id)
            result["repeat"] = metrics.repeat_errors(prev_rows, cur_rows)
            prev_exposed = {r["task"]["payload"].get("item") for r in prev_rows}
            prev_exposed.discard(None)
            result["new_item"] = metrics.new_item_rate(
                [r["task"] for r in cur_rows], prev_exposed
            )
        return result

    def _rows(self, run_id: str) -> list[dict]:
        outs = self.store.outcomes(run_id)
        rows = []
        for t in self.normalized_tasks(run_id):
            o = outs.get(t["id"])
            if not o:
                continue
            rows.append({"task": t, "done": bool(o["done"]), "reason": o["reason"]})
        return rows

    def status(self, subject_name: str) -> dict:
        sub = self.store.subject(subject_name)
        active = self.store.active_version(sub["id"], self.spec.name)
        if active is None:
            return {"subject": dict(sub), "active": None, "versions": []}
        snap = self.store.snapshot(active["id"])
        runs = self.store._q("SELECT * FROM runs WHERE subject_id=? ORDER BY created_at", (sub["id"],))
        return {
            "subject": dict(sub),
            "active": dict(active),
            "snapshot": snap,
            "summary": policy.View(snap).summary(),
            "versions": [dict(v) for v in self.store.versions(sub["id"], self.spec.name)],
            "runs": [dict(r) for r in runs],
            "events": len(self.store.events(sub["id"])),
        }

    def audit(self, subject_name: str) -> list[dict]:
        sub = self.store.subject(subject_name)
        rows = self.store.events(sub["id"])
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ 内部
    def _auto_key(self, run_id: str, payload: dict) -> str:
        import json

        h = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
        return f"{run_id}:{h}"
