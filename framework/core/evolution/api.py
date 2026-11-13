"""HTTP 契约层（纯函数）：前端不需要知道 SQLite 长什么样。

状态码约定（与队友 PRD 一致，前端按这个写就行）：

| 情况 | 码 | 说明 |
|---|---|---|
| 正常 | 200 / 201 | 创建运行返回 201 |
| 资源不存在 | 404 | 运行 id / 任务 key 找不到 |
| 基线已变 | 409 | 这条运行的策略版本已被推进，反馈不会覆盖新经验 |
| 输入非法 | 422 | 字段缺失、枚举越界、文本超长 |
| 闸门拒绝 | 200 | `disposition=gate_rejected`，不是错误，是可解释的拒绝 |
"""

from __future__ import annotations

from typing import Any

from . import domains, policy
from .runtime import RunConfig, Runtime
from .store import BaselineChanged, Store

MAX_FEEDBACK_TEXT = 120


class Api:
    def __init__(self, db_path: str, domain: str, subject: str) -> None:
        self.store = Store(db_path)
        self.domain = domains.get(domain)
        self.rt = Runtime(self.store, self.domain, RunConfig(model="api"))
        self.subject = subject
        self.rt.bootstrap(subject)

    # ------------------------------------------------------------ 运行
    def create_run(self, kind: str = "first") -> tuple[int, dict]:
        if kind not in ("first", "second"):
            return 422, {"error": "kind 只能是 first / second"}
        r = self.rt.start_run(self.subject, kind)
        return 201, {
            "run_id": r["run"]["id"],
            "policy_version": {"id": r["version"]["id"], "seq": r["version"]["seq"]},
            "asks": r["asks"],
            "tasks": [
                {
                    "key": t["key"],
                    "title": t["title"],
                    "prediction": t["prediction"],
                    "prediction_reason": t["prediction_reason"],
                    "kind": t["kind"],
                }
                for t in r["tasks"]
            ],
        }

    def get_run(self, run_id: str) -> tuple[int, dict]:
        run = self.store.run(run_id)
        if run is None:
            return 404, {"error": "运行不存在"}
        tasks = self.rt.normalized_tasks(run_id)
        outs = self.store.outcomes(run_id)
        return 200, {
            "run_id": run_id,
            "status": run["status"],
            "locked_version": run["policy_version_id"],
            "hashes": {
                "build": run["build_hash"],
                "prompt": run["prompt_hash"],
                "model": run["model_config_hash"],
            },
            "tasks": [
                {
                    "key": t["key"],
                    "title": t["title"],
                    "prediction": t["prediction"],
                    "done": bool(outs[t["id"]]["done"]) if t["id"] in outs else None,
                    "reason": outs[t["id"]]["reason"] if t["id"] in outs else None,
                }
                for t in tasks
            ],
        }

    def post_attempt(self, run_id: str, body: dict) -> tuple[int, dict]:
        key = body.get("task_key")
        if not key:
            return 422, {"error": "缺少 task_key"}
        if self.store.run(run_id) is None:
            return 404, {"error": "运行不存在"}
        if self.store.task_by_key(run_id, key) is None:
            return 404, {"error": f"任务 {key} 不存在"}
        if "done" not in body and not hasattr(self.domain, "judge"):
            return 422, {"error": "缺少 done"}
        row = self.rt.judge(run_id, key, body)
        return 200, {"task_key": key, "done": bool(row["done"]), "reason": row["reason"]}

    def finish_run(self, run_id: str) -> tuple[int, dict]:
        if self.store.run(run_id) is None:
            return 404, {"error": "运行不存在"}
        m = self.rt.finish(run_id)
        return 200, {
            "prediction": m["prediction"]["text"],
            "conflict": m["conflict"]["count"],
            "misses": m["prediction"]["misses"],
        }

    # ------------------------------------------------------------ 反馈 / 进化
    def post_feedback(self, run_id: str, body: dict) -> tuple[int, dict]:
        if self.store.run(run_id) is None:
            return 404, {"error": "运行不存在"}
        text = str(body.get("text", "") or "")
        if len(text) > MAX_FEEDBACK_TEXT:
            return 422, {"error": f"补充说明不得超过 {MAX_FEEDBACK_TEXT} 字"}
        if not body.get("stated") and not body.get("observed") and not body.get("choice"):
            return 422, {"error": "至少给一条反馈（choice / stated / observed）"}
        key = body.get("submission_key") or None
        res = self.rt.feedback(run_id, body, submission_key=key)
        code = 200
        if res["disposition"] == "baseline_changed":
            code = 409
        new_version = None
        if res.get("new_version_id"):
            seq = res.get("new_seq")
            if seq is None:  # 幂等重放：按 id 回查
                seq = self.store.version(res["new_version_id"])["seq"]
            new_version = {"id": res["new_version_id"], "seq": seq}
        return code, {
            "disposition": res["disposition"],
            "new_version": new_version,
            "changes": res.get("changes", []),
            "reasons": res.get("reasons", []),
            "idempotent": res.get("idempotent", False),
        }

    def get_evolution(self) -> tuple[int, dict]:
        st = self.rt.status(self.subject)
        return 200, {
            "active": {"id": st["active"]["id"], "seq": st["active"]["seq"]},
            "versions": [
                {
                    "seq": v["seq"],
                    "hash": v["snapshot_hash"],
                    "reason": v["reason"],
                    "restored_from": v["restored_from_id"],
                    "created_at": v["created_at"],
                }
                for v in st["versions"]
            ],
            "summary": st["summary"],
            "events": self.rt.audit(self.subject),
        }

    def post_rollback(self, body: dict) -> tuple[int, dict]:
        try:
            rb = self.rt.rollback(self.subject, expected_active=body.get("expected_active"), target=body.get("target"))
        except BaselineChanged as exc:
            return 409, {"error": str(exc)}
        except (KeyError, ValueError) as exc:
            return 422, {"error": str(exc)}
        return 200, rb

    def snapshot(self) -> dict:
        st = self.rt.status(self.subject)
        return st["snapshot"]

    def close(self) -> None:
        self.store.close()
