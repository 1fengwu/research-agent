"""HTTP 契约层：状态码与幂等行为。"""

from __future__ import annotations

import os
import tempfile

from _util import Checker, run

from core.evolution import domains
from core.evolution.api import Api


def new_api(tmp: str) -> tuple[Api, object]:
    db = os.path.join(tmp, "api.db")
    if os.path.exists(db):
        os.remove(db)
    api = Api(db, "habitfit", "u-api")
    return api, domains.get("habitfit")


def test_contract() -> int:
    c = Checker("API：创建 / 读回 / 上报 / 反馈 / 回退 的状态码")
    with tempfile.TemporaryDirectory() as tmp:
        api, domain = new_api(tmp)

        code, body = api.create_run("first")
        c.eq("创建运行 201", code, 201)
        c.check("返回任务带预测", all("prediction" in t for t in body["tasks"]), str(body["tasks"][:1]))
        rid = body["run_id"]

        code, run_body = api.get_run(rid)
        c.eq("读回 200", code, 200)
        c.check("三 hash 都在", all(run_body["hashes"].values()), str(run_body["hashes"]))
        c.eq("状态 running", run_body["status"], "running")

        code, _ = api.post_attempt(rid, {"task_key": "nope", "done": True})
        c.eq("未知任务 404", code, 404)
        code, _ = api.post_attempt(rid, {"task_key": "mon-ex"})
        c.eq("缺少 done 422", code, 422)

        for t in run_body["tasks"]:
            payload = next(x for x in api.rt.normalized_tasks(rid) if x["key"] == t["key"])["payload"]
            out = domain.simulate(payload, next(x for x in api.rt.normalized_tasks(rid) if x["key"] == t["key"])["hypothesis"])
            code, _ = api.post_attempt(rid, {"task_key": t["key"], "done": out.done, "reason": out.reason})
            c.check(f"上报 {t['key']} 200", code == 200)

        code, fin = api.finish_run(rid)
        c.eq("结束运行 200", code, 200)
        c.eq("命中率 1/7", fin["prediction"], "1/7")

        code, _ = api.post_feedback(rid, {"text": ""})
        c.eq("空反馈 422", code, 422)
        code, _ = api.post_feedback(rid, {"text": "x" * 200})
        c.eq("超长反馈 422", code, 422)

        payload = {
            "stated": domain.stated_feedback(api.rt._rows(rid)),
            "observed": domain.observations_from(api.rt.normalized_tasks(rid), api.rt._rows(rid)),
            "submission_key": "fb-1",
        }
        code, fb = api.post_feedback(rid, payload)
        c.eq("有效反馈 200", code, 200)
        c.eq("处置 applied", fb["disposition"], "applied")
        seq = fb["new_version"]["seq"]
        c.eq("版本推进到 1", seq, 1)

        code, fb2 = api.post_feedback(rid, payload)
        c.eq("同键重发仍 200", code, 200)
        c.check("标记幂等", fb2["idempotent"] is True)
        c.eq("版本没有再动", fb2["new_version"]["seq"], seq)

        code, evo = api.get_evolution()
        c.eq("进化记录 200", code, 200)
        c.eq("版本链 2 条", len(evo["versions"]), 2)
        c.check("事件里有 policy_updated", any(e["type"] == "policy_updated" for e in evo["events"]))

        code, rb = api.post_rollback({"expected_active": "pv_nope"})
        c.eq("预期版本不符 409", code, 409)
        code, rb = api.post_rollback({})
        c.eq("正常回退 200", code, 200)
        c.check("回退内容等同旧版", rb["content_equals"] is True)

        code, _ = api.post_rollback({"target": "pv_nope"})
        c.eq("目标版本不存在 422", code, 422)
        api.close()
    return c.done()


def main() -> int:
    return 0 if run(test_contract) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
