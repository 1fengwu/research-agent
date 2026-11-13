"""与 ARC 基础设施的接线：环境变量约定、.env 规则、事件镜像不重复。"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from _util import Checker, run

from core.evolution import domains
from core.evolution.arc_bridge import load_env, mirror_events, paths
from core.evolution.runtime import Runtime
from core.evolution.store import Store


def test_env_rules() -> int:
    c = Checker("接线：.env 只补缺不覆盖（与 core/config.py 同规则）")
    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text("OPENAI_KEY=sk-from-file\nARCBENCH_TEST_X=1\n", encoding="utf-8")
        os.environ["OPENAI_KEY"] = "sk-from-shell"
        os.environ.pop("ARCBENCH_TEST_X", None)
        loaded = load_env(env)
        c.eq("文件里的值被读到", loaded.get("ARCBENCH_TEST_X"), "1")
        c.eq("已有环境变量不被覆盖", os.environ["OPENAI_KEY"], "sk-from-shell")
        c.eq("补出 SDK 认的别名", os.environ.get("OPENAI_API_KEY"), "sk-from-shell")
        os.environ.pop("OPENAI_KEY", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ARCBENCH_TEST_X", None)
    return c.done()


def test_paths() -> int:
    c = Checker("接线：事件路径沿用 ARCBENCH_RUNNER_EVENTS_PATH")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["ARCBENCH_OUTPUT_DIR"] = tmp
        os.environ.pop("ARCBENCH_RUNNER_EVENTS_PATH", None)
        p = paths()
        c.eq("默认落 .arc/runner-events.jsonl", p["runner_events"], Path(tmp) / ".arc" / "runner-events.jsonl")
        os.environ["ARCBENCH_RUNNER_EVENTS_PATH"] = "logs/ev.jsonl"
        p2 = paths()
        c.eq("相对路径按 project_dir 解析", p2["runner_events"], Path(tmp) / "logs" / "ev.jsonl")
        for k in ("ARCBENCH_OUTPUT_DIR", "ARCBENCH_RUNNER_EVENTS_PATH"):
            os.environ.pop(k, None)
    return c.done()


def test_mirror() -> int:
    c = Checker("接线：事件镜像进 ARC jsonl，且不会重复写")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "runner-events.jsonl"
        store = Store(":memory:")
        domain = domains.get("habitfit")
        rt = Runtime(store, domain)
        rt.bootstrap("u")
        r1 = rt.start_run("u", "first")
        for t in rt.normalized_tasks(r1["run"]["id"]):
            o = domain.simulate(t["payload"], t["hypothesis"], t["kind"])
            rt.report(r1["run"]["id"], t["key"], o.done, o.reason, o.note)

        sub_id = store.subject("u")["id"]
        n1 = mirror_events(store, sub_id, out)
        n2 = mirror_events(store, sub_id, out)
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        c.check("首次写入事件", n1 > 0, str(n1))
        c.eq("重复调用不重复写", n2, 0)
        c.eq("文件行数等于首次条数", len(lines), n1)
        first = json.loads(lines[0])
        c.check("行格式沿用 ARC：type/timestamp", {"type", "timestamp"} <= set(first), str(first.keys()))
        c.check("type 带 evolution 前缀，便于和 ARC 事件区分", first["type"].startswith("evolution."), first["type"])
    return c.done()


def main() -> int:
    codes = [run(f) for f in (test_env_rules, test_paths, test_mirror)]
    return 0 if all(x == 0 for x in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
