"""与 agent-arc-based 既有基础设施的接线（全部可选，零第三方依赖）。

刻意**不 import** `core.config` 与 `arcbench_agent_runtime.events`：它们会拉起
colorama / langchain 等 ARC 全量依赖，而进化层必须能在裸 Python 下跑通
（现场断网、评审机器没装依赖时兜底演示）。

这里只复刻它们的**约定**，不复制代码：

| ARC 的约定 | 位置 | 这里怎么对齐 |
|---|---|---|
| `.env` 解析（KEY=VALUE，不覆盖已有变量） | `core/config.py::load_project_env` | `load_env()` 同规则 |
| 事件文件路径 `ARCBENCH_RUNNER_EVENTS_PATH`，默认 `.arc/runner-events.jsonl` | `arcbench_agent_runtime/context.py` | `paths()` 同变量名 |
| 事件行格式 `{"type","timestamp",...}` | `arcbench_agent_runtime/events.py` | `mirror_events()` 同格式 |

接上之后，进化过程会出现在 ARC 原本那套 `.arc/runner-events.jsonl` 里，
评审/排障时只需要看一个文件，不必学两套日志。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_RUNNER_EVENTS = ".arc/runner-events.jsonl"
DEFAULT_TRACEABILITY_DIR = ".arc/traceability"


def load_env(env_path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """与 `core/config.py::load_project_env` 同规则：只补缺，不覆盖。"""
    path = Path(env_path) if env_path else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if key not in os.environ:
            os.environ[key] = value
    # ARC 用 OPENAI_KEY / OPENAI_BASE_URL，SDK 认 OPENAI_API_KEY / OPENAI_API_BASE
    for src, dst in (("OPENAI_KEY", "OPENAI_API_KEY"), ("OPENAI_BASE_URL", "OPENAI_API_BASE")):
        if src in os.environ and dst not in os.environ:
            os.environ[dst] = os.environ[src]
    return loaded


def _resolve_under(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def paths(project_dir: str | None = None) -> dict[str, Path]:
    """对齐 `RuntimePaths.from_env` 的取值顺序。"""
    base = Path(
        project_dir
        or os.environ.get("ARCBENCH_OUTPUT_DIR", "").strip()
        or os.environ.get("ARCBENCH_PROJECT_DIR", "").strip()
        or os.environ.get("ARCBENCH_TEMPLATE_DIR", "").strip()
        or "."
    ).expanduser().resolve()
    events = _resolve_under(
        base, os.environ.get("ARCBENCH_RUNNER_EVENTS_PATH", "").strip() or DEFAULT_RUNNER_EVENTS
    )
    trace = _resolve_under(
        base, os.environ.get("ARCBENCH_TRACEABILITY_DIR", "").strip() or DEFAULT_TRACEABILITY_DIR
    )
    return {"project_dir": base, "runner_events": events, "traceability": trace}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def mirror_events(store: Any, subject_id: str, out_path: str | os.PathLike[str] | None = None) -> int:
    """把 SQLite 里的事件按 ARC 行格式追加到 jsonl；用游标文件保证不重复。"""
    target = Path(out_path) if out_path else paths()["runner_events"]
    target.parent.mkdir(parents=True, exist_ok=True)
    cursor = target.with_suffix(target.suffix + ".cursor")

    last_id = 0
    if cursor.exists():
        try:
            last_id = int(cursor.read_text(encoding="utf-8").strip() or 0)
        except ValueError:
            last_id = 0

    rows = store.conn.execute(
        "SELECT * FROM events WHERE subject_id=? AND id>? ORDER BY id", (subject_id, last_id)
    ).fetchall()
    if not rows:
        return 0

    with target.open("a", encoding="utf-8") as fh:
        for r in rows:
            payload = json.loads(r["payload_json"] or "{}")
            line = {
                "type": f"evolution.{r['type']}",
                "timestamp": _now(),
                "subject_id": r["subject_id"],
                "run_id": r["run_id"],
                "prev_version_id": r["prev_version_id"],
                "new_version_id": r["new_version_id"],
                "at": r["at"],
                "payload": payload,
            }
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    cursor.write_text(str(rows[-1]["id"]), encoding="utf-8")
    return len(rows)
