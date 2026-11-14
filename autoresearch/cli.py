"""AutoResearch 角色 D：CLI 回放/启动路径。

D 的界面退化时使用 `python -m autoresearch.cli ...`：
- `replay / report`：只读事件文件，输出回放与报告/复现包；
- `demo`：离线联调，使用 `mock_events.json`；
- `run`：真实后台启动，调用 A 的 orchestrator；C 的 oracle 由 `--oracle` 注入。
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from . import report as report_mod
from .budget_gate import BudgetGate
from .contracts import load_task_spec
from .experiment import ResearchBackend
from .llm_router import HttpJsonCloudClient, NullCloudClient, ResearchRouter
from .research_pv import ResearchPVPlanner

DEFAULT_MOCK_EVENTS = Path(__file__).resolve().parent / "mock_events.json"
DEFAULT_EVENTS = Path.cwd() / ".arc" / "runner-events.jsonl"
DEFAULT_LEDGER_DIR = Path.cwd() / ".arc" / "ledgers"
DEFAULT_OUTPUT_DIR = Path.cwd() / ".arc"
DEFAULT_ORACLE = "autoresearch.oracle:ResearchOracle"


def _print_timeline(events: List[Dict[str, Any]]) -> None:
    for event in events:
        timestamp = str(event.get("timestamp") or "")
        if timestamp:
            timestamp = timestamp[11:19]
        experiment = str(event.get("experiment_id") or "")
        exp = f" [{experiment}]" if experiment else ""
        print(
            "%s %-16s %-32s%s %s"
            % (
                timestamp,
                event.get("phase") or "",
                event.get("type") or "",
                exp,
                event.get("decision_reason") or "",
            )
        )


def _out_files(
    run_dir: Path,
    events_path: Path,
    out_dir: Path,
    summary: Dict[str, Any],
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(summary.get("run_id") or "run")
    report_md = report_mod.render_markdown(summary)
    report_path = out_dir / f"report-{run_id}.md"
    report_path.write_text(report_md, encoding="utf-8")
    command = report_mod.default_run_command(
        events_path,
        out_dir=out_dir,
    )
    package_path = report_mod.build_reproduce_package(
        summary,
        run_dir=run_dir,
        out_dir=out_dir,
        command=command,
        report_md=report_md,
    )
    return report_path, package_path


def cmd_replay(args: argparse.Namespace) -> int:
    events_path = Path(args.events)
    if not events_path.exists():
        print(f"事件文件不存在: {events_path}", file=sys.stderr)
        return 2
    events = report_mod.load_events(events_path)
    runs = report_mod.group_runs(events)
    if not runs:
        print("事件文件为空", file=sys.stderr)
        return 2
    for run in runs:
        summary = report_mod.summarize_run(run["events"])
        print("=" * 72)
        print(
            "RUN %s phase=%s stop=%s"
            % (
                summary.get("run_id"),
                summary.get("phase"),
                summary.get("stop_reason"),
            )
        )
        _print_timeline(run["events"])
    if args.out:
        out_dir = Path(args.out)
        for run in runs:
            summary = report_mod.summarize_run(run["events"])
            report_path, package_path = _out_files(
                Path.cwd(),
                events_path,
                out_dir,
                summary,
            )
            print("report=%s" % report_path)
            print("package=%s" % package_path)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    events_path = DEFAULT_MOCK_EVENTS
    if not events_path.exists():
        print(f"缺少 mock_events.json: {events_path}", file=sys.stderr)
        return 2
    args.events = str(events_path)
    return cmd_replay(args)


def cmd_report(args: argparse.Namespace) -> int:
    return cmd_replay(args)


def _resolve_attr(dotted: str) -> Any:
    module_name, _, attr_name = dotted.rpartition(":")
    if not module_name or not attr_name:
        raise ValueError("oracle 必须形如 module.path:ClassName")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _make_oracle(resolved: Any, backend: Any, output_dir: Path) -> Any:
    """实例化 oracle：优先使用 `(backend, output_dir=...)` 约定，兼容无参类。"""
    try:
        return resolved(backend, output_dir=output_dir)
    except TypeError:
        return resolved()


def cmd_run(args: argparse.Namespace) -> int:
    task_path = Path(args.task)
    if not task_path.exists():
        print(f"任务文件不存在: {task_path}", file=sys.stderr)
        return 2
    try:
        raw_task = json.loads(task_path.read_text(encoding="utf-8"))
        task = load_task_spec(raw_task)
    except Exception as exc:
        print(f"任务配置无效: {exc}", file=sys.stderr)
        return 2

    try:
        oracle_cls = _resolve_attr(args.oracle)
    except Exception as exc:
        print(
            "无法解析 oracle 类；请提供 `--oracle module:Class`，"
            "或使用 `python -m autoresearch.cli demo/replay` 离线联调。\n"
            f"详情: {exc}",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir)
    events_path = Path(args.events)
    ledger_dir = Path(args.ledger_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)

    ledger = BudgetGate(task.budget)
    cloud_client = HttpJsonCloudClient()
    if not cloud_client.configured():
        cloud_client = NullCloudClient()
    router = ResearchRouter(ledger, client=cloud_client)
    from .agent_loop import EventWriter, ResearchOrchestrator

    backend = ResearchBackend(task)
    orchestrator = ResearchOrchestrator(
        task,
        data=backend,
        oracle=_make_oracle(oracle_cls, backend, output_dir),
        planner=ResearchPVPlanner(),
        router=router,
        ledger=ledger,
        writer=EventWriter(str(events_path)),
        run_ledger_dir=str(ledger_dir),
    )
    summary = orchestrator.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if args.out:
        out_dir = Path(args.out)
        report_path, package_path = _out_files(
            Path.cwd(),
            events_path,
            out_dir,
            summary,
        )
        print("report=%s" % report_path)
        print("package=%s" % package_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m autoresearch.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="只读事件文件，输出时间线/报告/复现包")
    replay.add_argument("--events", default=str(DEFAULT_EVENTS))
    replay.add_argument("--out")

    demo = sub.add_parser("demo", help="离线联调，使用 mock_events.json")
    demo.add_argument("--out")
    report = sub.add_parser("report", help="从事件文件生成报告与复现包")
    report.add_argument("--events", default=str(DEFAULT_EVENTS))
    report.add_argument("--out")

    run = sub.add_parser("run", help="后台启动一次研究运行")
    run.add_argument("--task", required=True)
    run.add_argument("--events", default=str(DEFAULT_EVENTS))
    run.add_argument("--ledger-dir", default=str(DEFAULT_LEDGER_DIR))
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    run.add_argument("--oracle", default=DEFAULT_ORACLE)
    run.add_argument("--out")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        return cmd_replay(args)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "run":
        return cmd_run(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
