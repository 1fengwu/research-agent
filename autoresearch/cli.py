"""Helix 赫力 角色 D：CLI 回放/启动路径。

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
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict, List

from . import report as report_mod
from .budget_gate import BudgetGate
from .contracts import load_task_spec
from .discovery import apply_question, discover
from .experiment import ResearchBackend
from .evolution import ResearchIterationStore
from .llm_router import HttpJsonCloudClient, NullCloudClient, ResearchRouter
from .outputs import validate_event_list, validate_ledger, validate_summary
from .research_pv import PLAN_ORDER, ResearchPVPlanner

DEFAULT_MOCK_EVENTS = Path(__file__).resolve().parent / "mock_events.json"
DEFAULT_EVENTS = Path.cwd() / ".arc" / "runner-events.jsonl"
DEFAULT_LEDGER_DIR = Path.cwd() / ".arc" / "ledgers"
DEFAULT_OUTPUT_DIR = Path.cwd() / ".arc"
DEFAULT_EVOLUTION = Path.cwd() / ".arc" / "evolution" / "research-pv.db"
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
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(summary.get("run_id") or "run")
    report_md = report_mod.render_markdown(summary)
    report_path = out_dir / f"report-{run_id}.md"
    report_path.write_text(report_md, encoding="utf-8")
    command = report_mod.default_run_command(
        events_path,
        out_dir=out_dir,
    )
    git_commit = report_mod.repository_revision(Path.cwd())
    manifest = report_mod.build_delivery_manifest(summary, git_commit=git_commit)
    package_path = report_mod.build_reproduce_package(
        summary,
        run_dir=run_dir,
        out_dir=out_dir,
        command=command,
        report_md=report_md,
        delivery_manifest=manifest,
    )
    snapshot_path = report_mod.freeze_delivery_snapshot(
        summary,
        package_path,
        out_dir,
        delivery_manifest=manifest,
    )
    return report_path, package_path, snapshot_path


def cmd_replay(args: argparse.Namespace) -> int:
    events_path = Path(args.events)
    if not events_path.exists():
        print(f"事件文件不存在: {events_path}", file=sys.stderr)
        return 2
    events = report_mod.load_events(events_path)
    try:
        validate_event_list(events)
    except (TypeError, ValueError) as exc:
        print(f"事件流不符合后端契约: {exc}", file=sys.stderr)
        return 2
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
            report_path, package_path, snapshot_path = _out_files(
                Path.cwd(),
                events_path,
                out_dir,
                summary,
            )
            print("report=%s" % report_path)
            print("package=%s" % package_path)
            print("snapshot=%s" % snapshot_path)
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
    evolution_path = Path(args.evolution)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    evolution_path.parent.mkdir(parents=True, exist_ok=True)

    run_id = f"R{uuid4().hex}" if args.discover else None
    ledger = BudgetGate(task.budget, str(ledger_dir / f"{run_id}.json") if run_id else None)
    cloud_client = HttpJsonCloudClient()
    discovery = None
    discovery_call = None
    order = None
    if args.research_question:
        discovery = json.loads(Path(args.research_question).read_text(encoding="utf-8"))
        if discovery["status"] == "generated":
            task = apply_question(task, discovery)
            order = discovery["hypotheses"]
    elif args.discover:
        if cloud_client.configured():
            def publish_profile(facts):
                (output_dir / "discovery-profile.json").write_text(
                    json.dumps(facts, ensure_ascii=False), encoding="utf-8"
                )

            task, discovery, order, discovery_call = discover(
                task, cloud_client, ledger, [item[0] for item in PLAN_ORDER],
                on_profile=publish_profile,
            )
        else:
            discovery = {"status": "offline"}
    if not cloud_client.configured():
        cloud_client = NullCloudClient()
    router = ResearchRouter(
        ledger,
        client=cloud_client,
        cache_path=str(Path(args.output_dir) / "cache" / "decision-cache.json"),
        seed=task.seed,
    )
    from .agent_loop import EventWriter, ResearchOrchestrator

    with ResearchIterationStore(str(evolution_path)) as evolution_store:
        backend = ResearchBackend(task)
        if order:
            order = list(dict.fromkeys([*order, *(item[0] for item in PLAN_ORDER)]))
        planner = ResearchPVPlanner(order=order, evolution=evolution_store, defer_hypotheses=args.defer_hypothesis)
        if args.strategy_context:
            Path(args.strategy_context).write_text(json.dumps({
                "version": evolution_store.active_version_id,
                "candidate_order": planner.candidate_order(),
                "deferred_hypotheses": args.defer_hypothesis,
            }), encoding="utf-8")
        orchestrator = ResearchOrchestrator(
            task,
            data=backend,
            oracle=_make_oracle(oracle_cls, backend, output_dir),
            planner=planner,
            router=router,
            ledger=ledger,
            writer=EventWriter(str(events_path)),
            run_id=run_id,
            run_ledger_dir=None if run_id else str(ledger_dir),
        )
        if discovery is not None:
            (output_dir / "research-question.json").write_text(
                json.dumps(discovery, ensure_ascii=False), encoding="utf-8"
            )
            if discovery_call:
                orchestrator.writer.append(
                    "research.model_call", "PLANNING", orchestrator.run_id,
                    extra=discovery_call, remaining=ledger.snapshot()
                )
        summary = orchestrator.run()
        if run_id:
            summary["elapsed_seconds"] = round(ledger.elapsed_seconds(), 3)
        if discovery is not None:
            summary["research_question"] = discovery
            summary["label_horizon"] = task.label_horizon
    try:
        validate_summary(summary)
        validate_ledger(summary["ledger"])
        events = report_mod.load_events(events_path)
        validate_event_list(events)
    except (TypeError, ValueError) as exc:
        print(f"运行产物不符合后端契约: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))
    if args.summary_file:
        Path(args.summary_file).write_text(
            json.dumps(summary, ensure_ascii=False, default=str), encoding="utf-8"
        )
    if args.out:
        out_dir = Path(args.out)
        report_path, package_path, snapshot_path = _out_files(
            Path.cwd(),
            events_path,
            out_dir,
            summary,
        )
        print("report=%s" % report_path)
        print("package=%s" % package_path)
        print("snapshot=%s" % snapshot_path)
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
    run.add_argument("--evolution", default=str(DEFAULT_EVOLUTION))
    run.add_argument("--oracle", default=DEFAULT_ORACLE)
    run.add_argument("--out")
    discovery = run.add_mutually_exclusive_group()
    discovery.add_argument("--discover", action="store_true")
    discovery.add_argument("--research-question", help="Reuse a saved discovery result for comparison")
    run.add_argument("--strategy-context", help="Write the initial policy version and candidate order")
    run.add_argument("--defer-hypothesis", action="append", default=[],
                     help="Evaluate this hypothesis after other candidates")
    run.add_argument("--summary-file", help="Write the validated run summary to JSON")
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
