"""Run AutoResearch against a local CSV."""

import argparse
import json
from pathlib import Path
from uuid import uuid4

from .agent_loop import EventWriter, ResearchOrchestrator
from .budget_gate import BudgetGate
from .contracts import as_jsonable, load_task_spec
from .evolution import ResearchIterationStore
from .experiment import ResearchBackend
from .llm_router import HttpJsonCloudClient, NullCloudClient, ResearchRouter
from .oracle import ResearchOracle
from .research_pv import ResearchPVPlanner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path, help="input panel CSV")
    parser.add_argument("--output-dir", type=Path, default=Path(".arc"))
    parser.add_argument("--evolution", type=Path, default=Path(".arc/evolution/research-pv.db"))
    parser.add_argument("--seeds", type=str, default="42,2024,7")
    parser.add_argument("--cloud", action="store_true", help="enable the configured cloud model")
    args = parser.parse_args()

    client = HttpJsonCloudClient() if args.cloud else NullCloudClient()
    if args.cloud and not client.configured():
        parser.error("--cloud requires ARC_LLM_API_KEY, ARC_LLM_MODEL and a configured base URL")

    task = load_task_spec({
        "task_id": args.data_path.stem,
        "data_path": str(args.data_path.resolve()),
        "budget": {"cloud_calls": 3 if args.cloud else 0},
    })
    backend = ResearchBackend(task)
    ledger = BudgetGate(task.budget)
    run_id = f"R{uuid4().hex}"
    evolution_store = ResearchIterationStore(str(args.evolution))
    try:
        summary = ResearchOrchestrator(
            task,
            data=backend,
            oracle=ResearchOracle(backend, output_dir=args.output_dir),
            planner=ResearchPVPlanner(evolution=evolution_store),
            router=ResearchRouter(ledger, client=client),
            ledger=ledger,
            writer=EventWriter(args.output_dir / "runner-events.jsonl"),
            run_id=run_id,
            run_ledger_dir=args.output_dir / "ledgers",
        ).run()
    finally:
        evolution_store.close()
    try:
        locked_spec = summary.get("locked", {}).get("config") or {}
        seeds = [int(token) for token in args.seeds.split(",") if token.strip()]
        audit = ResearchOracle(backend, output_dir=None).holdout_audit(locked_spec, seeds=seeds)
        summary["seed_audit"] = audit
        (args.output_dir / f"seed-audit-{run_id}.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        summary["seed_audit"] = {"error": str(exc)}
    prepared = backend.prepared_split()
    summary["data_hash"] = prepared.meta.get("data_hash") or ""
    summary["split_fingerprint"] = prepared.fingerprint
    (args.output_dir / "task.json").write_text(
        json.dumps(as_jsonable(task), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / f"summary-{run_id}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if summary["tested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
