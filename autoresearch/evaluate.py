"""Helix 赫力 角色 C：三臂对照与系统指标。

三臂口径（与《AutoResearch-四人分工执行手册》§5.3 一致）：
- A 固定顺序本地搜索：无云调用，规则直出；
- B 贪婪云：每次要选实验都尝试云端，使用同一软预算口径；
- C 证据触发主臂：本地规则优先，只有结构触发才云端。

默认在没有云环境变量时使用确定性离线云客户端，指标里的 usage 标
`offline_simulated`；显式 `--cloud` 且环境变量齐备时走真实 OpenAI 兼容接口。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .agent_loop import EventWriter, ResearchOrchestrator
from .budget_gate import BudgetGate
from .contracts import TaskSpec, as_jsonable, load_task_spec
from .experiment import ResearchBackend
from .llm_router import (
    HttpJsonCloudClient,
    NullCloudClient,
    ResearchRouter,
    estimate_tokens,
)
from .oracle import ResearchOracle
from .research_pv import PLAN_ORDER, ResearchPVPlanner


class OfflineCloudClient:
    """Deterministic offline response used when no real cloud credentials exist."""

    def __init__(self, *, seed: int = 42, reply: Optional[Dict[str, Any]] = None) -> None:
        self.seed = seed
        self.reply = reply or {
            "decision": "keep current rule path",
            "evidence_ids": ["E00"],
            "actions": [{"action": "continue", "target": "rules"}],
        }
        self.calls = 0

    def call(
        self,
        summary: Dict[str, Any],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        del max_tokens, timeout_s
        self.calls += 1
        input_tokens = estimate_tokens(summary)
        output_tokens = max(1, estimate_tokens(self.reply))
        return {
            "content": json.dumps(self.reply, ensure_ascii=False),
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            },
            "offline": True,
        }


class AlwaysEscalateRouter:
    """B 贪婪云臂：每次编排器询问下一步时都强制进云端。"""

    def __init__(self, delegate: ResearchRouter) -> None:
        self.delegate = delegate

    def escalate_if_needed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.delegate.event_cb = self.event_cb
        return self.delegate.escalate_if_needed({**state, "critical_decision": True})

    event_cb = None

    def savings_summary(self) -> Dict[str, Any]:
        return self.delegate.savings_summary()


class LocalOnlyRouter:
    """A 固定顺序本地搜索：从不调用云，也从不写预算拦截事件。"""

    def escalate_if_needed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        del state
        return {"escalated": False, "valid": None, "reasons": [], "tokens": None}

    def savings_summary(self) -> Dict[str, Any]:
        return {
            "compared_decisions": 0,
            "prompt_baseline_tokens": 0,
            "prompt_compact_tokens": 0,
            "prompt_compression_saved": 0,
            "cache_hits": 0,
            "cache_saved_tokens": 0,
            "local_rule_decisions": 0,
            "local_rule_saved_tokens": 0,
            "cloud_calls_avoided": 0,
            "total_estimated_saved": 0,
            "cloud_calls_charged": 0,
            "cloud_tokens_charged": 0,
        }


def _usage_counts(ledger: Dict[str, Any]) -> Dict[str, Any]:
    entries = [entry for entry in ledger.get("entries", []) if entry.get("kind") == "cloud_call"]
    input_tokens = sum(int(entry.get("actual_input") or 0) for entry in entries)
    output_tokens = sum(int(entry.get("actual_output") or 0) for entry in entries)
    charged = sum(int(entry.get("charged") or 0) for entry in entries)
    return {
        "cloud_calls": int(ledger.get("used_cloud_calls") or 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "charged_tokens": charged,
    }


def _unwrap_event_value(value: Any) -> Any:
    """Event extras may store a truncated preview; unwrap it for metrics."""
    if isinstance(value, dict) and value.get("truncated") and isinstance(value.get("preview"), str):
        try:
            return json.loads(value["preview"])
        except Exception:
            return {}
    return value


def _extract_merged_auc(value: Any) -> Optional[float]:
    """Read merged_auc from a full baseline object or a truncated preview."""
    unwrapped = _unwrap_event_value(value)
    if isinstance(unwrapped, dict) and unwrapped.get("merged_auc") is not None:
        return float(unwrapped["merged_auc"])
    if isinstance(value, dict):
        preview = value.get("preview")
        if isinstance(preview, str):
            match = re.search(r'"merged_auc"\s*:\s*([0-9]+(?:\.[0-9]+)?)', preview)
            if match:
                return float(match.group(1))
    return None


def _metrics_row(arm: str, record: Dict[str, Any], price: Optional[float]) -> Dict[str, Any]:
    summary = record["summary"]
    test_verdict = summary.get("test_verdict") or {}
    metric = test_verdict.get("metric") or {}
    usage = _usage_counts(record["ledger"])
    if record["offline"]:
        usage_source = "offline_simulated"
    elif usage["cloud_calls"]:
        usage_source = "actual"
    else:
        usage_source = "none"
    cost = (
        round(float(usage["charged_tokens"]) * float(price) / 1000.0, 6)
        if price is not None
        else None
    )
    savings = summary.get("token_savings") or {}
    baseline_auc = None
    budget_blocked = 0
    for event in record.get("events") or []:
        event_type = event.get("type") or ""
        extra = event.get("extra") or {}
        if event_type == "research.baseline":
            baseline_auc = _extract_merged_auc(extra.get("baseline") or {})
        elif event_type == "research.escalation_blocked":
            budget_blocked += 1
    experiments = summary.get("experiments") or []
    total_experiments = len(experiments)
    negative_results = sum(
        1
        for item in experiments
        if (item.get("trial") or {}).get("error")
        or str((item.get("verdict") or {}).get("verdict") or "") in {"refuted", "blocked"}
    )
    attempts = int(summary.get("escalation_attempts") or 0)
    valid_escalations = int(summary.get("escalation_valid") or 0)
    auc = metric.get("auc")
    auc_gain = None
    per_10k_token_gain = None
    if auc is not None and baseline_auc is not None:
        auc_gain = float(auc) - baseline_auc
        charged = float(usage["charged_tokens"])
        if charged > 0:
            per_10k_token_gain = round(auc_gain * 10000.0 / charged, 6)
    return {
        "run_id": summary.get("run_id"),
        "arm": arm,
        "phase": summary.get("phase"),
        "completion": 1.0 if summary.get("tested") else 0.0,
        "test_auc": metric.get("auc"),
        "locked": (summary.get("locked") or {}).get("name"),
        "stop_reason": summary.get("stop_reason"),
        "candidates": (summary.get("budget") or {}).get("used_candidates", 0),
        "cloud_calls": usage["cloud_calls"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "charged_tokens": usage["charged_tokens"],
        "usage_source": usage_source,
        "cost": cost,
        "tokens_saved": savings.get("total_estimated_saved"),
        "cache_hits": savings.get("cache_hits", 0),
        "local_rules": savings.get("local_rule_decisions", 0),
        "cloud_calls_avoided": savings.get("cloud_calls_avoided", 0),
        "wall_seconds": summary.get("elapsed_seconds"),
        "baseline_auc": round(baseline_auc, 6) if baseline_auc is not None else None,
        "auc_gain_vs_baseline": round(auc_gain, 6) if auc_gain is not None else None,
        "per_10k_token_gain": per_10k_token_gain,
        "negative_result_rate": (
            round(negative_results / total_experiments, 6) if total_experiments else 0.0
        ),
        "escalation_attempts": attempts,
        "escalation_hit_rate": (
            round(valid_escalations / attempts, 6) if attempts else 0.0
        ),
        "budget_blocked_rate": (
            round(budget_blocked / attempts, 6) if attempts else 0.0
        ),
        "speedup_vs_greedy": None,
        "speedup_vs_local": None,
    }


def run_arm(
    task: TaskSpec,
    root: Path,
    arm: str,
    *,
    cloud_client: Optional[Any] = None,
    fixed_candidates: bool = False,
) -> Dict[str, Any]:
    run_id = f"R-{arm}-{uuid.uuid4().hex[:12]}"
    run_dir = root / arm / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    backend = ResearchBackend(task)
    ledger = BudgetGate(task.budget)
    offline = False
    if arm == "A":
        client: Any = NullCloudClient()
        router: Any = LocalOnlyRouter()
    elif arm == "B":
        client = cloud_client or OfflineCloudClient(seed=task.seed)
        offline = isinstance(client, OfflineCloudClient)
        router = AlwaysEscalateRouter(
            ResearchRouter(ledger, client=client, seed=task.seed)
        )
    else:
        client = cloud_client or OfflineCloudClient(seed=task.seed)
        offline = isinstance(client, OfflineCloudClient)
        router = ResearchRouter(ledger, client=client, seed=task.seed)

    oracle = ResearchOracle(backend, output_dir=run_dir)
    planner = ResearchPVPlanner(plan_size=len(PLAN_ORDER) if fixed_candidates else None)
    writer = EventWriter(run_dir / "events.jsonl")
    summary = ResearchOrchestrator(
        task,
        data=backend,
        oracle=oracle,
        planner=planner,
        router=router,
        ledger=ledger,
        writer=writer,
        run_id=run_id,
        run_ledger_dir=run_dir / "ledgers",
        candidate_limit=task.budget.candidates if fixed_candidates else None,
    ).run()
    prepared = backend.prepared_split()
    summary["data_hash"] = prepared.meta.get("data_hash") or ""
    summary["split_fingerprint"] = prepared.fingerprint

    (run_dir / "task.json").write_text(
        json.dumps(as_jsonable(task), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    events = []
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "ledger": ledger.usage_ledger(),
        "events": events,
        "offline": offline,
    }


def run_evaluate(
    task: TaskSpec,
    out_dir: Path,
    *,
    cloud: bool = False,
    price: Optional[float] = None,
    arms: Sequence[str] = ("A", "B", "C"),
    fixed_candidates: bool = False,
) -> Dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if cloud and not HttpJsonCloudClient().configured():
        raise ValueError(
            "--cloud requires ARC_LLM_BASE_URL, ARC_LLM_API_KEY and ARC_LLM_MODEL"
        )
    cloud_client: Any = HttpJsonCloudClient() if cloud else OfflineCloudClient(seed=task.seed)
    records = [
        run_arm(task, root, arm, cloud_client=cloud_client, fixed_candidates=fixed_candidates)
        for arm in arms
    ]
    rows = [_metrics_row(arm, record, price) for arm, record in zip(arms, records)]

    by_arm = {row["arm"]: row for row in rows}
    greedy_wall = float((by_arm.get("B") or {}).get("wall_seconds") or 0.0)
    local_wall = float((by_arm.get("A") or {}).get("wall_seconds") or 0.0)
    for row in rows:
        wall = float(row.get("wall_seconds") or 0.0)
        row["speedup_vs_greedy"] = (
            round(greedy_wall / wall, 4) if greedy_wall > 0 and wall > 0 else None
        )
        row["speedup_vs_local"] = (
            round(local_wall / wall, 4) if local_wall > 0 and wall > 0 else None
        )

    metrics = {
        "schema_version": "autoresearch-metrics-v2",
        "task_id": task.task_id,
        "data_path": task.data_path,
        "cloud": cloud,
        "candidate_limit": task.budget.candidates if fixed_candidates else None,
        "completion_rate": round(
            sum(float(row["completion"]) for row in rows) / len(rows), 6
        ),
        "arms": rows,
    }

    csv_path = root / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "task.json").write_text(
        json.dumps(as_jsonable(task), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="panel CSV")
    parser.add_argument("--out", type=Path, default=Path(".arc") / "eval")
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--price-per-1k-tokens", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--fixed-candidates", action="store_true", help="share the full candidate pool and enforce the same experiment cap")
    parser.add_argument("--cloud-calls", type=int, default=3)
    parser.add_argument("--cloud-tokens", type=int, default=6000)
    args = parser.parse_args(argv)

    task = load_task_spec(
        {
            "task_id": args.data.stem,
            "data_path": str(args.data.resolve()),
            "seed": args.seed,
            "budget": {
                "candidates": args.candidates,
                "cloud_calls": args.cloud_calls,
                "cloud_tokens": args.cloud_tokens,
            },
        }
    )
    metrics = run_evaluate(
        task,
        args.out,
        cloud=args.cloud,
        price=args.price_per_1k_tokens,
        fixed_candidates=args.fixed_candidates,
    )
    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    return 0 if float(metrics["completion_rate"]) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
