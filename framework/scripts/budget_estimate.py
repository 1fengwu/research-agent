#!/usr/bin/env python3
"""Estimate per-submission token budget and CNY cost vs the VOLO-AI target.

This is a planning helper, NOT runtime code. The real spend is hard-capped by
agents/model/token_budget.py at runtime; this script just projects what the
configured cap means in money and how it compares to the leaderboard target.
"""
from __future__ import annotations

import os

# Observed platform price from the last run: 759M tokens -> 200.92 CNY.
TOKEN_PRICE_CNY_PER_M = 200.92 / 759.0  # ~0.2645 CNY / 1M tokens

# Task scope (leaves) established from the requirement trees.
LEAVES_GITHUB = 47
LEAVES_SHEET = 24
TOTAL_LEAVES = LEAVES_GITHUB + LEAVES_SHEET

# VOLO-AI leaderboard reference (the target the user chose).
VOLO_COST_CNY = 13.99
VOLO_PASS_RATE = 0.515

# Per-task budget: each task (GitHub, Sheet) gets its own shared counter across
# its 4 shards. Default 26M/task => 52M total => ~13.8 CNY (matches VOLO-AI).
PER_TASK_BUDGET = int(os.environ.get("ARC_TOKEN_BUDGET", "26000000"))
SHARDS_PER_TASK = 4
TASKS = 2


def main() -> None:
    total_tokens = PER_TASK_BUDGET * TASKS
    total_cny = total_tokens / 1_000_000 * TOKEN_PRICE_CNY_PER_M
    per_task_cny = PER_TASK_BUDGET / 1_000_000 * TOKEN_PRICE_CNY_PER_M

    print("=" * 60)
    print("ARC token-budget projection (hard cap, enforced at runtime)")
    print("=" * 60)
    print(f"Per-task budget      : {PER_TASK_BUDGET:,} tokens  (~{per_task_cny:.2f} CNY/task)")
    print(f"Tasks x shards       : {TASKS} tasks x {SHARDS_PER_TASK} shards = {TASKS*SHARDS_PER_TASK} model processes")
    print(f"TOTAL hard cap       : {total_tokens:,} tokens  (~{total_cny:.2f} CNY)")
    print(f"VOLO-AI target       : {VOLO_COST_CNY:.2f} CNY  ({VOLO_PASS_RATE*100:.1f}% pass)")
    print(f"Budget vs VOLO-AI     : {total_cny/VOLO_COST_CNY*100:.0f}% of his spend")
    print("-" * 60)

    # Coverage projection (rough). Prior run used ~759M for 71 leaves before
    # dying => ~10.7M tokens/leaf at the OLD (un-trimmed) settings. Trims
    # (TDD 4->2, smaller context) plausibly cut 30-60%; show a range.
    for factor in (0.4, 0.6, 0.8):
        per_leaf = 10.7e6 * factor
        leaves = total_tokens / per_leaf
        print(
            f"  if trim saves {int((1-factor)*100):>2d}% (={per_leaf/1e6:.1f}M/leaf): "
            f"~{leaves:.0f} of {TOTAL_LEAVES} leaves covered"
        )
    print("-" * 60)
    print("NOTE: coverage is a projection. The hard cap GUARANTEES the CNY")
    print("spend above; real pass-rate comes from one live submission, which")
    print("also recalibrates the per-leaf cost for the next tuning pass.")
    print("=" * 60)


if __name__ == "__main__":
    main()
