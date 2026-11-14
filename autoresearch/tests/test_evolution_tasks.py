"""Helix 赫力 研究策略自进化桥测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from autoresearch.contracts import load_task_spec
from autoresearch.evolution import (
    AXIS_FOR_HYPOTHESIS,
    BaselineChanged,
    ResearchIterationStore,
)
from autoresearch.research_pv import PLAN_ORDER, ResearchPVPlanner


class TestResearchEvolution(unittest.TestCase):
    def _refute(self, hypothesis_id: str, reason: str = "window_drop") -> dict:
        return {
            "axis": AXIS_FOR_HYPOTHESIS[hypothesis_id],
            "reason": reason,
            "hypothesis_id": hypothesis_id,
            "refute": [hypothesis_id],
            "support": [],
        }

    def test_elimination_persists_across_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / "research-pv.db"
            store = ResearchIterationStore(str(db))
            before = [item[0] for item in store.active_plan_order(PLAN_ORDER)]
            self.assertIn("volume", before)

            store.record_signals([self._refute("volume")])
            store.record_signals([self._refute("volume")])
            after = [item[0] for item in store.active_plan_order(PLAN_ORDER)]
            self.assertNotIn("volume", after)
            store.close()

            store2 = ResearchIterationStore(str(db))
            reopened = [item[0] for item in store2.active_plan_order(PLAN_ORDER)]
            self.assertNotIn("volume", reopened)
            store2.close()

    def test_supported_hypothesis_prioritized(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResearchIterationStore(str(Path(folder) / "research-pv.db"))
            store.record_signals(
                [
                    {
                        "axis": "features",
                        "reason": "merged_improve",
                        "hypothesis_id": "volatility",
                        "refute": [],
                        "support": ["volatility"],
                    }
                ]
            )
            order = [item[0] for item in store.active_plan_order(PLAN_ORDER)]
            self.assertEqual(order[0], "volatility")
            store.close()

    def test_planner_skips_eliminated_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResearchIterationStore(str(Path(folder) / "research-pv.db"))
            store.record_signals([self._refute("volume")])
            store.record_signals([self._refute("volume")])
            planner = ResearchPVPlanner(evolution=store)
            task = load_task_spec(
                {
                    "task_id": "task-evo",
                    "data_path": "data/demo_panel.csv",
                    "budget": {"candidates": 4},
                }
            )
            specs = planner.plan(task, {"attempted": []}, "first")
            hypothesis_ids = [spec.hypothesis_id for spec in specs]
            self.assertNotIn("volume", hypothesis_ids)
            self.assertIn("volatility", hypothesis_ids)
            store.close()

    def test_rollback_restores_eliminated_hypothesis(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResearchIterationStore(str(Path(folder) / "research-pv.db"))
            store.record_signals([self._refute("volume")])
            store.record_signals([self._refute("volume")])
            excluded = [item[0] for item in store.active_plan_order(PLAN_ORDER)]
            self.assertNotIn("volume", excluded)
            result = store.rollback()
            self.assertTrue(result["content_equals"])
            restored = [item[0] for item in store.active_plan_order(PLAN_ORDER)]
            self.assertIn("volume", restored)
            store.close()

    def test_rollback_rejects_stale_expectation(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = ResearchIterationStore(str(Path(folder) / "research-pv.db"))
            store.record_signals([self._refute("volume")])
            with self.assertRaises(BaselineChanged):
                store.rollback(expected_active="does-not-exist")
            store.close()


if __name__ == "__main__":
    unittest.main()
