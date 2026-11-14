"""Helix 赫力 角色 D：事件展示、报告、复现包与 CLI 回放测试。"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from autoresearch import report

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_EVENTS = REPO_ROOT / "autoresearch" / "mock_events.json"


class TestDMockEvents(unittest.TestCase):
    def test_mock_events_timeline_order(self) -> None:
        events = report.load_events(MOCK_EVENTS)
        runs = report.group_runs(events)
        self.assertGreaterEqual(len(runs), 2)

        full = report.summarize_run(runs[0]["events"])
        phases = [str(event.get("phase")) for event in runs[0]["events"]]
        self.assertEqual(phases[0], "UNSEEN")
        self.assertEqual(full["phase"], "TESTED")
        self.assertTrue("REVISED" in phases)
        self.assertTrue("LOCKED" in phases)
        self.assertLess(phases.index("REVISED"), phases.index("LOCKED"))

    def test_summary_from_mock_full_run(self) -> None:
        events = report.load_events(MOCK_EVENTS)
        summary = report.summarize_run(report.group_runs(events)[0]["events"])
        self.assertEqual(summary["run_id"], "R20261114-001")
        self.assertEqual(float(summary["baseline"]["merged_auc"]), 0.661491)
        self.assertEqual(len(summary["experiments"]), 4)
        self.assertEqual(summary["locked"]["name"], "exp-004")
        self.assertTrue(summary["tested"])
        self.assertEqual(summary["test_verdict"]["verdict"], "supported")
        self.assertEqual(summary["evidence_ids"], ["exp-002:window_drop"])
        self.assertIn("total_estimated_saved", summary["token_savings"])
        self.assertGreater(
            int(summary["token_savings"]["total_estimated_saved"]), 0
        )

    def test_budget_blocked_visible_but_run_completes(self) -> None:
        events = report.load_events(MOCK_EVENTS)
        summary = report.summarize_run(report.group_runs(events)[1]["events"])
        self.assertEqual(summary["phase"], "TESTED")
        self.assertTrue(summary["tested"])
        self.assertEqual(len(summary["escalation_blocked"]), 1)
        self.assertIn(
            "budget_blocked",
            summary["escalation_blocked"][0]["decision_reason"],
        )
        self.assertEqual(summary["budget"]["remaining_cloud_tokens"], 300)

    def test_load_events_jsonl_roundtrip(self) -> None:
        source = report.load_events(MOCK_EVENTS)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.jsonl"
            with open(path, "w", encoding="utf-8") as handle:
                for event in source[:3]:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            loaded = report.load_events(path)
        self.assertEqual(loaded, source[:3])


class TestDReport(unittest.TestCase):
    def _summary(self):
        events = report.load_events(MOCK_EVENTS)
        return report.summarize_run(report.group_runs(events)[0]["events"])

    def test_report_markdown_contains_sections(self) -> None:
        markdown = report.render_markdown(self._summary())
        for section in ("## 1. 运行信息", "## 4. 假设与实验", "## 6. 审查结论", "## 7. 复现包"):
            self.assertIn(section, markdown)
        self.assertIn("R20261114-001", markdown)
        self.assertIn("0.661491", markdown)
        self.assertIn("上下文压缩节省", markdown)

    def test_reproduce_package_contains_expected(self) -> None:
        summary = self._summary()
        markdown = report.render_markdown(summary)
        with tempfile.TemporaryDirectory() as folder:
            package = report.build_reproduce_package(
                summary,
                run_dir=REPO_ROOT,
                out_dir=Path(folder),
                command=report.default_run_command("autoresearch/mock_events.json"),
                report_md=markdown,
            )
            with zipfile.ZipFile(package) as archive:
                names = set(archive.namelist())
        prefix = "reproduce-R20261114-001"
        self.assertIn(f"{prefix}/runner-events.jsonl", names)
        self.assertIn(f"{prefix}/report.md", names)
        self.assertIn(f"{prefix}/run-command.txt", names)
        self.assertIn(f"{prefix}/requirements.txt", names)
        self.assertIn(f"{prefix}/data/demo_panel.csv.sha256", names)


class TestDCLIFallback(unittest.TestCase):
    def test_cli_demo_replay_works(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "autoresearch.cli", "demo"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("R20261114-001", proc.stdout)
        self.assertIn("research.locked", proc.stdout)

    def test_cli_run_refuses_missing_oracle_class(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            task_path = Path(folder) / "task.json"
            task_path.write_text(
                json.dumps(
                    {
                        "task_id": "task-d-no-oracle",
                        "data_path": "data/demo_panel.csv",
                        "label_horizon": 5,
                        "split": {"train": 0.6, "val_windows": 2, "test": 0.2, "gap_days": 5},
                        "seed": 42,
                        "budget": {
                            "candidates": 1,
                            "cloud_calls": 1,
                            "cloud_tokens": 6000,
                            "wall_seconds": 180,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autoresearch.cli",
                    "run",
                    "--task",
                    str(task_path),
                    "--oracle",
                    "autoresearch.tests.no_such_oracle:Oracle",
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("oracle", proc.stderr.lower())

    def test_cli_run_with_c_oracle_works(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            task_path = base / "task.json"
            arc_dir = base / "arc"
            task_path.write_text(
                json.dumps(
                    {
                        "task_id": "task-d-e2e",
                        "data_path": "data/demo_panel.csv",
                        "label_horizon": 5,
                        "split": {
                            "train": 0.6,
                            "val_windows": 2,
                            "test": 0.2,
                            "gap_days": 5,
                        },
                        "seed": 42,
                        "budget": {
                            "candidates": 1,
                            "cloud_calls": 0,
                            "cloud_tokens": 6000,
                            "wall_seconds": 180,
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "autoresearch.cli",
                    "run",
                    "--task",
                    str(task_path),
                    "--output-dir",
                    str(arc_dir),
                    "--events",
                    str(arc_dir / "runner-events.jsonl"),
                    "--ledger-dir",
                    str(arc_dir / "ledgers"),
                    "--evolution",
                    str(arc_dir / "evolution-research-pv.db"),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("run_id", proc.stdout)
            self.assertTrue((arc_dir / "runner-events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
