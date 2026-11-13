"""极简断言器：不引第三方依赖，CI 里直接 python tests/run_all.py。"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # agent-arc-based/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Checker:
    def __init__(self, title: str) -> None:
        self.title = title
        self.passed = 0
        self.failed = 0
        print(f"\n===== {title} =====")

    def check(self, label: str, cond: bool, detail: str = "") -> bool:
        if cond:
            self.passed += 1
            print(f"[PASS] {label}")
        else:
            self.failed += 1
            print(f"[FAIL] {label}" + (f"  → {detail}" if detail else ""))
        return cond

    def eq(self, label: str, got, want) -> bool:
        return self.check(label, got == want, f"got={got!r} want={want!r}")

    def done(self) -> int:
        total = self.passed + self.failed
        print(f"----- {self.title}: {self.passed}/{total} passed -----")
        return 0 if self.failed == 0 else 1


def run(fn) -> int:
    """跑一个测试函数；异常也算失败，但要看到栈。"""
    try:
        return fn()
    except Exception:
        traceback.print_exc()
        print("[FAIL] 测试抛出异常")
        return 1
