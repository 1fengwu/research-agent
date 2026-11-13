"""跑全部测试：python tests/run_all.py"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ["test_policy_learn.py", "test_store.py", "test_e2e.py", "test_api.py", "test_bridge.py"]


def main() -> int:
    failed: list[str] = []
    total_pass = 0
    for case in CASES:
        print(f"\n##### {case} #####")
        proc = subprocess.run([sys.executable, str(HERE / case)], capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        print(out.rstrip())
        total_pass += out.count("[PASS]")
        if proc.returncode != 0:
            failed.append(case)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"ALL GREEN — {len(CASES)} 套件，{total_pass} 项断言")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
