from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _import_arc_main():
    # Lazily imported: the serial engine pulls heavy deps we don't need on the
    # parallel dispatch path (and it lets us unit-test the pure helpers cheaply).
    from arc_main import main as arc_main

    return arc_main


def _is_shard() -> bool:
    return os.environ.get("ARCBENCH_SHARD") == "1"


def _type_flag_present(arguments: list) -> bool:
    return any(a in ("-t", "--type") for a in arguments)


def _run_serial(arguments: list) -> None:
    if not _type_flag_present(arguments):
        arguments.extend(["--type", os.environ.get("ARCBENCH_TASK_TYPE", "web")])
    sys.argv = [sys.argv[0], "compile", *arguments]
    _import_arc_main()()


def _parse_platform_args(args: list):
    """Extract the fields parallel_compile needs from the platform's CLI argv.

    Mirrors arc_main's `compile` subparser: first positional is requirement_path,
    `-o`/`--output-dir` is the workspace, `--port` the web port, `-t`/`--type` the
    app type, and `--clean` wipes the workspace first.
    """
    req = None
    out = None
    port = None
    atype = None
    clean = False
    i = 0
    n = len(args)
    while i < n:
        a = args[i]
        if a in ("-o", "--output-dir"):
            if i + 1 < n:
                out = args[i + 1]
            i += 2
            continue
        if a == "--port":
            if i + 1 < n:
                try:
                    port = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
            continue
        if a in ("-t", "--type"):
            if i + 1 < n:
                atype = args[i + 1]
            i += 2
            continue
        if a == "--clean":
            clean = True
            i += 1
            continue
        if a.startswith("-"):
            # skip this flag and an optional non-flag value
            if i + 1 < n and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if req is None:
            req = a
        i += 1
    return req, out, port, atype, clean


def _derive_base_port(req, port):
    """Pick a localhost base port for shards that never collides across tasks.

    * If the platform hands a port P (GitHub=3000, Sheet=3001), use P*10 so the
      two tasks land in disjoint ranges: 30000-30003 vs 30010-30013.
    * Otherwise derive a stable per-task base from a hash of the requirement
      path, aligned to a multiple of 4 in 40000-43996 so shard ranges never
      overlap between concurrent tasks.
    """
    if port:
        return port * 10
    import zlib

    return 40000 + (abs(zlib.crc32(str(req).encode("utf-8"))) % 1000) * 4


def _run_parallel(arguments: list) -> None:
    req, out, port, atype, clean = _parse_platform_args(arguments)
    if not req or not out:
        raise RuntimeError("parallel mode requires requirement_path and -o/--output-dir")

    atype = atype or os.environ.get("ARCBENCH_TASK_TYPE", "web")
    # Budget-aware shard count. MEASURED CORRECTION (2026-09-27):
    # An earlier revision forced 1 serial shard whenever the per-task budget was
    # <=26M, on the theory that 4 shards sharing 26M would each run out mid-leaf
    # and emit half-built leaves. That theory was FALSIFIED by a real run
    # (sheet.docx, 25 leaves, shard ports 43016/43017/43019 i.e. genuinely
    # parallel): TokenBudgetExceeded / token-budget markers triggered ZERO times,
    # so the 26M ceiling was never reached. Serial mode therefore bought nothing
    # and cost ~4x wall-clock per leaf.
    # Default is now 4 parallel shards again. We only fall back to serial when
    # the budget is genuinely tiny (<8M), and SALVAGE merging plus the leaf-bound
    # budget precheck still protect against half-built output if the ceiling is
    # ever hit. ARCBENCH_SHARDS always wins if explicitly set.
    try:
        from agents.model.token_budget import DEFAULT_PER_TASK_BUDGET as _DEFAULT_BUDGET

        _budget_limit = int(str(os.environ.get("ARC_TOKEN_BUDGET", "")).strip() or 0) or _DEFAULT_BUDGET
    except Exception:
        _budget_limit = 26_000_000
    _forced_shards = os.environ.get("ARCBENCH_SHARDS", "").strip()
    shards = int(_forced_shards) if _forced_shards else (1 if _budget_limit < 8_000_000 else 4)
    base_override = os.environ.get("ARCBENCH_BASE_PORT", "").strip()
    base = int(base_override) if base_override else _derive_base_port(req, port)

    # --clean semantics on the master workspace. parallel_compile rebuilds
    # backend/frontend from the template but does not delete stale pre-existing
    # directories, so wipe them when the platform asks for a clean run.
    if clean:
        for sub in ("backend", "frontend"):
            d = Path(out) / sub
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "parallel_compile.py"),
        req,
        "-o", out,
        "--shards", str(shards),
        "--base-port", str(base),
        "--type", atype,
    ]
    print(
        f"[parallel] dispatch shards={shards} base_port={base} type={atype} req={req} out={out}",
        flush=True,
    )
    rc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"parallel_compile exited with code {rc}")


def main() -> None:
    arguments = sys.argv[1:]

    # Recursion guard: a process spawned as a parallel shard must run the serial
    # engine, never re-enter parallel_compile.
    if _is_shard():
        _run_serial(arguments)
        return

    # v4 default: leaf-level parallelism via parallel_compile.py.
    # Disable with ARCBENCH_PARALLEL=0 (e.g. to debug a single leaf), or rely on
    # the automatic fallback below if dispatch fails.
    if os.environ.get("ARCBENCH_PARALLEL", "1") != "0":
        try:
            _run_parallel(arguments)
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(
                f"[warn] parallel dispatch failed ({exc}); falling back to serial engine",
                flush=True,
            )

    _run_serial(arguments)


if __name__ == "__main__":
    main()
