from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .contracts import BudgetSpec


class BudgetBlocked(Exception):
    """Raised when a cloud call must be skipped before it starts."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class BudgetExceeded(Exception):
    """Raised when usage crosses the configured hard limit."""

    def __init__(self, used: int, limit: int) -> None:
        self.used = used
        self.limit = limit
        super().__init__(f"cloud tokens used={used} limit={limit}")


class BudgetGate:
    """Meter plus soft governor for candidates, cloud calls, tokens, and time.

    The ledger file is runtime state and must never be committed; keep it under
    `.arc/` so the repository ignores it. Soft semantics: limits steer routing
    and are metered, but never stop a run before it can finish. Set
    ``hard_limits`` to True to restore the original PRD hard-stop behavior.
    """

    def __init__(
        self,
        limits: BudgetSpec,
        ledger_path: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
        *,
        hard: bool = False,
    ) -> None:
        self.limits = limits
        self.hard = hard or bool(getattr(limits, "hard_limits", False))
        self.path = Path(ledger_path) if ledger_path else None
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._started = self._clock()
        self._used_candidates = 0
        self._used_cloud_calls = 0
        self._used_cloud_tokens = 0
        self._entries: list[Dict[str, Any]] = []
        self._load()

    def reset(self, path: Optional[str] = None) -> None:
        """Start a fresh per-run ledger at an optional path."""
        with self._lock:
            self.path = Path(path) if path else self.path
            self._started = self._clock()
            self._used_candidates = 0
            self._used_cloud_calls = 0
            self._used_cloud_tokens = 0
            self._entries = []
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._save()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._used_candidates = int(data.get("used_candidates", 0))
            self._used_cloud_calls = int(data.get("used_cloud_calls", 0))
            self._used_cloud_tokens = int(data.get("used_cloud_tokens", 0))
            entries = data.get("entries")
            if isinstance(entries, list):
                self._entries = [e for e in entries if isinstance(e, dict)]
        except Exception:
            # A broken runtime ledger should not block the demo; keep zero state.
            self._used_candidates = 0
            self._used_cloud_calls = 0
            self._used_cloud_tokens = 0
            self._entries = []

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "used_candidates": self._used_candidates,
            "used_cloud_calls": self._used_cloud_calls,
            "used_cloud_tokens": self._used_cloud_tokens,
            "entries": self._entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    def remaining_candidates(self) -> int:
        return max(0, self.limits.candidates - self._used_candidates)

    def remaining_cloud_calls(self) -> int:
        return max(0, self.limits.cloud_calls - self._used_cloud_calls)

    def remaining_tokens(self) -> int:
        return max(0, self.limits.cloud_tokens - self._used_cloud_tokens)

    def remaining_seconds(self) -> float:
        return max(0.0, float(self.limits.wall_seconds) - self.elapsed_seconds())

    def budget_mode(self) -> str:
        """normal / economy / critical / overrun for routing and display."""
        reserve_seconds = max(0, self.limits.reserve_seconds)
        reserve_tokens = max(0, self.limits.reserve_tokens)
        if self.overrun():
            return "overrun"
        if (
            self.remaining_seconds() <= 0
            or self.remaining_tokens() <= 0
            or (reserve_seconds > 0 and self.remaining_seconds() <= reserve_seconds)
            or (reserve_tokens > 0 and self.remaining_tokens() <= reserve_tokens)
            or (self.limits.cloud_calls > 0 and self.remaining_cloud_calls() <= 0)
        ):
            return "critical"
        if (
            (reserve_seconds > 0 and self.remaining_seconds() <= reserve_seconds * 2)
            or (reserve_tokens > 0 and self.remaining_tokens() <= reserve_tokens * 2)
            or (
                self.limits.cloud_calls > 0
                and self.remaining_cloud_calls() <= max(1, self.limits.cloud_calls // 2)
            )
        ):
            return "economy"
        return "normal"

    def overrun(self) -> bool:
        return self._used_cloud_tokens > self.limits.cloud_tokens or (
            self.limits.cloud_calls > 0 and self._used_cloud_calls > self.limits.cloud_calls
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "used_candidates": self._used_candidates,
            "remaining_candidates": self.remaining_candidates(),
            "used_cloud_calls": self._used_cloud_calls,
            "remaining_cloud_calls": self.remaining_cloud_calls(),
            "used_cloud_tokens": self._used_cloud_tokens,
            "remaining_cloud_tokens": self.remaining_tokens(),
            "elapsed_seconds": round(self.elapsed_seconds(), 3),
            "remaining_seconds": round(self.remaining_seconds(), 3),
            "mode": self.budget_mode(),
            "hard_limits": self.hard,
            "overrun": self.overrun(),
        }

    def can_run_experiment(self) -> bool:
        # Soft mode: candidate count is a planner target, not a stop switch.
        if self.hard:
            return self.remaining_candidates() > 0 and self.remaining_seconds() > 0
        # Wall time is a demo guard: when it is gone, stop starting candidates
        # and go straight to finalization.
        return self.remaining_seconds() > 0

    def can_finalize(self) -> bool:
        """Soft mode always allows finalization so a run can complete."""
        return self.finalize_status()["allow"]

    def finalize_status(self) -> Dict[str, Any]:
        mode = self.budget_mode()
        if self.hard:
            allow = (
                self.remaining_seconds() > self.limits.reserve_seconds
                and self.remaining_tokens() > self.limits.reserve_tokens
            )
            return {
                "allow": allow,
                "mode": mode,
                "warning": None if allow else "hard_limits_prevent_finalize",
            }
        warning = {
            "overrun": "cloud_budget_overrun",
            "critical": "cloud_budget_critical_but_finalizing",
            "economy": "cloud_budget_economy",
        }.get(mode)
        return {"allow": True, "mode": mode, "warning": warning}

    def record_experiment(self, note: str = "") -> Dict[str, Any]:
        with self._lock:
            self._used_candidates += 1
            entry = {
                "kind": "candidate",
                "ts": datetime.now(timezone.utc).isoformat(),
                "note": note,
                "snapshot": self.snapshot(),
            }
            self._entries.append(entry)
            self._save()
            return entry

    def preflight_cloud(
        self,
        input_tokens: int,
        max_output_tokens: int,
        *,
        allow_overrun: bool = False,
    ) -> Dict[str, Any]:
        """Decide whether one cloud call is allowed; never stops the run.

        Soft mode may breach the reserve to keep a critical decision on track;
        the UI sees the pressure through ``mode``/``overrun``. Hard mode keeps
        the original PRD reserve checks and raises BudgetBlocked.
        """
        if input_tokens < 0 or max_output_tokens < 0:
            raise ValueError("token estimates must be non-negative")
        needed = input_tokens + max_output_tokens
        remaining_tokens = self.remaining_tokens()
        remaining_seconds = self.remaining_seconds()
        mode = self.budget_mode()
        if remaining_seconds <= 0:
            raise BudgetBlocked("time_exhausted")
        available_after_reserve = remaining_tokens - self.limits.reserve_tokens
        if self.hard:
            if (
                remaining_seconds <= self.limits.reserve_seconds
                or available_after_reserve <= 0
                or needed > available_after_reserve
            ):
                raise BudgetBlocked(
                    f"cloud_call_need_{needed}_available_after_reserve_"
                    f"{max(0, available_after_reserve)}"
                )
        elif needed > remaining_tokens and not allow_overrun:
            raise BudgetBlocked(f"cloud_call_need_{needed}_remaining_{remaining_tokens}")
        overrun = (not self.hard) and allow_overrun and needed > remaining_tokens
        return {
            "allowed": True,
            "overrun": bool(overrun),
            "mode": mode,
            "estimated_input": input_tokens,
            "max_output": max_output_tokens,
            "needed": needed,
            "remaining_tokens": remaining_tokens,
            "remaining_seconds": remaining_seconds,
            "reserve_breached": needed > available_after_reserve,
        }

    def record_cloud(
        self,
        *,
        estimated_input: int,
        actual_input: int,
        actual_output: int,
        decision_note: str = "",
        overrun: bool = False,
    ) -> Dict[str, Any]:
        charged = int(actual_input) + int(actual_output)
        with self._lock:
            self._used_cloud_calls += 1
            self._used_cloud_tokens += charged
            entry = {
                "kind": "cloud_call",
                "ts": datetime.now(timezone.utc).isoformat(),
                "estimated_input": int(estimated_input),
                "actual_input": int(actual_input),
                "actual_output": int(actual_output),
                "charged": charged,
                "decision_note": decision_note,
                "mode": self.budget_mode(),
                "overrun": bool(overrun or self.overrun()),
            }
            self._entries.append(entry)
            used = self._used_cloud_tokens
            self._save()
        if self.hard and used > self.limits.cloud_tokens:
            raise BudgetExceeded(used, self.limits.cloud_tokens)
        return entry

    def stop_reason(self) -> Optional[str]:
        """Return a stop reason only in hard mode; soft mode never stops here.

        The orchestrator owns run completion: soft budgets are metered and can
        be overrun, so the loop finishes and the final test runs even when
        budget pressure is high.
        """
        if not self.hard:
            return None
        if self._used_candidates >= self.limits.candidates:
            return "candidate_budget"
        if self.limits.cloud_calls > 0 and self._used_cloud_calls >= self.limits.cloud_calls:
            return "cloud_call_budget"
        if self._used_cloud_tokens >= self.limits.cloud_tokens:
            return "cloud_tokens_budget"
        if self.elapsed_seconds() >= self.limits.wall_seconds:
            return "time_budget"
        return None

    def usage_ledger(self) -> Dict[str, Any]:
        return {
            "entries": list(self._entries),
            "used_candidates": self._used_candidates,
            "used_cloud_calls": self._used_cloud_calls,
            "used_cloud_tokens": self._used_cloud_tokens,
            "mode": self.budget_mode(),
            "hard_limits": self.hard,
            "overrun": self.overrun(),
        }
