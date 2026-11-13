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
    """Shared budget ledger for candidates, cloud calls, tokens, and wall time.

    The ledger file is runtime state and must never be committed; keep it under
    `.arc/` so the repository ignores it.
    """

    def __init__(
        self,
        limits: BudgetSpec,
        ledger_path: Optional[str] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.limits = limits
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
        }

    def can_run_experiment(self) -> bool:
        return self.remaining_candidates() > 0 and self.remaining_seconds() > 0

    def can_finalize(self) -> bool:
        return (
            self.remaining_seconds() > self.limits.reserve_seconds
            and self.remaining_tokens() > self.limits.reserve_tokens
        )

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

    def preflight_cloud(self, input_tokens: int, max_output_tokens: int) -> Dict[str, Any]:
        """Refuse a cloud call if it cannot be completed under the remaining budget."""
        if input_tokens < 0 or max_output_tokens < 0:
            raise ValueError("token estimates must be non-negative")
        needed = input_tokens + max_output_tokens
        remaining_tokens = self.remaining_tokens()
        remaining_seconds = self.remaining_seconds()
        available_after_reserve = remaining_tokens - self.limits.reserve_tokens
        if remaining_tokens <= 0:
            raise BudgetBlocked("cloud_tokens_exhausted")
        if available_after_reserve <= 0:
            raise BudgetBlocked("cloud_token_reserve_exhausted")
        if needed > remaining_tokens:
            raise BudgetBlocked(f"cloud_call_need_{needed}_remaining_{remaining_tokens}")
        if needed > available_after_reserve:
            raise BudgetBlocked(
                f"cloud_call_need_{needed}_available_after_reserve_{available_after_reserve}"
            )
        if remaining_seconds <= self.limits.reserve_seconds:
            raise BudgetBlocked("time_reserve_reached")
        return {
            "estimated_input": input_tokens,
            "max_output": max_output_tokens,
            "needed": needed,
            "remaining_tokens": remaining_tokens,
            "available_after_reserve": available_after_reserve,
            "remaining_seconds": remaining_seconds,
        }

    def record_cloud(
        self,
        *,
        estimated_input: int,
        actual_input: int,
        actual_output: int,
        decision_note: str = "",
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
            }
            self._entries.append(entry)
            used = self._used_cloud_tokens
            self._save()
        if used > self.limits.cloud_tokens:
            raise BudgetExceeded(used, self.limits.cloud_tokens)
        return entry

    def stop_reason(self) -> Optional[str]:
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
        }
