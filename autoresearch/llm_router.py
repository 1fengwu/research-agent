from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .budget_gate import BudgetBlocked, BudgetExceeded, BudgetGate


def _tokenize_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_env_file(env_file: str | Path) -> None:
    """读仓库根目录 `.env`，KEY=VALUE，不覆盖已有环境变量。"""
    path = Path(env_file)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, _tokenize_env_value(value))


def _load_dotenv() -> None:
    _load_env_file(Path(__file__).resolve().parents[1] / ".env")


_load_dotenv()


@runtime_checkable
class CloudClient(Protocol):
    def call(
        self,
        summary: Dict[str, Any],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        """Return {"content": str, "usage": {...}} or raise."""
        ...


def estimate_tokens(value: Any) -> int:
    """Cheap token estimate without pulling a tokenizer into the hot path."""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, math.ceil(len(text) / 4))


def parse_json_content(content: str) -> Any:
    text = str(content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        return None


def validate_reply(reply: Any) -> Tuple[bool, str, Dict[str, Any]]:
    """A cloud model may only steer via a structured, evidence-citing reply."""
    if not isinstance(reply, dict):
        return False, "reply must be an object", {}
    decision = str(reply.get("decision") or "").strip()
    evidence = reply.get("evidence_ids")
    actions = reply.get("actions")
    if not decision:
        return False, "reply.decision is required", {}
    if not isinstance(evidence, list) or not evidence:
        return False, "reply.evidence_ids must be a non-empty list", {}
    if not isinstance(actions, list) or not actions:
        return False, "reply.actions must be a non-empty list", {}
    if not all(isinstance(action, dict) for action in actions):
        return False, "reply.actions must be a list of objects", {}
    normalized = {
        "decision": decision,
        "evidence_ids": [str(x).strip() for x in evidence if str(x).strip()],
        "actions": [dict(action) for action in actions if isinstance(action, dict)],
        "note": str(reply.get("note") or "").strip(),
    }
    return True, "", normalized


def should_escalate(state: Dict[str, Any]) -> List[str]:
    """Structural triggers only; a rule path handles everything else."""
    reasons: List[str] = []
    if bool(state.get("data_anomaly")):
        reasons.append("data_anomaly")
    if int(state.get("stall_rounds", 0) or 0) >= 2:
        reasons.append("planning_stall")
    if bool(state.get("window_conflict")):
        reasons.append("window_conflict")
    if bool(state.get("critical_decision")):
        reasons.append("critical_decision")
    if bool(state.get("repeated_error_signature")):
        reasons.append("repeated_error_signature")
    if str(state.get("phase") or "").upper() == "WRITE_REPORT":
        reasons.append("finalize")
    return reasons


def build_compact_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the cloud prompt small: evidence ids and the current decision only."""
    conflicts = {
        "stall_rounds": int(state.get("stall_rounds", 0) or 0),
        "window_conflict": bool(state.get("window_conflict")),
        "repeated_error_signature": bool(state.get("repeated_error_signature")),
    }
    return {
        "run_id": state.get("run_id"),
        "phase": state.get("phase"),
        "active_config": state.get("active_config"),
        "evidence_ids": list(state.get("evidence_ids", [])),
        "last_trial": state.get("last_trial"),
        "conflicts": conflicts,
        "budget": state.get("budget_snapshot"),
    }


def apply_reply(state: Dict[str, Any], reply: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(state)
    updated["phase"] = "REVISED"
    updated["plan_hint"] = str(reply.get("decision") or "").strip()
    updated["evidence_ids"] = list(reply.get("evidence_ids", state.get("evidence_ids", [])))
    updated["actions"] = list(reply.get("actions", []))
    updated["last_escalation"] = reply.get("decision")
    return updated


class HttpJsonCloudClient:
    """Zero-dependency OpenAI-compatible client using the standard library."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("ARC_LLM_BASE_URL", "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("ARC_LLM_API_KEY", "").strip()
        self.model = model or os.environ.get("ARC_LLM_MODEL", "").strip()

    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def call(
        self,
        summary: Dict[str, Any],
        *,
        max_tokens: int,
        timeout_s: float,
    ) -> Dict[str, Any]:
        if not self.configured():
            raise RuntimeError("LLM client not configured (ARC_LLM_BASE_URL/API_KEY/MODEL)")
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": json.dumps(summary, ensure_ascii=False)}],
            "max_tokens": int(max_tokens),
        }
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urlopen(request, timeout=float(timeout_s)) as response:
                body = response.read().decode("utf-8", "replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"cloud call failed: {exc}") from exc
        try:
            parsed = json.loads(body)
            content = str(parsed["choices"][0]["message"]["content"] or "")
            usage = parsed.get("usage") or {}
            return {"content": content, "usage": usage}
        except Exception as exc:
            raise RuntimeError(f"cloud response not parseable: {exc}") from exc


class NullCloudClient:
    """Offline stand-in for tests and dry runs; never used for a real call."""

    def __init__(self, reply: Optional[Dict[str, Any]] = None) -> None:
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
        self.calls += 1
        return {
            "content": json.dumps(self.reply, ensure_ascii=False),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


EventCallback = Callable[
    [str, str, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]],
    None,
]


class ResearchRouter:
    """A-only cloud gate: decide, estimate, preflight, call, validate, record."""

    def __init__(
        self,
        ledger: BudgetGate,
        client: Optional[CloudClient] = None,
        event_cb: Optional[EventCallback] = None,
        *,
        max_output_tokens: int = 800,
        timeout_s: float = 30.0,
    ) -> None:
        self.ledger = ledger
        self.client = client
        self.event_cb = event_cb
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s

    def escalate_if_needed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        reasons = should_escalate(state)
        if not reasons:
            return {"escalated": False, "valid": None, "reasons": [], "tokens": None}

        phase = str(state.get("phase") or "UNSEEN")
        summary = build_compact_summary(state)
        input_est = estimate_tokens(summary)
        usable = self.ledger.remaining_tokens() - self.ledger.limits.reserve_tokens
        max_out = (
            max(0, min(self.max_output_tokens, usable - input_est))
            if usable > 0
            else 0
        )
        if max_out <= 0:
            note = (
                "cloud_token_reserve_exhausted"
                if usable <= 0
                else "cloud_tokens_exhausted"
            )
            self._emit("research.escalation_blocked", phase, note, extra={"trigger": reasons})
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": note,
                "reasons": reasons,
                "tokens": None,
            }

        try:
            self.ledger.preflight_cloud(input_est, max_out)
        except BudgetBlocked as exc:
            note = f"budget_blocked:{exc.reason}"
            self._emit("research.escalation_blocked", phase, note, extra={"trigger": reasons})
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": exc.reason,
                "reasons": reasons,
                "tokens": None,
            }

        if self.client is None:
            note = "cloud_client_missing"
            self._emit("research.escalation_blocked", phase, note, extra={"trigger": reasons})
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": note,
                "reasons": reasons,
                "tokens": None,
            }

        try:
            raw = self.client.call(summary, max_tokens=max_out, timeout_s=self.timeout_s)
        except Exception as exc:
            note = f"cloud_call_failed:{exc}"
            self._emit("research.escalation_failed", phase, note, extra={"trigger": reasons})
            return {
                "escalated": True,
                "valid": False,
                "error": str(exc),
                "reasons": reasons,
                "tokens": None,
            }

        content = str(raw.get("content") or raw.get("text") or "")
        usage = raw.get("usage") or {}
        try:
            actual_in = int(usage.get("prompt_tokens", usage.get("input_tokens") or 0))
            actual_out = int(usage.get("completion_tokens", usage.get("output_tokens") or 0))
        except (TypeError, ValueError):
            actual_in, actual_out = input_est, estimate_tokens(content)
        if actual_in <= 0:
            actual_in = input_est
        if actual_out <= 0:
            actual_out = estimate_tokens(content)
        tokens = {
            "estimated_input": input_est,
            "actual_input": actual_in,
            "actual_output": actual_out,
            "charged": int(actual_in) + int(actual_out),
        }
        try:
            self.ledger.record_cloud(
                estimated_input=input_est,
                actual_input=actual_in,
                actual_output=actual_out,
                decision_note=f"trigger={reasons[0]}",
            )
        except BudgetExceeded as exc:
            note = f"budget_exceeded:{exc}"
            self._emit("research.escalation_failed", phase, note, tokens=tokens)
            return {
                "escalated": True,
                "valid": False,
                "error": str(exc),
                "reasons": reasons,
                "tokens": tokens,
            }

        parsed = parse_json_content(content)
        ok, invalid_reason, reply = validate_reply(parsed)
        if not ok:
            self._emit(
                "research.escalation_invalid",
                phase,
                invalid_reason,
                tokens=tokens,
                extra={"trigger": reasons},
            )
            return {
                "escalated": True,
                "valid": False,
                "invalid_reason": invalid_reason,
                "reasons": reasons,
                "tokens": tokens,
            }

        self._emit(
            "research.escalation",
            phase,
            f"accepted:{reply['decision']}",
            tokens=tokens,
            extra={
                "trigger": reasons,
                "evidence_ids": reply["evidence_ids"],
                "actions": reply["actions"],
            },
        )
        return {
            "escalated": True,
            "valid": True,
            "reply": reply,
            "reasons": reasons,
            "tokens": tokens,
        }

    def _emit(
        self,
        kind: str,
        phase: str,
        reason: str,
        *,
        tokens: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_cb is not None:
            self.event_cb(kind, phase, reason, tokens, extra)
