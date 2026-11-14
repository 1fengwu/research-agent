from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .budget_gate import BudgetBlocked, BudgetExceeded, BudgetGate

DIFFICULTY_THRESHOLD = 2.0


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


def build_full_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Baseline for cost accounting: what a naive full-context prompt would send."""
    return {
        "run_id": state.get("run_id"),
        "baseline": state.get("baseline"),
        "experiments": list(state.get("experiments", [])),
        "attempted": list(state.get("attempted", [])),
        "evidence_ids": list(state.get("evidence_ids", [])),
        "error_signatures": dict(state.get("error_signatures", {})),
        "budget": state.get("budget_snapshot"),
    }


def cache_key_for_decision(state: Dict[str, Any], summary: Dict[str, Any]) -> str:
    """Stable decision key; run_id/budget are excluded so results replay across runs."""
    last_trial = summary.get("last_trial") or {}
    if isinstance(last_trial, dict) and last_trial.get("truncated"):
        last_trial = {}
    body = {
        "phase": summary.get("phase"),
        "active_config": summary.get("active_config"),
        "evidence_ids": list(summary.get("evidence_ids", []) or []),
        "last_trial": last_trial,
        "conflicts": summary.get("conflicts"),
        "critical_decision": bool(state.get("critical_decision")),
        "data_anomaly": bool(state.get("data_anomaly")),
        "pending_candidates": summary.get("pending_candidates", []),
    }
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


MAX_OUTPUT_BY_TRIGGER: Dict[str, int] = {
    "finalize": 220,
    "critical_decision": 180,
    "planning_stall": 180,
    "repeated_error_signature": 160,
    "window_conflict": 140,
    "data_anomaly": 140,
}


def output_cap_for(reasons: List[str]) -> int:
    """Phase-aware cap: planning/finalization never pays for verbose output."""
    return MAX_OUTPUT_BY_TRIGGER.get(str(reasons[0]) if reasons else "", 120)


class DecisionCache:
    """Bounded persistent cache of validated cloud decisions.

    A hit emits ``research.escalation_cached`` and charges zero cloud tokens.
    The file lives under ``.arc/cache/`` and is runtime state, not committed.
    """

    def __init__(self, path: Optional[str] = None, max_entries: int = 256) -> None:
        self.path = Path(path) if path else None
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            entries = data.get("entries")
            if isinstance(entries, dict):
                self._entries = {k: v for k, v in entries.items() if isinstance(v, dict)}
        except Exception:
            self._entries = {}

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": self._entries,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._entries.get(key)

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = payload
            if len(self._entries) > self.max_entries:
                for stale in list(self._entries)[: len(self._entries) - self.max_entries]:
                    self._entries.pop(stale, None)
            self._save()

    def size(self) -> int:
        with self._lock:
            return len(self._entries)


def local_rule_decision(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Deterministic zero-token fallback used when budget pressure is on.

    These are decisions the PRD already makes: a window conflict moves to the
    next hypothesis, and a repeated experiment error falls back to baseline.
    A cloud call under pressure would only add cost, not a new decision.
    """
    if bool(state.get("window_conflict")):
        return {
            "decision": "continue_with_next_hypothesis",
            "evidence_ids": list(state.get("evidence_ids", []) or []),
            "actions": [{"action": "replan", "reason": "window_conflict_rule"}],
            "note": "local rule: window conflict -> next candidate without cloud",
        }
    if bool(state.get("repeated_error_signature")):
        signatures = state.get("error_signatures") or {}
        error_signature = next(
            (
                key
                for key in sorted(signatures)
                if "error" in str(key).lower() or "failed" in str(key).lower()
            ),
            None,
        )
        if error_signature is not None:
            return {
                "decision": "fallback_to_baseline",
                "evidence_ids": list(state.get("evidence_ids", []) or []),
                "actions": [
                    {"action": "fallback_baseline", "reason": "repeated_error_rule"}
                ],
                "note": "local rule: repeated %s -> baseline without cloud" % error_signature,
            }
    return None


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
        pass
    decoder = json.JSONDecoder()
    for index in range(len(text)):
        if text[index] != "{":
            continue
        try:
            return decoder.raw_decode(text, index)[0]
        except Exception:
            continue
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


def difficulty_score(state: Dict[str, Any]) -> float:
    """根据执行反馈与剩余状态估算本轮任务难度。"""
    score = 0.0
    if bool(state.get("data_anomaly")):
        score += 3.0
    if int(state.get("stall_rounds", 0) or 0) >= 2:
        score += 2.0
    if bool(state.get("window_conflict")):
        score += 2.0
    if bool(state.get("critical_decision")):
        score += 3.0
    if bool(state.get("repeated_error_signature")):
        score += 2.0
    trial = state.get("last_trial")
    if not isinstance(trial, dict):
        baseline = state.get("baseline")
        trial = baseline.get("trial") if isinstance(baseline, dict) else {}
    missing = trial.get("missing_rates") if isinstance(trial, dict) else {}
    if isinstance(missing, dict) and missing:
        if max(float(value or 0.0) for value in missing.values()) >= 0.2:
            score += 1.0
    return min(10.0, score)


def should_escalate(state: Dict[str, Any], mode: str = "normal") -> List[str]:
    """Structural triggers filtered by budget pressure.

    Normal mode keeps every structural trigger. Economy mode only spends on an
    escalation that can unblock or finalize the loop. Critical mode keeps only
    decisions that affect the final outcome, so the run still completes under
    a tight budget.
    """
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
    if difficulty_score(state) >= DIFFICULTY_THRESHOLD:
        reasons.append("difficulty_high")
    if mode == "normal":
        return reasons
    if mode == "economy":
        keep = {
            "planning_stall",
            "repeated_error_signature",
            "critical_decision",
            "finalize",
            "difficulty_high",
        }
        return [reason for reason in reasons if reason in keep]
    keep = {"critical_decision", "finalize"}
    return [reason for reason in reasons if reason in keep]



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
        "pending_candidates": state.get("pending_candidates", []),
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


STRUCTURED_REPLY_PROMPT = (
    "You are a research-agent planner. "
    "Reply with a single JSON object only, no markdown fences and no prose. "
    'The object must contain "decision" (string), "evidence_ids" (non-empty list of strings), '
    '"actions" (non-empty list of objects). Use exactly one action. '
    'Use {"action":"select_candidate","target":"<pending candidate id>"} to choose '
    'the next experiment, {"action":"continue"} to keep the current order, or '
    '{"action":"replan"} to recover when planning failed and no candidates are listed. '
    'Keep the decision brief and cite evidence IDs from the input when available.'
)


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
            "messages": [
                {"role": "system", "content": STRUCTURED_REPLY_PROMPT},
                {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
            ],
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
            choice = parsed["choices"][0]
            content = str(choice["message"]["content"] or "")
            usage = parsed.get("usage") or {}
            return {
                "content": content,
                "usage": usage,
                "finish_reason": str(choice.get("finish_reason") or ""),
            }
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
        cache: Optional[DecisionCache] = None,
        cache_path: Optional[str] = None,
    ) -> None:
        self.ledger = ledger
        self.client = client
        self.event_cb = event_cb
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.cache = cache or DecisionCache(cache_path)
        self.cache_hits = 0
        self.cache_saved_tokens = 0
        self.local_rule_decisions = 0
        self.local_rule_saved_tokens = 0
        self._decision_count = 0
        self._baseline_input = 0
        self._compact_input = 0

    def escalate_if_needed(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.ledger.budget_mode()
        phase = str(state.get("phase") or "UNSEEN")
        reasons = should_escalate(state, mode=mode)
        raw_reasons = should_escalate(state) if mode != "normal" else reasons

        if not reasons and not raw_reasons:
            return {
                "escalated": False,
                "valid": None,
                "reasons": [],
                "tokens": None,
                "mode": mode,
            }

        summary = build_compact_summary(state)
        input_est = estimate_tokens(summary)
        if mode != "normal":
            self._emit(
                "research.budget_warning",
                phase,
                f"budget_mode:{mode}",
                extra={"mode": mode},
            )

        decision_key = cache_key_for_decision(state, summary)
        baseline_input = estimate_tokens(build_full_context(state))
        compact_input = input_est
        self._decision_count += 1
        self._baseline_input += baseline_input
        self._compact_input += compact_input
        decision_cap = output_cap_for(raw_reasons or reasons)

        if mode in {"economy", "critical", "overrun"}:
            local_reply = local_rule_decision(state)
            if local_reply is not None:
                saved = compact_input + decision_cap
                self.local_rule_decisions += 1
                self.local_rule_saved_tokens += saved
                self._emit(
                    "research.local_rule",
                    phase,
                    f"local_rule:{local_reply['decision']}",
                    extra={
                        "trigger": raw_reasons or reasons,
                        "mode": mode,
                        "saved_tokens_estimate": saved,
                    },
                )
                return {
                    "escalated": True,
                    "valid": True,
                    "reply": local_reply,
                    "reasons": raw_reasons or reasons,
                    "tokens": None,
                    "mode": mode,
                    "source": "local",
                    "saved_tokens_estimate": saved,
                }

        cached = self.cache.get(decision_key)
        if cached is not None and isinstance(cached.get("reply"), dict):
            cached_tokens = int(cached.get("charged") or 0)
            if cached_tokens <= 0:
                cached_tokens = compact_input + decision_cap
            self.cache_hits += 1
            self.cache_saved_tokens += cached_tokens
            self._emit(
                "research.escalation_cached",
                phase,
                f"cached:{cached['reply'].get('decision')}",
                extra={
                    "trigger": raw_reasons or reasons,
                    "mode": mode,
                    "saved_tokens_estimate": cached_tokens,
                },
            )
            return {
                "escalated": True,
                "valid": True,
                "reply": dict(cached["reply"]),
                "reasons": raw_reasons or reasons,
                "tokens": None,
                "mode": mode,
                "source": "cache",
                "saved_tokens_estimate": cached_tokens,
            }

        if not reasons:
            note = f"budget_blocked:budget_mode:{mode}"
            self._emit(
                "research.escalation_blocked",
                phase,
                note,
                extra={"trigger": raw_reasons, "mode": mode},
            )
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": note,
                "reasons": raw_reasons,
                "tokens": None,
                "mode": mode,
            }

        # Critical work may cross a soft target so the loop can still finish;
        # the overrun is metered and shown instead of stopping the research.
        allow_overrun = (
            not self.ledger.hard
            and mode in {"economy", "critical", "overrun"}
            and any(
                trigger in reasons
                for trigger in (
                    "critical_decision",
                    "finalize",
                    "planning_stall",
                    "repeated_error_signature",
                )
            )
        )
        cloud_cap = output_cap_for(reasons)
        if allow_overrun:
            max_out = min(self.max_output_tokens, cloud_cap)
        else:
            usable = self.ledger.remaining_tokens() - self.ledger.limits.reserve_tokens
            max_out = (
                max(0, min(self.max_output_tokens, cloud_cap, usable - input_est))
                if usable > 0
                else 0
            )
        if max_out <= 0 and not allow_overrun:
            note = (
                "cloud_token_reserve_exhausted"
                if self.ledger.remaining_tokens() <= self.ledger.limits.reserve_tokens
                else "cloud_tokens_exhausted"
            )
            self._emit(
                "research.escalation_blocked",
                phase,
                note,
                extra={"trigger": reasons, "mode": mode},
            )
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": note,
                "reasons": reasons,
                "tokens": None,
                "mode": mode,
            }

        try:
            self.ledger.preflight_cloud(input_est, max_out, allow_overrun=allow_overrun)
        except BudgetBlocked as exc:
            note = f"budget_blocked:{exc.reason}"
            self._emit(
                "research.escalation_blocked",
                phase,
                note,
                extra={"trigger": reasons, "mode": mode},
            )
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": exc.reason,
                "reasons": reasons,
                "tokens": None,
                "mode": mode,
            }

        if self.client is None:
            note = "cloud_client_missing"
            self._emit(
                "research.escalation_blocked",
                phase,
                note,
                extra={"trigger": reasons, "mode": mode},
            )
            return {
                "escalated": True,
                "valid": False,
                "blocked_reason": note,
                "reasons": reasons,
                "tokens": None,
                "mode": mode,
            }

        try:
            raw = self.client.call(summary, max_tokens=max_out, timeout_s=self.timeout_s)
        except Exception as exc:
            note = f"cloud_call_failed:{exc}"
            self._emit(
                "research.escalation_failed",
                phase,
                note,
                extra={"trigger": reasons, "mode": mode},
            )
            return {
                "escalated": True,
                "valid": False,
                "error": str(exc),
                "reasons": reasons,
                "tokens": None,
                "mode": mode,
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
            entry = self.ledger.record_cloud(
                estimated_input=input_est,
                actual_input=actual_in,
                actual_output=actual_out,
                decision_note=f"trigger={reasons[0]}",
                overrun=allow_overrun,
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
                "mode": mode,
            }
        if entry.get("overrun"):
            self._emit(
                "research.budget_overrun",
                phase,
                "cloud_budget_overrun",
                tokens=tokens,
                extra={"trigger": reasons, "mode": self.ledger.budget_mode()},
            )

        parsed = parse_json_content(content)
        ok, invalid_reason, reply = validate_reply(parsed)
        if not ok:
            self._emit(
                "research.escalation_invalid",
                phase,
                invalid_reason,
                tokens=tokens,
                extra={
                    "trigger": reasons,
                    "reply_preview": str(content)[:300],
                    "finish_reason": str(raw.get("finish_reason") or ""),
                    "content_length": len(content),
                },
            )
            return {
                "escalated": True,
                "valid": False,
                "invalid_reason": invalid_reason,
                "reasons": reasons,
                "tokens": tokens,
                "mode": mode,
            }

        self.cache.set(
            decision_key,
            {
                "reply": reply,
                "charged": int(tokens["charged"]),
                "input": int(tokens["actual_input"]),
                "output": int(tokens["actual_output"]),
                "ts": datetime.now(timezone.utc).isoformat(),
            },
        )
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
            "mode": mode,
            "source": "cloud",
            "saved_tokens_estimate": max(0, baseline_input - compact_input),
        }

    def savings_summary(self) -> Dict[str, Any]:
        """Measured token savings for the run; D renders this in the report/UI."""
        prompt_compression_saved = max(0, self._baseline_input - self._compact_input)
        return {
            "compared_decisions": self._decision_count,
            "prompt_baseline_tokens": self._baseline_input,
            "prompt_compact_tokens": self._compact_input,
            "prompt_compression_saved": prompt_compression_saved,
            "cache_hits": self.cache_hits,
            "cache_saved_tokens": self.cache_saved_tokens,
            "local_rule_decisions": self.local_rule_decisions,
            "local_rule_saved_tokens": self.local_rule_saved_tokens,
            "cloud_calls_avoided": self.cache_hits + self.local_rule_decisions,
            "total_estimated_saved": (
                prompt_compression_saved
                + self.cache_saved_tokens
                + self.local_rule_saved_tokens
            ),
            "cloud_calls_charged": int(
                (self.ledger.usage_ledger() or {}).get("used_cloud_calls") or 0
            ),
            "cloud_tokens_charged": int(
                (self.ledger.usage_ledger() or {}).get("used_cloud_tokens") or 0
            ),
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
