"""AutoResearch 研究策略自进化桥。

把 framework/core/evolution 的策略学习（elimination/UCB、闸门、版本提交）
接到 `ResearchPVPlanner`：每次 `interpret()` 产生的信号会写进 SQLite 策略库，
下一次 `plan()` 自动跳过已被反驳的假设；策略版本保留，可回退、可审计。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FRAMEWORK = _REPO_ROOT / "framework"
if str(_FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(_FRAMEWORK))

from core.evolution import learn, policy  # noqa: E402
from core.evolution.spec import Axis, DomainSpec, Hypothesis, Signal  # noqa: E402
from core.evolution.store import Store  # noqa: E402

SUBJECT = "autoresearch-research-pv"

# candidate_hid -> evolution axis
AXIS_FOR_HYPOTHESIS = {
    "volatility": "features",
    "volume": "features",
    "volatility_volume": "features",
    "tree_shallow": "model",
    "constant": "missing",
    "robust": "scaling",
    "momentum": "features",
    "cross": "features",
    "gradient_boost": "model",
    "random_forest": "model",
}

RESEARCH_SPEC = DomainSpec(
    name="autoresearch-research-pv",
    title="AutoResearch 量价研究策略",
    version="1.0",
    axes=[
        Axis(
            "features",
            "特征组",
            [
                Hypothesis("volatility", "+volatility"),
                Hypothesis("volume", "+volume"),
                Hypothesis("volatility_volume", "+volatility_volume"),
                Hypothesis("base_features", "固定基线特征"),
                Hypothesis("momentum", "+momentum"),
                Hypothesis("cross", "+cross"),
            ],
            default="volatility",
        ),
        Axis(
            "model",
            "模型设置",
            [
                Hypothesis("logistic", "LogisticRegression"),
                Hypothesis("tree_shallow", "浅层决策树"),
                Hypothesis("gradient_boost", "梯度提升树"),
                Hypothesis("random_forest", "随机森林"),
            ],
            default="logistic",
        ),
        Axis(
            "missing",
            "缺失处理",
            [
                Hypothesis("median", "中位数插补"),
                Hypothesis("constant", "常量填充"),
            ],
            default="median",
        ),
        Axis(
            "scaling",
            "缩放",
            [
                Hypothesis("standard", "StandardScaler"),
                Hypothesis("robust", "RobustScaler"),
            ],
            default="standard",
        ),
    ],
    learner="elimination",
)


def _signal_from_research(item: Dict[str, Any]) -> Optional[Signal]:
    if not isinstance(item, dict):
        return None
    axis = str(item.get("axis") or "")
    hypothesis = str(item.get("hypothesis_id") or "")
    if axis not in {"features", "model", "missing", "scaling"}:
        return None
    refutes = [hypothesis] if bool(item.get("refute")) else []
    supports = [hypothesis] if bool(item.get("support")) else []
    if not refutes and not supports:
        return None
    return Signal(
        axis=axis,
        reason=str(item.get("reason") or "observed_evidence"),
        refute=refutes,
        support=supports,
        source="observed",
        detail=str(item.get("detail") or ""),
    )


class ResearchIterationStore:
    """SQLite 持久化的研究策略库，给规划器提供学习后的候选顺序。"""

    def __init__(self, db_path: str | Path = ".arc/evolution/research-pv.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = Store(str(self.path))
        subject = self.store.subject(SUBJECT)
        self._subject_id = subject["id"]
        version = self.store.init_policy(self._subject_id, policy.build_initial(RESEARCH_SPEC))
        self._active_version_id = version["id"]

    def snapshot(self) -> Dict[str, Any]:
        return self.store.snapshot(self._active_version_id)

    def view(self) -> policy.View:
        return policy.View(self.snapshot())

    def hyp_state(self, hypothesis_id: str) -> Dict[str, Any]:
        axis = AXIS_FOR_HYPOTHESIS.get(hypothesis_id)
        if axis is None:
            return {"support": 0, "refute": 0, "excluded": False}
        return self.view().hyp_state(axis, hypothesis_id)

    def active_plan_order(self, order: List[Any]) -> List[Any]:
        """跳过已被反驳的候选，保留未试和支持过的候选。"""
        prioritized: List[Any] = []
        untested: List[Any] = []
        for item in order:
            hypothesis_id = str(item[0] if isinstance(item, (tuple, list)) else item)
            state = self.hyp_state(hypothesis_id)
            if state.get("excluded"):
                continue
            if int(state.get("support") or 0) > 0:
                prioritized.append(item)
            else:
                untested.append(item)
        return prioritized + untested

    def record_signals(self, research_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """把 research_pv.interpret() 的证据写入策略库，经过进化闸门后提交新版本。"""
        signals = [
            signal
            for signal in (_signal_from_research(item) for item in research_signals or [])
            if signal is not None
        ]
        if not signals:
            return {"ok": False, "note": "no usable evidence", "version_id": None}

        current = self.snapshot()
        candidate = learn.propose(current, signals, RESEARCH_SPEC)
        if not candidate.get("ok"):
            return {
                "ok": False,
                "note": str(candidate.get("note") or ""),
                "version_id": None,
            }

        accepted, reasons, info = learn.gate(
            candidate["snapshot"],
            current,
            RESEARCH_SPEC,
            signals,
        )
        if not accepted:
            return {
                "ok": False,
                "note": "; ".join(reasons),
                "info": info,
                "version_id": None,
            }

        version_id = self.store.commit_evolution(
            subject_id=self._subject_id,
            domain=RESEARCH_SPEC.name,
            snapshot=candidate["snapshot"],
            reason="evidence: " + ",".join(signal.reason for signal in signals),
            expected_active=self._active_version_id,
            event_type="policy_updated",
            payload={
                "axes_changed": candidate.get("axes_changed") or [],
                "signals": [signal.reason for signal in signals],
            },
        )
        self._active_version_id = version_id
        return {
            "ok": True,
            "version_id": version_id,
            "changes": candidate.get("changes") or [],
            "info": info,
        }

    def summary(self) -> Dict[str, Any]:
        return self.view().summary()

    def close(self) -> None:
        if hasattr(self, "store"):
            self.store.close()

    def __enter__(self) -> "ResearchIterationStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
