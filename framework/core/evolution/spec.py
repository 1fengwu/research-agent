"""领域契约：新领域只写一个 domain 包，不改框架代码。

一个领域包要交的东西只有四样：

1. `spec` —— 轴（axis）、假设（hypothesis）、学习参数；
2. `plan(view, run_kind)` —— 按当前策略生成任务，并**必须**给出预测；
3. `interpret(feedback, view)` —— 把人的反馈翻译成证据（Signal）；
4. `predict(view, payload)` —— 给定策略快照，这条任务会被预测成 high 还是 low（闸门回放用）。

`judge(task, attempt)` 可选：真实业务用它判分；不实现则调用方直接上报 done/reason。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

PREDICTIONS = ("high", "low")

DEFAULT_LEARN = {
    # 人明说的：一次就信（refute 2 即达淘汰线）
    "stated_refute": 2,
    "stated_support": 3,
    # 系统自己推断的：要两次
    "observed_refute": 1,
    "observed_support": 1,
    # 单次版本最多改动几个轴（可解释、可回退）
    "max_axis_changes": 3,
    # 闸门回放所需的最小历史样本数
    "min_replay_samples": 3,
}


@dataclass(frozen=True)
class Hypothesis:
    """一条可被证伪的假设，例如「晚间能坚持运动」。"""

    id: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Axis:
    """一个可学习维度，例如「运动时段」；同时只有一条假设生效。"""

    id: str
    question: str
    hypotheses: list[Hypothesis]
    default: str
    eliminate_refutes: int = 2

    def __post_init__(self) -> None:
        ids = [h.id for h in self.hypotheses]
        if not ids:
            raise ValueError(f"axis {self.id} 没有假设")
        if len(set(ids)) != len(ids):
            raise ValueError(f"axis {self.id} 假设 id 重复")
        if self.default not in ids:
            raise ValueError(f"axis {self.id} 的默认假设 {self.default} 不存在")

    def hyp(self, hid: str) -> Hypothesis:
        for h in self.hypotheses:
            if h.id == hid:
                return h
        raise KeyError(f"axis {self.id} 没有假设 {hid}")

    @property
    def ids(self) -> list[str]:
        return [h.id for h in self.hypotheses]

    @property
    def priority(self) -> list[str]:
        """淘汰后选谁：按声明顺序取第一个未被淘汰的。"""
        return self.ids


@dataclass
class Signal:
    """一条证据：在某轴上加/减某个假设的分。"""

    axis: str
    reason: str
    refute: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    source: str = "observed"  # observed | stated
    day: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.source not in ("observed", "stated"):
            raise ValueError(f"未知来源 {self.source}")


@dataclass
class Task:
    """一条计划任务，附带**事前预测**（预测先落库，无法事后编造）。"""

    key: str
    title: str
    axis: str
    hypothesis: str
    prediction: str
    prediction_reason: str
    payload: dict[str, Any] = field(default_factory=dict)
    kind: str = "default"

    def __post_init__(self) -> None:
        if self.prediction not in PREDICTIONS:
            raise ValueError(f"prediction 必须是 {PREDICTIONS}")


@dataclass
class Outcome:
    done: bool
    reason: str = "ok"
    note: str = ""


@dataclass
class DomainSpec:
    name: str
    title: str
    version: str
    axes: list[Axis]
    learn: dict[str, Any] = field(default_factory=dict)
    # 哪些失败原因算"与策略冲突"（不是用户懒）
    conflict_reasons: tuple[str, ...] = ("time_conflict",)
    # 学习算法：elimination（假设淘汰）| ucb（置信上界）
    learner: str = "elimination"

    def __post_init__(self) -> None:
        ids = [a.id for a in self.axes]
        if len(set(ids)) != len(ids):
            raise ValueError("axis id 重复")
        merged = dict(DEFAULT_LEARN)
        merged.update(self.learn)
        self.learn = merged

    def axis(self, aid: str) -> Axis:
        for a in self.axes:
            if a.id == aid:
                return a
        raise KeyError(f"未知 axis {aid}")


class Domain(Protocol):
    spec: DomainSpec

    def plan(self, view: Any, run_kind: str) -> list[Task]: ...

    def interpret(self, feedback: dict, view: Any) -> list[Signal]: ...


def check_domain(domain: Any) -> None:
    """领域包自检：缺东西就在启动时炸，别等到现场。"""
    for attr in ("spec", "plan", "interpret"):
        if not hasattr(domain, attr):
            raise TypeError(f"领域包缺少 {attr}（{type(domain).__name__}）")
    spec = domain.spec
    if not isinstance(spec, DomainSpec):
        raise TypeError("domain.spec 必须是 DomainSpec")
    for axis in spec.axes:
        for hid in axis.ids:
            axis.hyp(hid)
