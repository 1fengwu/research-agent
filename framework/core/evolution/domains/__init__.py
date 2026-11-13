"""领域包注册表：加一个领域 = 加一个文件 + 在这里登记一行。"""

from __future__ import annotations

from .habitfit import HabitFit
from .zhijie import Zhijie

REGISTRY = {
    "habitfit": HabitFit(),
    "zhijie": Zhijie(),
}


def get(name: str):
    if name not in REGISTRY:
        raise KeyError(f"未知领域 {name}，可选：{', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]


def names() -> list[str]:
    return sorted(REGISTRY)
