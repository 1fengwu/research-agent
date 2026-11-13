"""框架门面：Store / Runtime 之外，其余模块按需导入。"""

from __future__ import annotations

__version__ = "1.0.0"

from .runtime import RunConfig, Runtime  # noqa: E402
from .store import BaselineChanged, Store  # noqa: E402

__all__ = ["Store", "Runtime", "RunConfig", "BaselineChanged", "__version__"]
