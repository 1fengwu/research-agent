"""策略快照：只追加、可比较、可回退。

快照长这样（`snapshot_json`）：

```json
{
  "domain": "habitfit",
  "seq": 0,
  "axes": {
    "exercise_time": {
      "active": "H1",
      "unknown": false,
      "hypotheses": {
        "H1": {"support": 0, "refute": 0, "excluded": false,
               "params": {"slot": "evening"}, "days": []},
        "H2": {"support": 0, "refute": 0, "excluded": false,
               "params": {"slot": "noon"}, "days": []}
      }
    }
  },
  "meta": {"origin": "init", "note": "", "changed_axes": []}
}
```

三个刻意设计：

- **假设不删除只标记** `excluded`：回退能原样恢复，答辩时也能解释"为什么否掉它"；
- **active 与 hypotheses 分开**：策略是"选谁"，证据是"每条的支持/反驳"，两者不混；
- **`unknown`**：全被淘汰时置 true，运行时会**主动问人**，而不是硬猜一个。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .spec import DomainSpec


def build_initial(spec: DomainSpec) -> dict[str, Any]:
    axes: dict[str, Any] = {}
    for axis in spec.axes:
        axes[axis.id] = {
            "active": axis.default,
            "unknown": False,
            "hypotheses": {
                h.id: {
                    "support": 0,
                    "refute": 0,
                    "excluded": False,
                    "params": dict(h.params),
                    "days": [],
                }
                for h in axis.hypotheses
            },
        }
    return {
        "domain": spec.name,
        "seq": 0,
        "axes": axes,
        "meta": {"origin": "init", "note": "", "changed_axes": []},
    }


def canonical(snapshot: dict) -> str:
    """**内容**的稳定序列化：同一内容必须得到同一 hash。

    刻意排除 `seq` 与 `meta`：版本号与来源不是内容。
    回退生成的新版本 seq 变了、内容没变 —— 这时 hash 必须相等，
    否则"V2 内容等同 V0"这句话在数据库里就无法举证。
    """
    body = {k: v for k, v in snapshot.items() if k not in ("meta", "seq")}
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_hash(snapshot: dict) -> str:
    return hashlib.sha256(canonical(snapshot).encode("utf-8")).hexdigest()[:16]


def clone(snapshot: dict) -> dict:
    return json.loads(json.dumps(snapshot, ensure_ascii=False))


class View:
    """只读视图：领域包拿它生成计划和预测，不许它改快照。"""

    def __init__(self, snapshot: dict) -> None:
        self._s = snapshot

    @property
    def raw(self) -> dict:
        return self._s

    def active(self, axis: str) -> str | None:
        return self._s["axes"][axis]["active"]

    def unknown(self, axis: str) -> bool:
        return bool(self._s["axes"][axis].get("unknown"))

    def active_params(self, axis: str) -> dict:
        hid = self.active(axis)
        if hid is None:
            return {}
        return dict(self._s["axes"][axis]["hypotheses"][hid]["params"])

    def hyp_state(self, axis: str, hid: str) -> dict:
        return self._s["axes"][axis]["hypotheses"][hid]

    def excluded(self, axis: str) -> list[str]:
        return [h for h, st in self._s["axes"][axis]["hypotheses"].items() if st["excluded"]]

    def days(self, axis: str, hid: str) -> list[str]:
        return list(self._s["axes"][axis]["hypotheses"][hid]["days"])

    def summary(self) -> dict[str, Any]:
        out = {}
        for aid, a in self._s["axes"].items():
            out[aid] = {
                "active": a["active"],
                "unknown": a["unknown"],
                "excluded": [h for h, st in a["hypotheses"].items() if st["excluded"]],
                "evidence": {h: f"+{st['support']}/-{st['refute']}" for h, st in a["hypotheses"].items()},
            }
        return out


def diff(old: dict, new: dict) -> list[dict[str, Any]]:
    """只报"变了什么"，供控制台和答辩使用。"""
    changes: list[dict[str, Any]] = []
    for aid in sorted(set(old["axes"]) | set(new["axes"])):
        o = old["axes"].get(aid, {})
        n = new["axes"].get(aid, {})
        if o.get("active") != n.get("active"):
            changes.append({"axis": aid, "field": "active", "from": o.get("active"), "to": n.get("active")})
        if bool(o.get("unknown")) != bool(n.get("unknown")):
            changes.append({"axis": aid, "field": "unknown", "from": bool(o.get("unknown")), "to": bool(n.get("unknown"))})
        for hid in sorted(set(o.get("hypotheses", {})) | set(n.get("hypotheses", {}))):
            oh = o.get("hypotheses", {}).get(hid, {})
            nh = n.get("hypotheses", {}).get(hid, {})
            for f in ("support", "refute", "excluded"):
                if oh.get(f) != nh.get(f):
                    changes.append({"axis": aid, "hypothesis": hid, "field": f, "from": oh.get(f), "to": nh.get(f)})
    return changes


def restore(snapshot: dict, restored_from: str) -> dict:
    """回退 = 复制目标内容生成**新版本**，不是把指针往回拨。

    这样 V1 的事件、反馈、运行记录全都还在，新运行读到的却是 V0 的内容。
    """
    s = clone(snapshot)
    s["meta"] = dict(s.get("meta", {}))
    s["meta"]["origin"] = "rollback"
    s["meta"]["restored_from"] = restored_from
    return s
