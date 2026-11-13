"""HabitFit 领域包：运动时段 / 运动时长 / 睡前时点，三个轴共用一份学习代码。

这就是回答"换个失败原因还会不会学"的地方：

- 失败原因是「时间冲突」→ 淘汰 H1（晚间），换 H2（午间）；
- 失败原因是「太累」→ 淘汰 H6（23:20），换 H7（22:50）；
- 两条淘汰走的是**同一段逻辑**，只是轴不同。

刻意留的一处"不完美"：周三（例会日）系统**没有证据**，于是预测 low + 请你确认。
它实际也冲突了 —— 算命中（它本来就说不知道），但仍算冲突。这才是诚实的样子。
"""

from __future__ import annotations

from ..spec import Axis, DomainSpec, Hypothesis, Outcome, Signal, Task

AXES = [
    Axis(
        "exercise_time",
        "运动安排在什么时候？",
        [Hypothesis("H1", "晚间", {"slot": "evening"}), Hypothesis("H2", "午间", {"slot": "noon"})],
        default="H1",
    ),
    Axis(
        "exercise_len",
        "一次运动多久？",
        [Hypothesis("H3", "30 分钟", {"minutes": 30}), Hypothesis("H4", "15 分钟", {"minutes": 15})],
        default="H3",
    ),
    Axis(
        "sleep_time",
        "睡前准备排在几点？",
        [Hypothesis("H6", "23:20", {"time": "23:20"}), Hypothesis("H7", "22:50", {"time": "22:50"})],
        default="H6",
    ),
]

SPEC = DomainSpec(
    name="habitfit",
    title="HabitFit · 生活节律教练",
    version="2.1",
    axes=AXES,
    conflict_reasons=("time_conflict",),
)

# key, 标题, 轴, 日, 类型
PLAN = [
    ("mon-ex", "周一 运动 30 分钟", "exercise_time", "mon", "task"),
    ("tue-ex", "周二 运动 30 分钟", "exercise_time", "tue", "task"),
    ("wed-ex", "周三 运动 30 分钟", "exercise_time", "wed", "task"),
    ("sat-ex", "周六 运动 30 分钟", "exercise_time", "sat", "task"),
    ("mon-sleep", "周一 睡前准备 10 分钟", "sleep_time", "mon", "task"),
    ("tue-sleep", "周二 睡前准备 10 分钟", "sleep_time", "tue", "task"),
    ("wed-sleep", "周三 睡前准备 10 分钟", "sleep_time", "wed", "task"),
]


class HabitFit:
    spec = SPEC

    # ------------------------------------------------------------ 预测
    def _learned(self, view) -> bool:
        """有没有从证据里学过东西（用于区分"默认乐观"和"知道自己不知道"）。"""
        for axis in SPEC.axes:
            for hid in axis.ids:
                st = view.hyp_state(axis.id, hid)
                if st["refute"] or st["support"]:
                    return True
        return False

    def predict(self, view, payload: dict) -> tuple[str, str]:
        axis = payload["axis"]
        day = payload["day"]
        hid = view.active(axis)
        label = view.hyp_state(axis, hid)["params"] if hid else {}
        if not self._learned(view):
            return "high", f"默认假设 {hid} {label}，还没有反例"
        if payload.get("uncertain"):
            return "low", f"{day} 这天没有任何证据，不敢替你保证，请确认"
        return "high", f"{hid} {label} 在已观察到的日子里成立"

    # ------------------------------------------------------------ 计划
    def plan(self, view, run_kind: str) -> list[Task]:
        learned = self._learned(view)
        tasks: list[Task] = []
        for key, title, axis_id, day, kind in PLAN:
            hid = view.active(axis_id)
            unknown = view.unknown(axis_id)
            exercise = axis_id == "exercise_time"
            payload = {
                "axis": axis_id,
                "day": day,
                # 周三有例会：只有"学过之后"才承认自己不知道，第一次不装懂
                "uncertain": bool(learned and exercise and day == "wed"),
                "hypothesis": hid,
            }
            if unknown:
                pred, why = "low", f"{axis_id} 的假设全被否掉了，需要你告诉我一个可行方案"
            else:
                pred, why = self.predict(view, payload)
            tasks.append(
                Task(
                    key=key,
                    title=title,
                    axis=axis_id,
                    hypothesis=hid or "",
                    prediction=pred,
                    prediction_reason=why,
                    payload=payload,
                    kind=kind,
                )
            )
        return tasks

    # ------------------------------------------------------------ 反馈 → 证据
    def interpret(self, feedback: dict, view) -> list[Signal]:
        signals: list[Signal] = []
        for item in feedback.get("stated", []):
            signals.append(
                Signal(
                    axis=item["axis"],
                    reason=item.get("reason", "stated_preference"),
                    refute=list(item.get("refute", [])),
                    support=list(item.get("support", [])),
                    source="stated",
                    day=item.get("day"),
                    detail=item.get("text", ""),
                )
            )
        for item in feedback.get("observed", []):
            signals.append(
                Signal(
                    axis=item["axis"],
                    reason=item.get("reason", "observed"),
                    refute=list(item.get("refute", [])),
                    support=list(item.get("support", [])),
                    source="observed",
                    day=item.get("day"),
                    detail=item.get("detail", ""),
                )
            )
        return signals

    # ------------------------------------------------------------ 演示用"真实世界"
    def simulate(self, payload: dict, hypothesis: str, kind: str = "task") -> Outcome:
        axis, day = payload["axis"], payload["day"]
        params = {}
        for a in SPEC.axes:
            if a.id == axis and hypothesis in a.ids:
                params = dict(a.hyp(hypothesis).params)
        if axis == "exercise_time":
            slot = params.get("slot")
            if day in ("mon", "tue", "wed") and slot == "evening":
                return Outcome(False, "time_conflict", "晚间被临时安排占了")
            if day == "wed" and slot == "noon":
                return Outcome(False, "time_conflict", "周三中午有例会（系统事先不知道）")
            return Outcome(True, "ok", "按时完成")
        if axis == "sleep_time":
            if params.get("time") == "23:20":
                return Outcome(False, "too_tired", "到 23:20 已经太累，直接躺下了")
            return Outcome(True, "ok", "22:50 完成准备")
        return Outcome(True, "ok", "")

    # ------------------------------------------------------------ 证据提取助手
    def stated_feedback(self, rows: list[dict]) -> list[dict]:
        """演示里"人明说的一句"——框架按 1 条即生效处理。"""
        return [
            {
                "axis": "exercise_time",
                "refute": ["H1"],
                "support": ["H2"],
                "reason": "time_conflict",
                "text": "晚上要带娃，中午其实可以",
            },
            {
                "axis": "sleep_time",
                "refute": ["H6"],
                "support": ["H7"],
                "reason": "too_tired",
                "text": "23:20 太晚了，想早点准备",
            },
        ]

    def observations_from(self, tasks: list[dict], rows: list[dict]) -> list[dict]:
        """把本轮失败自动归纳成证据（对应 PRD 的"自动反馈分析"）。"""
        failed = {r["task"]["key"]: r for r in rows if not r["done"]}
        observed = []
        for t in tasks:
            r = failed.get(t["key"])
            if not r:
                continue
            if r["reason"] not in SPEC.conflict_reasons and r["reason"] != "too_tired":
                continue
            observed.append(
                {
                    "axis": t["axis"],
                    "refute": [t["hypothesis"]] if t["hypothesis"] else [],
                    "reason": r["reason"],
                    "day": t["payload"].get("day"),
                    "detail": "",
                }
            )
        return observed


DOMAIN = HabitFit()
