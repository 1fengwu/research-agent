"""知阶领域包：三模块 × 三类困难 × 三种呈现策略。

放这个包的目的是**证明框架通用**，而不是给数学教学做完整实现：

- 同一套 `plan / interpret / predict`，换成"呈现方式"这条轴；
- 验证题带 `item`，于是"新题率"这条指标才有意义（HabitFit 没有题目概念，报 n/a）；
- 回退语义、事务、幂等全部复用核心，领域包里没有一行版本逻辑。
"""

from __future__ import annotations

from ..spec import Axis, DomainSpec, Hypothesis, Outcome, Signal, Task

AXES = [
    Axis(
        "d1_presentation",
        "D1「说不清为什么」用哪种呈现？",
        [Hypothesis("S1_sym", "符号对照", {"variant": "symbol"}), Hypothesis("S1_balance", "天平图示", {"variant": "balance"})],
        default="S1_sym",
    ),
    Axis(
        "d2_presentation",
        "D2「步骤中出错」用哪种提示粒度？",
        [Hypothesis("S2_coarse", "整步提示", {"granularity": "coarse"}), Hypothesis("S2_step", "分步提示", {"granularity": "step"})],
        default="S2_coarse",
    ),
    Axis(
        "d3_presentation",
        "D3「新情境卡住」用哪种支架？",
        [Hypothesis("S3_direct", "直接给式", {"scaffold": "direct"}), Hypothesis("S3_table", "关系表支架", {"scaffold": "table"})],
        default="S3_direct",
    ),
]

SPEC = DomainSpec(
    name="zhijie",
    title="知阶 · AI 自进化数学教师（核心闭环）",
    version="1.1",
    axes=AXES,
    conflict_reasons=("misconception",),
)

# 真实世界里每个困难真正需要的呈现（只有"学生"知道，系统要靠证据摸出来）
TRUTH = {"d1_presentation": "S1_balance", "d2_presentation": "S2_step", "d3_presentation": "S3_table"}
DIFFICULTY = {"d1_presentation": "D1", "d2_presentation": "D2", "d3_presentation": "D3"}


class Zhijie:
    spec = SPEC

    def _learned(self, view) -> bool:
        for axis in SPEC.axes:
            for hid in axis.ids:
                st = view.hyp_state(axis.id, hid)
                if st["refute"] or st["support"]:
                    return True
        return False

    def predict(self, view, payload: dict) -> tuple[str, str]:
        axis = payload["axis"]
        hid = view.active(axis)
        if view.unknown(axis):
            return "low", f"{axis} 没有可用策略，需要老师介入"
        return "high", f"按 {hid} 呈现，预期这次能独立完成"

    def plan(self, view, run_kind: str) -> list[Task]:
        seq = view.raw.get("seq", 0)
        tasks: list[Task] = []
        for axis in SPEC.axes:
            difficulty = DIFFICULTY[axis.id]
            hid = view.active(axis.id)
            pred, why = self.predict(view, {"axis": axis.id})
            tasks.append(
                Task(
                    key=f"{difficulty.lower()}-teach",
                    title=f"{difficulty} 教学：{axis.hyp(hid).label}",
                    axis=axis.id,
                    hypothesis=hid or "",
                    prediction=pred,
                    prediction_reason=why,
                    payload={"axis": axis.id, "difficulty": difficulty, "hypothesis": hid},
                    kind="teach",
                )
            )
            # 验证题用"本版本号"编号：第二次运行时必然是一道新题
            tasks.append(
                Task(
                    key=f"{difficulty.lower()}-verify",
                    title=f"{difficulty} 独立验证（新题）",
                    axis=axis.id,
                    hypothesis=hid or "",
                    prediction=pred,
                    prediction_reason=why,
                    payload={
                        "axis": axis.id,
                        "difficulty": difficulty,
                        "hypothesis": hid,
                        "item": f"{difficulty}-v{seq + 1}",
                    },
                    kind="verify",
                )
            )
        return tasks

    def interpret(self, feedback: dict, view) -> list[Signal]:
        out: list[Signal] = []
        for item in feedback.get("stated", []):
            out.append(
                Signal(
                    axis=item["axis"],
                    reason=item.get("reason", "principle_unclear"),
                    refute=list(item.get("refute", [])),
                    support=list(item.get("support", [])),
                    source="stated",
                    detail=item.get("text", ""),
                )
            )
        for item in feedback.get("observed", []):
            out.append(
                Signal(
                    axis=item["axis"],
                    reason=item.get("reason", "misconception"),
                    refute=list(item.get("refute", [])),
                    support=list(item.get("support", [])),
                    source="observed",
                    day=item.get("day"),
                    detail=item.get("detail", ""),
                )
            )
        return out

    # 演示用"学生"：策略对了就做对，错了就错在同一个地方
    def simulate(self, payload: dict, hypothesis: str, kind: str = "teach") -> Outcome:
        axis = payload["axis"]
        if hypothesis == TRUTH[axis]:
            return Outcome(True, "ok", "独立完成")
        return Outcome(False, "misconception", "换了呈现方式仍然卡在同一步")

    def stated_feedback(self, rows: list[dict]) -> list[dict]:
        return [
            {
                "axis": "d1_presentation",
                "refute": ["S1_sym"],
                "support": ["S1_balance"],
                "reason": "principle_unclear",
                "text": "原理没看懂，能画天平吗",
            },
            {
                "axis": "d2_presentation",
                "refute": ["S2_coarse"],
                "support": ["S2_step"],
                "reason": "step_error",
                "text": "提示太粗，我要一步步来",
            },
            {
                "axis": "d3_presentation",
                "refute": ["S3_direct"],
                "support": ["S3_table"],
                "reason": "modeling",
                "text": "文字题不会列式，给我关系表",
            },
        ]

    def observations_from(self, tasks: list[dict], rows: list[dict]) -> list[dict]:
        failed = {r["task"]["key"]: r for r in rows if not r["done"]}
        obs = []
        for t in tasks:
            r = failed.get(t["key"])
            if not r or not t["hypothesis"]:
                continue
            obs.append({"axis": t["axis"], "refute": [t["hypothesis"]], "reason": r["reason"], "detail": ""})
        return obs


DOMAIN = Zhijie()
