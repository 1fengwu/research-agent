# A：编排与推理模块

这是 A 负责的运行时模块：`agent_loop`（状态机与主循环）、`llm_router`（云调用与升级路由）、`budget_gate`（预算账本与拦截）、`contracts`（任务/实验/判决/事件契约）。

## A 只做这些

- 决定下一步谁来、什么时候可以调用云模型、预算不够时怎么降级；
- 把每一步写成 `RunEvent` 追加进 `.arc/runner-events.jsonl`；
- 把云调用和实验用量写进 `.arc/token_ledger.json`，对外只给只读快照；
- 调用 B 的 `data/experiment`、C 的 `oracle/evaluate`，不自己实现数据或裁判。

A 不碰：holdout、实验评分、特征工程、verdicts 写入、UI 业务逻辑。

## 契约

- `TaskSpec`：任务、数据、切分、seed、预算；
- `ExperimentSpec`：假设编号、证据引用、特征组、模型、参数、缺失/缩放处理；
- `TrialResult`：两个验证窗、有效天数、排除数、缺失率、耗时、错误；
- `Verdict`：`supported / refuted / inconclusive / blocked / partial_completed`；
- `RunEvent`：类型、阶段、时间戳、决策原因、实验编号、token、剩余预算。

任务启动前先 `load_task_spec`，契约不合法就拒绝开工。

## 主循环

```
UNSEEN → PLANNING → PLANNED → EXPERIMENTING → VERIFIED → REVISED → LOCKED → TESTED
                                              ↘ CONVERGED / FAILED / PARTIAL_COMPLETED
```

停止条件：候选实验数、云调用次数、云 token、墙钟时间、无新候选、预算不足且不允许升级。

## 升级触发（`llm_router`）

- `data_anomaly`：数据异常；
- `planning_stall`：连续两轮无进展；
- `window_conflict`：两个验证窗方向相反且规则无法归因；
- `critical_decision`：影响后续 ≥3 步；
- `repeated_error_signature`：同一失败签名两次；
- `finalize`：收尾成文。

云回包必须是结构化 JSON，必须带 `decision`、`evidence_ids` 和 `actions`；无效回包记录 `research.escalation_invalid` 并回规则路径。

## 事件类型

- `research.run_started / data_profile / baseline / hypothesis_registered`
- `research.experiment_started / experiment_result / experiment_skipped / experiment_verdict`
- `research.revised / escalation / escalation_blocked / escalation_failed / escalation_invalid`
- `research.locked / tested / completed / partial_completed`

## B/C/D 怎么对接

```python
from autoresearch.ports import DataPort, OraclePort, PlannerPort
from autoresearch.agent_loop import EventWriter, ResearchOrchestrator
from autoresearch.budget_gate import BudgetGate
from autoresearch.llm_router import ResearchRouter, HttpJsonCloudClient

ledger = BudgetGate(task.budget, ledger_path=".arc/token_ledger.json")
router = ResearchRouter(ledger, client=HttpJsonCloudClient())  # 或 NullCloudClient 离线
orchestrator = ResearchOrchestrator(task, data=DataPort(), oracle=OraclePort(),
                                    planner=PlannerPort(), router=router,
                                    ledger=ledger, writer=EventWriter())
summary = orchestrator.run()
```

## 测试

```bash
python -m unittest discover -s autoresearch/tests -v
```

覆盖：状态机到 TESTED、预算拦截、重复指纹、非法云回包降级、重复失败升级、候选上限。
