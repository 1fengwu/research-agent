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
- `RunEvent`：类型、阶段、时间戳、`run_id`、决策原因、实验编号、token、剩余预算。

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

## 运行量价研究

在仓库根目录使用 Python 安装依赖并运行离线演示。

```bash
python -m pip install -r requirements.txt
python -m autoresearch data/demo_panel.csv
```

命令使用真实数据后端、规划器和裁判。它在标准输出返回 JSON 摘要，独立测试指标位于 `test_verdict.metric`，流程未完成时返回非零退出码。默认把事件和预算账本写入 `.arc`，可用 `--output-dir` 指定输出目录。

如需启用云模型，先设置 `ARC_LLM_BASE_URL`（或 `OPENAI_BASE_URL`）、`ARC_LLM_API_KEY` 和 `ARC_LLM_MODEL`，再添加 `--cloud`。

每次 `run()` 会先把 ledger 重置为 `.arc/ledgers/<run_id>.json`，R2 不会继承 R1 的预算；事件流用 `run_id` 区分多轮运行。

## 测试

```bash
python -m unittest discover -s autoresearch/tests -v
```

覆盖：状态机到 TESTED、预算拦截、重复指纹、非法云回包降级、重复失败升级、候选上限。

## B 对接说明

角色 B 的正式实现也在本包内：

- `data.py`：CSV 接入、数据画像、`fwd5` 标签、时间切分与滚动特征；
- `experiment.py`：`ResearchBackend` 实现 `DataPort` 的 `describe_data / baseline / run_candidate`；
- `research_pv.py`：`ResearchPVPlanner` 实现 `PlannerPort` 的 `plan / interpret`；
- `demo_data.py`：生成固定 seed 的 `data/demo_panel.csv`、`data/demo_split.json` 与 `data/sample.csv`。

```bash
python -m autoresearch.demo_data
python -m unittest autoresearch.tests.test_b_tasks -v
```

## C：评测与复现

角色 C 的交付模块：

- `pre_gate.py`：前置门，固定基线 + 3 候选，两窗配对 bootstrap 95% CI，输出 `.arc/pre_gate.json`；
- `oracle.py`：holdout 唯一裁判，锁定前拒绝测试、锁定后只测一次，并重跑验证窗拒绝伪造提升；
- `evaluate.py`：A/B/C 三臂对照，输出 `.arc/eval/metrics.csv` 与 `metrics.json`；
- `replay.py`：无云复现，重跑一次 holdout 并与原记录对账，输出 `.arc/repro/<run_id>/summary.json`。

```bash
python -m autoresearch.pre_gate --data data/demo_panel.csv --out .arc
python -m autoresearch.evaluate --data data/demo_panel.csv --out .arc/eval
python -m autoresearch.replay --run-id <run_id> --output-dir .arc
```

默认离线可复现；`evaluate.py` 加 `--cloud` 且配置 `ARC_LLM_BASE_URL/ARC_LLM_API_KEY/ARC_LLM_MODEL` 时才使用真实云。

```bash
python -m unittest autoresearch.tests.test_c_tasks -v
```
