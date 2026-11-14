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

如需启用云模型，可在仓库根目录创建 `.env`（参考 `.env.example`）或设置环境变量 `ARC_LLM_BASE_URL`（或 `OPENAI_BASE_URL`）、`ARC_LLM_API_KEY`、`ARC_LLM_MODEL`，再添加 `--cloud`；运行时自动读取 `.env`，且不覆盖已存在的环境变量。

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

## 后端输出契约

- 契约说明：[docs/后端数据契约.md](../docs/后端数据契约.md)
- 契约校验器：[`autoresearch/outputs.py`](outputs.py)
- 契约测试：[`autoresearch/tests/test_output_contracts.py`](tests/test_output_contracts.py)

前端、API 队友和评审回放都按这套结构消费事件流、summary、ledger、verdicts、
metrics、pre_gate 与 replay 输出；字段变更必须同步更新契约文档与测试。

## D 说明：界面与报告

角色 D 的交付位于本包：

- `app.py`：单页 Streamlit 评审工作台（创建任务 / 跟踪研究 / 审查结论）；
- `report.py`：事件聚合、Markdown 报告与复现包打包；
- `mock_events.json`：离线联调事件流，UI 不依赖云模型也能走完整流程；
- `cli.py`：界面异常时的 CLI 回放路径，以及接 C oracle 后的真实启动入口。

D 只读 A 的事件、B 的数据摘要和 C 的指标，不在界面里生成研究结论。

```bash
pip install -r requirements-app.txt
python -m streamlit run autoresearch/app.py

# 离线联调 / 回放
python -m autoresearch.cli demo
python -m autoresearch.cli replay --events .arc/runner-events.jsonl

# 真实后台启动（C 的 oracle 已由上游实现）
python -m autoresearch.cli run --task .arc/task.json --oracle autoresearch.oracle:ResearchOracle
```

测试：

```bash
python -m unittest autoresearch.tests.test_d_tasks -v
```

## 自进化迭代

`autoresearch/evolution.py` 把 `framework/core/evolution` 的策略学习接到研究规划器：

- 每次 `interpret()` 产生的证据会写进 SQLite 策略库（默认 `.arc/evolution/research-pv.db`）；
- 规划器自动跳过已被反驳的假设，优先复测已有支持证据的假设；
- 策略版本只追加，可回退、可审计；
- 不同运行共用同一策略库时，上一轮学到的结论会作用于下一轮。

```bash
# 直接跑研究（默认使用自进化策略库）
python -m autoresearch data/demo_panel.csv

# 通过 CLI 启动，自定义策略库位置
python -m autoresearch.cli run --task .arc/task.json \
  --evolution .arc/evolution/research-pv.db
```

```bash
python -m unittest autoresearch.tests.test_evolution_tasks -v
```

## 增强预测可信度

- 特征新增 `+momentum`（`r10 / r60 / ma_dist5 / ma_dist20`）和 `+cross`（同日横截面排名 `x_rank_r5 / x_rank_r20 / x_rank_volume`）；
- 模型新增 `gradient_boost`（`HistGradientBoostingClassifier`）和 `random_forest`（`RandomForestClassifier`）；
- holdout 判决的 `metric` 额外输出 `spearman`、`brier`、`pr_auc`，用于检验排序能力、概率校准和正例排序质量。
