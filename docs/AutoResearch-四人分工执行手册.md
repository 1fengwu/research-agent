# Helix 赫力 四人分工执行手册

**版本**：对应《ResearchAgent-系统设计-v5.md》的落地执行版

**窗口**：10 小时，4 人（A/B/C/D），P0 冻结，一个共享云 LLM

**原则**：所有实验、预算、结论都只能发生在规定角色、规定文件、规定接口里；分工清楚到“谁能写哪个文件、谁能调哪个函数、谁不能碰什么”。

---

## 0. 公共契约（四个人先一起定死，H1 之后不许单方面改）

1. `TaskSpec / ExperimentSpec / TrialResult / RunEvent` 在 0-1 小时冻结，附样例 JSON。冻结后任何字段变更都要四个人当面确认，并同步改示例文件与校验 schema。
2. 同一数据、同一 `sha256`、同一切分、同一组 seed、同一套依赖 lock，三个臂才能比；任何一臂换了数据口径，其他臂作废。
3. 数据、标签、切分、裁判只由 C 的 `oracle` 出分；A 只负责调度，B 只负责产生实验，D 只负责展示。谁都不能“顺便自己评一下”。
4. 任何云调用必须经过 A 的 `llm_router`，并且只记在 A 的 `token_ledger.json` 里；没有 ledger 记录的调用等于不存在。
5. H6 如果主流程没跑通：全部人停做 P1、动画、图表美化，D 切 CLI 回放路径，A/B/C 全力保证闭环。
6. 每个节点 commit 一次；现场禁止为了演示效果改历史事件、指标或 holdout 结果。

## 1. 文件归属总表

| 文件 | 谁写 | 谁评审 | 谁调用 | 核心职责 |
|---|---|---|---|---|
| `contracts.py`（schema） | C 起草校验，A 起草字段 | 四人 | 全部 | 所有接口的类型、默认值和示例 |
| `pre_gate.py` | C | B | A 启动前 | 前置门：数据有没有可研究空间 |
| `data.py` | B | C | A、C | CSV、画像、量价特征、时间切分 |
| `experiment.py` | B | C | A | 基线/候选实验、日均 AUC |
| `research_pv.py`（领域包） | B 实现，A 定计划契约 | A | A、C | 假设轴、预测、证据、judge 回调 |
| `agent_loop.py` | A | B、D | CLI/UI | 状态机、循环、停止条件 |
| `llm_router.py` | A | C | agent_loop | 预算门、云调用、回包校验、降级 |
| `token_ledger.*` / `budget_gate.py` | A | C | A、D | token/时间/实验数记账与拦截 |
| `oracle.py` | C | A、B | A、C | holdout 裁判，唯一出分口 |
| `evaluate.py` | C | B | A、D | 三臂对照和各项系统指标 |
| `replay.py` | C | A | D | 无云复现、复现包校验 |
| `app.py` | D | A、C | 评审/用户 | 单页 Streamlit |
| `report.py` | D | C | app.py | 报告与复现包导出 |
| 演示数据生成器（合成面板） | B | C | D | 固定 seed 的演示 CSV |
| `mock_events.json` | D | A | D | 前端离线联调 |

## 2. 每个人不能碰什么（红线）

| 人 | 不能碰 | 否则 |
|---|---|---|
| A | holdout、实验评分、特征工程细节、直接改 B/C/D 的业务文件 | 裁判被污染，闭环无法自证 |
| B | LLM 调用、ledger、verdicts、holdout 读取、切分冻结后改切分 | token 账本失效 / 反作弊失效 |
| C | 实验设计、假设生成、训练超参、UI 业务展示 | 被当成参与方，裁判不再可信 |
| D | LLM、实验、verdicts、任何研究数据写入、业务逻辑 | 界面变成裁判，推翻反作弊结论 |

---

## 3. A：编排与推理（agent_loop + llm_router）

### 3.1 一句话职责

A 是唯一的“大脑调度者”：决定下一步谁做、什么时候可以请云模型、预算不够时怎么降级，并保证每一步都有事件记录。

### 3.2 交付文件

- `autoresearch/agent_loop.py`
- `autoresearch/llm_router.py`
- `autoresearch/budget_gate.py`
- `autoresearch/token_ledger.json`（运行态）
- `contracts.py` 中 RunEvent、TaskSpec 的字段提案

### 3.3 具体要做的事

1. 实现研究状态机：`UNSEEN → PLANNING → PLANNED → EXPERIMENTING → VERIFIED → REVISED → LOCKED → TESTED → PASSED / CONVERGED / FAILED`，以及仅在硬限制/不可恢复错误时保留的 `partial_completed` 分支。
2. 实现主循环：

```
load TaskSpec
→ B.describe_data()
→ B.baseline()
→ request plan (research_pv.plan)
→ for each ExperimentSpec: B.run_candidate()
→ C.oracle.judge()
→ if evidence changes plan: research_pv.interpret() → REVISED
→ lock candidate or baseline
→ C.oracle.judge(test=True)
→ stop and emit summary
```

3. 实现停止条件（软预算下都不是硬停）：候选计划耗尽、无新候选、180 秒保护、用户取消、数据/裁判不可恢复失败；预算只切换到 economy/critical，必要时 overrun 并完成最终测试。
4. 实现 `llm_router`：
   - `should_escalate(state)`：只有 v5 §7.3 的结构性触发才返回 True；
   - `build_summary(state)`：紧凑摘要，只含证据编号、当前策略、候选、冲突；
   - `call_cloud(summary, max_tokens)`：受 `budget_gate` 约束；
   - `validate_reply(reply)`：必须是可执行 JSON，且引用证据编号；无效就记为失败并回规则路径；
   - `apply_reply(state, reply)`：改成下一步计划，记录 `escalation` 事件；
   - `local_rule_decision(state)`：economy/critical/overrun 下 window_conflict → 下一假设、重复实验错误 → 回退基线，0 云调用并写 `research.local_rule`；
   - `DecisionCache`：`.arc/cache/decision-cache.json` 跨轮次复用有效回包，命中 0 token 并写 `research.escalation_cached`；
   - `savings_summary()`：汇总压缩节省/缓存/本地规则/避免调用/总节省。
5. 实现预算门：调用前估算输入 token + 输出上限；预算压力决定 normal/economy/critical/overrun；soft 默认不拦死，economy 计划 2 个、critical/overrun 计划 1 个新候选；软目标不足只跳过本次云调用并写 `budget_blocked`，不终止研究，最终测试总会被执行；`hard_limits=True` 才恢复硬停。
6. 实现事件写入：阶段、决策原因、实验编号、调用 token、剩余预算全部写 `.arc/runner-events.jsonl`。
7. 保证重试/回退语义：同一错误签名两次才云诊断；云回包无效则回规则；`rollback` 只由版本链生成新版本。
8. 输出 token 节省汇总：completed summary 与 `research.token_savings` 事件包含 `prompt_compression_saved / cache_hits / cache_saved_tokens / local_rule_decisions / local_rule_saved_tokens / cloud_calls_avoided / total_estimated_saved / cloud_tokens_charged`。

### 3.4 消费与提供

| 方向 | 内容 |
|---|---|
| 消费 B | `data.py.describe_data()`、`experiment.py.baseline()`、`experiment.py.run_candidate(spec)` |
| 消费 C | `oracle.judge(...)`、`evaluate.py` 输出 |
| 提供 D | 完整 RunEvent 流、预算实时读数、停止原因 |
| 提供 C | 每次运行的全量事件与 ledger，供 evaluate/replay |

### 3.5 小时级安排

| 时间 | 完成条件 |
|---|---|
| 0-1h | 与所有人冻结 TaskSpec/RunEvent；画出状态表；搭 agent_loop 骨架和 ledger 空文件 |
| 1-4h | 状态机、停止条件、llm_router 规则、预算门全部可单测；H4 时能让 B 的基线结果走完第一个状态 |
| 4-6h | 与 B 联调首个端到端；至少一次证据驱动的 REVISED；H6 出第一版复现包 |
| 6-8h | 跑 A/B/C 三臂、低预算、重复指纹、超时、云回包无效这些分支；H8 冻结路由参数 |
| 8-10h | 彩排真实云调用与离线回放；确认事件流与 ledger 可被 C 复算 |

### 3.6 A 自己的测试

- 状态机每步转移有测试；
- 预算不足时不发起低价值云调用，事件里有 `budget_blocked`，但研究仍走规则并完成最终测试；
- 云回包缺证据编号会被拒绝；
- 同一错误签名第二次才升级；
- 重复实验指纹会跳过而不是重跑；
- 相同决策状态二次运行命中缓存，云端调用数不增加，`research.escalation_cached` 事件存在；
- 预算压力下 window_conflict/重复实验错误走本地规则，云端调用数为 0，`research.local_rule` 事件存在；
- completed summary 与报告能输出 `token_savings` 各项读数；
- 事件流里每个阶段都有 `decision_reason`。

### 3.7 A 的验收标准

- 一个任务能从头跑到报告，且日志里能指出“哪一步证据改变了哪一步配置”；
- 预算设得太低时能展示拦截 + economy/critical/overrun，且最终结果为 TESTED；
- D 的 UI 读到的所有状态都来自 A 的事件，而不是 D 自己拼的。

---

## 4. B：数据与模型（data + experiment + research_pv）

### 4.1 一句话职责

B 是“研究的身体”：负责把 CSV 变成干净、可切分、可复现的实验面板，并忠实地执行 A 批准的候选实验，不自己发明结论。

### 4.2 交付文件

- `autoresearch/data.py`
- `autoresearch/experiment.py`
- `core/evolution/domains/research_pv.py`（领域包）
- 演示数据生成器 + 固定 seed 的 `data/demo_panel.csv` + `sha256`
- `data/sample.csv`（给 D 联通用）

### 4.3 具体要做的事

1. CSV 接入与校验：
   - 必含 `date/symbol/close/volume`；
   - `(date, symbol)` 唯一；
   - 支持列名映射（如 `adj_close → close`）；
   - 输出数据画像：日期范围、标的数、缺失率、类别平衡、标签可计算比例。
2. 标签生成：
   - t 日特征截止，`fwd5 = close[t+5]/close[t] - 1`；
   - 同日截面中位数，严格大于记为 1；
   - 缺 t+5 的样本排除并计数；禁止用下一条行情补。
3. 切分（冻结后不许改）：
   - 按日期整组：60% train / 各自 10% 的两个验证窗 / 20% 独立测试；
   - 每个分段尾部清除标签跨界的样本，按 5 个交易日预留；
   - 输出 `split.json` + 切分指纹 + seed。
4. 特征组：
   - 基础收益：`r5, r20`；
   - 历史波动率：滚动标准差、波动比例；
   - 成交量变化：相对成交量、量比、量能变化率；
   - 滚动特征必须按标的、只用历史和当日。
5. 实验执行：
   - 基线 `LogisticRegression[r5, r20]`；
   - 候选由 `ExperimentSpec` 指定，只能改一个特征组或一个模型设置；
   - 插补、缩放、模型只 fit 在 train；
   - 输出 `TrialResult`：两窗 `daily_auc`, valid_days, excluded, missing_rates, seconds, error。
6. 领域包 `research_pv.py`：
   - `spec`：features / model / missing / scaling 轴；
   - `plan(view, run_kind)`：按当前策略生成下一批实验；
   - `interpret(feedback, view)`：把 TrialResult 变成 Signal；
   - `predict(view, payload)`：判断 high/low；
   - `judge(task, attempt)`：交给 oracle 出最终判决。

### 4.4 消费与提供

| 方向 | 内容 |
|---|---|
| 消费 A | ExperimentSpec、停止/重试指令（不消费 LLM 权限） |
| 提供 A | `describe_data()`、`baseline()`、`run_candidate(spec)` |
| 提供 C | 锁定配置的独立测试入口、TrialResult、数据指纹、切分指纹 |
| 提供 D | 数据画像、日期范围、字段说明（只读摘要） |

### 4.5 小时级安排

| 时间 | 完成条件 |
|---|---|
| 0-1h | 确认 CSV 与列名映射；生成 `sample.csv`/`demo_panel.csv`；写 split 骨架与指纹 |
| 1-4h | 标签成熟、特征组、基线、候选、日均 AUC 跑通；H4 基线数字进入 UI |
| 4-6h | 与 A 联调第一个端到端；至少演示一次“相对成交量让 val2 下滑 → REVISED”路径 |
| 6-8h | 边界用例：缺字段、缺失率>30%、唯一性破、重复指纹；锁演示数据和参数 |
| 8-10h | 与 C 对指纹；确认任何一臂改了数据都会导致不一致报警 |

### 4.6 B 自己的测试

- `(date,symbol)` 唯一；
- t+5 缺失被排除并计数；
- 标签尾部按 5 日清除；
- 滚动特征只用历史与当日；
- 插补/缩放/模型只在训练段 fit；
- 单轮只改一个特征组或一个模型设置；
- 日均 AUC 只统计两类都存在的有效日。

### 4.7 B 的验收标准

- 同一输入 + 同一 seed 能重建同一 `TrialResult`；
- `split.json / data hash / feature fingerprint` 与 C 完全一致；
- 演示数据前置门通过，且有一组可被证伪的候选。

---

## 5. C：评测与复现（oracle + evaluate + replay + pre_gate）

### 5.1 一句话职责

C 是“裁判和账房”：决定 holdout 怎么切、分数怎么出、三臂怎么比、复现怎么验；C 不出研究结论，只对事实盖章。

### 5.2 交付文件

- `autoresearch/pre_gate.py`
- `autoresearch/oracle.py`
- `autoresearch/evaluate.py`
- `autoresearch/replay.py`
- `contracts.py` 的 pydantic/jsonschema 校验与样例 JSON
- `tests/`（评测侧测试）

### 5.3 具体要做的事

1. `pre_gate.py`：
   - 固定基线 + 3 个候选，在两个验证窗各跑一遍；
   - 算合并日均 AUC，配对 bootstrap 95% CI；
   - 输出 `pre_gate.json`：两窗 AUC、CI、有效天数、排除样本数、结论；
   - 不通过就返回失败，A 直接拒绝开工。
2. `oracle.py`：
   - 持有切分指纹、数据指纹、holdout 哈希；
   - 拒绝锁定前的独立测试请求；
   - 计算 `daily_auc` 均值、有效天数、排除数；
   - 做泄漏检查：id/时间列泄漏、全量 fit、目标进特征、测试段被访问；
   - 返回 `supported / refuted / inconclusive / blocked / partial_completed`；
   - 每个 verdict 记录证据编号与 `artifacts_hash`。
3. `evaluate.py`：
   - 三臂：A 固定顺序本地搜索 / B 贪婪云 / C 证据触发主臂；
   - 每臂记录：闭环完成、独立测试 AUC、候选数、云调用数、input/output token、费用（有单价才填）、墙钟、完成率；
   - 输出 `metrics.csv` / `metrics.json`；缺失 usage 标 `estimated/unknown`。
4. `replay.py`：
   - 用事件流、ledger、配置、seed 重建流程；
   - 不重新调用云模型；
   - 校验复现包是否与原运行一致。
5. `contracts.py` 校验：
   - `TaskSpec / ExperimentSpec / TrialResult / RunEvent` 的 schema；
   - 保存一份合法样例 + 非法样例；非法样例必须被拒绝。

### 5.4 消费与提供

| 方向 | 内容 |
|---|---|
| 消费 B | 锁定配置的独立测试入口、TrialResult、指纹 |
| 消费 A | RunEvent、token_ledger、状态版本 |
| 提供 A | oracle.judge、pre_gate 结论、三臂指标 |
| 提供 D | 指标 CSV/JSON、复现包校验结果、回放命令 |

### 5.5 小时级安排

| 时间 | 完成条件 |
|---|---|
| 0-1h | 冻结 contracts 与样例；oracle 骨架 + fake experiment 能跑；预演“伪造实验被拒” |
| 1-4h | pre_gate 在演示数据上跑出结论；oracle 完整实现；H4 时 B 的基线分数由 oracle 复核 |
| 4-6h | evaluate 三臂骨架；首轮端到端的 verdict 与 metrics 可复算 |
| 6-8h | 三臂正式跑；对抗用例：伪造无泄漏实验、预算超限、切分被改；H8 冻结评测口径 |
| 8-10h | replay 全量复现；复现包校验通过；现场评分脚本唯一可用 |

### 5.6 C 自己的测试

- 伪造“无泄漏但分数变好”的实验必须被拒绝；
- 锁定前请求 holdout 必须失败；
- holdout 被改哈希必须报警；
- CI 跨 0 返回 `inconclusive`；
- 预算拦截在 ledger 上可核对；
- replay 结果与原运行一致，且没有云调用。

### 5.7 C 的验收标准

- `pre_gate.json` 通过；
- 评审问“提升是不是真的”时，C 能当场用 oracle + replay 复出同样的数字；
- 三臂指标表不是 PPT，是 `evaluate.py` 的输出。

---

## 6. D：产品界面与集成（app + report）

### 6.1 一句话职责

D 是“脸”：把 A 的事件、C 的指标变成一页能操作的 Streamlit，并负责导出报告和复现包；D 不做任何研究决策。

### 6.2 交付文件

- `autoresearch/app.py`
- `autoresearch/report.py`
- `autoresearch/mock_events.json`（离线联调）
- 页面截图/演示脚本

### 6.3 具体要做的事

1. 单页 Streamlit 三个区块：
   - 创建任务：选择演示数据/导入 CSV、显示日期范围与字段、设置预算后启动；
   - 跟踪研究：逐轮显示假设、数据证据、验证窗分数、失败原因和下一步；右侧显示 token、调用次数、耗时、停止原因；
   - 审查结论：固定基线与锁定候选的独立测试对照、保留基线或结束研究理由、下载复现包。
2. 运行方式：
   - 用 `session_state + 子进程 + 事件文件轮询`；
   - 页面轮询 `.arc/runner-events.jsonl` 显示进度；
   - 不用长连接，不阻塞页面 180 秒。
3. `report.py`：
   - 从 plan/hypotheses/experiments/verdicts/metrics 生成 Markdown 报告；
   - 打包复现目录：数据哈希、split、seed、事件、ledger、指标、依赖 lock、运行命令；
   - 提供“下载复现包”按钮。
4. 离线联调：
   - 前几小时用 `mock_events.json` 模拟研究时间线和结果；
   - UI 不依赖真实 LLM 也能展示全部页面状态。
5. 降级：
   - Streamlit 环境损坏时，提供 `python -m autoresearch.cli run` 的 CLI 路径；
   - 评审现场 UI 失败时明确标“回放”，并用 CLI 顶替。

### 6.4 消费与提供

| 方向 | 内容 |
|---|---|
| 消费 A | RunEvent、剩余预算、停止原因、`token_savings` 读数 |
| 消费 C | metrics、verdicts、复现包校验结果 |
| 消费 B | 数据画像、日期范围（只读摘要） |
| 提供评审 | 可操作的评审界面、报告下载、复现包下载 |

### 6.5 小时级安排

| 时间 | 完成条件 |
|---|---|
| 0-1h | 页面三区块布局 + `mock_events.json`；下载按钮先接假文件 |
| 1-4h | 用 mock 事件驱动时间线和预算面板；H4 时 B 的基线能在页面显示 |
| 4-6h | 接 A 真实事件流；第一个端到端结果可下载报告；H6 出第一版复现包按钮 |
| 6-8h | 低预算拦截可视化、budget_mode/overrun、停止原因；H8 页面功能冻结，只做细节 |
| 8-10h | 彩排：创建→跟踪→审查→下载；确认 CLI 回放路径可用 |

### 6.6 D 自己的测试

- mock 事件下时间线顺序正确；
- 预算不足时页面显示剩余预算、`budget_mode` 与 `budget_blocked`，且运行仍 TESTED；
- 跟踪页/报告显示 `token_savings`：上下文压缩、缓存命中、本地规则、避免云调用与总节省；
- 报告下载包含复现目录；
- 页面刷新后状态能从事件文件恢复；
- 不引入任何会写研究数据的功能。

### 6.7 D 的验收标准

- 一次端到端：创建任务 → 跟踪研究 → 审查结论 → 下载复现包；
- 现场 3 分钟演示可以由 D 的页面主导；
- UI 异常时 CLI 回放路径完整。

---

## 7. 四人交叉验收时间点

| 时间 | 验收 | 谁验收谁 |
|---|---|---|
| H1 | contracts 冻结、样例 JSON 合法/非法样例通过 | C 出校验，四人签字 |
| H4 | 基线在 UI 可见；数据/切分指纹在 B 与 C 一致 | D 验收 B 的摘要，C 验收 B 的指纹 |
| H6 | 第一个端到端闭环 + 复现包 | A 验收 C 的 verdict，C 验收 A 的 ledger，B 验收 A 的路由，D 验收 A 的事件流 |
| H8 | 三臂、低预算、边界用例全部跑过；P0 冻结 | C 出指标，B 锁数据，A 锁路由，D 锁页面 |
| H10 | 干净环境启动 + 全流程彩排 | 四人各按自己的验收标准过一遍 |

## 8. H6 未跑通时的降级

触发条件：主闭环（基线→候选→验证→修正→锁定→测试→报告）H6 仍未跑通。

立即执行：

1. D 停止 UI 美化，只保留单页表格 + 下载按钮；准备 CLI 回放；
2. C 停做 P1/第二数据/额外图表，只保证 oracle/evaluate/replay 可用；
3. B 停做新特征组，锁住已经能出分的特征与数据；
4. A 取消所有非必要云路由分支，只保留“两窗矛盾 → 云诊断”和“收尾成文”两个触发；
5. 若 30 分钟内仍未跑通，四人按 A↔B、C↔D 结对排障，不再各自开新分支。

## 9. 三分钟演示各人负责的环节

| 时间 | 环节 | 谁 | 具体动作 |
|---|---|---|---|
| 0:00-0:35 | 研究问题、数据、预算 | A 讲，B 投数据画像，C 投 pre_gate | A 报任务与预算；B 展示日期范围/字段/缺失；C 展示前置门 3 个数字 |
| 0:35-1:35 | 被推翻的假设 + 真实云调用或跳过 | A 操作，B 解释实验，C 出 verdict | A 触发或回放 REVISED；B 指出哪组特征导致 val2 下滑；C 展示证据编号和拦截记录 |
| 1:35-2:25 | 独立测试与成本对照 | C 讲，D 投界面 | C 解释三臂表；D 打开审查结论页展示对比和下载 |
| 2:25-3:00 | 复现包 + 低预算降级 | D 操作，C 复核 | D 下载复现包并演示低预算 `budget_blocked` 后仍 TESTED；C 跑一条 `replay` 命令 |

若现场断网/UI 异常：D 明确说“现在切 CLI 回放路径”，A 继续用本地规则路径跑；不能假装是实时云调用。

## 10. 最终交付包清单（每个人签收自己的部分）

| 项 | 负责人 | 状态 |
|---|---|---|
| `pre_gate.json` + 前置门日志 | C | ☐ |
| `demo_panel.csv` + `sha256` + 生成 seed | B | ☐ |
| `task.json / plan.json / hypotheses.json` | A | ☐ |
| `experiments/*.json`（TrialResult） | B | ☐ |
| `verdicts.json` + oracle 日志 | C | ☐ |
| `.arc/runner-events.jsonl` + `token_ledger.json` | A | ☐ |
| `metrics.csv / metrics.json`（三臂，含每 10k token 提升） | C | ☐ |
| `research.token_savings` 事件与 summary 字段 | A（D 在报告展示） | ☐ |
| `report/final.md` | D（数据来自 A/C） | ☐ |
| 复现包 zip + `replay` 输出 | D（内容来自 C） | ☐ |
| requirements lock + 离线 wheel | B/C 协同 | ☐ |
| 干净环境启动记录 | 全员按 D 脚本验收 | ☐ |
| P0 验收清单逐项打勾 | 四人 | ☐ |

---

## 11. 一句话给每人

- A：你只回答“下一步该干什么，这笔 token 花不花”，别自己动手做数据或评分。
- B：你只回答“这份数据长什么样、这个实验跑出什么”，别碰预算和裁判。
- C：你只回答“这结论能不能算数、能不能复现”，别下场做假设或训练。
- D：你只回答“让评委怎么看得懂”，永远别在你的页面里写研究结论。

