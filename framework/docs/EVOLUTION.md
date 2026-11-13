# 进化层（core/evolution）——在 agent-arc-based 之上

> 这份文档回答一个问题：**为什么进化层长在 ARC 仓库里，却几乎没用到 ARC 的流水线代码。**
> 结论先说：以 ARC 的**分层、约定、事件格式、配置规则**为基础；不把它的**代码生成流水线**搬过来，因为那是另一件事。

---

## 1. 先看清 ARC 到底是什么

| 项 | 事实（读了代码之后的） |
|---|---|
| 体量 | 24 个 py 文件 / 14284 行（不含模板） |
| 核心对象 | **requirement node**（需求树节点），不是「一次运行」 |
| 主流程 | `DESIGN`（interface_designer）→ `TEST_GENERATION` → `IMPLEMENT`（TDD developer），见 `core/phases.py`、`core/workflow.py` |
| 技术栈 | langchain / deepagents / openai（`requirements.txt`），当前这台机器的 Python 环境**没装**（实测 `langchain`、`deepagents`、`openai` 均 ModuleNotFound） |
| 产出 | 一个能跑的应用（web / android / cli），由 `app_type_handler/*` 分别验收 |
| 状态机 | 节点级：`UNSEEN → DESIGNING → DESIGNED → IMPLEMENTING → PASSED / CONVERGED / FAILED` |
| 一个易混点 | `core/workflow.py` 里的 `WORKSPACE_MODE_EVOLUTION` 指的是**「在已有应用上继续加需求」的代码演进**，不是「策略随反馈变好」的策略进化 |

所以它的强项是「**把需求写成代码，并证明代码能跑**」；它没有「同一主体第二次运行要比第一次好」这个概念，
也没有版本化策略、预测、回退语义。这正是评审必问的三件事（两次指标 / 学到什么 / 怎么退回）落空的地方。

---

## 2. 用了什么、没用什么

### 2.1 用了（约定与基础设施）

| ARC 的资产 | 位置 | 进化层怎么用 |
|---|---|---|
| 仓库分层（`core/` 放编排与内核、`agents/` 放模型与技能） | 仓库根 | 进化层放 `core/evolution/`，与 `core/phases.py` 平级，不另起仓库 |
| `.env` 解析规则（KEY=VALUE、不覆盖已有变量、`OPENAI_KEY→OPENAI_API_KEY`） | `core/config.py::load_project_env` | `arc_bridge.load_env()` 同规则重写（不 import，见 §3） |
| 事件文件路径变量 `ARCBENCH_RUNNER_EVENTS_PATH`（默认 `.arc/runner-events.jsonl`） | `arcbench_agent_runtime/context.py` | `arc_bridge.paths()` 同变量名、同默认值 |
| 事件行格式 `{"type","timestamp",...}` | `arcbench_agent_runtime/events.py` | `arc_bridge.mirror_events()` 输出 `evolution.*` 前缀行，与 ARC 事件同文件 |
| 「阶段纪律」思想：模型只能写允许写的路径、有写配额 | `agents/runtime/stage_discipline.py` | 换成**闸门**：模型/规则只能通过 `learn.gate()` 改策略，不能直接写库 |
| 「按应用类型分别验收」思想 | `app_type_handler/*.py` | 换成 `verify.py`：按指标验收（命中率 / 冲突 / 重复犯错 / 新题率） |
| 「可追溯」思想：产物对应需求 | `arcbench_agent_runtime/traceability.py` | 换成**版本链 + 事件**：每次策略变动对应哪批证据、哪次运行 |

### 2.2 没用（有意为之）

| 资产 | 为什么不用 |
|---|---|
| `core/phases.py`（1491 行）、`core/workflow.py`（1338 行） | 它们的输入是需求树、输出是代码；我们的输入是执行证据、输出是策略快照。硬接会让 90% 的分支永远走不到 |
| `agents/runtime/factory.py`（584 行）+ `deepagents` 文件权限 | 依赖 langchain 全家桶；现场断网/没装依赖就演不了 |
| `app_type_handler/web.py`（1296 行）的 Playwright 验收 | 它验的是「生成的网站能不能点」，我们验的是「预测准不准」 |
| `scripts/parallel_compile.py`（826 行）分片并行 | 单主体的两次运行天然串行，不需要分片 |

**如果一定要一句话**：ARC 是"代码工厂"，进化层是"产品运行时"。共用仓库、共用约定、共用日志；不共用流水线。

---

## 3. 为什么不直接 `import core.config` / `arcbench_agent_runtime.events`

因为它们会把 langchain / colorama 一起拉进来。实测这台机器：

```
MISS langchain / langchain_openai / deepagents / openai
 OK  pydantic / yaml
```

而进化层的第一条要求是**裸 Python 就能跑**（现场断网、评审机器、备用笔记本）。
所以 `arc_bridge.py` 只**复刻约定**（约 120 行），不 import。代价是两份 .env 解析逻辑，收益是 `python -m core.evolution.cli demo` 在任何机器上都能跑。

---

## 4. 目录与入口

```
agent-arc-based/
  core/
    phases.py            # ARC 原有：DESIGN / IMPLEMENT
    workflow.py          # ARC 原有：需求树调度
    evolution/           # 新增：自进化运行时（零第三方依赖）
      spec.py            # 领域契约：轴 / 假设 / 证据 / 任务 / 结果
      policy.py          # 快照构建、内容 hash、diff、restore
      learn.py           # 假设表学习器（elimination / UCB）+ 闸门
      metrics.py         # 命中率 / 冲突 / 重复犯错 / 新题率 / 回放
      db.py store.py     # SQLite：版本只追加、事务、幂等
      runtime.py         # start_run → plan → report → feedback → rollback
      verify.py          # 程序化验收
      api.py server.py   # 零依赖 HTTP（给前端联调）
      cli.py             # demo / status / audit / rollback / verify / serve / export
      arc_bridge.py      # 与 ARC 的接线（.env / 事件路径 / jsonl 镜像）
      domains/           # habitfit.py / zhijie.py（业务只写这里）
      tests/             # 5 套件 / 107 项断言，全离线
  docs/EVOLUTION.md      # 本文
```

```bash
# 在仓库根目录执行
python -m core.evolution.cli demo --domain habitfit --db core/evolution/_data/demo.db --fresh
python -m core.evolution.cli status --domain habitfit
python -m core.evolution.cli audit --db core/evolution/_data/demo.db --verbose
python -m core.evolution.cli export            # 事件镜像进 .arc/runner-events.jsonl
python -m core.evolution.cli serve --port 8000 # 前端联调
python core/evolution/tests/run_all.py         # 107 项断言
```

演示输出（与 PRD 数字一致）：

```
[1] 第一次运行 R1  锁定 V0   预测命中率 1/7 ｜ 冲突 3
[2] 反馈 → applied：exercise_time H1→H2，sleep_time H6→H7，闸门回放 1/7→1/7
[3] 第二次运行 R2  锁定 V1   预测命中率 7/7 ｜ 冲突 1 ｜ 重复犯错 0 类
[4] 回退 V2 内容等同 V0：True
```

---

## 5. 契约（要改先改文档，再改代码）

### 5.1 表

| 表 | 不变量 |
|---|---|
| `policy_versions` | 只追加；`UNIQUE(subject,domain,seq)`；`snapshot_hash` 排除 `seq/meta`，所以「V2 内容等同 V0」可用两行相同 hash 举证 |
| `policy_state` | 单行指针，与版本、事件、反馈**同事务**更新 |
| `runs` | 创建时锁定 `policy_version_id`，之后版本推进不影响它 |
| `tasks` | `prediction ∈ {high,low}`，在结果之前落库 |
| `outcomes` | 一任务一结果，重复上报返回原值 |
| `feedback` | `UNIQUE(run_id,submission_key)`：重发不产生第二份版本 |
| `events` | 追加写；`policy_init / policy_updated / rollback / run_* / task_outcome` |
| `exposures` | 发题前登记，用于算新题率 |

### 5.2 HTTP

`POST /api/runs`、`GET /api/runs/{id}`、`POST /api/runs/{id}/attempts`、`POST /api/runs/{id}/finish`、
`POST /api/runs/{id}/feedback`、`GET /api/admin/evolution`、`POST /api/admin/rollback`。

`disposition` 只有五个值：`applied` / `no_change` / `gate_rejected` / `baseline_changed` / `idempotent`（标记）。

### 5.3 领域包（业务只写这四个方法）

`spec` / `plan(view, run_kind)` / `interpret(feedback, view)` / `predict(view, payload)`（+ 可选 `judge`）。

**规则**：`plan` 必须给每条任务附预测；写不出预测的领域不该上场，因为主指标（命中率）失去意义。

---

## 6. 评审要问的三件事，落在哪

| 评审问 | 代码位置 | 现场怎么演 |
|---|---|---|
| 指标两次各是多少 | `metrics.prediction_stats` + `tests/test_e2e.py` | demo 第 1、3 步；`status` 也行 |
| 学到了什么 | `policy.diff` + `events` | demo 第 2 步打出 `H1→H2`、`refute 0→5` |
| 怎么退回 | `Runtime.rollback` | `python -m core.evolution.cli rollback`，20 秒 |

---

## 7. 红线

1. 两次运行之间**只允许**：给反馈、等事务、重启服务。不许改代码 / 参数 / 模型 / 提示词。
2. 不许为了好看改指标；要改先改本文 §5，再改代码，最后改 PPT。
3. `gate_rejected` 要展示，不许藏——被自己的闸门拒掉是"有纪律"的证据。
4. 领域包不许直接写 `policy_*` 表；只能返回证据，由 `learn.gate()` 决定是否落库。
5. 不许让 `core/evolution` 依赖 langchain / deepagents：它必须能在裸 Python 下跑完两次运行。
