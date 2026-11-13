# ARC Agent Framework

本目录来自 `agent-arc-based`：ARC 智能体框架（需求树、阶段调度、阶段纪律、可追溯、事件流）以及在其基础上扩展的进化层 `core/evolution`。

## 关键目录

| 路径 | 内容 |
|---|---|
| `core/evolution/` | 自进化运行时：领域契约、策略版本、闸门、命中率、回放、回退 |
| `agents/` | 阶段 Agent、模型适配、工具与技能 |
| `app_type_handler/` | 按应用类型验收 |
| `arcbench_agent_runtime/` | 事件、上下文、可追溯存储 |
| `skills/` | 阶段技能说明 |
| `template/`、`arc-template/` | 项目模板 |
| `docs/EVOLUTION.md` | 进化层设计说明 |

## 常用命令

```bash
python -m core.evolution.cli demo --domain habitfit --db core/evolution/_data/demo.db --fresh
python -m core.evolution.cli status --domain habitfit
python core/evolution/tests/run_all.py
```

## 说明

- `.env`、`.arc/`、`__pycache__`、`.pyc`、SQLite 数据库等运行产物不入库；
- `core/evolution` 保持零第三方依赖可运行，不依赖 langchain/deepagents；
- 新领域包只实现 `spec / plan / interpret / predict / judge`。

