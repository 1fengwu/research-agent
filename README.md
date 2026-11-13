# Research Agent System（AutoResearch）

Research Agent System 的竞赛交付仓库：面向量价信号验证场景，以 ARC agent 框架为内核，把“规划—收集证据—提出假设—实验/分析—验证—修正”做成一条可追溯、可复现、可回退、可计费的研究闭环。

## 目录

| 目录 | 内容 |
|---|---|
| `docs/ResearchAgent-系统设计-v5.md` | 系统设计 v5：AutoResearch 量价信号验证工作台 |
| `docs/AutoResearch-四人分工执行手册.md` | 四人 A/B/C/D 分工、小时级排期、接口、测试、红线、演示 |
| `framework/` | ARC agent 框架源码（agent-arc-based） |

## 快速入口

- 设计总览：[docs/ResearchAgent-系统设计-v5.md](docs/ResearchAgent-系统设计-v5.md)
- 四人执行手册：[docs/AutoResearch-四人分工执行手册.md](docs/AutoResearch-四人分工执行手册.md)
- ARC 框架说明：[framework/docs/EVOLUTION.md](framework/docs/EVOLUTION.md)

## 上传注意

仓库不包含 `.env`、`.arc` 运行目录、`__pycache__`、`.pyc`、SQLite 数据库、依赖缓存和构建产物；相关文件由 `.gitignore` 排除。

