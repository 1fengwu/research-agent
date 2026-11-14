# Research Agent System（AutoResearch）

Research Agent System 的竞赛交付仓库：面向量价信号验证场景，以 ARC agent 框架为内核，把“规划—收集证据—提出假设—实验/分析—验证—修正”做成一条可追溯、可复现、可回退、可计费的研究闭环。

## 目录

| 目录 | 内容 |
|---|---|
| `docs/ResearchAgent-系统设计-v5.md` | 系统设计 v5：AutoResearch 量价信号验证工作台 |
| `docs/AutoResearch-四人分工执行手册.md` | 四人 A/B/C/D 分工、小时级排期、接口、测试、红线、演示 |
| `docs/后端数据契约.md` | 前端、API、评测/复现共同消费的输出字段契约 |
| `framework/` | ARC agent 框架源码（agent-arc-based） |
| `autoresearch/` | A/B/C/D 已实现模块：A 编排/路由/预算门/契约，B 数据/实验/规划器，C 前置门/裁判/三臂指标/复现，D 界面/报告/复现包，研究策略自进化桥 |
| `data/` | B 固定 seed 演示面板、切分与画像 |
| `requirements.txt` | 根目录 Python 依赖（pandas、scikit-learn） |
| `requirements-app.txt` | D 页面额外依赖（streamlit） |

## 快速入口

- 设计总览：[docs/ResearchAgent-系统设计-v5.md](docs/ResearchAgent-系统设计-v5.md)
- 四人执行手册：[docs/AutoResearch-四人分工执行手册.md](docs/AutoResearch-四人分工执行手册.md)
- ARC 框架说明：[framework/docs/EVOLUTION.md](framework/docs/EVOLUTION.md)
- A/B/C/D 模块说明：[autoresearch/README.md](autoresearch/README.md)
- D 页面入口：[autoresearch/app.py](autoresearch/app.py)

## 上传注意

仓库不包含 `.env`、`.arc` 运行目录、`__pycache__`、`.pyc`、SQLite 数据库、依赖缓存和构建产物；相关文件由 `.gitignore` 排除。

