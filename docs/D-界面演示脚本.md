# D 界面评审演示脚本（3 分钟）

## 准备

```bash
pip install -r requirements-app.txt
python -m streamlit run autoresearch/app.py
```

离线联调不需要真实 LLM 或 C 的 oracle；`autoresearch/mock_events.json` 已内置完整闭环与低预算两个场景。

## 0:00 - 0:30 创建任务

- 左侧数据源选“离线演示（mock_events.json）”。
- 在“创建任务”里使用内置演示数据 `data/demo_panel.csv`，预算候选数 4、云 token 6000。
- 点击“启动研究”，页面把任务与演示事件写入 `.arc/demo/`。

## 0:30 - 1:30 跟踪研究

- 选择完整运行 `R20261114-001`，查看时间线：基线 → 假设 → 实验 → 证据修正 → 锁定 → 独立测试。
- 右侧展示剩余实验、云调用、token、时间与停止原因。
- 切换到低预算运行 `R20261114-002`，展示 `budget_blocked:cloud_token_reserve_exhausted` 与 `PARTIAL_COMPLETED`。
- 可勾选“自动刷新（2 秒）”验证事件文件轮询。

## 1:30 - 2:20 审查结论

- 对比固定基线 `0.661491` 与锁定候选 `exp-004`（`tree_shallow`，merged `0.675808`）。
- 查看独立测试判决 `supported` 与测试 AUC `0.6842`。
- 点击“下载 Markdown 报告”和“下载复现包 zip”。

## 2:20 - 3:00 CLI 降级回放

若 Streamlit 或现场环境异常：

```bash
python -m autoresearch.cli demo
```

同一份 `mock_events.json` 会以时间线形式输出，可继续评审；真实后台运行使用 C 已实现的 oracle：

```bash
python -m autoresearch.cli run --task .arc/task.json --oracle autoresearch.oracle:ResearchOracle
```
