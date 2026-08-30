# Evals

`evals` 用固定合成项目验证四类业务能力，与测试 Agent 自身代码的 `tests` 分开。

## 任务分布

- C1-C2：从零创建计算器 CLI 和 TODO CLI。
- F1-F2：增加幂运算和删除任务功能。
- B1-B2：修复分页边界和 JSON 持久化问题。
- E1-E2：只读解释分层项目和带配置的项目。
- M1：跨 CLI、Service 和测试增加关键词筛选。
- M2：先复现已有失败测试，再修复折扣边界 Bug。
- M3：从零创建包含五个职责模块的记账 CLI。
- M4：使用自然语言需求从零创建番茄钟桌面 App，工程要求由 Agent 主动完成。

外置验收器和参考答案不会复制到 Agent 工作区。每次运行都使用新的工作区副本。

## 命令

以下命令不调用模型：

```powershell
.\.venv\Scripts\python.exe -m evals --list
.\.venv\Scripts\python.exe -m unittest tests.test_evals -v
```

以下命令会读取项目根目录的 `.env` 并产生 DeepSeek API 调用：

```powershell
# 单题试运行
.\.venv\Scripts\python.exe -m evals --run C1

# 8 题核心基线
.\.venv\Scripts\python.exe -m evals --run C1 C2 F1 F2 B1 B2 E1 E2

# 3 题中等难度基线
.\.venv\Scripts\python.exe -m evals --run M1 M2 M3

# 全部 12 题
.\.venv\Scripts\python.exe -m evals --run all

# 最终稳定性评测，共 22 次
.\.venv\Scripts\python.exe -m evals --run all --repeat 2
```

生成结果保存在被 Git 忽略的 `evals/results/<时间>/`，包括每题 Trace、回答、文件补丁、验收输出和结果，以及整轮 `results.json`、`report.md` 和 `failure_cases.jsonl`。失败后结合 Trace，按 `failure_categories.json` 填写 `diagnosis` 和后续改进，不根据单个答案编写特例。
