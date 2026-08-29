# Evals

`evals` 用固定合成项目验证四类业务能力，与测试 Agent 自身代码的 `tests` 分开。

## 任务分布

- C1-C2：从零创建计算器 CLI 和 TODO CLI。
- F1-F2：增加幂运算和删除任务功能。
- B1-B2：修复分页边界和 JSON 持久化问题。
- E1-E2：只读解释分层项目和带配置的项目。

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

# 8 题基线
.\.venv\Scripts\python.exe -m evals --run all

# 最终稳定性评测，共 16 次
.\.venv\Scripts\python.exe -m evals --run all --repeat 2
```

生成结果保存在被 Git 忽略的 `evals/results/<时间>/`，包括每题 Trace、回答、文件补丁、验收输出和结果，以及整轮 `results.json`、`report.md` 和 `failure_cases.jsonl`。失败后结合 Trace，按 `failure_categories.json` 填写 `diagnosis` 和后续改进，不根据单个答案编写特例。
