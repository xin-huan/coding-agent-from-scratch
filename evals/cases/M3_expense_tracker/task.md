请在当前空目录中从零创建一个仅使用 Python 标准库的记账 CLI 项目。

项目包名必须是 `expense_tracker`，并至少按职责拆分为：

- `models.py`：支出数据结构。
- `storage.py`：JSON 读取和保存。
- `service.py`：新增支出和计算总额的业务逻辑。
- `cli.py`：参数解析与输出。
- `__main__.py`：支持模块启动。

必须支持：

- `python -m expense_tracker --data expenses.json add 12.50 "Lunch"`
- `python -m expense_tracker --data expenses.json list`，每项格式为 `1 | 12.50 | Lunch`
- `python -m expense_tracker --data expenses.json total`，只输出两位小数总额，例如 `12.50`

数据必须在多个进程间通过合法 JSON 持久化。金额必须大于 0，说明不能为空；非法输入返回非零退出码。项目还应包含 README，以及至少覆盖新增、持久化、汇总和错误输入的测试，可通过 `python -m unittest discover -s tests -v` 运行。

完成后请主动运行全部测试并如实报告结果。
