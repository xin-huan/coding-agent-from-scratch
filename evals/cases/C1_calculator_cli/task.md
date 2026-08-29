请在当前空目录中从零创建一个仅使用 Python 标准库的计算器 CLI 项目。

必须支持以下调用：

- `python -m calculator add 2 3` 输出 `5`
- `python -m calculator subtract 8 3` 输出 `5`
- `python -m calculator multiply 2.5 4` 输出 `10`
- `python -m calculator divide 10 4` 输出 `2.5`

未知运算、非数字参数和除以零必须返回非零退出码，并输出简明错误信息。项目还应包含 README 和可通过 `python -m unittest discover -s tests -v` 运行的测试。完成后请主动运行测试并如实报告结果。
