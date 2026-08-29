请在当前空目录中从零创建一个仅使用 Python 标准库的 TODO CLI 项目，任务保存在 `--data` 指定的 JSON 文件中。

必须支持：

- `python -m todo --data tasks.json add "Write tests"`
- `python -m todo --data tasks.json list`，未完成任务显示为 `[ ] 1: Write tests`
- `python -m todo --data tasks.json done 1`
- 再次 list 时已完成任务显示为 `[x] 1: Write tests`

多个进程之间必须能正确保存和重新读取数据。空标题、不存在的任务编号等错误输入必须返回非零退出码。项目还应包含 README 和可通过 `python -m unittest discover -s tests -v` 运行的测试。完成后请主动运行测试并如实报告结果。
