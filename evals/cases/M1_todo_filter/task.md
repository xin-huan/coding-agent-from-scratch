请为当前分层 TODO 项目增加按关键词筛选任务的功能，并保持已有功能不变。

具体要求：

- `TodoService.list_tasks(keyword=None)` 接受可选关键词。
- 关键词匹配任务标题，不区分大小写。
- 未提供关键词或关键词只有空白时，返回全部任务；没有匹配时返回空列表。
- CLI 的 `list` 命令支持 `--keyword TEXT`，例如 `python -m todo_app --data tasks.json list --keyword code`。
- 修改应遵守现有分层职责，并为新行为增加项目测试。
- 运行全部项目测试并如实报告结果。

不要改变 JSON 存储格式，也不要破坏现有的 `add` 和无参数 `list` 行为。
