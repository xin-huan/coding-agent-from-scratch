请为当前 TODO 项目增加删除任务功能，同时保持已有功能不变。

具体要求：

- 核心模块提供 `delete_task(tasks, task_id)`。
- CLI 支持 `python -m todo_app --data tasks.json delete 1`。
- 删除成功后数据文件中不再包含该任务。
- 删除不存在的编号必须返回非零退出码并给出明确错误。
- 增加测试并运行全部项目测试。

完成后请如实说明修改内容和测试结果。
