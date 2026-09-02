# coding-agent-from-scratch

Git 仓库：https://github.com/xin-huan/coding-agent-from-scratch.git

这是一个本地 Coding Agent 系统。它直接调用 OpenAI 兼容模型接口，但 Agent 循环、工具定义、本地文件/命令执行、上下文压缩、项目记忆、SubAgent、聊天界面、Session 树和工作区快照都由项目自己实现。

## 如何运行

先参考 `.env.example` 配置 API Key，并保存为 `.env`，然后执行：
```powershell
cd 项目文件夹
.\.venv\Scripts\python.exe -m coding_agent.chat --port 8767
```

然后打开 http://127.0.0.1:8767 。

## 特色功能

1.**树状 Session 与代码快照恢复**。每轮对话保存为带 id/parent_id 的树节点，用户可以从历史节点继续形成新分支，同时保留旧分支。Agent 完成任务后会保存相关文件快照，用户可显式恢复到某一轮完成后的代码状态，并在恢复前备份当前文件。

2.**基于生命周期 Hook 的 Extension 架构**。主循环只负责模型调用、工具执行、状态推进和终止判断；项目记忆、skill 选择、subagent 和上下文压缩都通过 Extension 接入，便于扩展复杂能力而不重写核心循环。

3.**面向可靠交付的任务状态机**。TaskState 会维护任务阶段、修改文件、最近命令和测试状态。文件修改后必须完成相关验证；测试失败会进入修复阶段；只有没有缺失交付物、没有未验证实现改动，并且模型给出最终回复时才结束循环。

4.**上下文与记忆管理**。系统区分单次任务上下文、对话级历史和项目记忆：单次任务上下文服务当前执行，对话级历史支持回看、分支和恢复，项目记忆让同一项目的不同对话保持连续性。长工具输出会外置为 id，历史过长时会压缩较早工具交互并保留最近工作集。

5.**SubAgent 审查机制**。复杂任务中可以委派 researcher、tester、reviewer。Reviewer SubAgent 会在复杂改动后独立审查 patch、风险和验证建议，但最终决策仍由主 Agent 负责。


## 实验验证

项目包含多组本地 Eval，覆盖创建项目、增加功能、修复 Bug、解释项目和异常恢复。实验中发现部分失败来自 Runtime 收尾和评分器误判，而不只是模型能力不足，因此加入确定性终止、澄清门、检查点恢复和交付验证机制。结果显示这些机制能提升 Agent 完成真实编程任务的稳定性。