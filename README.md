# coding-agent-from-scratch
A lightweight coding agent built from scratch with native LLM tool calling, local tool execution, context management, and a custom agent loop.

## Usage

Run the terminal agent:

```powershell
.\.venv\Scripts\coding-agent.exe --workspace D:\path\to\project
```

Run the minimal local chat UI:

```powershell
.\.venv\Scripts\python.exe -m coding_agent.chat
```

Then open `http://127.0.0.1:8765`.

Project memory and chat history are saved under `.coding-agent/` in the directory where the agent is launched. The memory stores project structure, user decisions, known launch commands, and historical task summaries.

## Skills

The agent has a lightweight built-in skill registry. Skills live as Markdown files under `src/coding_agent/skills/builtin/`, and the registry deterministically selects up to two relevant skills for each task based on trigger phrases. Selected skills are injected as task-specific system messages, recorded in the trace, and shown in live chat events.

Built-in skills:

- `diagnose`: reproduce bugs, failures, exceptions, regressions, and performance issues before fixing them.
- `verification`: compare delivery against requirements, add/update tests, and run real verification before completion.
- `desktop-python`: keep Python desktop apps testable and cover UI callbacks where possible.
- `web-ui`: handle immediate UI feedback, progress states, errors, and request/rendering paths.
- `project-summary`: explain the live project using memory as orientation, not as a substitute for inspection.
