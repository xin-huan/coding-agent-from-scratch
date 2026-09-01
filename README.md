# coding-agent-from-scratch
A lightweight coding agent built from scratch with native LLM tool calling, local tool execution, extension hooks, project memory, skills, and a custom agent loop.

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

Project memory, chat history, long context outputs, and workspace snapshots are saved under `.coding-agent/` in the directory where the agent is launched. The memory stores project structure, user decisions, known launch commands, and historical task summaries.

## Skills

The agent has a lightweight built-in skill registry implemented as an extension. Skills live as Markdown files under `src/coding_agent/skills/builtin/`, and the registry deterministically selects up to two relevant skills for each task based on trigger phrases. Selected skills are injected as task-specific system messages, recorded in the trace, and shown in live chat events.

Built-in skills:

- `diagnose`: reproduce bugs, failures, exceptions, regressions, and performance issues before fixing them.
- `verification`: compare delivery against requirements, add/update tests, and run real verification before completion.
- `desktop-python`: keep Python desktop apps testable and cover UI callbacks where possible.
- `web-ui`: handle immediate UI feedback, progress states, errors, and request/rendering paths.
- `project-summary`: explain the live project using memory as orientation, not as a substitute for inspection.

## Context Pack

Long conversations are handled by a default `ContextPackExtension`. It keeps raw long outputs in a local `.coding-agent/context-outputs/` store, puts only a compact summary and `output_id` into the model context, and registers `read_context_output` so the model can fetch exact line ranges when needed. When a model request grows beyond roughly 40,000 characters, it also replaces older assistant/tool-call exchanges with structured `<compacted_history>` summaries.

The older v2/v3 single-task context manager and read cache experiments have been removed. Context packing changes the request copy sent to the model and offloads oversized tool outputs, but the exact long output remains recoverable by id for diagnosis and re-summary.

## Extensions

The agent exposes lifecycle hooks in `src/coding_agent/extensions.py` so new framework behavior can be added without editing the core loop. Extensions can inject context, add tool definitions, observe LLM calls, subscribe to context compaction, intercept or implement tool calls, and save state at session end. Project memory, skill selection, and context packing are now ordinary extensions.

## Tasks and Subagents

Every run exposes a lightweight task list through live `[任务]` events and the runtime `<task_state>`. From-scratch project work uses the reviewed implementation plan as its task list; other work gets a default inspect/change/verify/finalize list.

The default `SubAgentExtension` enables `delegate_subagent` only when the request or workspace appears complex enough, such as multi-domain failure analysis, large projects, architecture reviews, performance/security investigations, or an explicit subagent request. The system intentionally supports only three bounded roles: `researcher` for read-only module and architecture discovery, `tester` for verification strategy and restricted test execution, and `reviewer` for patch review after non-trivial changes. None of them can edit files or make the final decision. After complex file changes, final verification is gated: a premature verification command is converted into reviewer delegation first, then the main agent judges the report, fixes anything needed, and runs verification itself.

## Tree Sessions

Chat history is stored as a message tree instead of only a flat list. Each message has an `id` and `parent_id`, and each conversation tracks `current_message_id`. The UI renders the current branch as normal chat while the Session tree shows all user-turn nodes as a history directory. Left-clicking a node only scrolls to the user request when it is visible; right-clicking opens explicit actions such as continuing from that turn. Normal linear history stays aligned on one vertical line, while real sibling branches are indented.

When an Agent run changes text files, the chat UI also stores a lightweight workspace snapshot under `.coding-agent/workspace-snapshots/` and binds it to that turn. Restoring code is always explicit: right-click a turn, choose restore snapshot, confirm the warning, then the current related files are backed up before restoration.
