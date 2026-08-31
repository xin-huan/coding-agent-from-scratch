import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.context import ContextManager


def tool_turn(
    index: int,
    *,
    payload_size: int = 4_000,
    output_size: int = 0,
) -> list[dict[str, object]]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": f"step {index}",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps(
                            {
                                "path": f"module_{index}.py",
                                "content": "x" * payload_size,
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": (
                f"Wrote module_{index}.py ({payload_size} characters)"
                + "y" * output_size
            ),
        },
    ]


class ContextManagerTests(unittest.TestCase):
    def test_v3_replaces_written_source_with_a_structural_ledger(self) -> None:
        manager = ContextManager(mode="v3", max_prompt_tokens=2_000)
        source = (
            "class Timer:\n"
            "    def start(self):\n"
            "        return True\n"
            + "# implementation detail\n" * 500
        )
        manager.record_tool(
            step=1,
            name="write_file",
            arguments={"path": "timer.py", "content": source},
            result="Wrote timer.py",
            success=True,
            version="version-1",
            file_content=source,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "build project"},
            *tool_turn(1, payload_size=10_000),
        ]

        built = manager.build(messages)

        rendered = json.dumps(built, ensure_ascii=False)
        self.assertIn("<artifact_ledger>", rendered)
        self.assertIn("timer.py", rendered)
        self.assertIn("Timer", rendered)
        self.assertIn("start", rendered)
        self.assertNotIn("implementation detail", rendered)
        self.assertNotIn("x" * 1_000, rendered)

    def test_v3_retrieves_last_real_file_after_a_failed_patch(self) -> None:
        manager = ContextManager(mode="v3")
        manager.record_tool(
            step=1,
            name="write_file",
            arguments={"path": "timer.py", "content": "VALUE = 'saved'\n"},
            result="Wrote timer.py",
            success=True,
            version="saved-version",
            file_content="VALUE = 'saved'\n",
        )
        manager.record_tool(
            step=2,
            name="apply_patch",
            arguments={
                "path": "timer.py",
                "old_text": "missing",
                "new_text": "VALUE = 'not-saved'\n",
            },
            result="ERROR: old_text was not found",
            success=False,
        )

        built = manager.build(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "repair the failed patch"},
            ]
        )

        rendered = json.dumps(built, ensure_ascii=False)
        self.assertIn("<retrieved_history>", rendered)
        self.assertIn("VALUE = 'saved'", rendered)
        self.assertNotIn("VALUE = 'not-saved'", rendered)

    def test_keeps_contract_and_state_inside_a_token_budget(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "build project"},
            *tool_turn(1),
            *tool_turn(2),
            *tool_turn(3),
        ]
        manager = ContextManager(max_prompt_tokens=2_000, recent_turns=1)

        built = manager.build(
            messages,
            contract_messages=[
                {"role": "system", "content": "<task_contract>must keep</task_contract>"}
            ],
            state_messages=[
                {"role": "system", "content": "<work_state>must keep</work_state>"}
            ],
        )

        rendered = json.dumps(built, ensure_ascii=False)
        self.assertIn("<task_contract>must keep</task_contract>", rendered)
        self.assertIn("<work_state>must keep</work_state>", rendered)
        self.assertLessEqual(manager.last_stats.sent_tokens, 2_000)
        self.assertGreater(manager.last_stats.saved_tokens, 0)

    def test_compacts_old_turns_and_keeps_latest_tool_pair(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "build project"},
            *tool_turn(1, output_size=4_000),
            *tool_turn(2, output_size=4_000),
            *tool_turn(3, output_size=4_000),
            *tool_turn(4, output_size=4_000),
        ]
        manager = ContextManager(max_prompt_tokens=1_200, recent_turns=1)

        built = manager.build(
            messages,
            state_messages=[{"role": "system", "content": "current state"}],
        )

        self.assertTrue(
            any(message.get("tool_call_id") == "call-4" for message in built)
        )
        self.assertEqual(built[-1]["content"], "current state")
        self.assertGreater(manager.last_stats.summarized_messages, 0)
        self.assertGreater(manager.last_stats.saved_tokens, 0)

    def test_persists_and_retrieves_relevant_file_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "context-history.jsonl"
            writer = ContextManager(mode="v2", archive_path=archive)
            writer.start_task(reset=True)
            writer.record_tool(
                step=1,
                name="read_file",
                arguments={"path": "src/timer.py"},
                result="1 | class Timer:\n2 |     def tick(self): pass",
                success=True,
                version="abc123",
            )

            reader = ContextManager(mode="v2", archive_path=archive)
            built = reader.build(
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "fix Timer.tick"},
                ],
                state_messages=[
                    {"role": "system", "content": "Current file: src/timer.py"}
                ],
            )

        rendered = json.dumps(built, ensure_ascii=False)
        self.assertIn("<retrieved_history>", rendered)
        self.assertIn("src/timer.py", rendered)
        self.assertIn("version=abc123", rendered)
        self.assertIn("class Timer", rendered)
        self.assertEqual(reader.last_stats.retrieved_entries, 1)

    def test_retrieval_uses_only_the_latest_file_version(self) -> None:
        manager = ContextManager(mode="v2")
        manager.record_tool(
            step=1,
            name="write_file",
            arguments={"path": "src/timer.py", "content": "OLD = True"},
            result="Wrote src/timer.py",
            success=True,
            version="old-version",
        )
        manager.record_tool(
            step=2,
            name="write_file",
            arguments={"path": "src/timer.py", "content": "NEW = True"},
            result="Wrote src/timer.py",
            success=True,
            version="new-version",
        )

        built = manager.build(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "inspect src/timer.py"},
            ]
        )

        rendered = json.dumps(built, ensure_ascii=False)
        self.assertIn("NEW = True", rendered)
        self.assertIn("version=new-version", rendered)
        self.assertNotIn("OLD = True", rendered)
        self.assertEqual(manager.last_stats.retrieved_entries, 1)

    def test_cache_hit_does_not_replace_the_retrievable_file_snapshot(self) -> None:
        manager = ContextManager(mode="v2")
        manager.record_tool(
            step=1,
            name="read_file",
            arguments={"path": "src/timer.py"},
            result="1 | class Timer:\n2 |     pass",
            success=True,
            version="same-version",
        )
        manager.record_tool(
            step=2,
            name="read_file",
            arguments={"path": "src/timer.py", "start_line": 1, "end_line": 2},
            result=(
                "unchanged read cache hit: src/timer.py (version same-version). "
                "Reuse the earlier content."
            ),
            success=True,
            version="same-version",
        )

        built = manager.build(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "inspect src/timer.py"},
            ]
        )

        rendered = json.dumps(built, ensure_ascii=False)
        self.assertIn("class Timer", rendered)
        self.assertNotIn("unchanged read cache hit", rendered)

    def test_distills_command_output_but_preserves_failure_evidence(self) -> None:
        manager = ContextManager()
        result = (
            "Exit code: 1\nSTDOUT:\n"
            + "unimportant log\n" * 2_000
            + "Ran 8 tests\nFAILED (failures=1)\nAssertionError: expected 2"
        )

        distilled = manager.distill_tool_result(
            "run_command",
            {"argv": ["python", "-m", "unittest"]},
            result,
            success=True,
        )

        self.assertIn("Exit code: 1", distilled)
        self.assertIn("Ran 8 tests", distilled)
        self.assertIn("FAILED (failures=1)", distilled)
        self.assertIn("AssertionError", distilled)
        self.assertNotIn("unimportant log\nunimportant log", distilled)


if __name__ == "__main__":
    unittest.main()
