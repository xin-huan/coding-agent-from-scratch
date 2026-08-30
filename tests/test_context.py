import json
import unittest

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
    def test_omits_large_historical_write_payload_without_mutating_history(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "build project"},
            *tool_turn(1),
        ]
        original_arguments = str(messages[2]["tool_calls"][0]["function"]["arguments"])

        built = ContextManager(max_characters=20_000).build(messages)

        sent_arguments = str(built[2]["tool_calls"][0]["function"]["arguments"])
        self.assertIn("omitted 4000 characters", sent_arguments)
        self.assertEqual(
            str(messages[2]["tool_calls"][0]["function"]["arguments"]),
            original_arguments,
        )

    def test_compacts_old_turns_and_keeps_latest_tool_pair(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "build project"},
            *tool_turn(1, output_size=4_000),
            *tool_turn(2, output_size=4_000),
            *tool_turn(3, output_size=4_000),
            *tool_turn(4, output_size=4_000),
        ]
        manager = ContextManager(max_characters=4_000, recent_turns=1)

        built = manager.build(
            messages,
            [{"role": "system", "content": "current state"}],
        )

        summary = next(
            str(message["content"])
            for message in built
            if "<history_summary>" in str(message.get("content", ""))
        )
        self.assertIn("module_1.py", summary)
        self.assertIn("module_3.py", summary)
        self.assertTrue(
            any(message.get("tool_call_id") == "call-4" for message in built)
        )
        self.assertEqual(built[-1]["content"], "current state")
        self.assertGreater(manager.last_stats.summarized_messages, 0)
        self.assertGreater(manager.last_stats.saved_characters, 0)


if __name__ == "__main__":
    unittest.main()
