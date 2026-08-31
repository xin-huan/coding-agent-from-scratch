import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.trace import JsonlTrace


class TraceTests(unittest.TestCase):
    def test_writes_json_event_and_redacts_secret_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            trace = JsonlTrace(path)

            trace.record("tool_start", tool="read_file", api_key="test-secret")

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["event"], "tool_start")
            self.assertEqual(record["data"]["tool"], "read_file")
            self.assertEqual(record["data"]["api_key"], "[REDACTED]")
            self.assertNotIn("test-secret", path.read_text(encoding="utf-8"))

    def test_keeps_token_counts_but_redacts_token_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            trace = JsonlTrace(path)

            trace.record(
                "token_usage",
                prompt_tokens=120,
                completion_tokens=30,
                access_token="secret-value",
            )

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["data"]["prompt_tokens"], 120)
            self.assertEqual(record["data"]["completion_tokens"], 30)
            self.assertEqual(record["data"]["access_token"], "[REDACTED]")

    def test_keeps_context_token_metrics_without_exposing_token_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            trace = JsonlTrace(path)

            trace.record(
                "context_built",
                original_tokens=4_000,
                sent_tokens=1_500,
                saved_tokens=2_500,
                tool_definition_tokens=600,
                access_token="secret-value",
            )

            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["data"]["original_tokens"], 4_000)
            self.assertEqual(record["data"]["sent_tokens"], 1_500)
            self.assertEqual(record["data"]["saved_tokens"], 2_500)
            self.assertEqual(record["data"]["tool_definition_tokens"], 600)
            self.assertEqual(record["data"]["access_token"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
