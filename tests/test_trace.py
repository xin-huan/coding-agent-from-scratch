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

    def test_replaces_invalid_unicode_surrogates_before_writing_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.jsonl"
            trace = JsonlTrace(path)

            trace.record("task_start", task="bad\udcaa input")

            content = path.read_text(encoding="utf-8")
            record = json.loads(content)
            self.assertEqual(record["data"]["task"], "bad\ufffd input")
            self.assertNotIn("\udcaa", content)


if __name__ == "__main__":
    unittest.main()
