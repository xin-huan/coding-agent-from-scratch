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


if __name__ == "__main__":
    unittest.main()
