import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent.checkpoint import CheckpointStore


class CheckpointStoreTests(unittest.TestCase):
    def test_failed_atomic_replace_preserves_the_previous_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CheckpointStore(Path(temp_dir) / "checkpoint.json")
            store.save({"step": 1})

            with patch(
                "coding_agent.checkpoint.os.replace",
                side_effect=OSError("interrupted replace"),
            ):
                with self.assertRaises(OSError):
                    store.save({"step": 2})

            self.assertEqual(store.load(), {"step": 1})
            self.assertEqual(
                list(Path(temp_dir).glob("checkpoint.json.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
