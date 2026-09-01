import json
import tempfile
import unittest
from pathlib import Path

from coding_agent.workspace import Workspace
from coding_agent.workspace_snapshot import WorkspaceSnapshotStore


class WorkspaceSnapshotTests(unittest.TestCase):
    def test_captures_changed_text_files_and_restores_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / "app.py").write_text("print('before')\n", encoding="utf-8")
            workspace = Workspace(root)
            store = WorkspaceSnapshotStore(Path(temp_dir) / "snapshots")

            before = store.capture_state(workspace)
            (root / "app.py").write_text("print('after')\n", encoding="utf-8")
            snapshot = store.create_for_changes(
                before=before,
                after=store.capture_state(workspace),
                project_id="project-1",
                conversation_id="conversation-1",
                message_id="assistant-1",
                user_message_id="user-1",
                created_at="2026-09-01T00:00:00+00:00",
            )

            self.assertIsNotNone(snapshot)
            (root / "app.py").write_text("print('current')\n", encoding="utf-8")
            result = store.restore(
                workspace,
                snapshot.id,
                created_at="2026-09-01T00:01:00+00:00",
            )

            self.assertEqual((root / "app.py").read_text(encoding="utf-8"), "print('after')\n")
            self.assertEqual(result.restored_files, ["app.py"])
            backup = json.loads((Path(temp_dir) / "snapshots" / f"{result.backup_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(backup["backup_of"], snapshot.id)
            self.assertEqual(backup["files"][0]["content"], "print('current')\n")

    def test_snapshot_state_skips_protected_and_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project"
            root.mkdir()
            (root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "HEAD").write_text("main\n", encoding="utf-8")
            (root / "image.bin").write_bytes(b"\xff\x00\xff")
            (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
            store = WorkspaceSnapshotStore(Path(temp_dir) / "snapshots")

            state = store.capture_state(Workspace(root))

            self.assertEqual(state, {"app.py": "print('ok')\n"})


if __name__ == "__main__":
    unittest.main()
