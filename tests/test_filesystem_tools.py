import tempfile
import unittest
from pathlib import Path

from coding_agent.tools.filesystem import (
    apply_patch,
    FileToolError,
    list_files,
    read_file,
    search_text,
    write_file,
)
from coding_agent.workspace import Workspace


class FilesystemToolTests(unittest.TestCase):
    def test_lists_project_files_and_ignores_internal_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "main.py").write_text("print('hello')\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src" / "helper.py").write_text("", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("secret", encoding="utf-8")

            result = list_files(Workspace(root))

            self.assertEqual(result, "main.py\nsrc/\nsrc/helper.py")

    def test_reads_text_file_with_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.txt").write_text("first\nsecond\n", encoding="utf-8")

            result = read_file(Workspace(root), "notes.txt")

            self.assertEqual(result, "1 | first\n2 | second")

    def test_reads_only_the_requested_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.txt").write_text(
                "first\nsecond\nthird\nfourth\n",
                encoding="utf-8",
            )

            result = read_file(
                Workspace(root),
                "notes.txt",
                start_line=2,
                end_line=3,
            )

            self.assertEqual(result, "2 | second\n3 | third")

    def test_large_file_read_is_truncated_with_continuation_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "large.txt").write_text(
                "".join(f"line {index:04d}\n" for index in range(3_000)),
                encoding="utf-8",
            )

            result = read_file(Workspace(root), "large.txt")

            self.assertIn("file output truncated", result)
            self.assertIn("start_line=", result)

    def test_file_listing_hides_local_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("SECRET=value\n", encoding="utf-8")
            (root / ".env.example").write_text("SECRET=\n", encoding="utf-8")

            result = list_files(Workspace(root))

            self.assertEqual(result, ".env.example")

    def test_read_file_rejects_local_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text("SECRET=value\n", encoding="utf-8")

            with self.assertRaisesRegex(FileToolError, "protected"):
                read_file(Workspace(root), ".env")

    def test_read_file_rejects_internal_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("private\n", encoding="utf-8")

            with self.assertRaisesRegex(FileToolError, "protected"):
                read_file(Workspace(root), ".git/config")

    def test_searches_project_text_without_exposing_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "app.py").write_text(
                "def complete_task():\n    return True\n", encoding="utf-8"
            )
            (root / ".env").write_text("complete_task=secret\n", encoding="utf-8")

            result = search_text(Workspace(root), "complete_task")

            self.assertEqual(result, "app.py:1: def complete_task():")

    def test_writes_a_new_file_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            result = write_file(
                Workspace(root), "src/main.py", "print('hello')\n"
            )

            self.assertIn("src/main.py", result)
            self.assertEqual(
                (root / "src" / "main.py").read_text(encoding="utf-8"),
                "print('hello')\n",
            )

    def test_write_file_rejects_local_secret_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Workspace(Path(temp_dir))

            with self.assertRaisesRegex(FileToolError, "protected"):
                write_file(workspace, ".env", "DEEPSEEK_API_KEY=changed\n")

    def test_applies_one_exact_text_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "config.py"
            path.write_text("MODE = 'old'\n", encoding="utf-8")

            result = apply_patch(
                Workspace(root), "config.py", "MODE = 'old'", "MODE = 'new'"
            )

            self.assertIn("config.py", result)
            self.assertEqual(path.read_text(encoding="utf-8"), "MODE = 'new'\n")

    def test_patch_rejects_ambiguous_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "values.txt").write_text("same\nsame\n", encoding="utf-8")

            with self.assertRaisesRegex(FileToolError, "exactly once"):
                apply_patch(Workspace(root), "values.txt", "same", "changed")


if __name__ == "__main__":
    unittest.main()
