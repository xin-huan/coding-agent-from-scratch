import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evals.case import EvalCase, load_cases
from evals.cli import main as eval_main
from evals.runner import EvalResult, evaluate_case, write_summary


CASES_DIR = Path(__file__).parents[1] / "evals" / "cases"


class EvalCaseTests(unittest.TestCase):
    def test_cli_lists_cases_without_running_a_model(self) -> None:
        output = StringIO()

        with redirect_stdout(output):
            exit_code = eval_main(["--list"])

        self.assertEqual(exit_code, 0)
        self.assertIn("C1", output.getvalue())
        self.assertIn("E2", output.getvalue())

    def test_existing_eval_workspaces_have_passing_baseline_tests(self) -> None:
        for case in load_cases(CASES_DIR):
            if case.workspace is None:
                continue
            workspace = case.root / case.workspace
            tests_dir = workspace / "project_tests"
            if not tests_dir.exists():
                continue
            with self.subTest(case=case.id):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "project_tests",
                        "-v",
                    ],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_loads_the_fixed_eval_case_manifest(self) -> None:
        cases = load_cases(CASES_DIR)

        self.assertEqual(
            [case.id for case in cases],
            ["C1", "C2", "F1", "F2", "B1", "B2", "E1", "E2"],
        )
        self.assertTrue(all(case.task.strip() for case in cases))

    def test_evaluates_a_case_in_an_isolated_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_root = root / "case"
            source = case_root / "workspace"
            source.mkdir(parents=True)
            (source / "input.txt").write_text("start\n", encoding="utf-8")
            (case_root / "grader.py").write_text(
                """import sys
from pathlib import Path

workspace = Path(sys.argv[1])
raise SystemExit(0 if (workspace / "answer.txt").read_text() == "done\\n" else 1)
""",
                encoding="utf-8",
            )
            case = EvalCase(
                id="T1",
                category="test",
                title="Offline test",
                task="Create answer.txt",
                root=case_root,
                workspace="workspace",
                grader={"kind": "python", "path": "grader.py", "timeout": 5},
            )

            def fake_agent(workspace: Path, task: str, trace_path: Path) -> str:
                self.assertEqual(task, "Create answer.txt")
                self.assertEqual((workspace / "input.txt").read_text(), "start\n")
                (workspace / "answer.txt").write_text("done\n", encoding="utf-8")
                trace_path.write_text(
                    '{"event":"model_request"}\n'
                    '{"event":"tool_start"}\n'
                    '{"event":"task_complete"}\n',
                    encoding="utf-8",
                )
                return "Completed"

            run_dir = root / "result"
            result = evaluate_case(case, fake_agent, run_dir)

            self.assertTrue(result.passed)
            self.assertEqual(result.score, 100.0)
            self.assertEqual(result.changed_files, ["answer.txt"])
            self.assertEqual(result.model_calls, 1)
            self.assertEqual(result.tool_calls, 1)
            saved = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["case_id"], "T1")
            self.assertIn("answer.txt", (run_dir / "changes.patch").read_text())
            self.assertIn("+done", (run_dir / "changes.patch").read_text())
            self.assertFalse((source / "answer.txt").exists())

    def test_programming_cases_start_unsolved(self) -> None:
        cases = [case for case in load_cases(CASES_DIR) if case.category != "explain"]

        with tempfile.TemporaryDirectory() as temp_dir:
            for case in cases:
                with self.subTest(case=case.id):
                    result = evaluate_case(
                        case,
                        lambda _workspace, _task, _trace: "No changes",
                        Path(temp_dir) / case.id,
                    )
                    self.assertFalse(result.passed)
                    self.assertEqual(result.failure_reason, "acceptance_failed")

    def test_c1_grader_accepts_a_contract_compliant_project(self) -> None:
        case = load_cases(CASES_DIR)[0]
        solution = CASES_DIR / "F1_calculator_power" / "workspace"

        def create_solution(workspace: Path, _task: str, _trace: Path) -> str:
            shutil.copytree(solution / "calculator", workspace / "calculator")
            shutil.copytree(solution / "project_tests", workspace / "tests")
            shutil.copy2(solution / "README.md", workspace / "README.md")
            return "Completed"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, create_solution, Path(temp_dir) / "C1")

        self.assertTrue(result.passed, result.grader_output)

    def test_c2_grader_accepts_a_contract_compliant_project(self) -> None:
        case = load_cases(CASES_DIR)[1]
        solution = CASES_DIR / "F2_todo_delete" / "workspace"

        def create_solution(workspace: Path, _task: str, _trace: Path) -> str:
            shutil.copytree(solution / "todo_app", workspace / "todo")
            shutil.copytree(solution / "project_tests", workspace / "tests")
            shutil.copy2(solution / "README.md", workspace / "README.md")
            for path in workspace.rglob("*.py"):
                path.write_text(
                    path.read_text(encoding="utf-8").replace("todo_app", "todo"),
                    encoding="utf-8",
                )
            return "Completed"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, create_solution, Path(temp_dir) / "C2")

        self.assertTrue(result.passed, result.grader_output)

    def test_f1_grader_accepts_power_feature(self) -> None:
        case = load_cases(CASES_DIR)[2]

        def add_power(workspace: Path, _task: str, _trace: Path) -> str:
            core = workspace / "calculator" / "core.py"
            text = core.read_text(encoding="utf-8")
            text = text.replace(
                "\n\nOPERATIONS = {",
                "\n\ndef power(left: float, right: float) -> float:\n"
                "    return left ** right\n\n\nOPERATIONS = {",
            ).replace('    "divide": divide,', '    "divide": divide,\n    "power": power,')
            core.write_text(text, encoding="utf-8")
            (workspace / "project_tests" / "test_power.py").write_text(
                """import unittest

from calculator.core import power


class PowerTests(unittest.TestCase):
    def test_power(self) -> None:
        self.assertEqual(power(2, 3), 8)
""",
                encoding="utf-8",
            )
            return "Completed"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, add_power, Path(temp_dir) / "F1")

        self.assertTrue(result.passed, result.grader_output)

    def test_f2_grader_accepts_delete_feature(self) -> None:
        case = load_cases(CASES_DIR)[3]

        def add_delete(workspace: Path, _task: str, _trace: Path) -> str:
            core = workspace / "todo_app" / "core.py"
            core.write_text(
                core.read_text(encoding="utf-8")
                + """

def delete_task(tasks: list[Task], task_id: int) -> list[Task]:
    remaining = [task for task in tasks if task["id"] != task_id]
    if len(remaining) == len(tasks):
        raise ValueError(f"Task does not exist: {task_id}")
    return remaining
""",
                encoding="utf-8",
            )
            cli = workspace / "todo_app" / "cli.py"
            text = cli.read_text(encoding="utf-8")
            text = text.replace(
                "from todo_app.core import add_task, complete_task",
                "from todo_app.core import add_task, complete_task, delete_task",
            ).replace(
                '    done.add_argument("task_id", type=int)\n',
                '    done.add_argument("task_id", type=int)\n'
                '    delete = commands.add_parser("delete")\n'
                '    delete.add_argument("task_id", type=int)\n',
            ).replace(
                '        else:\n            for task in tasks:',
                '        elif args.command == "delete":\n'
                '            tasks = delete_task(tasks, args.task_id)\n'
                '            save_tasks(args.data, tasks)\n'
                '        else:\n            for task in tasks:',
            )
            cli.write_text(text, encoding="utf-8")
            (workspace / "project_tests" / "test_delete.py").write_text(
                """import unittest

from todo_app.core import delete_task


class DeleteTests(unittest.TestCase):
    def test_delete(self) -> None:
        self.assertEqual(delete_task([{"id": 1}], 1), [])
""",
                encoding="utf-8",
            )
            return "Completed"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, add_delete, Path(temp_dir) / "F2")

        self.assertTrue(result.passed, result.grader_output)

    def test_b1_grader_accepts_pagination_fix(self) -> None:
        case = load_cases(CASES_DIR)[4]

        def fix_pagination(workspace: Path, _task: str, _trace: Path) -> str:
            pager = workspace / "pager.py"
            pager.write_text(
                pager.read_text(encoding="utf-8").replace(
                    "len(items) - 1", "len(items)"
                ),
                encoding="utf-8",
            )
            (workspace / "project_tests" / "test_boundary.py").write_text(
                """import unittest

from pager import paginate


class BoundaryTests(unittest.TestCase):
    def test_last_page(self) -> None:
        self.assertEqual(paginate([1, 2, 3], 2, 2), [3])
""",
                encoding="utf-8",
            )
            return "Completed"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, fix_pagination, Path(temp_dir) / "B1")

        self.assertTrue(result.passed, result.grader_output)

    def test_b2_grader_accepts_persistence_fix(self) -> None:
        case = load_cases(CASES_DIR)[5]

        def fix_persistence(workspace: Path, _task: str, _trace: Path) -> str:
            storage = workspace / "todo_app" / "storage.py"
            storage.write_text(
                storage.read_text(encoding="utf-8").replace(
                    "repr(tasks)", "json.dumps(tasks, ensure_ascii=False, indent=2)"
                ),
                encoding="utf-8",
            )
            (workspace / "project_tests" / "test_round_trip.py").write_text(
                """import tempfile
import unittest
from pathlib import Path

from todo_app.storage import load_tasks, save_tasks


class RoundTripTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tasks.json"
            tasks = [{"id": 1, "title": "Saved", "done": False}]
            save_tasks(path, tasks)
            self.assertEqual(load_tasks(path), tasks)
""",
                encoding="utf-8",
            )
            return "Completed"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, fix_persistence, Path(temp_dir) / "B2")

        self.assertTrue(result.passed, result.grader_output)

    def test_scores_a_read_only_explanation_against_reference_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_root = root / "case"
            workspace = case_root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "main.py").write_text("print('demo')\n", encoding="utf-8")
            (case_root / "reference.json").write_text(
                json.dumps(
                    {
                        "questions": [
                            {"id": 1, "all_of": ["main.py"]},
                            {"id": 2, "all_of": ["JSON", "data.json"]},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            case = EvalCase(
                id="E0",
                category="explain",
                title="Explain",
                task="Answer two questions",
                root=case_root,
                workspace="workspace",
                grader={"kind": "facts", "reference": "reference.json"},
            )

            result = evaluate_case(
                case,
                lambda _workspace, _task, _trace: (
                    "1. 入口是 main.py。\n2. 数据保存在 data.json，格式为 JSON。"
                ),
                root / "result",
            )

            self.assertTrue(result.passed, result.grader_output)
            self.assertEqual(result.score, 100.0)
            self.assertEqual(result.changed_files, [])

    def test_explanation_scoring_accepts_markdown_numbered_headings(self) -> None:
        case = load_cases(CASES_DIR)[7]
        answer = """**1. 项目的入口文件在哪里？**
notes_app/__main__.py 调用 notes_app.app.main。
**2. 主要模块分别负责什么？**
notes_app/app.py、notes_app/config.py、notes_app/service.py、notes_app/validation.py、notes_app/repository.py。
**3. 一次典型请求经过哪些模块？**
app.py、service.py、validation.py、repository.py。
1. app.py 解析请求。
2. service.py 调用 validation.py 和 repository.py。
**4. 数据保存在哪里？**
notes.json，格式为 JSON。
**5. 项目如何启动？**
python -m notes_app
**6. 测试如何运行？**
python -m unittest discover -s project_tests -v
**7. 配置从哪里加载？**
notes.json、config.json、NOTES_DATA_PATH。
1. notes.json 是默认值。
2. config.json 随后覆盖。
3. NOTES_DATA_PATH 优先级最高。
**8. 修改哪些位置？**
notes_app/validation.py 和 project_tests。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(
                case,
                lambda _workspace, _task, _trace: answer,
                Path(temp_dir) / "E2",
            )

        self.assertTrue(result.passed, result.grader_output)
        self.assertEqual(result.score, 100.0)

    def test_explanation_cases_accept_code_supported_answers(self) -> None:
        cases = {case.id: case for case in load_cases(CASES_DIR)}
        answers = {
            "E1": """1. 入口是 todo_app/__main__.py，它调用 cli.main。
2. todo_app/cli.py 解析命令，todo_app/service.py 负责用例，todo_app/repository.py 读写，todo_app/models.py 定义数据形状。
3. 命令依次经过 cli.py、service.py、repository.py。
4. 默认保存在 tasks.json，格式是 JSON。
5. 使用 python -m todo_app 启动。
6. 使用 python -m unittest discover -s project_tests -v 测试。
7. todo_app/cli.py 从 --data 读取路径，默认 tasks.json。
8. 修改 todo_app/cli.py 和 todo_app/service.py，并在 project_tests 增加测试。""",
            "E2": """1. 入口是 notes_app/__main__.py，它调用 app.main。
2. notes_app/app.py 分发命令，notes_app/config.py 配置，notes_app/service.py 处理用例，notes_app/validation.py 校验，notes_app/repository.py 存储。
3. 请求经过 app.py、service.py、validation.py、repository.py。
4. 默认数据文件是 notes.json，格式是 JSON。
5. 使用 python -m notes_app 启动。
6. 使用 python -m unittest discover -s project_tests -v 测试。
7. 默认 notes.json，随后读取 config.json，最后 NOTES_DATA_PATH 环境变量覆盖。
8. 修改 notes_app/validation.py，并在 project_tests 增加测试。""",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            for case_id in ("E1", "E2"):
                with self.subTest(case=case_id):
                    result = evaluate_case(
                        cases[case_id],
                        lambda _workspace, _task, _trace, answer=answers[case_id]: answer,
                        Path(temp_dir) / case_id,
                    )
                    self.assertTrue(result.passed, result.grader_output)
                    self.assertEqual(result.score, 100.0)

    def test_explanation_score_penalises_unnecessary_file_changes(self) -> None:
        case = load_cases(CASES_DIR)[6]
        answer = "\n".join(
            f"{item['id']}. " + "、".join(item["all_of"])
            for item in json.loads(
                (case.root / "reference.json").read_text(encoding="utf-8")
            )["questions"]
        )

        def modifying_agent(workspace: Path, _task: str, _trace: Path) -> str:
            (workspace / "README.md").write_text("unnecessary\n", encoding="utf-8")
            return answer

        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate_case(case, modifying_agent, Path(temp_dir) / "E1")

        self.assertTrue(result.passed)
        self.assertEqual(result.score, 90.0)
        self.assertEqual(result.changed_files, ["README.md"])

    def test_writes_run_summary_and_failure_cases(self) -> None:
        results = [
            EvalResult("C1", "create", True, 100.0, 1.2, ["main.py"], None, "ok", "ok"),
            EvalResult(
                "B1",
                "bugfix",
                False,
                0.0,
                2.3,
                [],
                "acceptance_failed",
                "done",
                "failed",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            write_summary(results, output)

            saved = json.loads((output / "results.json").read_text(encoding="utf-8"))
            report = (output / "report.md").read_text(encoding="utf-8")
            failures = (output / "failure_cases.jsonl").read_text(encoding="utf-8")

        self.assertEqual(len(saved), 2)
        self.assertIn("50.0%", report)
        self.assertIn("| create | 1 | 1 | 100.0% |", report)
        self.assertIn("| bugfix | 1 | 0 | 0.0% |", report)
        self.assertIn('"case_id": "B1"', failures)
        self.assertNotIn('"case_id": "C1"', failures)

    def test_programming_grader_keeps_partial_score_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_root = root / "case"
            case_root.mkdir()
            (case_root / "grader.py").write_text(
                """import sys
import unittest
from pathlib import Path

Path(sys.argv.pop(1))

class Checks(unittest.TestCase):
    def test_passes(self):
        self.assertTrue(True)

    def test_fails(self):
        self.assertTrue(False)

unittest.main()
""",
                encoding="utf-8",
            )
            case = EvalCase(
                "T2",
                "test",
                "Partial",
                "Do work",
                case_root,
                None,
                {"kind": "python", "path": "grader.py", "timeout": 5},
            )

            result = evaluate_case(
                case,
                lambda _workspace, _task, _trace: "Tried",
                root / "result",
            )

        self.assertFalse(result.passed)
        self.assertEqual(result.score, 50.0)

    def test_runs_acceptance_even_when_agent_does_not_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            case_root = root / "case"
            case_root.mkdir()
            (case_root / "grader.py").write_text(
                """import sys
import unittest
from pathlib import Path

workspace = Path(sys.argv.pop(1))

class Checks(unittest.TestCase):
    def test_artifact_exists(self):
        self.assertTrue((workspace / "answer.txt").exists())

unittest.main()
""",
                encoding="utf-8",
            )
            case = EvalCase(
                "T3",
                "test",
                "Interrupted",
                "Create answer",
                case_root,
                None,
                {"kind": "python", "path": "grader.py", "timeout": 5},
            )

            def interrupted_agent(workspace: Path, _task: str, _trace: Path) -> str:
                (workspace / "answer.txt").write_text("done\n", encoding="utf-8")
                raise RuntimeError("tool budget exhausted")

            result = evaluate_case(case, interrupted_agent, root / "result")

        self.assertFalse(result.passed)
        self.assertFalse(result.agent_completed)
        self.assertTrue(result.acceptance_passed)
        self.assertEqual(result.score, 100.0)


if __name__ == "__main__":
    unittest.main()
