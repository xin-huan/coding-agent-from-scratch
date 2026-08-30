import unittest
from types import SimpleNamespace

from coding_agent.config import Settings
from coding_agent.model import DeepSeekModel


class FakeCompletions:
    def __init__(self) -> None:
        self.request: dict[str, object] = {}

    def create(self, **request: object) -> object:
        self.request = request
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DeepSeekModelTests(unittest.TestCase):
    def test_converts_provider_tool_call_to_agent_format(self) -> None:
        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        model = DeepSeekModel(Settings(api_key="test-secret"), client=client)

        reply = model.complete([], [{"type": "function"}])

        self.assertEqual(reply.tool_calls[0].name, "read_file")
        self.assertEqual(reply.tool_calls[0].arguments, {"path": "README.md"})
        self.assertIn("tools", completions.request)

    def test_omits_tools_from_a_finalization_request(self) -> None:
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        model = DeepSeekModel(Settings(api_key="test-secret"), client=client)

        model.complete([], [])

        self.assertNotIn("tools", completions.request)


if __name__ == "__main__":
    unittest.main()
