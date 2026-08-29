import unittest
from types import SimpleNamespace

from coding_agent.config import Settings
from coding_agent.model import DeepSeekModel


class FakeCompletions:
    def create(self, **_request: object) -> object:
        tool_call = SimpleNamespace(
            id="call-1",
            function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class DeepSeekModelTests(unittest.TestCase):
    def test_converts_provider_tool_call_to_agent_format(self) -> None:
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )
        model = DeepSeekModel(Settings(api_key="test-secret"), client=client)

        reply = model.complete([], [])

        self.assertEqual(reply.tool_calls[0].name, "read_file")
        self.assertEqual(reply.tool_calls[0].arguments, {"path": "README.md"})


if __name__ == "__main__":
    unittest.main()
