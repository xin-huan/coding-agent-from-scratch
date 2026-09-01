import unittest
from types import SimpleNamespace

from coding_agent.config import Settings
from coding_agent.model import DeepSeekModel, ModelError


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
        usage = SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            total_tokens=150,
            prompt_cache_hit_tokens=40,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=usage,
        )


class InvalidCompletions(FakeCompletions):
    def create(self, **request: object) -> object:
        response = super().create(**request)
        response.choices[0].message.tool_calls[0].function.arguments = "{invalid"
        return response


class TrailingTextCompletions(FakeCompletions):
    def create(self, **request: object) -> object:
        response = super().create(**request)
        response.choices[0].message.tool_calls[0].function.arguments = (
            '{"path":"README.md"}\nextra commentary'
        )
        return response


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
        self.assertEqual(reply.usage.prompt_tokens, 120)
        self.assertEqual(reply.usage.completion_tokens, 30)
        self.assertEqual(reply.usage.cache_hit_tokens, 40)
        self.assertIn("tools", completions.request)

    def test_recovers_first_tool_argument_object_with_trailing_text(self) -> None:
        completions = TrailingTextCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        model = DeepSeekModel(Settings(api_key="test-secret"), client=client)

        reply = model.complete([], [{"type": "function"}])

        self.assertEqual(reply.tool_calls[0].arguments, {"path": "README.md"})

    def test_omits_tools_from_a_finalization_request(self) -> None:
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        model = DeepSeekModel(Settings(api_key="test-secret"), client=client)

        model.complete([], [])

        self.assertNotIn("tools", completions.request)

    def test_invalid_tool_arguments_keep_billable_usage(self) -> None:
        completions = InvalidCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        model = DeepSeekModel(Settings(api_key="test-secret"), client=client)

        with self.assertRaises(ModelError) as raised:
            model.complete([], [{"type": "function"}])

        self.assertEqual(raised.exception.usage.prompt_tokens, 120)
        self.assertEqual(raised.exception.usage.completion_tokens, 30)


if __name__ == "__main__":
    unittest.main()
