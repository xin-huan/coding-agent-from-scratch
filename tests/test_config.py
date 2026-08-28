import tempfile
import unittest
from pathlib import Path

from coding_agent.config import ConfigError, Settings


class SettingsTests(unittest.TestCase):
    def test_missing_api_key_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
                Settings.load(Path(temp_dir), environ={})

    def test_loads_deepseek_settings_without_exposing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            (config_dir / ".env").write_text(
                "DEEPSEEK_API_KEY=test-secret\n"
                "DEEPSEEK_BASE_URL=https://example.test\n"
                "DEEPSEEK_MODEL=test-model\n",
                encoding="utf-8",
            )

            settings = Settings.load(config_dir, environ={})

            self.assertEqual(settings.api_key, "test-secret")
            self.assertEqual(settings.base_url, "https://example.test")
            self.assertEqual(settings.model, "test-model")
            self.assertNotIn("test-secret", repr(settings))


if __name__ == "__main__":
    unittest.main()
