"""配置层测试：.env 解析与优先级。"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from mini_agent.config import ConfigError, Settings, load_dotenv, write_env
from tests.support import temporary_directory

MANAGED_ENV = [
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_BASE_URL",
    "MINI_AGENT_MAX_STEPS",
    "MINI_AGENT_TEMPERATURE",
    "MINI_AGENT_TIMEOUT",
    "MINI_AGENT_STREAM",
    "MINI_AGENT_THINKING",
    "MINI_AGENT_WORKSPACE",
    "DOTENV_A",
    "DOTENV_B",
    "DOTENV_C",
    "DOTENV_D",
]


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.pop(key, None) for key in MANAGED_ENV}
        self._tmp = temporary_directory()
        self.missing_env = Path(self._tmp.name) / "absent.env"

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def write_env(self, content: str) -> Path:
        path = Path(self._tmp.name) / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    # ---------------------------------------------------------------- .env 解析

    def test_dotenv_parses_quotes_export_and_comments(self) -> None:
        path = self.write_env(
            "\n".join(
                [
                    "# 注释行",
                    "",
                    "DOTENV_A=plain",
                    "export DOTENV_B=\"带 空格的值\"",
                    "DOTENV_C='单引号'  # 行内注释",
                    "DOTENV_D=with # comment",
                ]
            )
        )

        load_dotenv(path)

        self.assertEqual(os.environ["DOTENV_A"], "plain")
        self.assertEqual(os.environ["DOTENV_B"], "带 空格的值")
        self.assertEqual(os.environ["DOTENV_C"], "单引号")
        self.assertEqual(os.environ["DOTENV_D"], "with")

    def test_dotenv_does_not_override_existing_env(self) -> None:
        os.environ["DOTENV_A"] = "from-shell"
        path = self.write_env("DOTENV_A=from-file")

        load_dotenv(path)

        self.assertEqual(os.environ["DOTENV_A"], "from-shell")

    def test_missing_dotenv_is_not_an_error(self) -> None:
        load_dotenv(self.missing_env)  # 不应抛异常

    def test_write_env_replaces_keeps_and_appends(self) -> None:
        path = Path(self._tmp.name) / "custom.env"
        path.write_text("# 注释\nDOTENV_A=old\nOTHER=keep\n", encoding="utf-8")

        write_env(path, {"DOTENV_A": "new", "DOTENV_B": "added"})

        text = path.read_text(encoding="utf-8")
        self.assertIn("# 注释", text)
        self.assertIn("DOTENV_A=new", text)
        self.assertIn("OTHER=keep", text)
        self.assertIn("DOTENV_B=added", text)
        self.assertEqual(text.count("DOTENV_A="), 1)

    def test_write_env_roundtrips_through_loader(self) -> None:
        path = Path(self._tmp.name) / "roundtrip.env"
        path.write_text("DOTENV_A=old\n", encoding="utf-8")

        write_env(path, {"DOTENV_A": "x y", "DOTENV_B": "z"})
        load_dotenv(path)

        self.assertEqual(os.environ["DOTENV_A"], "x y")
        self.assertEqual(os.environ["DOTENV_B"], "z")

    # ---------------------------------------------------------------- Settings

    def test_defaults(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"

        settings = Settings.from_env(env_file=self.missing_env)

        self.assertEqual(settings.model, "deepseek-flash")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")
        self.assertEqual(settings.max_steps, 20)
        self.assertTrue(settings.stream)
        self.assertEqual(settings.thinking, "disabled")
        self.assertEqual(settings.workspace, Path.cwd().resolve())

    def test_missing_api_key_raises_with_hint(self) -> None:
        with self.assertRaises(ConfigError) as ctx:
            Settings.from_env(env_file=self.missing_env)

        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_overrides_beat_environment(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "sk-env"
        os.environ["DEEPSEEK_MODEL"] = "model-from-env"

        settings = Settings.from_env(
            env_file=self.missing_env,
            model="deepseek-flash",
            max_steps=3,
            stream=False,
            thinking="enabled",
        )

        self.assertEqual(settings.model, "deepseek-flash")
        self.assertEqual(settings.max_steps, 3)
        self.assertFalse(settings.stream)
        self.assertEqual(settings.thinking, "enabled")

    def test_env_variables_are_read(self) -> None:
        os.environ.update(
            {
                "DEEPSEEK_API_KEY": "sk-test",
                "DEEPSEEK_BASE_URL": "https://example.com/v1/",
                "MINI_AGENT_MAX_STEPS": "7",
                "MINI_AGENT_TEMPERATURE": "0.2",
                "MINI_AGENT_STREAM": "0",
            }
        )

        settings = Settings.from_env(env_file=self.missing_env)

        self.assertEqual(settings.base_url, "https://example.com/v1")
        self.assertEqual(settings.chat_completions_url, "https://example.com/v1/chat/completions")
        self.assertEqual(settings.max_steps, 7)
        self.assertAlmostEqual(settings.temperature, 0.2)
        self.assertFalse(settings.stream)

    def test_thinking_mode_from_env_and_flag(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        os.environ["MINI_AGENT_THINKING"] = "enabled"

        from_env = Settings.from_env(env_file=self.missing_env)
        overridden = Settings.from_env(env_file=self.missing_env, thinking="auto")

        self.assertEqual(from_env.thinking, "enabled")
        self.assertEqual(overridden.thinking, "auto")

    def test_invalid_thinking_mode_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            Settings(api_key="sk-test", thinking="on")

    def test_bad_number_raises(self) -> None:
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        os.environ["MINI_AGENT_MAX_STEPS"] = "many"

        with self.assertRaises(ConfigError):
            Settings.from_env(env_file=self.missing_env)

    def test_invalid_temperature_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            Settings(api_key="sk-test", temperature=3.0)

    def test_workspace_is_absolutized(self) -> None:
        settings = Settings(api_key="sk-test", workspace=Path("."))

        self.assertTrue(settings.workspace.is_absolute())

    def test_env_file_is_loaded_by_from_env(self) -> None:
        path = self.write_env("DEEPSEEK_API_KEY=sk-from-file\nDEEPSEEK_MODEL=model-from-file")

        settings = Settings.from_env(env_file=path)

        self.assertEqual(settings.api_key, "sk-from-file")
        self.assertEqual(settings.model, "model-from-file")


if __name__ == "__main__":
    unittest.main()
