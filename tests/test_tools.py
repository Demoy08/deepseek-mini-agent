"""工具层测试：沙箱、错误转文本、表达式安全。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mini_agent.tools import Tool, ToolRegistry, default_tools
from tests.support import temporary_directory


class ToolSandboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = temporary_directory()
        self.workspace = Path(self._tmp.name)
        self.tools = default_tools(self.workspace)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_then_read_with_line_numbers(self) -> None:
        written = self.tools.execute("write_file", '{"path": "notes.txt", "content": "第一行\\n第二行"}')
        self.assertIn("已写入", written)

        content = self.tools.execute("read_file", '{"path": "notes.txt"}')
        self.assertIn("    1| 第一行", content)
        self.assertIn("    2| 第二行", content)

    def test_read_supports_offset_and_limit(self) -> None:
        (self.workspace / "lines.txt").write_text("\n".join(str(i) for i in range(1, 11)), encoding="utf-8")

        content = self.tools.execute("read_file", '{"path": "lines.txt", "offset": 3, "limit": 2}')

        self.assertIn("    3| 3", content)
        self.assertIn("    4| 4", content)
        self.assertNotIn("    5| 5", content)
        self.assertIn("还有 6 行", content)

    def test_read_outside_workspace_is_denied(self) -> None:
        observation = self.tools.execute("read_file", '{"path": "../secret.txt"}')

        self.assertTrue(observation.startswith("错误："))
        self.assertIn("越出了工作目录", observation)

    def test_absolute_path_outside_workspace_is_denied(self) -> None:
        target = Path(tempfile.gettempdir()) / "outside.txt"
        observation = self.tools.execute("write_file", f'{{"path": "{target.as_posix()}", "content": "x"}}')

        self.assertTrue(observation.startswith("错误："))
        self.assertFalse(target.exists())

    def test_list_dir(self) -> None:
        (self.workspace / "a.txt").write_text("x", encoding="utf-8")
        (self.workspace / "sub").mkdir()

        listing = self.tools.execute("list_dir", "{}")

        self.assertIn("a.txt", listing)
        self.assertIn("sub/", listing)

    def test_unknown_tool_lists_available_tools(self) -> None:
        observation = self.tools.execute("nope", "{}")

        self.assertTrue(observation.startswith("错误：未知工具"))
        self.assertIn("calculator", observation)

    def test_arguments_must_be_json_object(self) -> None:
        self.assertIn("不是合法 JSON", self.tools.execute("calculator", "{oops"))
        self.assertIn("必须是 JSON 对象", self.tools.execute("calculator", "[1, 2]"))

    def test_missing_argument_reports_schema(self) -> None:
        observation = self.tools.execute("read_file", "{}")

        self.assertIn("参数不匹配", observation)
        self.assertIn("schema", observation)


class CalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = temporary_directory()
        self.tools = default_tools(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def calc(self, expression: str) -> str:
        return self.tools.execute("calculator", json.dumps({"expression": expression}))

    def test_arithmetic(self) -> None:
        self.assertEqual(self.calc("1+2*3"), "7")
        self.assertEqual(self.calc("(1+2)*3"), "9")
        self.assertEqual(self.calc("2**10"), "1024")
        self.assertEqual(self.calc("7/2"), "3.5")
        self.assertEqual(self.calc("10 % 4"), "2")
        self.assertEqual(self.calc("-(3-5)"), "2")

    def test_division_by_zero_is_observation(self) -> None:
        observation = self.calc("1/0")

        self.assertIn("ZeroDivisionError", observation)

    def test_code_execution_is_rejected(self) -> None:
        for expression in ("__import__('os').system('echo hi')", "open('x')", "os.system('x')"):
            observation = self.calc(expression)
            self.assertTrue(observation.startswith("错误："), observation)

    def test_huge_exponent_is_rejected(self) -> None:
        self.assertIn("指数过大", self.calc("2**10000"))

    def test_boolean_constant_is_rejected(self) -> None:
        self.assertTrue(self.calc("True + 1").startswith("错误："))


class RegistryTest(unittest.TestCase):
    def test_duplicate_tool_name_rejected(self) -> None:
        def noop() -> str:
            return ""

        registry = ToolRegistry()
        tool = Tool(name="x", description="d", parameters={"type": "object"}, func=noop)
        registry.add(tool)

        with self.assertRaises(ValueError):
            registry.add(tool)

    def test_tool_returning_non_string_is_coerced(self) -> None:
        registry = ToolRegistry(
            [
                Tool(
                    name="answer",
                    description="d",
                    parameters={"type": "object"},
                    func=lambda: 42,  # type: ignore[arg-type]
                )
            ]
        )

        self.assertEqual(registry.execute("answer", "{}"), "42")

    def test_long_output_is_truncated(self) -> None:
        registry = ToolRegistry(
            [
                Tool(
                    name="spam",
                    description="d",
                    parameters={"type": "object"},
                    func=lambda: "x" * 100_000,
                )
            ]
        )

        observation = registry.execute("spam", "{}")

        self.assertLessEqual(len(observation), 17_000)
        self.assertIn("已截断", observation)


if __name__ == "__main__":
    unittest.main()
