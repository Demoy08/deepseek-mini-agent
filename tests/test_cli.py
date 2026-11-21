"""CLI 端到端测试：真的启动子进程跑 ``python -m mini_agent``。

前面几层测的是模块，这一层测的是用户实际敲命令的路径：
REPL 输入、流式打印、工具展示、退出码、错误提示。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from tests.support import temporary_directory
from tests.test_llm_http import Handler, Scenario, chunk, sse

REPO_ROOT = Path(__file__).resolve().parent.parent


def tool_call_chunks(name: str, arguments: str, call_id: str = "call_1") -> bytes:
    return sse(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]}),
        chunk({}, finish="tool_calls"),
    )


def text_chunks(*parts: str) -> bytes:
    return sse(*[chunk({"content": part}) for part in parts], chunk({}, finish="stop"))


class CLITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.scenario = Scenario()  # type: ignore[attr-defined]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        self.server.scenario.requests.clear()  # type: ignore[attr-defined]
        self.server.scripts = []  # type: ignore[attr-defined]
        self._tmp = temporary_directory()
        self.workspace = self._tmp.name

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---------------------------------------------------------------- helpers

    def script(self, *entries) -> None:
        self.server.scripts = list(entries)  # type: ignore[attr-defined]

    def run_cli(self, *args: str, stdin: str = "", api_key: str | None = "sk-test", cwd: str | None = None):
        env = os.environ.copy()
        env.update(
            {
                "DEEPSEEK_BASE_URL": self.base_url,
                "DEEPSEEK_MODEL": "deepseek-flash",
                "MINI_AGENT_THINKING": "disabled",
                "PYTHONIOENCODING": "utf-8",
                "NO_COLOR": "1",
                # 让子进程从仓库导入包，同时把 .env 查找留在临时目录
                "PYTHONPATH": str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
            }
        )
        env.pop("MINI_AGENT_STREAM", None)
        env.pop("MINI_AGENT_WORKSPACE", None)
        if api_key is None:
            env.pop("DEEPSEEK_API_KEY", None)
        else:
            env["DEEPSEEK_API_KEY"] = api_key

        return subprocess.run(
            [sys.executable, "-m", "mini_agent", *args],
            cwd=cwd or str(REPO_ROOT),
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
        )

    def requests(self) -> list[dict]:
        return self.server.scenario.requests  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- 一次性执行

    def test_one_shot_tool_call_streaming(self) -> None:
        self.script(
            (200, "text/event-stream", [tool_call_chunks("calculator", '{"expression": "6*7"}')]),
            (
                200,
                "text/event-stream",
                [
                    sse(
                        chunk({"content": "6×7 "}),
                        chunk({"content": "等于 42。"}),
                        chunk({}, finish="stop"),
                        {
                            "choices": [],
                            "usage": {"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59},
                        },
                    )
                ],
            ),
        )

        result = self.run_cli("-p", "算一下 6*7", "--workspace", self.workspace)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("→ calculator", result.stdout)
        self.assertIn("← 42", result.stdout)
        self.assertIn("6×7 等于 42。", result.stdout)
        self.assertIn("[tokens]", result.stdout)

        # 第二次请求必须带上工具观察结果
        second = self.requests()[1]["body"]["messages"]
        self.assertEqual(second[-1]["role"], "tool")
        self.assertEqual(second[-1]["content"], "42")

        # 请求体里应该带上工具 schema、关闭思考、指定模型
        first = self.requests()[0]["body"]
        self.assertEqual(first["model"], "deepseek-flash")
        self.assertEqual(first["thinking"], {"type": "disabled"})
        self.assertEqual(first["tool_choice"], "auto")
        self.assertTrue(first["stream"])

    # ---------------------------------------------------------------- REPL

    def test_repl_keeps_history_between_turns(self) -> None:
        self.script(
            (200, "text/event-stream", [text_chunks("第一答")]),
            (200, "text/event-stream", [text_chunks("第二答")]),
        )

        result = self.run_cli(stdin="第一问\n第二问\n/exit\n", cwd=self.workspace)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("第一答", result.stdout)
        self.assertIn("第二答", result.stdout)

        messages = self.requests()[1]["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertEqual(messages[1]["content"], "第一问")
        self.assertEqual(messages[3]["content"], "第二问")

    def test_repl_reset_clears_history(self) -> None:
        self.script((200, "text/event-stream", [text_chunks("新对话")]))

        result = self.run_cli(stdin="旧问题\n/reset\n新问题\n/exit\n", cwd=self.workspace)

        self.assertEqual(result.returncode, 0, result.stderr)
        messages = self.requests()[-1]["body"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "新问题")

    def test_repl_history_command(self) -> None:
        self.script((200, "text/event-stream", [text_chunks("你好")]))

        result = self.run_cli(stdin="问一句\n/history\n/exit\n", cwd=self.workspace)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[1] user", result.stdout)
        self.assertIn("问一句", result.stdout)

    # ---------------------------------------------------------------- 非流式

    def test_non_streaming_mode(self) -> None:
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "非流式回答"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
        self.script((200, "application/json", [json.dumps(payload, ensure_ascii=False).encode()]))

        result = self.run_cli("-p", "你好", "--no-stream", cwd=self.workspace)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("非流式回答", result.stdout)
        self.assertIn("prompt=3", result.stdout)

    # ---------------------------------------------------------------- 失败路径

    def test_step_limit_exits_nonzero(self) -> None:
        endless = tool_call_chunks("calculator", '{"expression": "1"}')
        self.script(*[(200, "text/event-stream", [endless]) for _ in range(4)])

        result = self.run_cli("-p", "死循环", "--max-steps", "2", cwd=self.workspace)

        self.assertEqual(result.returncode, 1)
        self.assertIn("已连续调用 2 次模型", result.stderr)

    def test_401_shows_actionable_hint(self) -> None:
        self.script((401, "application/json", [b'{"error": {"message": "Authentication Fails"}}']))

        result = self.run_cli("-p", "你好", cwd=self.workspace)

        self.assertEqual(result.returncode, 1)
        self.assertIn("401", result.stderr)
        self.assertIn("DEEPSEEK_API_KEY", result.stderr)

    def test_missing_key_exits_with_config_error(self) -> None:
        result = self.run_cli("-p", "你好", api_key=None, cwd=self.workspace)

        self.assertEqual(result.returncode, 2)
        self.assertIn("配置错误", result.stderr)


if __name__ == "__main__":
    unittest.main()
