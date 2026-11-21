"""HTTP 集成测试：起一个本地假服务，验证流式/非流式/重试/错误处理。

这一层不碰外网，也不消耗额度，但走的是真实的 urllib + SSE 路径。
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mini_agent.config import Settings
from mini_agent.llm import DeepSeekClient, LLMError


class Scenario:
    """一次测试用的服务端剧本。"""

    def __init__(self, status: int = 200, body: bytes = b"", content_type: str = "application/json"):
        self.status = status
        self.body = body
        self.content_type = content_type
        self.requests: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # 静音
        pass

    def do_GET(self):  # noqa: N802
        scenario: Scenario = self.server.scenario  # type: ignore[attr-defined]
        if self.path.endswith("/models"):
            payload = {"object": "list", "data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]}
            self._send(200, "application/json", json.dumps(payload).encode())
            return
        self._send(404, "application/json", b'{"error": {"message": "not found"}}')

    def do_POST(self):  # noqa: N802
        scenario: Scenario = self.server.scenario  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        scenario.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(raw) if raw else {},
            }
        )

        script = getattr(self.server, "scripts", [])
        if script:
            status, content_type, chunks = script.pop(0)
        else:
            status, content_type, chunks = scenario.status, scenario.content_type, [scenario.body]

        if content_type == "text/event-stream":
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            self._send(status, content_type, chunks[0] if chunks else b"")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def chunk(delta: dict, finish: str | None = None) -> dict:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class HTTPClientTest(unittest.TestCase):
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

    def settings(self, *, stream: bool = True) -> Settings:
        return Settings(
            api_key="sk-test",
            model="deepseek-flash",
            base_url=self.base_url,
            stream=stream,
            timeout=5,
        )

    def script(self, *entries) -> None:
        self.server.scripts = list(entries)  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- 流式

    def test_streaming_text(self) -> None:
        self.script(
            (
                200,
                "text/event-stream",
                [
                    b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n',
                    'data: {"choices":[{"index":0,"delta":{"content":"你好"}}]}\n\n'.encode(),
                    'data: {"choices":[{"index":0,"delta":{"content":"，世界"}}]}\n\n'.encode(),
                    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                    b"data: [DONE]\n\n",
                ],
            )
        )

        received: list[str] = []
        result = DeepSeekClient(self.settings()).chat(
            [{"role": "user", "content": "hi"}], on_text=received.append
        )

        self.assertEqual(result.content, "你好，世界")
        self.assertEqual(received, ["你好", "，世界"])
        self.assertEqual(result.finish_reason, "stop")

    def test_streaming_tool_call(self) -> None:
        self.script(
            (
                200,
                "text/event-stream",
                [
                    sse(
                        chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "calculator", "arguments": ""},
                                    }
                                ]
                            }
                        ),
                        chunk(
                            {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '{"expression": "2+2"}'}}
                                ]
                            }
                        ),
                        chunk({}, finish="tool_calls"),
                    )
                ],
            )
        )

        result = DeepSeekClient(self.settings()).chat(
            [{"role": "user", "content": "算一下"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "calculator", "description": "计算", "parameters": {"type": "object"}},
                }
            ],
        )

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "calculator")
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"expression": "2+2"})

        body = self.server.scenario.requests[0]["body"]  # type: ignore[attr-defined]
        self.assertTrue(body["stream"])
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["tools"][0]["function"]["name"], "calculator")
        self.assertEqual(body["thinking"], {"type": "disabled"})

    def test_streaming_usage_is_captured(self) -> None:
        self.script(
            (
                200,
                "text/event-stream",
                [
                    'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'.encode(),
                    b'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":1,"total_tokens":10}}\n\n',
                    b"data: [DONE]\n\n",
                ],
            )
        )

        result = DeepSeekClient(self.settings()).chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result.usage["total_tokens"], 10)

    # ---------------------------------------------------------------- 非流式

    def test_non_streaming_with_tool_calls(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "type": "function",
                                "function": {"name": "list_dir", "arguments": '{"path": "."}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }
        self.script((200, "application/json", [json.dumps(payload).encode()]))

        result = DeepSeekClient(self.settings(stream=False)).chat([{"role": "user", "content": "ls"}])

        self.assertEqual(result.content, "")
        self.assertEqual(result.tool_calls[0]["function"]["name"], "list_dir")
        self.assertEqual(result.usage["total_tokens"], 11)
        self.assertFalse(self.server.scenario.requests[0]["body"]["stream"])  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- 重试与错误

    def test_retries_on_500_then_succeeds(self) -> None:
        ok = {
            "choices": [{"message": {"role": "assistant", "content": "恢复了"}, "finish_reason": "stop"}],
            "usage": {},
        }
        self.script(
            (500, "application/json", [b'{"error": {"message": "boom"}}']),
            (200, "application/json", [json.dumps(ok).encode()]),
        )
        client = DeepSeekClient(self.settings(stream=False), retry_delay=0.01)

        result = client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "恢复了")
        self.assertEqual(len(self.server.scenario.requests), 2)  # type: ignore[attr-defined]

    def test_auth_error_is_not_retried(self) -> None:
        self.script((401, "application/json", [b'{"error": {"message": "invalid api key"}}']))
        client = DeepSeekClient(self.settings(stream=False), retry_delay=0.01)

        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(ctx.exception.status, 401)
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("invalid api key", str(ctx.exception))
        self.assertEqual(len(self.server.scenario.requests), 1)  # type: ignore[attr-defined]

    def test_list_models(self) -> None:
        models = DeepSeekClient(self.settings()).list_models()

        self.assertIn("deepseek-flash", models)

    def test_non_stream_dict_arguments_are_normalized(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "calculator", "arguments": {"expression": "1+1"}},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        self.script((200, "application/json", [json.dumps(payload).encode()]))

        result = DeepSeekClient(self.settings(stream=False)).chat([{"role": "user", "content": "hi"}])

        arguments = result.tool_calls[0]["function"]["arguments"]
        self.assertIsInstance(arguments, str)
        self.assertEqual(json.loads(arguments), {"expression": "1+1"})

    def test_stream_rejects_non_sse_response(self) -> None:
        self.script((200, "application/json", [b'{"error": {"message": "not a stream"}}']))

        with self.assertRaises(LLMError) as ctx:
            DeepSeekClient(self.settings()).chat([{"role": "user", "content": "hi"}])

        self.assertIn("SSE", str(ctx.exception))

    def test_non_stream_invalid_json_raises_llm_error(self) -> None:
        self.script((200, "application/json", [b"<html>upstream broke</html>"]))

        with self.assertRaises(LLMError) as ctx:
            DeepSeekClient(self.settings(stream=False)).chat([{"role": "user", "content": "hi"}])

        self.assertIn("不是合法 JSON", str(ctx.exception))

    def test_thinking_auto_omits_field(self) -> None:
        self.script(
            (
                200,
                "application/json",
                [b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'],
            )
        )
        settings = Settings(
            api_key="sk-test",
            model="deepseek-flash",
            base_url=self.base_url,
            stream=False,
            thinking="auto",
            timeout=5,
        )

        DeepSeekClient(settings).chat([{"role": "user", "content": "hi"}])

        body = self.server.scenario.requests[0]["body"]  # type: ignore[attr-defined]
        self.assertNotIn("thinking", body)

    def test_authorization_header_is_sent(self) -> None:
        self.script((200, "application/json", [b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}']))

        DeepSeekClient(self.settings(stream=False)).chat([{"role": "user", "content": "hi"}])

        headers = self.server.scenario.requests[0]["headers"]  # type: ignore[attr-defined]
        self.assertEqual(headers.get("Authorization"), "Bearer sk-test")


if __name__ == "__main__":
    unittest.main()
