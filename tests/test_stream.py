"""流式拼装测试：真实 SSE 的分片形态都覆盖一遍。"""

from __future__ import annotations

import unittest

from mini_agent.llm import ChatResult, StreamAssembler


def chunk(delta: dict, finish_reason: str | None = None) -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


class StreamAssemblerTest(unittest.TestCase):
    def test_text_deltas_accumulate(self) -> None:
        assembler = StreamAssembler()
        self.assertEqual(assembler.feed(chunk({"role": "assistant", "content": ""})), "")
        self.assertEqual(assembler.feed(chunk({"content": "你好"})), "你好")
        self.assertEqual(assembler.feed(chunk({"content": "，世界"})), "，世界")
        assembler.feed(chunk({}, finish_reason="stop"))

        result = assembler.build()

        self.assertEqual(result.content, "你好，世界")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.tool_calls, [])

    def test_tool_call_fragments_are_joined(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "read_", "arguments": ""},
                        }
                    ]
                }
            )
        )
        assembler.feed(
            chunk({"tool_calls": [{"index": 0, "function": {"name": "file", "arguments": '{"path": '}}]})
        )
        assembler.feed(
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"a.txt"}'}}]})
        )
        assembler.feed(chunk({}, finish_reason="tool_calls"))

        result = assembler.build()

        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call["id"], "call_abc")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(call["function"]["arguments"], '{"path": "a.txt"}')
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertTrue(result.has_tool_calls)

    def test_multiple_tool_calls_are_ordered_by_index(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(
            chunk(
                {
                    "tool_calls": [
                        {"index": 1, "id": "call_2", "function": {"name": "list_dir", "arguments": "{}"}},
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "calculator", "arguments": '{"expression": "1"}'},
                        },
                    ]
                }
            )
        )

        calls = assembler.build().tool_calls

        self.assertEqual([c["id"] for c in calls], ["call_1", "call_2"])

    def test_usage_is_captured_from_final_chunk(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(chunk({"content": "hi"}))
        assembler.feed(
            {
                "choices": [],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            }
        )

        result = assembler.build()

        self.assertEqual(result.usage["total_tokens"], 13)

    def test_empty_arguments_become_empty_object(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "call_x", "function": {"name": "list_dir", "arguments": ""}}
                    ]
                }
            )
        )

        self.assertEqual(assembler.build().tool_calls[0]["function"]["arguments"], "{}")

    def test_missing_id_gets_fallback(self) -> None:
        assembler = StreamAssembler()
        assembler.feed(chunk({"tool_calls": [{"index": 0, "function": {"name": "list_dir"}}]}))

        self.assertTrue(assembler.build().tool_calls[0]["id"])


class ChatResultTest(unittest.TestCase):
    def test_message_with_tool_calls_uses_null_content(self) -> None:
        result = ChatResult(
            content="",
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}}
            ],
        )

        message = result.to_message()

        self.assertEqual(message["role"], "assistant")
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["id"], "c1")

    def test_plain_message_has_content(self) -> None:
        message = ChatResult(content="你好").to_message()

        self.assertEqual(message, {"role": "assistant", "content": "你好"})


if __name__ == "__main__":
    unittest.main()
