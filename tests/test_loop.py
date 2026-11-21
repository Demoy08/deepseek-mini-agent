"""用脚本化的假模型验证 agent loop 本身，不产生任何网络请求。"""

from __future__ import annotations

import unittest
from pathlib import Path

from mini_agent.llm import ChatResult
from mini_agent.loop import Agent, StepLimitExceeded
from mini_agent.tools import default_tools
from tests.support import temporary_directory


def tool_result(name: str, arguments: str, call_id: str = "call_1") -> ChatResult:
    return ChatResult(
        content="",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
        finish_reason="tool_calls",
    )


class FakeLLM:
    """按脚本依次返回 ChatResult，并记录每次收到的上下文。"""

    def __init__(self, results: list[ChatResult]) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, on_text=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self.results:
            raise AssertionError("脚本里的模型回复已经用完")
        result = self.results.pop(0)
        if on_text and result.content:
            on_text(result.content)
        return result


class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = temporary_directory()
        self.workspace = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_agent(self, results: list[ChatResult], max_steps: int = 6) -> tuple[Agent, FakeLLM]:
        llm = FakeLLM(results)
        agent = Agent(llm=llm, tools=default_tools(self.workspace), max_steps=max_steps)
        return agent, llm

    # ---------------------------------------------------------------- 基本路径

    def test_plain_answer_ends_turn(self) -> None:
        agent, llm = self.make_agent([ChatResult(content="你好，我是 agent。")])

        answer = agent.run("你好")

        self.assertEqual(answer, "你好，我是 agent。")
        self.assertEqual([m["role"] for m in agent.history], ["system", "user", "assistant"])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["messages"][-1]["content"], "你好")

    def test_tools_are_advertised_to_model(self) -> None:
        agent, llm = self.make_agent([ChatResult(content="ok")])

        agent.run("随便聊聊")

        schemas = llm.calls[0]["tools"]
        self.assertIsNotNone(schemas)
        names = {item["function"]["name"] for item in schemas}
        self.assertIn("read_file", names)
        self.assertIn("calculator", names)

    # ---------------------------------------------------------------- 工具往返

    def test_tool_call_is_executed_and_fed_back(self) -> None:
        agent, llm = self.make_agent(
            [
                tool_result("calculator", '{"expression": "1+2*3"}'),
                ChatResult(content="结果是 7。"),
            ]
        )

        answer = agent.run("1+2*3 等于几？")

        self.assertEqual(answer, "结果是 7。")
        self.assertEqual(
            [m["role"] for m in agent.history],
            ["system", "user", "assistant", "tool", "assistant"],
        )

        tool_message = agent.history[3]
        self.assertEqual(tool_message["tool_call_id"], "call_1")
        self.assertEqual(tool_message["content"], "7")

        # 第二次请求必须带上工具结果
        second_messages = llm.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["content"], "7")

    def test_parallel_tool_calls_are_all_executed(self) -> None:
        agent, _ = self.make_agent(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression": "2**5"}'},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "list_dir", "arguments": "{}"},
                        },
                    ],
                    finish_reason="tool_calls",
                ),
                ChatResult(content="算完了。"),
            ]
        )

        agent.run("并行做两件事")

        tool_messages = [m for m in agent.history if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)
        self.assertEqual(tool_messages[0]["content"], "32")
        self.assertEqual(tool_messages[1]["tool_call_id"], "call_b")

    def test_unknown_tool_becomes_observation(self) -> None:
        agent, _ = self.make_agent(
            [
                tool_result("delete_everything", "{}"),
                ChatResult(content="这个工具不存在，我换个办法。"),
            ]
        )

        answer = agent.run("删库")

        self.assertEqual(answer, "这个工具不存在，我换个办法。")
        observation = agent.history[3]["content"]
        self.assertTrue(observation.startswith("错误：未知工具"))
        self.assertIn("read_file", observation)

    def test_malformed_arguments_become_observation(self) -> None:
        agent, _ = self.make_agent(
            [
                tool_result("read_file", "{not json"),
                ChatResult(content="参数写错了。"),
            ]
        )

        agent.run("读文件")

        observation = agent.history[3]["content"]
        self.assertIn("不是合法 JSON", observation)

    def test_tool_python_error_becomes_observation(self) -> None:
        agent, _ = self.make_agent(
            [
                tool_result("read_file", '{"path": "nope.txt"}'),
                ChatResult(content="文件不存在。"),
            ]
        )

        agent.run("读一个不存在的文件")

        observation = agent.history[3]["content"]
        self.assertIn("FileNotFoundError", observation)

    # ---------------------------------------------------------------- 边界

    def test_step_limit_raises_and_history_stays_valid(self) -> None:
        loop_result = tool_result("calculator", '{"expression": "1"}')
        agent, llm = self.make_agent(
            [loop_result, loop_result, loop_result],
            max_steps=2,
        )

        with self.assertRaises(StepLimitExceeded):
            agent.run("死循环测试")

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(agent.history[-1]["role"], "tool")

    def test_empty_input_rejected(self) -> None:
        agent, _ = self.make_agent([ChatResult(content="不该被调用")])

        with self.assertRaises(ValueError):
            agent.run("   ")

    def test_usage_accumulates(self) -> None:
        agent, _ = self.make_agent(
            [
                ChatResult(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": '{"expression": "1"}'},
                        }
                    ],
                    usage={"prompt_tokens": 10, "completion_tokens": 5},
                ),
                ChatResult(content="好了", usage={"prompt_tokens": 12, "completion_tokens": 3}),
            ]
        )

        agent.run("算一下")

        self.assertEqual(agent.total_usage["prompt_tokens"], 22)
        self.assertEqual(agent.total_usage["completion_tokens"], 8)

    def test_reset_restores_system_prompt(self) -> None:
        agent, _ = self.make_agent([ChatResult(content="一")])
        agent.run("第一轮")

        agent.reset()

        self.assertEqual(len(agent.history), 1)
        self.assertEqual(agent.history[0]["role"], "system")
        self.assertEqual(agent.total_usage, {})

    def test_repair_tail_fills_interrupted_tool_call(self) -> None:
        agent, _ = self.make_agent([ChatResult(content="ok")])
        agent.history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_x",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{}"},
                    }
                ],
            }
        )

        agent.repair_tail()

        self.assertEqual(agent.history[-1]["role"], "tool")
        self.assertEqual(agent.history[-1]["tool_call_id"], "call_x")

    def test_events_are_emitted_in_order(self) -> None:
        events: list[tuple[str, dict]] = []
        agent, _ = self.make_agent(
            [
                tool_result("calculator", '{"expression": "1+1"}'),
                ChatResult(content="等于 2。"),
            ]
        )
        agent.on_event = lambda kind, **data: events.append((kind, data))

        agent.run("1+1")

        kinds = [kind for kind, _ in events]
        self.assertIn("tool_call", kinds)
        self.assertIn("tool_result", kinds)
        self.assertLess(kinds.index("tool_call"), kinds.index("tool_result"))
        self.assertEqual(kinds[-1], "final")
        self.assertEqual(events[-1][1]["text"], "等于 2。")

    def test_system_prompt_can_be_disabled(self) -> None:
        llm = FakeLLM([ChatResult(content="ok")])
        agent = Agent(llm=llm, tools=default_tools(self.workspace), system_prompt="")

        agent.run("hi")

        self.assertEqual([m["role"] for m in agent.history], ["user", "assistant"])


if __name__ == "__main__":
    unittest.main()
