"""Agent loop：整个项目的核心。

循环本身只有四步：

1. 调模型，拿到 assistant 消息（可能带 tool_calls）；
2. 没有 tool_calls → 这就是最终回答，结束；
3. 有 tool_calls → 逐个执行，把结果作为 role=tool 的消息追加进历史；
4. 带着新历史回到第 1 步，直到收敛或触发 ``max_steps``。

模型、工具、UI 都被挡在循环外面：换模型只改 ``llm``，换工具只改 ``tools``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .llm import ChatResult
from .tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = """你是一个运行在用户电脑上的命令行助手，通过调用工具完成任务。

规则：
1. 需要读取文件、写文件或做精确计算时，调用工具，不要凭记忆猜测结果。
2. 你可以在一次回复里并行调用多个工具。
3. 工具返回以「错误：」开头时，读懂原因后修正参数再试，不要原样重复。
4. 信息足够后直接给出简洁的中文回答；不要复述工具调用过程，也不要编造工具结果。"""


class ChatModel(Protocol):
    """``loop`` 需要的最小模型接口，方便测试时替换成假模型。"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatResult: ...


class StepLimitExceeded(RuntimeError):
    """达到单轮步数上限，通常意味着模型在原地打转。"""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        super().__init__(f"已连续调用 {steps} 次模型仍未给出最终回答，为防死循环已停止。")


@dataclass
class Agent:
    """把模型、工具和对话历史绑在一起。"""

    llm: ChatModel
    tools: ToolRegistry
    max_steps: int = 20
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    on_event: Callable[..., None] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    total_usage: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.history and self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    # ------------------------------------------------------------------ 主循环

    def run(
        self,
        user_text: str,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """处理一条用户输入，返回最终回答；期间可能调用任意多次工具。"""
        if not user_text.strip():
            raise ValueError("输入不能为空")

        self.repair_tail()
        self.history.append({"role": "user", "content": user_text})
        tools = self.tools.schemas() or None

        for step in range(1, self.max_steps + 1):
            self._emit("llm_start", step=step)
            result = self.llm.chat(self.history, tools=tools, on_text=on_text)
            self._add_usage(result.usage)
            self.history.append(result.to_message())
            self._emit(
                "assistant",
                step=step,
                content=result.content,
                tool_calls=result.tool_calls,
                usage=result.usage,
            )

            if not result.tool_calls:
                self._emit("final", step=step, text=result.content)
                return result.content

            for call in result.tool_calls:
                function = call.get("function") or {}
                name = function.get("name") or ""
                arguments = function.get("arguments") or "{}"
                call_id = call.get("id") or ""

                self._emit("tool_call", step=step, id=call_id, name=name, arguments=arguments)
                observation = self.tools.execute(name, arguments)
                self._emit("tool_result", step=step, id=call_id, name=name, result=observation)

                self.history.append(
                    {"role": "tool", "tool_call_id": call_id, "content": observation}
                )

        raise StepLimitExceeded(self.max_steps)

    def reset(self) -> None:
        """清空对话，但保留 system prompt。"""
        self.history = []
        self.total_usage = {}
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    def repair_tail(self) -> None:
        """历史若停在未回填的 tool_calls 上（如用户 Ctrl-C），补占位结果。

        OpenAI 兼容接口要求每个 tool_call 都有对应的 tool 消息，
        否则下一轮请求会直接 400。
        """
        if not self.history:
            return
        last = self.history[-1]
        if last.get("role") != "assistant":
            return
        for call in last.get("tool_calls") or []:
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": "错误：该工具调用被中断，未执行。",
                }
            )

    # ------------------------------------------------------------------ 内部

    def _add_usage(self, usage: dict[str, int] | None) -> None:
        if not usage:
            return
        for key, value in usage.items():
            if isinstance(value, int):
                self.total_usage[key] = self.total_usage.get(key, 0) + value

    def _emit(self, kind: str, **payload: Any) -> None:
        if self.on_event is not None:
            self.on_event(kind, **payload)
