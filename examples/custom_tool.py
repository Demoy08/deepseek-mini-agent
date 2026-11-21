"""演示如何注册自己的工具：一个统计字符数的 word_count。

    python examples/custom_tool.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_agent import Agent, DeepSeekClient, Settings, Tool, default_tools


def word_count(text: str) -> str:
    return f"字符数 {len(text)}，空白分隔词数 {len(text.split())}"


def build_agent(settings: Settings) -> Agent:
    tools = default_tools(settings.workspace)
    tools.add(
        Tool(
            name="word_count",
            description="统计一段文本的字符数与词数",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "待统计的文本"}},
                "required": ["text"],
            },
            func=word_count,
        )
    )
    return Agent(llm=DeepSeekClient(settings), tools=tools, max_steps=settings.max_steps)


def main() -> int:
    settings = Settings.from_env()
    agent = build_agent(settings)

    prompt = "用 word_count 统计这句话：agent loop 就是模型与工具之间的循环，直到模型不再调用工具。"
    answer = agent.run(prompt, on_text=lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush()))
    print()
    if not settings.stream:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
