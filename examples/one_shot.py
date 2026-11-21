"""以库的方式驱动 agent，而不是走命令行。

    python examples/one_shot.py "用 list_dir 看看当前目录"
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接从仓库根目录运行，无需先 pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_agent import Agent, DeepSeekClient, Settings, default_tools


def main() -> int:
    prompt = " ".join(sys.argv[1:]).strip() or "用 list_dir 看看当前目录，再用一句话总结。"

    settings = Settings.from_env()
    agent = Agent(
        llm=DeepSeekClient(settings),
        tools=default_tools(settings.workspace),
        max_steps=settings.max_steps,
    )

    answer = agent.run(prompt, on_text=lambda chunk: (sys.stdout.write(chunk), sys.stdout.flush()))
    print()

    if not settings.stream:
        print(answer)

    print(f"\n[history] {len(agent.history)} 条消息")
    print(f"[usage]   {agent.total_usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
