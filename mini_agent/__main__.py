"""命令行入口：交互式 REPL、一次性执行、列出可用模型。

    python -m mini_agent            # 进入 REPL
    python -m mini_agent -p "..."   # 只跑一句
    python -m mini_agent --models   # 查看账号下的模型 id
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from . import __version__
from .config import ConfigError, Settings
from .llm import DeepSeekClient, LLMError
from .loop import Agent, StepLimitExceeded
from .tools import default_tools

TOOL_ARG_PREVIEW = 200
TOOL_RESULT_PREVIEW = 160


# --------------------------------------------------------------------------- 终端显示


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return False
    if not sys.stdout.isatty():
        return False
    # 老式 Windows 控制台默认不解析 ANSI，避免把转义序列原样打出来。
    if os.name == "nt":
        return bool(
            os.environ.get("WT_SESSION")
            or os.environ.get("ANSICON")
            or os.environ.get("ConEmuANSI") == "ON"
        )
    return True


class Console:
    """负责所有终端输出，保证 agent loop 保持纯净。"""

    def __init__(self, stream: bool, verbose: bool = False, color: bool | None = None) -> None:
        self.stream = stream
        self.verbose = verbose
        self.use_color = _supports_color() if color is None else color
        self._wrote_text = False
        self._at_line_start = True

    # -- 着色

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.use_color else text

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    # -- 基础输出

    def info(self, text: str) -> None:
        print(self.dim(text))

    def error(self, text: str) -> None:
        print(self.red(text), file=sys.stderr)

    def _break_line(self) -> None:
        """只在「已经输出过内容且不在行首」时补一个换行，避免多出空行。"""
        if self._wrote_text and not self._at_line_start:
            print()
        self._wrote_text = False
        self._at_line_start = True

    # -- 流式文本

    def on_text(self, chunk: str) -> None:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        self._wrote_text = True
        self._at_line_start = chunk.endswith("\n")

    # -- 循环事件

    def on_event(self, kind: str, **data: Any) -> None:
        if kind == "tool_call":
            self._break_line()
            arguments = str(data.get("arguments") or "")
            if len(arguments) > TOOL_ARG_PREVIEW and not self.verbose:
                arguments = arguments[:TOOL_ARG_PREVIEW] + "…"
            print(self.dim("  → ") + self.cyan(str(data.get("name"))) + self.dim(f"({arguments})"))
        elif kind == "tool_result":
            self._break_line()
            result = str(data.get("result") or "")
            if self.verbose:
                indented = result.replace("\n", "\n    ")
                print(self.dim("  ← ") + indented)
            else:
                lines = result.splitlines() or [""]
                preview = lines[0][:TOOL_RESULT_PREVIEW]
                extra = f"  …(+{len(lines) - 1} 行)" if len(lines) > 1 else ""
                print(self.dim("  ← ") + preview + extra)
        elif kind == "assistant" and not self.stream:
            content = data.get("content") or ""
            if content:
                print(content)

    # -- 回合收尾

    def finish_turn(self, usage: dict[str, int]) -> None:
        self._break_line()
        if usage:
            print(self.dim(f"  [tokens] {_format_usage(usage)}"))

    # -- 固定区块

    def banner(self, settings: Settings, tools: list[str]) -> None:
        print(self.bold(f"deepseek-mini-agent v{__version__}"))
        print(self.dim(f"  model   : {settings.model}"))
        print(self.dim(f"  endpoint: {settings.base_url}"))
        print(self.dim(f"  workdir : {settings.workspace}"))
        print(self.dim(f"  thinking: {settings.thinking}"))
        print(self.dim(f"  tools   : {', '.join(tools) or '（无）'}"))
        print(self.dim("  输入 /help 查看命令，/exit 退出。"))
        print()

    def show_help(self) -> None:
        print(
            "命令：\n"
            "  /help     显示帮助\n"
            "  /reset    清空对话历史（保留 system prompt）\n"
            "  /history  打印当前消息历史\n"
            "  /exit     退出（/quit、/q 同义）\n"
            "其他输入一律作为用户消息交给 agent。"
        )

    def show_history(self, history: list[dict[str, Any]]) -> None:
        if not history:
            self.info("（历史为空）")
            return
        for index, message in enumerate(history):
            role = message.get("role")
            content = message.get("content")
            text = content if isinstance(content, str) else ""
            if len(text) > 400:
                text = text[:400] + "…"
            print(self.dim(f"[{index}] {role}"))
            if text:
                print(text)
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                print(self.cyan(f"  tool_call: {function.get('name')}({function.get('arguments')})"))
        print()

    def prompt(self) -> str:
        return self.cyan("你 › ") if self.use_color else "你 › "


def _format_usage(usage: dict[str, int]) -> str:
    parts = []
    for key, label in (
        ("prompt_tokens", "prompt"),
        ("completion_tokens", "completion"),
        ("prompt_cache_hit_tokens", "cache_hit"),
        ("total_tokens", "total"),
    ):
        if key in usage:
            parts.append(f"{label}={usage[key]}")
    return " ".join(parts) or json.dumps(usage, ensure_ascii=False)


# --------------------------------------------------------------------------- 主流程


def _configure_stdio() -> None:
    """输出被重定向时用 UTF-8，避免中文在 GBK 管道里编码失败。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty() and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mini-agent",
        description="由 agent loop 驱动的极简 DeepSeek agent",
    )
    parser.add_argument("-p", "--prompt", help="一次性执行这句指令，打印结果后退出")
    parser.add_argument("--gui", action="store_true", help="启动图形界面（Tkinter，Python 自带）")
    parser.add_argument("-m", "--model", help="模型 id，默认取 DEEPSEEK_MODEL 或 deepseek-flash")
    parser.add_argument("--base-url", help="API 地址，默认 https://api.deepseek.com")
    parser.add_argument("--workspace", help="工具可读写的工作目录，默认当前目录")
    parser.add_argument("--max-steps", type=int, help="单轮最多调用模型次数，默认 20")
    parser.add_argument("--temperature", type=float, help="采样温度，默认 0.7（思考模式下无效）")
    parser.add_argument(
        "--thinking",
        choices=["auto", "enabled", "disabled"],
        help="深度思考模式：默认 disabled（更快更省），enabled 更会推理，auto 交给服务端默认",
    )
    parser.add_argument("--no-stream", action="store_true", help="关闭流式输出")
    parser.add_argument("--models", action="store_true", help="列出该 API key 可用的模型 id 后退出")
    parser.add_argument("-v", "--verbose", action="store_true", help="显示完整工具参数与结果")
    parser.add_argument("--no-color", action="store_true", help="禁用彩色输出")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        model=args.model,
        base_url=args.base_url,
        workspace=args.workspace,
        max_steps=args.max_steps,
        temperature=args.temperature,
        thinking=args.thinking,
        stream=False if args.no_stream else None,
    )


def _hint_for_status(status: int | None) -> str:
    if status == 401:
        return "（key 无效、已删除或没被读到，请检查 .env 里的 DEEPSEEK_API_KEY）"
    if status == 402:
        return "（账户余额不足，请到 platform.deepseek.com 充值）"
    if status == 429:
        return "（请求太频繁，稍等再试）"
    return ""


def _humanize_llm_error(exc: LLMError) -> str:
    """把常见 HTTP 状态翻成给新手看得懂的提示。"""
    return f"模型调用失败：{exc}{_hint_for_status(exc.status)}"


def run_turn(agent: Agent, console: Console, text: str) -> bool:
    before = dict(agent.total_usage)
    try:
        answer = agent.run(text, on_text=console.on_text if console.stream else None)
        if not answer:
            console.info("（模型没有返回文字内容）")
    except StepLimitExceeded as exc:
        console.error(str(exc))
        return False
    except LLMError as exc:
        console.error(_humanize_llm_error(exc))
        return False
    except KeyboardInterrupt:
        agent.repair_tail()
        print()
        console.error("已中断本轮。")
        return False

    usage = {key: value - before.get(key, 0) for key, value in agent.total_usage.items()}
    console.finish_turn(usage)
    return True


def repl(agent: Agent, console: Console, settings: Settings) -> int:
    console.banner(settings, agent.tools.names)
    while True:
        try:
            line = input(console.prompt())
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not sys.stdin.isatty():
            # 管道输入时没有回显的换行，补一个，避免提示符和工具日志挤在一行
            print()

        text = line.strip()
        if not text:
            continue
        if text in {"/exit", "/quit", "/q"}:
            return 0
        if text in {"/help", "/?"}:
            console.show_help()
            continue
        if text == "/reset":
            agent.reset()
            console.info("对话历史已重置。")
            continue
        if text == "/history":
            console.show_history(agent.history)
            continue

        run_turn(agent, console, text)
        print()


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    args = _build_parser().parse_args(argv)

    if args.gui:
        from .gui import run_gui

        return run_gui()

    try:
        settings = _settings_from_args(args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2

    client = DeepSeekClient(settings)

    if args.models:
        try:
            models = client.list_models()
        except LLMError as exc:
            print(f"获取模型列表失败：{exc}{_hint_for_status(exc.status)}", file=sys.stderr)
            return 1
        print("\n".join(models) or "（没有返回任何模型）")
        if settings.model not in models:
            print(
                f"提示：当前配置的模型 {settings.model!r} 不在上面的列表里，"
                "如果调用报 model not found，请用 --model 换一个。",
                file=sys.stderr,
            )
        return 0

    try:
        tools = default_tools(settings.workspace)
    except OSError as exc:
        print(f"无法准备工作目录 {settings.workspace}：{exc}", file=sys.stderr)
        return 2
    console = Console(
        stream=settings.stream,
        verbose=args.verbose,
        color=False if args.no_color else None,
    )
    agent = Agent(
        llm=client,
        tools=tools,
        max_steps=settings.max_steps,
        on_event=console.on_event,
    )

    if args.prompt:
        ok = run_turn(agent, console, args.prompt)
        return 0 if ok else 1

    return repl(agent, console, settings)


if __name__ == "__main__":
    raise SystemExit(main())
