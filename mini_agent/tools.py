"""工具层：声明 JSON Schema、执行、沙箱约束。

设计原则：

* 工具执行 **永不抛出** —— 任何异常都变成给模型看的 `错误：...` 文本，
  让 agent 自己决定重试还是换方案；
* 文件工具限制在 workspace 目录内，防止 ``../`` 逃逸；
* calculator 用 AST 白名单求值，不碰 ``eval``。
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

MAX_READ_BYTES = 64 * 1024
MAX_READ_LINES = 400
MAX_TOOL_OUTPUT = 16 * 1024

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


@dataclass(frozen=True)
class Tool:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI / DeepSeek 的 tool 定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：负责 schema 汇总与安全执行。"""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重名：{tool.name}")
        self._tools[tool.name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names]

    def execute(self, name: str, arguments: str) -> str:
        """执行工具并返回观察文本；任何异常都被转成错误文本。"""
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未知工具 {name!r}。可用工具：{', '.join(self.names) or '（无）'}"

        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return (
                f"错误：arguments 不是合法 JSON（{exc.msg}）。"
                f"请提供一个 JSON 对象，例如 {{\"path\": \"a.txt\"}}。"
            )
        if not isinstance(args, dict):
            return "错误：arguments 必须是 JSON 对象（键值对），不能是数组或标量。"

        try:
            result = tool.func(**args)
        except TypeError as exc:
            schema = json.dumps(tool.parameters, ensure_ascii=False)
            return f"错误：调用参数不匹配（{exc}）。该工具的参数 schema：{schema}"
        except Exception as exc:  # noqa: BLE001 - 工具失败要变成观察文本
            return f"错误：工具 {name} 执行失败：{type(exc).__name__}: {exc}"

        text = result if isinstance(result, str) else str(result)
        if len(text) > MAX_TOOL_OUTPUT:
            text = text[:MAX_TOOL_OUTPUT] + f"\n...（已截断，原始长度 {len(text)} 字符）"
        return text


# --------------------------------------------------------------------------- 内置工具


def _resolve_in(root: Path, raw: str) -> Path:
    """把用户给的相对路径限制在 root 内。"""
    candidate = Path(raw).expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PermissionError(f"路径 {raw!r} 越出了工作目录 {root}")
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix() or "."
    except ValueError:
        return str(path)


def default_tools(workspace: str | Path) -> ToolRegistry:
    """构建默认工具集，所有文件操作都固定在 ``workspace`` 内。"""
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def read_file(path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        target = _resolve_in(root, path)
        if target.is_dir():
            raise IsADirectoryError(f"{path} 是目录，请用 list_dir")
        if not target.exists():
            raise FileNotFoundError(f"文件不存在：{path}")
        if target.stat().st_size > MAX_READ_BYTES:
            raise ValueError(f"文件超过 {MAX_READ_BYTES // 1024}KB，请改用更小的文件")

        offset = max(1, int(offset))
        limit = max(1, min(int(limit), MAX_READ_LINES))
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        chunk = lines[offset - 1 : offset - 1 + limit]
        if not chunk:
            return f"（{path} 的第 {offset} 行之后没有内容，文件共 {len(lines)} 行）"

        numbered = "\n".join(f"{offset + i:>5}| {line}" for i, line in enumerate(chunk))
        remaining = len(lines) - (offset - 1 + len(chunk))
        if remaining > 0:
            numbered += f"\n...（还有 {remaining} 行，用 offset={offset + len(chunk)} 继续读）"
        return numbered

    def write_file(path: str, content: str) -> str:
        target = _resolve_in(root, path)
        if target.is_dir():
            raise IsADirectoryError(f"{path} 是目录")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {len(content)} 个字符到 {_relative(root, target)}"

    def list_dir(path: str = ".") -> str:
        target = _resolve_in(root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"{path} 不是目录")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        if not entries:
            return f"（{_relative(root, target)} 是空目录）"
        lines = []
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                lines.append(f"{entry.name} ({size} B)")
        return "\n".join(lines)

    def calculator(expression: str) -> str:
        try:
            tree = ast.parse(str(expression), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"不是合法的算术表达式：{exc.msg}") from exc
        value = _eval_arithmetic(tree.body)
        if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
            value = int(value)
        return str(value)

    def get_current_time(timezone: str = "local") -> str:
        moment = dt.datetime.now(dt.timezone.utc)
        if timezone.lower() == "local":
            moment = moment.astimezone()
        elif timezone.lower() in {"utc", "z"}:
            pass
        else:
            raise ValueError(f"不支持的时区 {timezone!r}，只支持 local / utc")
        return moment.isoformat(timespec="seconds")

    return ToolRegistry(
        [
            Tool(
                name="read_file",
                description="读取工作目录内文本文件的内容，按行号返回。支持分页读取。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的文件路径"},
                        "offset": {"type": "integer", "description": "起始行号，从 1 开始，默认 1"},
                        "limit": {"type": "integer", "description": f"最多读取行数，默认 {MAX_READ_LINES}"},
                    },
                    "required": ["path"],
                },
                func=read_file,
            ),
            Tool(
                name="write_file",
                description="把文本内容写入工作目录内的文件（覆盖写，父目录自动创建）。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的文件路径"},
                        "content": {"type": "string", "description": "要写入的完整文本内容"},
                    },
                    "required": ["path", "content"],
                },
                func=write_file,
            ),
            Tool(
                name="list_dir",
                description="列出工作目录内某个目录的文件与子目录。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录的目录路径，默认当前目录 ."}
                    },
                },
                func=list_dir,
            ),
            Tool(
                name="calculator",
                description="计算四则运算、幂、取模等算术表达式，例如 (3+4)*2**3。",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "只包含数字、运算符和括号的表达式"}
                    },
                    "required": ["expression"],
                },
                func=calculator,
            ),
            Tool(
                name="get_current_time",
                description="获取当前时间，返回 ISO 8601 字符串。",
                parameters={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "enum": ["local", "utc"],
                            "description": "local 为本地时区，utc 为 UTC；默认 local",
                        }
                    },
                },
                func=get_current_time,
            ),
        ]
    )


def _eval_arithmetic(node: ast.AST) -> int | float:
    """只允许数字、括号与算术运算符的 AST 求值。"""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("表达式里只能出现数字")
        return node.value

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_arithmetic(node.left)
        right = _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大")
        result = _BIN_OPS[type(node.op)](left, right)
        if isinstance(result, complex):
            raise ValueError("不支持复数")
        if abs(result) > 1e100:
            raise ValueError("结果过大")
        return result

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arithmetic(node.operand))

    raise ValueError(f"表达式包含不支持的语法：{type(node).__name__}")
