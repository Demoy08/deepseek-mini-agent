"""DeepSeek Chat Completions 客户端（OpenAI 兼容协议，纯标准库实现）。

这一层只做三件事：

1. 把 ``messages`` + ``tools`` 发到 ``/chat/completions``；
2. 解析流式 SSE，把 ``tool_calls`` 的增量片段拼回完整调用；
3. 对 429 / 5xx / 网络抖动做有限重试。

Agent loop 本身在 ``loop.py``，这里不掺任何策略。
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Settings

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """调用模型失败。``retryable`` 表示同样的请求值得再试一次。"""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ChatResult:
    """一次模型回复。"""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> dict[str, Any]:
        """转成可回填进 ``messages`` 的 assistant 消息。"""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message


class StreamAssembler:
    """把流式 chunk 拼装成完整回复。

    流式响应里 ``tool_calls`` 是按 ``index`` 分片下发的：
    第一片带 ``id`` 和函数名，后续片只追加 ``arguments`` 字符串。
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None
        self._usage: dict[str, int] = {}

    def feed(self, chunk: dict[str, Any]) -> str:
        """吃进一个 chunk，返回本片新增的文本（可能为空）。"""
        if chunk.get("usage"):
            self._usage.update(chunk["usage"])

        text_delta = ""
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                text_delta += delta["content"]
                self._text.append(delta["content"])

            for tc in delta.get("tool_calls") or []:
                index = int(tc.get("index", 0))
                slot = self._tool_calls.setdefault(
                    index,
                    {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                if tc.get("type"):
                    slot["type"] = tc["type"]
                function = tc.get("function") or {}
                # 协议规定是字符串，但个别兼容实现会直接给对象，这里都兜住。
                if isinstance(function.get("name"), str):
                    slot["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    slot["function"]["arguments"] += function["arguments"]

            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]

        return text_delta

    def build(self) -> ChatResult:
        calls: list[dict[str, Any]] = []
        for index in sorted(self._tool_calls):
            slot = self._tool_calls[index]
            if not slot["id"]:
                slot["id"] = f"call_{index}"
            if not slot["function"]["arguments"]:
                slot["function"]["arguments"] = "{}"
            calls.append(slot)
        return ChatResult(
            content="".join(self._text),
            tool_calls=calls,
            finish_reason=self._finish_reason,
            usage=dict(self._usage),
        )


class DeepSeekClient:
    """最小可用的 DeepSeek 对话客户端。"""

    def __init__(self, settings: Settings, *, retry_delay: float = 1.5) -> None:
        self.settings = settings
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------ 公共接口

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> ChatResult:
        """发一次请求；``on_text`` 会在每个文本增量到达时被调用（仅流式）。"""
        body = self._build_body(messages, tools)
        if self.settings.stream:
            return self._chat_stream(body, on_text)
        return self._chat_once(body)

    def list_models(self) -> list[str]:
        """GET /models，用来确认账号下可用的模型 id。"""
        request = urllib.request.Request(
            self.settings.models_url, headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise LLMError(f"网络错误：{exc}", retryable=True) from exc

        items = data.get("data")
        if not isinstance(items, list):
            snippet = json.dumps(data, ensure_ascii=False)[:300]
            raise LLMError(f"模型列表格式异常：{snippet}")
        return [
            item["id"]
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
        ]

    # ------------------------------------------------------------------ 请求构造

    def _build_body(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": self.settings.stream,
            "temperature": self.settings.temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.settings.thinking != "auto":
            # DeepSeek V4.1 默认开启深度思考；显式关闭可显著降低延迟和费用。
            body["thinking"] = {"type": self.settings.thinking}
        if self.settings.stream:
            # 让服务端在最后一帧带上 token 用量；不支持的实现会忽略它。
            body["stream_options"] = {"include_usage": True}
        return body

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if self.settings.stream else "application/json",
            "User-Agent": "deepseek-mini-agent/0.1",
        }

    def _open(self, body: dict[str, Any]):
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.settings.chat_completions_url,
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=self.settings.timeout)

    # ------------------------------------------------------------------ 非流式

    def _chat_once(self, body: dict[str, Any]) -> ChatResult:
        last_error: LLMError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with self._open(body) as response:
                    raw = response.read()
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    snippet = raw[:300].decode("utf-8", "replace")
                    raise LLMError(f"响应不是合法 JSON：{snippet}") from exc
                return self._parse_completion(payload)
            except urllib.error.HTTPError as exc:
                last_error = self._http_error(exc)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                last_error = LLMError(f"网络错误：{exc}", retryable=True)

            if last_error is None or not last_error.retryable or attempt == MAX_ATTEMPTS:
                break
            time.sleep(self.retry_delay * 2 ** (attempt - 1))

        assert last_error is not None
        raise last_error

    def _parse_completion(self, data: dict[str, Any]) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            snippet = json.dumps(data, ensure_ascii=False)[:400]
            raise LLMError(f"模型没有返回 choices：{snippet}")

        choice = choices[0]
        message = choice.get("message") or {}
        tool_calls: list[dict[str, Any]] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = "{}" if arguments is None else json.dumps(arguments, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": item.get("id") or f"call_{len(tool_calls)}",
                    "type": item.get("type") or "function",
                    "function": {
                        "name": str(function.get("name") or ""),
                        "arguments": arguments or "{}",
                    },
                }
            )

        return ChatResult(
            content=message.get("content") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage") or {},
        )

    # ------------------------------------------------------------------ 流式

    def _chat_stream(
        self, body: dict[str, Any], on_text: Callable[[str], None] | None
    ) -> ChatResult:
        assembler = StreamAssembler()
        received = False
        first_line = ""
        last_error: LLMError | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with self._open(body) as response:
                    for raw_line in response:
                        line = raw_line.decode("utf-8", "replace").strip()
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            if not first_line and line:
                                first_line = line[:300]
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        if not data:
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        received = True
                        delta = assembler.feed(chunk)
                        if delta and on_text is not None:
                            on_text(delta)
                if not received:
                    snippet = f"：{first_line}" if first_line else ""
                    raise LLMError(f"服务端没有返回 SSE 数据，可能不是流式接口或返回了错误内容{snippet}")
                return assembler.build()
            except urllib.error.HTTPError as exc:
                last_error = self._http_error(exc)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
                last_error = LLMError(f"网络错误：{exc}", retryable=True)

            # 只要收到过任何 chunk，重试就会重复文本或工具调用，直接失败。
            if received or not last_error.retryable or attempt == MAX_ATTEMPTS:
                raise last_error

            # 重试前清空拼装器，避免残留半截数据。
            assembler = StreamAssembler()
            time.sleep(self.retry_delay * 2 ** (attempt - 1))

        raise last_error or LLMError("流式请求失败")

    # ------------------------------------------------------------------ 错误

    def _http_error(self, exc: urllib.error.HTTPError) -> LLMError:
        try:
            raw = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - 读错误体失败不该掩盖原始错误
            raw = ""
        finally:
            exc.close()

        detail = raw.strip()
        try:
            payload = json.loads(raw)
            detail = str((payload.get("error") or {}).get("message") or payload.get("message") or raw)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

        status = exc.code
        text = f"HTTP {status}：{detail[:500]}" if detail else f"HTTP {status}：{exc.reason}"
        return LLMError(text, status=status, retryable=status in RETRYABLE_STATUS)
