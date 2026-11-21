"""Tkinter 图形界面：零第三方依赖的聊天窗口。

启动方式::

    python -m mini_agent --gui
    python -m mini_agent.gui

界面只有三块：顶部状态与按钮、中间对话区、底部输入框。
设置在右上角，首次运行会先弹窗让你填 key。
"""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from typing import Any

from .config import DEFAULT_MODEL, ConfigError, Settings, write_env
from .llm import DeepSeekClient, LLMError
from .loop import Agent, StepLimitExceeded
from .tools import default_tools

BG = "#f4f5f7"
CARD = "#ffffff"
BORDER = "#e5e7eb"
USER_BG = "#e8f1ff"
AGENT_BG = "#f2f3f5"
TEXT = "#111827"
MUTED = "#6b7280"
ACCENT = "#2563eb"
GREEN = "#047857"
RED = "#b91c1c"


# --------------------------------------------------------------------------- 小工具


def _font_family(root: tk.Misc) -> str:
    available = set(tkfont.families(root))
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "PingFang SC", "Noto Sans CJK SC"):
        if name in available:
            return name
    return "TkDefaultFont"


def _looks_like_key(value: str) -> bool:
    value = value.strip()
    return value.startswith("sk-") and len(value) >= 20 and "your-key" not in value


def _hint_for_status(status: int | None) -> str:
    if status == 401:
        return "\n\n请点右上角「设置」检查 API key。"
    if status == 402:
        return "\n\n账户余额不足，请到 platform.deepseek.com 充值。"
    if status == 429:
        return "\n\n请求太频繁，稍后再试。"
    return ""


def _humanize_error(exc: LLMError) -> str:
    return f"模型调用失败：{exc}{_hint_for_status(exc.status)}"


def _preview(text: str, limit: int) -> str:
    first = text.splitlines()[0] if text.splitlines() else ""
    extra = len(text.splitlines()) - 1
    if len(first) > limit:
        first = first[:limit] + "…"
    return first + (f"  (+{extra} 行)" if extra > 0 else "")


def _format_usage(usage: dict[str, int]) -> str:
    labels = (
        ("prompt_tokens", "prompt"),
        ("completion_tokens", "completion"),
        ("prompt_cache_hit_tokens", "cache_hit"),
        ("total_tokens", "total"),
    )
    return " ".join(f"{label}={usage[key]}" for key, label in labels if key in usage) or "tokens: -"


def _short_path(path: Path, limit: int = 52) -> str:
    text = str(path)
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


# --------------------------------------------------------------------------- 设置弹窗


def settings_dialog(parent: tk.Misc, current: Settings | None) -> Settings | None:
    """读取 key / 模型 / 工作目录 / 思考开关，写回 .env 后返回新配置。"""
    dialog = tk.Toplevel(parent)
    dialog.title("设置")
    dialog.configure(bg=CARD)
    dialog.resizable(False, False)
    dialog.transient(parent)
    dialog.grab_set()

    family = _font_family(dialog)
    result: dict[str, Settings | None] = {"value": None}

    api_key = tk.StringVar(value=current.api_key if current else "")
    model = tk.StringVar(value=current.model if current else DEFAULT_MODEL)
    workspace = tk.StringVar(value=str(current.workspace) if current else str(Path.cwd()))
    thinking = tk.BooleanVar(value=bool(current and current.thinking == "enabled"))

    form = tk.Frame(dialog, bg=CARD, padx=18, pady=16)
    form.pack(fill="both", expand=True)

    def add_row(row: int, label: str, widget: tk.Widget) -> None:
        tk.Label(form, text=label, bg=CARD, fg=MUTED, font=(family, 9)).grid(
            row=row, column=0, sticky="w", pady=(8, 2)
        )
        widget.grid(row=row + 1, column=0, sticky="ew")

    add_row(
        0,
        "API key（sk- 开头，只保存在本机 .env）",
        tk.Entry(form, textvariable=api_key, show="*", width=52, font=(family, 10), relief="flat",
                 highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT),
    )
    add_row(
        2,
        "模型 id",
        tk.Entry(form, textvariable=model, width=52, font=(family, 10), relief="flat",
                 highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT),
    )

    tk.Label(form, text="工作目录（agent 只能读写这里）", bg=CARD, fg=MUTED, font=(family, 9)).grid(
        row=4, column=0, sticky="w", pady=(8, 2)
    )
    path_row = tk.Frame(form, bg=CARD)
    path_row.grid(row=5, column=0, sticky="ew")
    tk.Entry(path_row, textvariable=workspace, font=(family, 10), relief="flat",
             highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT).pack(
        side="left", fill="x", expand=True
    )
    def browse() -> None:
        chosen = filedialog.askdirectory(initialdir=workspace.get() or str(Path.cwd()))
        if chosen:
            workspace.set(chosen)

    tk.Button(
        path_row,
        text="浏览…",
        command=browse,
        relief="flat",
        bg="#eef0f3",
        activebackground="#e2e5e9",
        font=(family, 9),
        cursor="hand2",
        padx=10,
    ).pack(side="left", padx=(8, 0))

    tk.Checkbutton(
        form,
        text="深度思考（更慢、更贵，推理更强）",
        variable=thinking,
        bg=CARD,
        activebackground=CARD,
        fg=TEXT,
        font=(family, 9),
        highlightthickness=0,
    ).grid(row=6, column=0, sticky="w", pady=(12, 0))

    buttons = tk.Frame(form, bg=CARD)
    buttons.grid(row=7, column=0, sticky="e", pady=(16, 0))

    def on_cancel() -> None:
        result["value"] = None
        dialog.destroy()

    def on_save() -> None:
        key = api_key.get().strip()
        if not _looks_like_key(key):
            messagebox.showerror("设置", "API key 看起来不对：应当以 sk- 开头，长度 20 以上。", parent=dialog)
            return
        model_value = model.get().strip() or DEFAULT_MODEL
        workspace_value = workspace.get().strip() or str(Path.cwd())
        values = {
            "DEEPSEEK_API_KEY": key,
            "DEEPSEEK_MODEL": model_value,
            "MINI_AGENT_WORKSPACE": workspace_value,
            "MINI_AGENT_THINKING": "enabled" if thinking.get() else "disabled",
        }
        try:
            Path(workspace_value).expanduser().resolve().mkdir(parents=True, exist_ok=True)
            # 同一进程内 os.environ 优先于 .env，必须同步更新，否则新设置要重启才生效
            os.environ.update(values)
            write_env(Path.cwd() / ".env", values)
            result["value"] = Settings.from_env()
        except (OSError, ConfigError) as exc:
            messagebox.showerror("设置", f"保存失败：{exc}", parent=dialog)
            return
        dialog.destroy()

    tk.Button(
        buttons, text="取消", command=on_cancel, relief="flat", bg="#eef0f3", activebackground="#e2e5e9",
        font=(family, 9), padx=14, pady=4, cursor="hand2",
    ).pack(side="right", padx=(8, 0))
    tk.Button(
        buttons, text="保存", command=on_save, relief="flat", bg=ACCENT, fg="white", activebackground="#1d4ed8",
        activeforeground="white", font=(family, 9), padx=14, pady=4, cursor="hand2",
    ).pack(side="right")

    dialog.protocol("WM_DELETE_WINDOW", on_cancel)
    dialog.bind("<Escape>", lambda _event: on_cancel())
    dialog.update_idletasks()
    dialog.geometry(f"+{parent.winfo_rootx() + 120}+{parent.winfo_rooty() + 100}")
    dialog.wait_window()
    return result["value"]


# --------------------------------------------------------------------------- 主窗口


class AgentGUI:
    """把 Agent 包成一个聊天窗口。"""

    def __init__(self, root: tk.Tk, settings: Settings) -> None:
        self.root = root
        self.settings = settings
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False
        self.streaming = False
        self.usage_before: dict[str, int] = {}
        self._at_line_start = True
        self.family = _font_family(root)

        self.agent = self._build_agent(settings)
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._drain()

    # ------------------------------------------------------------------ 构建

    def _build_agent(self, settings: Settings) -> Agent:
        return Agent(
            llm=DeepSeekClient(settings),
            tools=default_tools(settings.workspace),
            max_steps=settings.max_steps,
            on_event=lambda kind, **data: self.queue.put(("event", (kind, data))),
        )

    def _model_text(self) -> str:
        thinking = "思考开" if self.settings.thinking == "enabled" else "思考关"
        return f"{self.settings.model} · {thinking}"

    def _build_ui(self) -> None:
        self.root.title("deepseek-mini-agent")
        self.root.geometry("920x680")
        self.root.minsize(720, 540)
        self.root.configure(bg=BG)

        header = tk.Frame(self.root, bg=CARD, padx=16, pady=10)
        header.pack(fill="x")

        # 先占住右侧按钮的位置，避免左侧长文本把它们挤出窗口
        actions = tk.Frame(header, bg=CARD)
        actions.pack(side="right")

        def flat_button(parent: tk.Misc, text: str, command) -> tk.Button:
            return tk.Button(
                parent, text=text, command=command, relief="flat", bg="#eef0f3", fg=TEXT,
                activebackground="#e2e5e9", font=(self.family, 9), padx=12, pady=3, cursor="hand2",
            )

        self.settings_btn = flat_button(actions, "设置", self._open_settings)
        self.settings_btn.pack(side="right")
        self.clear_btn = flat_button(actions, "清空对话", self._clear)
        self.clear_btn.pack(side="right", padx=(0, 8))

        tk.Label(header, text="deepseek-mini-agent", bg=CARD, fg=TEXT, font=(self.family, 13, "bold")).pack(side="left")
        self.model_label = tk.Label(header, text=self._model_text(), bg=CARD, fg=MUTED, font=(self.family, 9))
        self.model_label.pack(side="left", padx=(12, 0), pady=(4, 0))

        body = tk.Frame(self.root, bg=BG, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        self.chat = tk.Text(
            body, wrap="word", state="disabled", bg=CARD, fg=TEXT, relief="flat",
            highlightthickness=1, highlightbackground=BORDER, font=(self.family, 10),
            padx=10, pady=10, spacing2=3, cursor="arrow",
        )
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.chat.yview)
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.chat.tag_configure("role_user", foreground=ACCENT, font=(self.family, 9, "bold"), spacing1=10)
        self.chat.tag_configure("role_agent", foreground=GREEN, font=(self.family, 9, "bold"), spacing1=10)
        self.chat.tag_configure(
            "user", background=USER_BG, lmargin1=14, lmargin2=14, rmargin=14, spacing1=6, spacing3=6
        )
        self.chat.tag_configure(
            "assistant", background=AGENT_BG, lmargin1=14, lmargin2=14, rmargin=14, spacing1=6, spacing3=6
        )
        self.chat.tag_configure("tool", foreground=MUTED, font=("Consolas", 9), lmargin1=28, lmargin2=28, spacing3=1)
        self.chat.tag_configure(
            "meta", foreground="#9ca3af", font=(self.family, 8), lmargin1=14, spacing1=4, spacing3=8
        )
        self.chat.tag_configure("error", foreground=RED, lmargin1=14, lmargin2=14, spacing3=6)

        footer = tk.Frame(self.root, bg=BG, padx=16)
        footer.pack(fill="x")
        self.input = tk.Text(
            footer, height=3, wrap="word", font=(self.family, 10), relief="flat",
            highlightthickness=1, highlightbackground="#d1d5db", highlightcolor=ACCENT, padx=8, pady=6,
        )
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_return)

        self.send_btn = tk.Button(
            footer, text="发送", command=self._send, relief="flat", bg=ACCENT, fg="white",
            activebackground="#1d4ed8", activeforeground="white", font=(self.family, 10),
            width=8, cursor="hand2",
        )
        self.send_btn.pack(side="right", padx=(10, 0), fill="y")

        status = tk.Frame(self.root, bg=BG, padx=16, pady=8)
        status.pack(fill="x")
        self.status_label = tk.Label(status, text="就绪", bg=BG, fg=MUTED, font=(self.family, 9))
        self.status_label.pack(side="left")
        self.path_label = tk.Label(
            status, text=_short_path(self.settings.workspace), bg=BG, fg="#9ca3af", font=(self.family, 8)
        )
        self.path_label.pack(side="left", padx=(12, 0))
        tk.Label(status, text="Enter 发送 · Shift+Enter 换行", bg=BG, fg="#9ca3af", font=(self.family, 8)).pack(
            side="right"
        )

        self._append(
            "直接输入你的需求，例如「看看当前目录有什么」或「算一下 (3+4)*2**3」。\n",
            "meta",
        )
        self.input.focus_set()

    # ------------------------------------------------------------------ 交互

    def _on_return(self, event: tk.Event) -> str | None:
        if event.state & 0x0001:  # Shift
            return None
        self._send()
        return "break"

    def _send(self) -> None:
        if self.busy:
            return
        prompt = self.input.get("1.0", "end").strip()
        if not prompt:
            return

        self.input.delete("1.0", "end")
        self.busy = True
        self.streaming = False
        self.usage_before = dict(self.agent.total_usage)
        self._append_line("你", "role_user")
        self._append(prompt + "\n", "user")
        self._set_status("正在请求…")
        threading.Thread(target=self._worker, args=(prompt,), daemon=True).start()

    def _worker(self, prompt: str) -> None:
        try:
            self.agent.run(prompt, on_text=lambda chunk: self.queue.put(("delta", chunk)))
            usage = {key: value - self.usage_before.get(key, 0) for key, value in self.agent.total_usage.items()}
            self.queue.put(("done", usage))
        except StepLimitExceeded as exc:
            self.queue.put(("error", str(exc)))
        except LLMError as exc:
            self.queue.put(("error", _humanize_error(exc)))
        except Exception as exc:  # noqa: BLE001 - 界面必须把异常显示出来而不是静默
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break

            if kind == "delta":
                if not self.streaming:
                    self._append_line("Agent", "role_agent")
                    self.streaming = True
                self._append(str(payload), "assistant")
            elif kind == "event":
                event_kind, data = payload
                self._on_agent_event(event_kind, data)
            elif kind == "done":
                if payload:
                    self._append_line(f"  {_format_usage(payload)}", "meta")
                self.busy = False
                self.streaming = False
                self._set_status("就绪")
            elif kind == "error":
                self._append_line(payload, "error")
                self.busy = False
                self.streaming = False
                self._set_status("出错")

        self.root.after(60, self._drain)

    def _on_agent_event(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "tool_call":
            self.streaming = False
            arguments = str(data.get("arguments") or "")
            if len(arguments) > 300:
                arguments = arguments[:300] + "…"
            self._append_line(f"  → {data.get('name')}({arguments})", "tool")
        elif kind == "tool_result":
            self._append_line(f"  ← {_preview(str(data.get('result') or ''), 180)}", "tool")

    def _append(self, text: str, tag: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.configure(state="disabled")
        self.chat.see("end")
        if text:
            self._at_line_start = text.endswith("\n")

    def _ends_with_newline(self) -> bool:
        return self._at_line_start

    def _append_line(self, text: str, tag: str) -> None:
        """整行追加：保证内容从新行开始，且以换行结束。"""
        if not self._ends_with_newline():
            self._append("\n", tag)
        self._append(text + "\n", tag)

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)
        self.send_btn.configure(state="disabled" if self.busy else "normal")

    def _clear(self) -> None:
        if self.busy:
            messagebox.showinfo("清空对话", "正在请求中，请等这一轮结束。", parent=self.root)
            return
        self.agent.reset()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._at_line_start = True
        self.streaming = False
        self._set_status("就绪")

    def _open_settings(self) -> None:
        if self.busy:
            messagebox.showinfo("设置", "正在请求中，请等这一轮结束。", parent=self.root)
            return
        new_settings = settings_dialog(self.root, self.settings)
        if new_settings is None:
            return
        self.settings = new_settings
        self.agent = self._build_agent(new_settings)
        self.model_label.configure(text=self._model_text())
        self.path_label.configure(text=_short_path(new_settings.workspace))
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._at_line_start = True
        self.streaming = False
        self._append_line("（设置已更新，对话已重置）", "meta")

    def _on_close(self) -> None:
        if self.busy and not messagebox.askyesno("退出", "正在请求中，确定退出吗？", parent=self.root):
            return
        self.root.destroy()


# --------------------------------------------------------------------------- 入口


def run_gui() -> int:
    """启动图形界面；缺少有效 key 时先弹设置窗口。"""
    root = tk.Tk()
    root.withdraw()

    try:
        settings = Settings.from_env()
        if not _looks_like_key(settings.api_key):
            raise ConfigError("API key 还没填")
    except ConfigError:
        settings = settings_dialog(root, None)
        if settings is None:
            root.destroy()
            return 1

    root.deiconify()
    AgentGUI(root, settings)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gui())
