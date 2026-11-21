"""图形界面冒烟测试。

CI 或没有显示环境时自动跳过；这里只验证控件能建出来、消息能渲染。
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tests.support import temporary_directory


def _tk_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.update()
        root.destroy()
        return True
    except Exception:  # noqa: BLE001 - 没显示器、缺 tkinter 都算不可用
        return False


TK_AVAILABLE = _tk_available()


@unittest.skipUnless(TK_AVAILABLE, "没有可用的 Tk 显示环境")
class GuiSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        import tkinter as tk

        from mini_agent.config import Settings
        from mini_agent.gui import AgentGUI

        self._tmp = temporary_directory()
        self.root = tk.Tk()
        self.root.withdraw()
        settings = Settings(
            api_key="sk-test-0123456789abcdef",
            workspace=Path(self._tmp.name),
            stream=True,
        )
        self.app = AgentGUI(self.root, settings)

    def tearDown(self) -> None:
        self.root.destroy()
        self._tmp.cleanup()

    def test_widgets_are_created(self) -> None:
        self.assertEqual(self.root.title(), "deepseek-mini-agent")
        self.assertIn("deepseek-flash", self.app.model_label.cget("text"))
        self.assertEqual(self.app.send_btn.cget("text"), "发送")
        self.assertEqual(self.app.status_label.cget("text"), "就绪")

    def test_append_and_clear(self) -> None:
        self.app._append("hello", "user")
        self.assertIn("hello", self.app.chat.get("1.0", "end"))

        self.app._clear()

        self.assertEqual(self.app.chat.get("1.0", "end").strip(), "")

    def test_events_render_tool_calls_and_usage(self) -> None:
        self.app.queue.put(("event", ("tool_call", {"name": "calculator", "arguments": '{"expression": "1+1"}'})))
        self.app.queue.put(("event", ("tool_result", {"result": "2"})))
        self.app.queue.put(("done", {"total_tokens": 12}))

        self.app._drain()

        content = self.app.chat.get("1.0", "end")
        self.assertIn("→ calculator", content)
        self.assertIn("← 2", content)
        self.assertIn("total=12", content)


class GuiHelpersTest(unittest.TestCase):
    def test_key_shape_check(self) -> None:
        from mini_agent.gui import _looks_like_key

        self.assertTrue(_looks_like_key("sk-0123456789abcdef0123456789abcdef"))
        self.assertFalse(_looks_like_key(""))
        self.assertFalse(_looks_like_key("sk-your-key-here"))
        self.assertFalse(_looks_like_key("16b13b4e-96cd-4904-8f96-d7427e7138ea"))

    def test_preview_truncates(self) -> None:
        from mini_agent.gui import _preview

        self.assertEqual(_preview("a\nb\nc", 10), "a  (+2 行)")
        self.assertEqual(_preview("x" * 30, 10), "x" * 10 + "…")


if __name__ == "__main__":
    unittest.main()
