"""Windows 双击启动图形界面。

.pyw 文件由 pythonw 打开，不会弹出黑色控制台窗口。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # 保证能读到同目录的 .env
sys.path.insert(0, str(ROOT))

from mini_agent.gui import run_gui

raise SystemExit(run_gui())
