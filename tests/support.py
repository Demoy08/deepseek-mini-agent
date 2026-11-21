"""测试辅助：把临时目录放进仓库内。

不用 ``tempfile.TemporaryDirectory``：在受限的 Windows 沙箱里
它用 mode=0o700 创建目录，会得到一个连自己都写不进去的 ACL。
这里手动建一个普通目录，清理失败也不影响测试结论。
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parent / "_work"


class WorkspaceTempDir:
    def __init__(self, prefix: str = "case-") -> None:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        self.name = str(WORK_ROOT / f"{prefix}{uuid.uuid4().hex[:10]}")
        Path(self.name).mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)


def temporary_directory() -> WorkspaceTempDir:
    return WorkspaceTempDir()
