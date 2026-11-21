"""配置层：从环境变量和 .env 文件读取运行参数。

刻意保持零第三方依赖 —— 一个 20 行的 .env 解析器足够本项目使用，
也方便读者 clone 下来直接跑。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://api.deepseek.com"
# 注意：API 的模型 id 是 deepseek-flash，DeepSeek-V4.1-Flash 只是展示名。
DEFAULT_MODEL = "deepseek-flash"
THINKING_MODES = ("auto", "enabled", "disabled")


class ConfigError(RuntimeError):
    """配置缺失或取值非法。"""


def load_dotenv(path: str | Path = ".env") -> None:
    """读取极简 .env 文件。

    支持 ``KEY=VALUE``、``export KEY=VALUE``、单/双引号包裹的值、
    行内 ``#`` 注释以及空行。已存在的环境变量优先，不会被覆盖。
    """
    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue

        value = value.strip()
        if value[:1] in {"'", '"'}:
            # 引号包裹的值：允许 `KEY="a b"  # 注释` 这种写法
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            # 未加引号时，允许写法：KEY=value  # 注释
            value = value.split(" #", 1)[0].rstrip()

        os.environ.setdefault(key, value)


def write_env(path: str | Path, values: dict[str, str]) -> None:
    """把若干键值写回 .env，保留原有注释与其它行。

    图形界面的设置弹窗只更新传入的键，其余内容原样保留。
    """
    env_path = Path(path)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    remaining = dict(values)

    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            if key in remaining:
                output.append(f"{key}={remaining.pop(key)}")
                continue
        output.append(line)

    for key, value in remaining.items():
        output.append(f"{key}={value}")

    env_path.write_text("\n".join(output).rstrip("\n") + "\n", encoding="utf-8")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是数字，当前为 {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {name} 必须是整数，当前为 {raw!r}") from exc


@dataclass
class Settings:
    """一次运行所需的全部配置。"""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_steps: int = 20
    temperature: float = 0.7
    timeout: float = 120.0
    stream: bool = True
    thinking: str = "disabled"
    workspace: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.workspace = Path(self.workspace).expanduser().resolve()
        if self.max_steps < 1:
            raise ConfigError("max_steps 至少为 1")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("temperature 取值应在 0.0 ~ 2.0 之间")
        if self.thinking not in THINKING_MODES:
            raise ConfigError(f"thinking 只能是 {', '.join(THINKING_MODES)}，当前为 {self.thinking!r}")
        if not self.api_key:
            raise ConfigError("缺少 API key")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    @classmethod
    def from_env(cls, env_file: str | Path = ".env", **overrides: object) -> "Settings":
        """从环境变量构造配置；``overrides`` 用于命令行参数覆盖。"""
        load_dotenv(env_file)

        api_key = str(overrides.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        if not api_key:
            raise ConfigError(
                "没有找到 API key。请复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY，"
                "或设置环境变量 DEEPSEEK_API_KEY。"
            )

        workspace = overrides.get("workspace") or os.environ.get("MINI_AGENT_WORKSPACE") or Path.cwd()
        max_steps = overrides.get("max_steps")
        temperature = overrides.get("temperature")
        timeout = overrides.get("timeout")
        stream = overrides.get("stream")

        return cls(
            api_key=api_key,
            model=str(overrides.get("model") or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL),
            base_url=str(overrides.get("base_url") or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL),
            max_steps=int(max_steps) if max_steps is not None else _env_int("MINI_AGENT_MAX_STEPS", 20),
            temperature=float(temperature) if temperature is not None else _env_float("MINI_AGENT_TEMPERATURE", 0.7),
            timeout=float(timeout) if timeout is not None else _env_float("MINI_AGENT_TIMEOUT", 120.0),
            stream=bool(stream) if stream is not None else os.environ.get("MINI_AGENT_STREAM", "1") != "0",
            thinking=str(overrides.get("thinking") or os.environ.get("MINI_AGENT_THINKING") or "disabled").lower(),
            workspace=Path(workspace),
        )
