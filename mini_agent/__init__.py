"""deepseek-mini-agent —— 一个由 agent loop 驱动的极简命令行 agent。

对外只暴露四样东西：配置、模型客户端、工具注册表、循环本体。
"""

from .config import ConfigError, Settings
from .llm import ChatResult, DeepSeekClient, LLMError, StreamAssembler
from .loop import Agent, StepLimitExceeded
from .tools import Tool, ToolRegistry, default_tools

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "ChatResult",
    "ConfigError",
    "DeepSeekClient",
    "LLMError",
    "Settings",
    "StepLimitExceeded",
    "StreamAssembler",
    "Tool",
    "ToolRegistry",
    "default_tools",
    "__version__",
]
