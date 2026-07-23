"""Configuration system."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 8788
    debug: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "pretty"  # pretty or json


@dataclass
class StatsConfig:
    """Statistics configuration."""
    enabled: bool = True
    file: str = ""


@dataclass
class ProxyConfig:
    """Main proxy configuration."""
    server: ServerConfig = field(default_factory=ServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    # Default provider
    default_provider: str = "nvidia"
    default_model: str = "deepseek-ai/deepseek-v4-pro"

    # Request settings
    max_body_size: int = 256 * 1024 * 1024  # 256MB
    keepalive_interval: float = 15.0
    request_timeout: float = 300.0

    @classmethod
    def from_env(cls) -> ProxyConfig:
        """Load configuration from environment variables."""
        return cls(
            server=ServerConfig(
                host=os.getenv("ATLAS_PROXY_HOST", "127.0.0.1"),
                port=int(os.getenv("ATLAS_PROXY_PORT", "8788")),
                debug=os.getenv("ATLAS_PROXY_DEBUG", "0") == "1",
            ),
            logging=LoggingConfig(
                level=os.getenv("ATLAS_LOG_LEVEL", "INFO"),
                format=os.getenv("ATLAS_LOG_FORMAT", "pretty"),
            ),
            stats=StatsConfig(
                enabled=True,
                file=os.getenv("ATLAS_STATS_FILE", ""),
            ),
            default_provider=os.getenv("ATLAS_DEFAULT_PROVIDER", "nvidia"),
            default_model=os.getenv("ATLAS_NVIDIA_MODEL", "deepseek-ai/deepseek-v4-pro"),
            max_body_size=int(os.getenv("ATLAS_MAX_BODY_SIZE", str(256 * 1024 * 1024))),
            keepalive_interval=float(os.getenv("ATLAS_KEEPALIVE_INTERVAL", "15.0")),
            request_timeout=float(os.getenv("ATLAS_REQUEST_TIMEOUT", "300.0")),
        )


# Global config
_config: Optional[ProxyConfig] = None


def get_config() -> ProxyConfig:
    """Get the global configuration."""
    global _config
    if _config is None:
        _config = ProxyConfig.from_env()
    return _config
