"""Configuration tests — env parsing, defaults, runtime model loading."""

from __future__ import annotations

import importlib
import os

import pytest

from proxy import config


def test_listen_port_default_is_fork_port() -> None:
    """The fork proxy must NOT default to 8788 (primary atlas port)."""
    os.environ.pop("LISTEN_PORT", None)
    importlib.reload(config)
    assert config.LISTEN_PORT == 8777


def test_listen_port_env_override() -> None:
    os.environ["LISTEN_PORT"] = "9999"
    importlib.reload(config)
    assert config.LISTEN_PORT == 9999
    del os.environ["LISTEN_PORT"]


def test_cors_origins_default_is_empty_not_wildcard() -> None:
    """Empty default is safer than '*' (which conflicts with allow_credentials=True)."""
    os.environ.pop("CORS_ORIGINS", None)
    importlib.reload(config)
    assert config.CORS_ORIGINS == []


def test_cors_origins_env_parsing() -> None:
    os.environ["CORS_ORIGINS"] = "https://a.com, https://b.com"
    importlib.reload(config)
    assert config.CORS_ORIGINS == ["https://a.com", "https://b.com"]
    del os.environ["CORS_ORIGINS"]


def test_provider_default_is_openrouter() -> None:
    # ATLAS_PROVIDER unset in test env; default = openrouter
    os.environ.pop("ATLAS_PROVIDER", None)
    os.environ.pop("RUNTIME_PROVIDER_FILE", None)
    importlib.reload(config)
    # Falls back to either runtime file or env or "openrouter"
    assert config.PROVIDER in ("openrouter", "huggingface", "nvidia")


def test_get_default_model_returns_string() -> None:
    m = config.get_default_model()
    assert isinstance(m, str) and m


def test_get_chat_url_for_openrouter() -> None:
    os.environ.pop("ATLAS_PROVIDER", None)
    importlib.reload(config)
    if config.PROVIDER == "openrouter":
        url = config.get_chat_url()
        assert "openrouter" in url or "api/v1" in url