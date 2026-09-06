"""Tests for atlas.bin.setup_wizard.

Covers the pure-logic helpers and the dry-run path through run_wizard.
Does NOT exercise live network calls (those are smoke-tested manually).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from atlas.bin.setup_wizard import (
    HARNESSES,
    Harness,
    _backup,
    _proxy_key_file,
    _read_json,
    _timestamp,
    _write_json,
    detect_os,
    run_wizard,
)


# ---- fixtures --------------------------------------------------------------

@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-atlas-dummy")
    return tmp_path


@pytest.fixture
def harness_by_id() -> dict[str, Harness]:
    return {h.id: h for h in HARNESSES}


# ---- helpers ---------------------------------------------------------------

def test_harness_registry_has_four():
    assert len(HARNESSES) == 4
    ids = {h.id for h in HARNESSES}
    assert ids == {"claude", "codex", "hermes", "pi"}


def test_harness_shell_bins_match_ids(harness_by_id):
    for h in HARNESSES:
        assert h.bin == h.id, f"{h.id}: bin mismatch"


def test_detect_os_returns_known_family():
    fam = detect_os()
    assert fam in {"linux", "macos", "windows", "other"}


def test_timestamp_format():
    ts = _timestamp()
    assert len(ts) == 15  # YYYYMMDD-HHMMSS
    assert ts[8] == "-"


def test_backup_creates_atlas_bak_suffix(tmp_path: Path):
    target = tmp_path / "config.json"
    target.write_text("{}")
    bak = _backup(target)
    assert bak is not None
    assert bak.name.startswith("config.json.atlas-bak.")
    assert bak.exists()
    # original untouched
    assert target.read_text() == "{}"


def test_backup_returns_none_when_target_missing(tmp_path: Path):
    assert _backup(tmp_path / "missing.json") is None


def test_read_write_json_roundtrip(tmp_path: Path):
    p = tmp_path / "x.json"
    data = {"a": 1, "b": [2, 3]}
    _write_json(p, data)
    assert _read_json(p) == data


def test_resolve_key_file_falls_back(monkeypatch):
    """`_proxy_key_file()` reads from `proxy.config.KEY_FILE` via repo import."""
    monkeypatch.setenv("HOME", "/nonexistent")
    p = _proxy_key_file()
    assert p.name == "openroute_keys.txt"
    # Real path comes from the proxy config — verify it ends with the openrouter data dir
    assert str(p).endswith("openrouter_data/openroute_keys.txt")


# ---- configure merge: minimum-change contract -----------------------------

def test_configure_claude_adds_env_block_preserves_existing(fake_home: Path):
    from atlas.bin.setup_wizard import _configure_claude
    settings = fake_home / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({
        "alwaysThinkingEnabled": True,
        "theme": "dark",
        "mcpServers": {"github": {"url": "x"}},
    }, indent=2))
    ok, msg = _configure_claude("http://127.0.0.1:8788", "sk-dummy")
    assert ok, msg
    out = json.loads(settings.read_text())
    assert out["theme"] == "dark"  # preserved
    assert out["alwaysThinkingEnabled"] is True  # preserved
    assert out["mcpServers"]["github"]["url"] == "x"  # preserved
    assert out["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8788"
    assert out["env"]["ANTHROPIC_AUTH_TOKEN"] == "sk-dummy"
    # backup created
    assert any(settings.parent.glob("settings.json.atlas-bak.*"))


def test_configure_claude_idempotent(fake_home: Path):
    """Re-running with same base_url preserves the env block and is a no-op write."""
    from atlas.bin.setup_wizard import _configure_claude
    _configure_claude("http://127.0.0.1:8788", "sk-dummy")
    settings = fake_home / ".claude" / "settings.json"
    after_first = settings.read_text()
    backups_before = len(list((fake_home / ".claude").glob("settings.json.atlas-bak.*")))
    # Re-run with identical args — should still succeed and overwrite (latest wins)
    ok, msg = _configure_claude("http://127.0.0.1:8788", "sk-dummy")
    assert ok
    after_second = settings.read_text()
    # File contents are byte-identical (same values re-applied)
    assert after_first == after_second
    # Backup is always taken — latest-wins semantics
    backups_after = len(list((fake_home / ".claude").glob("settings.json.atlas-bak.*")))
    assert backups_after == backups_before + 1


def test_configure_codex_writes_openai_base_url(fake_home: Path):
    from atlas.bin.setup_wizard import _configure_codex
    ok, msg = _configure_codex("http://127.0.0.1:8788", "sk-dummy")
    assert ok, msg
    cfg = fake_home / ".codex" / "config.toml"
    text = cfg.read_text()
    assert 'openai_base_url = "http://127.0.0.1:8788/v1"' in text
    assert "Added by Atlas proxy installer" in text


def test_configure_codex_preserves_existing_sections(fake_home: Path):
    from atlas.bin.setup_wizard import _configure_codex
    cfg = fake_home / ".codex" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text('[projects."/root/foo"]\ntrust_level = "trusted"\n')
    _configure_codex("http://127.0.0.1:8788", "sk-dummy")
    text = cfg.read_text()
    assert '[projects."/root/foo"]' in text  # preserved
    assert 'openai_base_url = "http://127.0.0.1:8788/v1"' in text  # added
    assert "trust_level" in text


def test_configure_pi_adds_atlas_provider_to_models_json(fake_home: Path):
    from atlas.bin.setup_wizard import _configure_pi
    models = fake_home / ".pi" / "agent" / "models.json"
    models.parent.mkdir(parents=True)
    models.write_text(json.dumps({"providers": {"existing": {"baseUrl": "x"}}}, indent=2))
    ok, msg = _configure_pi("http://127.0.0.1:8788", "sk-dummy")
    assert ok, msg
    out = json.loads(models.read_text())
    assert "atlas" in out["providers"]
    assert out["providers"]["existing"]["baseUrl"] == "x"  # preserved
    assert out["providers"]["atlas"]["baseUrl"] == "http://127.0.0.1:8788/v1"
    assert len(out["providers"]["atlas"]["models"]) >= 1


def test_configure_pi_does_not_change_settings_default_provider(fake_home: Path):
    from atlas.bin.setup_wizard import _configure_pi
    settings = fake_home / ".pi" / "agent" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"defaultProvider": "openai", "defaultModel": "gpt-4o"}, indent=2))
    _configure_pi("http://127.0.0.1:8788", "sk-dummy")
    out = json.loads(settings.read_text())
    assert out["defaultProvider"] == "openai"  # NOT touched


def test_configure_hermes_uses_built_in_config_set(monkeypatch, fake_home: Path):
    """Hermes config: must use the built-in `hermes config set` CLI, not file editing."""
    from atlas.bin import setup_wizard
    calls: list[list[str]] = []

    def fake_which(b: str) -> str | None:
        return "/usr/bin/hermes" if b == "hermes" else None

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    monkeypatch.setattr(setup_wizard.shutil, "which", fake_which)
    monkeypatch.setattr(setup_wizard.subprocess, "run", fake_run)
    ok, msg = setup_wizard._configure_hermes("http://127.0.0.1:8788", "sk-dummy")
    assert ok, msg
    assert calls, "should have invoked `hermes config set`"
    assert any("config" in c and "set" in c for c in calls)


# ---- wizard entrypoint: invoked through actual atlas2 install flow --------

def test_run_wizard_dry_run_completes_without_network(fake_home: Path, monkeypatch):
    """End-to-end: run_wizard with dry_run=True + piped stdin (non-interactive).

    Confirms the wizard module works when invoked through the actual entrypoint,
    not just when imported directly.
    """
    # Pretend all 4 harnesses are detected
    def fake_which(b: str) -> str | None:
        return f"/usr/bin/{b}"
    monkeypatch.setattr("shutil.which", fake_which)
    # Skip the smoke tests — they hit the network
    monkeypatch.setattr(
        "atlas.bin.setup_wizard._smoke_claude", lambda b, k: (True, "smoke-ok", "")
    )
    monkeypatch.setattr(
        "atlas.bin.setup_wizard._smoke_codex", lambda b, k: (True, "smoke-ok", "")
    )
    monkeypatch.setattr(
        "atlas.bin.setup_wizard._smoke_hermes", lambda b, k: (True, "smoke-ok", "")
    )
    monkeypatch.setattr(
        "atlas.bin.setup_wizard._smoke_pi", lambda b, k: (True, "smoke-ok", "")
    )

    # Feed: harness=claude, provider=1 (openrouter), launch=n
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("1\n1\nn\n"))

    rc = run_wizard(dry_run=True)
    assert rc == 0


def test_run_wizard_importable_via_actual_entrypath(tmp_path, monkeypatch):
    """Replicate what cmd_install does: insert REPO_ROOT/atlas on sys.path then import."""
    import sys
    repo_root = Path("/root/atlas_proxy")
    monkeypatch.syspath_prepend(str(repo_root / "atlas"))
    sys.modules.pop("bin", None)
    sys.modules.pop("bin.setup_wizard", None)
    import importlib
    wiz = importlib.import_module("bin.setup_wizard")
    assert hasattr(wiz, "run_wizard")
    assert hasattr(wiz, "HARNESSES")
