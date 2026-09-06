# Atlas

**One proxy. Every harness. Zero API bills.**

Atlas is a self-hosted OpenAI/Anthropic-compatible reverse proxy that lets
every coding harness on your machine share a single pool of free-tier
provider keys. Run Claude Code, Codex, Hermes, Pi — point them all at
`http://127.0.0.1:8788` and Atlas round-robins across OpenRouter, Hugging
Face inference providers, and NVIDIA NIM, replacing the model each harness
asks for with one you actually have keys for.

If you have a working key on the upstream, Atlas just works. If you don't,
Atlas's interactive model picker tells you which models are actually free
right now and switches the proxy to one of them in a single command.

---

**Quick links:** [Install](#installation) · [Providers](#supported-providers) · [Harnesses](#supported-harnesses) · [Configuration](#configuration) · [`atlas switch`](#switching-providers-and-models) · [Troubleshooting](#troubleshooting) · [Architecture](#how-the-architecture-works) · [Security](#security)

---

## What Atlas does

- **Provider aggregation.** A single OpenAI-compatible endpoint
  (`/v1/chat/completions`, `/v1/responses`, `/v1/models`, etc.) fronts
  OpenRouter and Hugging Face. NVIDIA is exposed as a first-class provider
  in the picker.
- **Model rewriting.** Whatever model the harness asks for, Atlas can
  substitute a configured default — useful when the harness's `model`
  field is a paid model and you only have free-tier keys.
- **Key pooling with dead-key quarantine.** Each provider has a key file
  (`openroute_keys.txt`, `hf_keys.txt`). Atlas round-robins across the
  pool, retires dead keys to `dead_hf_keys.txt`, and keeps traffic
  flowing.
- **System prompt override.** A built-in additive prompt injection layer
  (`proxy/system_prompt.py`) — loaded from
  `data/proxy_data/prompt_override.txt` — runs *after* the harness's
  native system messages so existing tool/date/context stays intact.
- **Harness auto-config.** `atlas install` walks you through wiring up
  Claude Code, Codex, Hermes, or Pi to point at the proxy and verify the
  end-to-end flow with a real smoke test.

## Supported providers

| Provider         | Key source                              | Auth required for catalogue |
|------------------|-----------------------------------------|-----------------------------|
| OpenRouter       | `data/openrouter_data/openroute_keys.txt` | No (public `/api/v1/models`) |
| Hugging Face     | `data/huggingface_data/hf_keys.txt`       | No (public `/api/models`)    |
| NVIDIA NIM       | (read from `proxy/config.py`)             | No (public catalogue)        |

Free-tier filtering on OpenRouter verifies both `prompt` and `completion`
prices are exactly $0 — not just `:free` in the slug.

## Supported harnesses

`atlas install` can configure any of:

- **Claude Code** — `~/.claude/settings.json` `env` block.
- **Codex** — `~/.codex/config.toml` + `OPENAI_API_KEY` export.
- **Hermes** — `hermes config set` for `model.base_url` / `model.api_key`.
- **Pi** — `~/.pi/agent/models.json` custom provider entry.

## How the architecture works

```
  ┌────────────────────┐         ┌──────────────────┐         ┌─────────────────────┐
  │  Claude Code       │         │                  │         │  OpenRouter         │
  │  Codex             │──HTTP──▶│  Atlas proxy     │──HTTP──▶│  Hugging Face       │
  │  Hermes            │         │  :8788           │         │  NVIDIA NIM         │
  │  Pi                │         │                  │         │                     │
  └────────────────────┘         └──────────────────┘         └─────────────────────┘
                                       │
                                       │  system prompt
                                       │  override (additive)
                                       ▼
                                data/proxy_data/
                                ├─ runtime_provider.json   (current provider/model)
                                └─ prompt_override.txt     (system prompt file)
```

The proxy is plain FastAPI. The CLI (`atlas`) is plain Python with `rich`.

## Key features

- **Interactive model switcher.** `atlas switch` shows live, ranked,
  free-filtered models from any provider with a single-prompt picker —
  no menus, no arrow keys, works in any TTY including piped stdin.
- **Runtime config.** `data/proxy_data/runtime_provider.json` is the
  single source of truth for "which provider / which model". Edit it by
  hand, by `atlas switch`, or by `atlas restart --huggingface foo/bar`.
- **Append-only key import.** `atlas import-key` and
  `atlas import-key-file` add new keys to the existing key file,
  dedupe, and report existing/imported/duplicates/added/total — never
  overwrite, even if the file is already huge.
- **Runtime portability.** Cross-platform runtime layer
  (`atlas/bin/runtime.py`) auto-detects systemd / systemd-user / tmux /
  nohup / manual. Works on Linux, macOS, and (via tmux-equivalents) on
  Windows.
- **Public provider catalogues.** All three providers serve their model
  lists without auth. No API keys needed just to browse models.

## Requirements

- **Python 3.11+** (the proxy itself; the CLI works on 3.12+)
- `pip install -r requirements.txt` (FastAPI, uvicorn, httpx, orjson,
  rich, pydantic)
- A working OpenRouter or Hugging Face API key (for actually routing
  traffic). The model picker works without one.

## Installation

Atlas is a Python project with a bash installer. It runs natively on Linux
and macOS, and on Windows via the cross-platform runtime layer
(`atlas/bin/runtime.py`) which auto-detects the best fallback (tmux-equivalent
→ nohup-equivalent → manual).

### Prerequisites

- **Python 3.11+**
- **Git**
- **Linux:** systemd (system or user mode) **or** tmux
- **macOS:** tmux (`brew install tmux`) or just leave it — the runtime
  layer picks the right fallback automatically
- **Windows:** one of `tmux`, `psmux`, `tmuxw`, `lumux`, `wmux`, or
  `qscreen` on `PATH` (for detached operation). Foreground works
  without any of these.

### One-line install — Linux / macOS

```bash
git clone https://github.com/dommurphy155/Atlas.git
cd Atlas
./setup/install.sh
```

Non-interactive equivalent (user-scope systemd, no harness setup):

```bash
./setup/install.sh --user
```

### One-line install — Windows (PowerShell)

```powershell
git clone https://github.com/dommurphy155/Atlas.git
cd Atlas
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m proxy.main
```

The Windows path skips the bash installer and the systemd unit; the proxy
runs as a foreground process (or under a tmux-equivalent for detachment).

### What the installer does

1. Creates `.venv/` and installs `requirements.txt`.
2. Writes a systemd unit (system scope when root, `--user` otherwise) and
   enables it.
3. Symlinks `atlas` into your `PATH`:
   - `/usr/local/bin/atlas2` (system scope, root)
   - `~/.local/bin/atlas2` (user scope, with a hint if `~/.local/bin`
     is not on `PATH`)
4. Starts the proxy and prints a smoke-test summary.

### After install

```bash
atlas2 status       # verify the proxy is up
atlas2 doctor       # diagnose common issues
atlas2 import-key sk-or-v1-...   # add an OpenRouter key
atlas2 switch       # interactive model picker
```

## Configuration

The proxy reads environment variables and `data/proxy_data/*.json`. The
template is `.env.example` — copy to `.env` and edit. All values are
optional; defaults are sensible.

| Variable                     | Default                       | Purpose                                   |
|------------------------------|-------------------------------|-------------------------------------------|
| `LISTEN_HOST`                | `0.0.0.0`                     | Bind address                              |
| `LISTEN_PORT`                | `8788`                        | Bind port                                 |
| `ATLAS_PROVIDER`             | `openrouter`                  | Fallback if no runtime file               |
| `ATLAS_OPENROUTER_MODEL`     | (set in runtime file)         | Default model for OpenRouter              |
| `ATLAS_HF_MODEL`             | (set in runtime file)         | Default model for HF                      |
| `FORCE_DEFAULT_MODEL`        | `1`                           | Replace client-requested model with default |
| `ATLAS_OPENROUTER_KEYS_FILE` | `data/openrouter_data/openroute_keys.txt` | Path to OR keys            |
| `ATLAS_HF_KEYS_FILE`         | `data/huggingface_data/hf_keys.txt`     | Path to HF keys            |
| `SYSTEM_PROMPT_OVERRIDE_FILE`| `data/proxy_data/prompt_override.txt`   | System prompt source        |
| `SYSTEM_PROMPT_OVERRIDE_ENABLED` | `1`                       | Toggle the override                        |

## API key setup

```bash
# Interactive, appends + dedupes:
atlas import-key sk-or-v1-...
atlas import-key-file /path/to/keys.txt
```

The CLI resolves the actual key file path via `proxy.config.KEY_FILE` —
no hardcoded paths. Existing keys are preserved; new unique keys are
appended.

## Running Atlas

```bash
atlas start                # start the proxy (systemd / run.sh fallback)
atlas status               # show runtime info, mode, service state, /health
atlas logs                 # tail proxy logs
atlas doctor               # diagnose the install
atlas stop                 # stop the proxy
atlas restart              # restart (no provider change)
```

The CLI displays as `atlas` everywhere — `atlas` is the local command
name on a box where the upstream `atlas` is also installed. The two
do not collide.

## Switching providers and models

The interactive `atlas switch` is the easiest path:

```bash
atlas switch
```

```
Atlas Model Switch

  1. OpenRouter — 400+ models, free tier, agentic leaderboard
  2. NVIDIA — NIM inference endpoints, no key required
  3. Hugging Face — Inference Providers — frontier models, free tier

Choose a provider (1): 1
16 models for OpenRouter

OpenRouter — 16 models total
    #  ID                                  Tasks            Ctx  Price
    1  minimax/minimax-m3:free            agentic,coding     1M  FREE
    2  nvidia/nemotron-3-ultra-...:free    reasoning          1M  FREE
    ...
n/p next/prev • /term search • f<N> ★ row N • r refresh • q back • <number> pick
>:
```

**Keyboard controls** (single prompt — works in any TTY, including piped stdin):

| Input        | Action                                       |
|--------------|----------------------------------------------|
| `<number>`   | Pick the model on that row of the current page |
| (empty) Enter | Pick the first model on the current page    |
| `n` / `p`    | Next / previous page                         |
| `/<term>`    | Search (substring on id, name, or task)      |
| `f<N>`       | Toggle ★ favourite on row N                  |
| `r`          | Bust the cache and refetch                   |
| `q`          | Back to the provider menu                    |

After you pick a model, Atlas saves the selection to
`data/proxy_data/runtime_provider.json` and offers to restart the proxy
in place.

The script-driven equivalent (no UI) is:

```bash
atlas restart --openrouter minimax/minimax-m3:free
atlas restart --huggingface deepseek-ai/DeepSeek-V4-Flash:deepinfra
atlas restart --skip                        # just restart, no change
```

## Examples

```bash
# Browse and switch interactively
atlas switch

# Set model directly
atlas restart --openrouter minimax/minimax-m3:free

# Toggle provider
atlas restart --huggingface deepseek-ai/DeepSeek-V4-Flash:deepinfra

# Just bounce the proxy
atlas restart

# Add more keys (append-only, dedupe)
echo "sk-or-v1-..." | atlas import-key
atlas import-key-file ~/Downloads/more-keys.txt

# Inspect a running install
atlas status
atlas doctor
atlas logs
```

## Runtime configuration

`data/proxy_data/runtime_provider.json` is the source of truth for
"which provider / which model" at boot. It's a tiny file:

```json
{
  "provider": "openrouter",
  "model": "minimax/minimax-m3:free",
  "selected_at": 1788728066
}
```

Edit it by hand, by `atlas switch`, or by the CLI's `restart --<provider>
<model>`. The proxy reads it on startup, refuses to load it if it's
world-writable (security), and validates that the provider name is in
the known set.

## Troubleshooting

**`atlas status` shows inactive.** Check `atlas logs`. The most common
cause is a missing key file or an upstream 401 on first request.

**Models show up as `$0.01/...` in the picker but I asked for free only.**
The filter verifies both `prompt` and `completion` are exactly $0. If
either has a non-zero price (or is `null`), the model is excluded. That
filters out models where one direction is paid (e.g. paid input, free
output) — by design.

**`No models available for OpenRouter`.** The cached `openrouter_models.json`
may have hit an upstream error. Run `atlas switch` and choose `r` to
bust the cache and refetch, or delete `.cache/openrouter_models.json`
manually.

**Two `atlas` repos on the same box?** `atlas` is the local command for
this fork. `SERVICE_NAME` defaults to `atlas-proxy-fork.service`
(deliberately distinct from upstream's `atlas-proxy.service`). Override
with `ATLAS_SERVICE_NAME=atlas-proxy.service` if you really want the
upstream name.

**Picker loop is "stuck" on `Fetching ...`.** The transient spinner
clears on its own as soon as the prompt is ready. If the prompt never
shows, check `pip install textual` — no, actually you don't need
textual; the picker is plain rich. If it's truly stuck, the upstream
catalogue endpoint is timing out. `Ctrl-C` exits.

## Development

```bash
# Install dev deps
.venv/bin/pip install -r requirements.txt pytest

# Run the test suite
.venv/bin/python -m pytest tests/ -q
```

The project is organised as:

```
atlas_proxy/
├── atlas/bin/                  # CLI
│   ├── atlas                   # Main entry point (display name: "atlas")
│   ├── runtime.py              # Cross-platform runtime detection
│   ├── setup_wizard.py         # Interactive harness setup
│   ├── models.py               # Provider discovery + ranking + caching
│   ├── switch.py               # Interactive model picker
│   └── __init__.py
├── proxy/                      # The reverse proxy itself (FastAPI)
│   ├── main.py                 # App factory
│   ├── routes.py               # /v1/chat/completions etc.
│   ├── providers.py            # OpenRouter / HF provider adapters
│   ├── config.py               # All env-overridable settings
│   ├── keypool.py              # Key round-robin + dead-key quarantine
│   ├── system_prompt.py        # Additive system-prompt override layer
│   ├── translation/            # Cross-provider message translation
│   └── ...
├── tests/                      # 157+ tests
├── setup/install.sh            # Bash installer (systemd unit, venv, symlink)
├── run.sh                      # Foreground / nohup fallback
├── requirements.txt
├── .env.example                # Documented config template
└── data/                       # Runtime data (gitignored)
    ├── openrouter_data/        # openroute_keys.txt
    ├── huggingface_data/       # hf_keys.txt + dead_hf_keys.txt
    ├── proxy_data/             # runtime_provider.json + prompt_override.txt
    └── payloads/               # Captured requests (gitignored)
```

The **system prompt** is at `proxy/system_prompt.py`. It implements
additive injection (prepends the override as the first system message
without stripping existing tool/date/context content). The override
text itself is loaded at runtime from
`data/proxy_data/prompt_override.txt` (gitignored) so you can swap
personas without touching code.

## Security

- API key files (`data/openrouter_data/openroute_keys.txt`,
  `data/huggingface_data/hf_keys.txt`) are gitignored and never
  committed. The gitignore also covers `.env` and any `*.txt` key
  file in `data/`.
- `runtime_provider.json` is gitignored. The proxy refuses to load it
  if it is world-writable (local user privilege escalation guard).
- `data/payloads/` is gitignored — captured request bodies can contain
  user data.
- `proxy/logs/atlas-proxy.log` is gitignored — proxy logs may include
  full request bodies.
- Body-size limits, header allowlists, and origin checks live in
  `proxy/main.py` and are exercised by
  `tests/test_security_hardening.py`.

If you have accidentally committed a secret, the recommended recovery is
to **rotate the key**, not rewrite history. `git filter-repo` against a
public remote is a leak, not a cleanup.

## Contributing / license

See `LICENSE` for the licence terms. Pull requests that touch the
public CLI surface (`atlas/bin/atlas`) should keep the displayed name
as `atlas` even when invoked as `atlas` — that's a hard rule, not a
convention.
