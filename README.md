<p align="center">
  <img src="assets/git_banner.svg" alt="Atlas" width="100%">
</p>

<h1 align="center">Atlas</h1>

<p align="center">
  <strong>Run Claude Code for free on NVIDIA API keys.</strong><br>
  A drop-in OpenAI/Anthropic-compatible proxy that routes every request straight to NVIDIA's integrate API — no Anthropic billing, no middleman.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-what-it-does">What It Does</a> ·
  <a href="#-wire-up-claude-code">Claude Code</a> ·
  <a href="#-cli">CLI</a> ·
  <a href="#-endpoints">Endpoints</a> ·
  <a href="#-configuration">Config</a> ·
  <a href="#-nvidia-keys">Keys</a> ·
  <a href="#-system-prompt-override">Override</a> ·
  <a href="#-architecture">Architecture</a> ·
  <a href="#-uninstall">Uninstall</a>
</p>

---

## ✨ What it does

Atlas is a single-provider NVIDIA proxy. It speaks both **OpenAI** and **Anthropic** wire formats, so anything that targets either SDK — including Claude Code — can point at it and get answers back from an NVIDIA-hosted model. No fallback, no provider switching, no HuggingFace.

- **OpenAI-compatible** at `/v1/chat/completions` (stream + non-stream)
- **Anthropic-compatible** at `/v1/messages` (stream + non-stream, real-time OpenAI→Anthropic SSE translation)
- Routes everything to NVIDIA's `integrate.api.nvidia.com` endpoint
- Rotates `nvapi-` keys from `data/keys.txt` with automatic cooldown + failover
- Hot-reloads the key file — edit it live, no restart
- Injects a **system prompt override** and strips `<system-reminder>` harness blocks before forwarding
- SSE keepalive comments so reasoning models don't trip middlebox idle timers while they think
- `/health`, `/stats`, and a `atlas tokens` usage dashboard
- Runs as a standalone systemd service on `127.0.0.1:8788`

## 🚀 Quick start

```bash
cd ~/claude/atlas && bash setup/install.sh   # venv + deps + systemd unit + CLI + .env
atlas start                                   # fire it up
atlas status                                  # service + /health + /stats
```

Drop your NVIDIA keys in `data/keys.txt` (one `nvapi-…` per line) and you're live. The installer also wires `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` into your shell rc so Claude Code picks it up automatically — see [Wire up Claude Code](#-wire-up-claude-code).

## 🔌 Wire up Claude Code

The installer appends this to `~/.bashrc` / `~/.zshrc` (skipped if already present):

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8788
export ANTHROPIC_API_KEY=atlas
```

Source your rc (or open a new shell), launch `claude`, and it now talks to the NVIDIA-backed model through Atlas instead of the Anthropic API. The key value (`atlas`) is a placeholder — the proxy ignores it and uses your `nvapi-` keys upstream.

## 🧱 Repository structure

```
atlas/
├── bin/
│   └── atlas                 # operator CLI (start/stop/status/logs/tokens)
├── proxy/
│   ├── atlas_proxy.py        # FastAPI app — endpoints, streaming, failover loop
│   ├── nvidia_client.py      # httpx client + SSE streaming + prewarm
│   ├── nvidia_key_store.py   # hot-reload, rotation, cooldown
│   ├── openai_compat.py      # OpenAI↔Anthropic message + SSE translation
│   ├── system_prompt.py      # override injection + <system-reminder> stripping
│   ├── stats.py              # request/token counters (all-time + since-restart)
│   └── token_tracker.py      # `atlas tokens` dashboard renderer
├── data/
│   ├── keys.txt              # one nvapi- key per line
│   ├── system_prompt_override.txt
│   └── proxy_stats.json      # written after every request
├── setup/
│   ├── install.sh            # bootstrap: venv + pip, hands off to installer.py
│   ├── installer.py          # systemd unit, CLI symlink, .env, shell wiring
│   └── requirements.txt
└── systemd/
    └── atlas-proxy.service
```

## 🖥️ CLI

| Command | Description |
| --- | --- |
| `atlas start` | Start the systemd service |
| `atlas stop` | Stop the systemd service |
| `atlas restart` | Restart the systemd service |
| `atlas status` | Service status + probe `/health` and `/stats` |
| `atlas logs` | Follow the journal (Ctrl-C to exit) — pass extra flags for one-shot: `atlas logs -n 100`, `atlas logs --since '10 min ago'`, `atlas logs -o json \| jq .` |
| `atlas tokens` | Clean since-restart token/usage summary (requests, success rate, in/out/total tokens, tool calls, per-model breakdown) |

## 📡 Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check — service, model, available keys |
| GET | `/stats` | Proxy request/token stats + key-store stats |
| GET | `/v1/models` | List the backing NVIDIA model |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/messages` | Anthropic-compatible messages |

## ⚙️ Configuration

Env vars (all `ATLAS_`-prefixed, set in `.env` or the systemd unit):

| Variable | Default | Description |
| --- | --- | --- |
| `ATLAS_PROXY_HOST` | `127.0.0.1` | Bind host |
| `ATLAS_PROXY_PORT` | `8788` | Bind port |
| `ATLAS_KEYS_FILE` | `data/keys.txt` | Path to NVIDIA keys file |
| `ATLAS_NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1/chat/completions` | Upstream NVIDIA endpoint |
| `ATLAS_NVIDIA_MODEL` | `z-ai/glm-5.2` | Default NVIDIA model |
| `ATLAS_PROXY_RELOAD_SECONDS` | `5` | Key file reload interval |
| `ATLAS_PROXY_REQUEST_TIMEOUT` | `300` | Per-request timeout (s) |
| `ATLAS_PROXY_CONNECT_TIMEOUT` | `10` | Connect timeout (s) |
| `ATLAS_PROXY_READ_TIMEOUT` | `180` | Stream read deadline — the dead-stream backstop, not the thinking-gap limit |
| `ATLAS_PROXY_KEEPALIVE_SECONDS` | `15` | SSE keepalive comment cadence during upstream silence |
| `ATLAS_PROXY_MAX_RETRIES` | `2` | Max same-key retries on transient 5xx |
| `ATLAS_PROXY_MAX_KEY_FAILOVERS` | `3` | Max key failovers per request |
| `ATLAS_PROXY_DEBUG` | `0` | Debug logging toggle |
| `ATLAS_PROXY_LOG_FORMAT` | `pretty` | `pretty` (colored one-liner) or `json` (one object per record for jq/aggregators) |

`.env` is auto-created by the installer and never overwritten on subsequent runs.

## 🔑 NVIDIA keys

`data/keys.txt` holds one `nvapi-` key per line. The file is hot-reloaded every `ATLAS_PROXY_RELOAD_SECONDS` — edit it live, no restart. Keys that fail upstream (401/403/402/429, transport errors, mid-stream timeouts) are briefly cooled down before returning to rotation, and the proxy transparently fails the request over to the next key up to `ATLAS_PROXY_MAX_KEY_FAILOVERS`.

## 🧬 System prompt override

`data/system_prompt_override.txt`, if present, is forced into every request:

- `<system-reminder>` blocks are stripped from **all** messages before forwarding
- Any existing `system` message is **replaced** with the override
- If there's no `system` message, the override is **inserted** at the start
- The override is also **prepended to the first user message** for double primacy

This is the mechanism that lets you run the backing model with a fixed persona/instruction set regardless of what the client sends. Edit the file, save — the next request picks it up (it's read fresh per request, not cached).

## 🏗️ Architecture

```
client (Claude Code / OpenAI SDK / Anthropic SDK)
  │  OpenAI or Anthropic wire format
  ▼
Atlas (FastAPI / uvicorn, 127.0.0.1:8788)
  ├── parse + normalize messages
  ├── system_prompt.replace_system_prompt()   ← inject override, strip <system-reminder>
  ├── NvidiaKeyStore.acquire()                ← rotate, skip cooled keys
  ├── NvidiaClient.chat() / stream_chat()      ← httpx → NVIDIA integrate API
  ├── keepalive() SSE comments during silence  ← keeps middleboxes alive
  ├── openai_sse_to_anthropic_sse()           ← /v1/messages stream translation
  └── stats.record_success/failure()          ← persisted to proxy_stats.json
        │
        ▼
NVIDIA integrate API (z-ai/glm-5.2 by default)
```

Streaming is real translation, not buffer-then-fake: OpenAI SSE chunks from NVIDIA are converted to Anthropic SSE events on the fly for `/v1/messages`, and usage is accumulated across chunk boundaries so token counts stay accurate even when the upstream splits them.

**Resilience loop** (both stream and non-stream): acquire key → call NVIDIA → on 401/403/402/429 cool the key and fail over to the next → on 500/502/503/504 retry on the same pool up to `MAX_RETRIES` → on mid-stream timeout cool the key and record the failure. Bounded by `MAX_KEY_FAILOVERS` so a bad pool can't loop forever.

## 🧭 Running side-by-side

Atlas runs on `8788` and is fully independent — separate venv, data directory, systemd unit, and `ATLAS_` env prefix. It can run alongside any other proxy on a different port with zero shared state.

## 🗑️ Uninstall

```bash
atlas stop
sudo systemctl disable atlas-proxy.service
sudo rm /etc/systemd/system/atlas-proxy.service
sudo systemctl daemon-reload
rm /usr/local/bin/atlas
# optionally
rm -rf ~/claude/atlas
```

Then remove the Atlas block from `~/.bashrc` / `~/.zshrc` (the `# --- Atlas NVIDIA proxy ---` … `# --- end Atlas ---` lines) if you let the installer wire it.
