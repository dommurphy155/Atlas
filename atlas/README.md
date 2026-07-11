# Atlas

A minimal NVIDIA-only OpenAI/Anthropic-compatible API proxy.

## What It Does

- Accepts OpenAI-compatible requests at `/v1/chat/completions`
- Accepts Anthropic-compatible requests at `/v1/messages`
- Routes all requests directly to NVIDIA's API
- Uses NVIDIA `nvapi-` keys from `data/keys.txt`
- Applies the system prompt override from `data/system_prompt_override.txt`
- Exposes `/health` and `/stats`
- Runs as a standalone systemd service on `127.0.0.1:8788`
- Single provider, no fallback, no HF

## Repository Structure

```
atlas/
├── README.md
├── bin/
│   └── atlas
├── proxy/
│   ├── __init__.py
│   ├── atlas_proxy.py
│   ├── nvidia_client.py
│   ├── nvidia_key_store.py
│   ├── openai_compat.py
│   ├── stats.py
│   └── system_prompt.py
├── data/
│   ├── keys.txt
│   ├── proxy_stats.json
│   └── system_prompt_override.txt
├── setup/
│   ├── install.sh
│   ├── installer.py
│   └── requirements.txt
└── systemd/
    └── atlas-proxy.service
```

There is also `proxy/nvidia_key_store.py` — the key store that backs `data/keys.txt`.

## Install

```bash
cd ~/claude/atlas && bash setup/install.sh
atlas start
```

The installer creates a venv, installs deps, installs the systemd unit, and symlinks the CLI.

## CLI

| Command | Description |
| --- | --- |
| `atlas start` | Start the systemd service |
| `atlas stop` | Stop the systemd service |
| `atlas restart` | Restart the systemd service |
| `atlas status` | Show service status |
| `atlas logs` | Tail service logs |

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check |
| GET | `/stats` | Proxy request/token stats |
| GET | `/v1/models` | List available NVIDIA models |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions |
| POST | `/v1/messages` | Anthropic-compatible messages |

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `ATLAS_PROXY_HOST` | `127.0.0.1` | Bind host |
| `ATLAS_PROXY_PORT` | `8788` | Bind port |
| `ATLAS_KEYS_FILE` | `data/keys.txt` | Path to NVIDIA keys file |
| `ATLAS_NVIDIA_BASE_URL` | NVIDIA API base URL | Upstream NVIDIA endpoint |
| `ATLAS_NVIDIA_MODEL` | — | Default NVIDIA model |
| `ATLAS_PROXY_RELOAD_SECONDS` | — | Key file reload interval |
| `ATLAS_PROXY_REQUEST_TIMEOUT` | — | Per-request timeout |
| `ATLAS_PROXY_MAX_RETRIES` | — | Max retries per request |
| `ATLAS_PROXY_MAX_KEY_FAILOVERS` | — | Max key failovers per request |
| `ATLAS_PROXY_DEBUG` | — | Debug logging toggle |

`.env` is auto-created by the installer and not overwritten.

## NVIDIA Keys

`data/keys.txt` holds one `nvapi-` key per line. The file is hot-reloaded on change — edit it live, no restart needed. Keys that fail upstream are briefly cooled down before being returned to rotation.

## System Prompt Override

`data/system_prompt_override.txt`, if present, replaces the system message on every request and is prepended to the first user message. `<system-reminder>` blocks are stripped before forwarding.

## Running Side-by-Side

Atlas runs on `8788` and is fully independent. It can run alongside any other proxy on a different port with no shared state — separate venv, data directory, systemd unit, and env prefix.

## Uninstall

```bash
atlas stop
sudo systemctl disable atlas-proxy.service
sudo rm /etc/systemd/system/atlas-proxy.service
rm /usr/local/bin/atlas
# optionally
rm -rf ~/claude/atlas
```
