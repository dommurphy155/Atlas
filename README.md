# Atlas Proxy

OpenAI/Anthropic-compatible LLM proxy with provider failover, key rotation,
and OpenRouter ⇄ HuggingFace switching. Sits between Claude Code / Hermes
and upstream APIs, transparently retrying on dead keys and rotating models.

## Install

```bash
./setup/install.sh            # full: venv + systemd unit (requires root)
./setup/install.sh --user     # venv only, run with run.sh
./setup/install.sh --check    # verify a healthy install
./setup/install.sh --uninstall
```

The installer is repo-location agnostic. Put this anywhere (`~/atlas`,
`~/proxy`, `/opt/atlas_proxy`) and it works.

## Run

```bash
./run.sh                # foreground
./run.sh --bg           # background, log to data/run.log
./run.sh --status
./run.sh --stop
LISTEN_PORT=9000 ./run.sh
```

## Configure

```bash
cp .env.example .env
$EDITOR .env
```

Key vars:

| var | default | meaning |
|---|---|---|
| `LISTEN_HOST` / `LISTEN_PORT` | `0.0.0.0` / `8788` | bind address |
| `ATLAS_PROVIDER` | `openrouter` | `openrouter` or `huggingface` |
| `ATLAS_OPENROUTER_MODEL` | `z-ai/glm-5.2:free` | default OR model |
| `ATLAS_HF_MODEL` | `deepseek-ai/DeepSeek-V4-Flash:deepinfra` | default HF model |
| `FORCE_DEFAULT_MODEL` | `1` | override client-sent model |
| `ATLAS_OPENROUTER_KEYS_FILE` | `data/openrouter_data/openroute_keys.txt` | OR keys, one per line |
| `ATLAS_HF_KEYS_FILE` | `data/huggingface_data/hf_keys.txt` | HF keys, one per line |
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

Full env-var reference: see `.env.example`.

## Switch providers

The `atlas` CLI flips provider without restarting anything else:

```bash
atlas restart --huggingface   # writes data/proxy_data/runtime_provider.json
atlas restart                 # back to OpenRouter
```

The proxy hot-picks the new provider on the next reload (or on full restart).

## Endpoints

- `POST /v1/chat/completions` — OpenAI chat
- `POST /v1/messages`         — Anthropic messages
- `POST /v1/responses`        — OpenAI Responses (mapped → chat/completions)
- `GET  /v1/models`           — model list (returns active default)
- `GET  /healthz`             — liveness

## Layout

```
atlas_proxy/
├── proxy/                # the package
│   ├── main.py             # FastAPI app + uvicorn entry
│   ├── routes.py           # HTTP routes
│   ├── proxy.py            # streaming + retry core
│   ├── translation.py      # OpenAI ↔ Anthropic ↔ Responses translation
│   ├── keypool.py          # key rotation + health
│   ├── config.py           # all env-overridable settings
│   ├── logger.py           # root-logger setup
│   ├── prettylog.py        # pretty formatter + request tracing
│   ├── system_prompt.py    # override / strip / reinforce
│   └── utils.py
├── data/                 # runtime state (gitignored except placeholders)
│   ├── openrouter_data/    # OR keys
│   ├── huggingface_data/   # HF keys + dead keys
│   ├── proxy_data/         # runtime_provider.json, prompt override
│   └── payloads/           # debug payload dumps (if enabled)
├── setup/install.sh      # installer
├── run.sh                # runner
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

All paths resolve relative to the repo root (`Path(__file__).resolve().parent.parent`),
so the proxy is location-agnostic.

## Logs

- foreground → stdout
- `--bg` / systemd → `proxy/logs/atlas-proxy.log` (rotating, 10MB × 3 backups)
- systemd → `journalctl -u atlas-proxy -f`

## License

MIT — see `LICENSE`.