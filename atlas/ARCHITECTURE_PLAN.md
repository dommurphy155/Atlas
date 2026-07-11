# Atlas — Architecture & Extraction Plan

> **Atlas** is a new, standalone NVIDIA-only API proxy extracted from the working
> parts of KeyHive (`~/api_maker`). KeyHive is **frozen and untouched** — still
> running on `127.0.0.1:8787` (PID confirmed live). Atlas runs on `127.0.0.1:8788`.
> The two coexist side-by-side with zero shared state.

---

## 1. Architecture Plan

Atlas is a FastAPI + uvicorn + httpx service that exposes an
OpenAI-compatible **and** Anthropic-compatible chat API, routing every request
directly to NVIDIA's `integrate.api.nvidia.com` endpoint. One provider, no
fallback, no router.

```
~/claude/atlas/
├── README.md
├── bin/atlas                      operator CLI: start|stop|restart|status|logs
├── proxy/
│   ├── __init__.py
│   ├── atlas_proxy.py             FastAPI app + entrypoint (NVIDIA-only)
│   ├── nvidia_client.py           thin httpx wrapper → NVIDIA chat-completions
│   ├── openai_compat.py           OpenAI↔Anthropic↔router JSON translation
│   ├── stats.py                   per-request stats → data/proxy_stats.json
│   └── system_prompt.py           system-reminder strip + override injection
├── data/
│   ├── keys.txt                   NVIDIA nvapi- keys (one per line)
│   ├── proxy_stats.json           runtime stats (fresh, not copied from KeyHive)
│   └── system_prompt_override.txt  system-prompt override text
├── setup/
│   ├── install.sh                 venv + pip bootstrap (no node/npm)
│   ├── installer.py               systemd install + CLI symlink + Claude env wiring
│   └── requirements.txt           httpx, python-dotenv, fastapi, uvicorn[standard], rich
└── systemd/
    └── atlas-proxy.service        port 8788, WorkingDirectory ~/claude/atlas
```

### Request flow (both endpoints)

```
client → /v1/chat/completions  ─┐
client → /v1/messages          ─┤  → atlas_proxy.py
                                 │    ├─ parse + size guard
                                 │    ├─ normalize_messages (openai_compat)
                                 │    ├─ replace_system_prompt (system_prompt)
                                 │    ├─ acquire key from NvidiaKeyStore (round-robin)
                                 │    ├─ NvidiaClient.chat / stream_chat
                                 │    ├─ openai_response_from_router (non-stream)
                                 │    │   or stream_router_sse (stream)
                                 │    ├─ record_success/failure (stats)
                                 │    └─ (Anthropic endpoint) openai_response_to_anthropic
                                 └→ NVIDIA integrate.api.nvidia.com
```

No `FallbackManager`. No provider selection. No HF. `/v1/messages` routes to
NVIDIA directly (this fixes the latent KeyHive bug where it was hardcoded to HF).

---

## 2. Files to Extract (verbatim logic, cleaned comments)

| Atlas file | KeyHive source | Treatment |
|---|---|---|
| `proxy/openai_compat.py` | `proxy/openai_compat.py` | copy verbatim; drop the "Hugging Face and NVIDIA" comment to "NVIDIA" |
| `proxy/stats.py` | `proxy/stats.py` | copy verbatim; rename env `KEYHIVE_STATS_DIR` → `ATLAS_STATS_DIR`; drop `provider_hf` bucket (single provider) |
| `proxy/system_prompt.py` | `proxy/system_prompt.py` | copy verbatim; docstring says "jailbreak persona" — keep neutral wording, logic unchanged |
| `proxy/nvidia_client.py` | `proxy/fallback/nvidia_client.py` | copy verbatim (self-contained, no HF refs) |
| `proxy/__init__.py` | `proxy/__init__.py` | new package marker |

---

## 3. Files to Rewrite

| Atlas file | Source | What changes |
|---|---|---|
| `proxy/atlas_proxy.py` | `proxy/keyhive_proxy.py` | **major rewrite.** Drop all HF imports (`HFClient`, `KeyStore`, `KeyState`, `FallbackManager`). Promote `handle_nvidia_non_stream`/`handle_nvidia_stream` to the sole handlers. `/v1/messages` routes to NVIDIA (was hardcoded `"hf"`). `/v1/models` lists the NVIDIA model, `owned_by: "nvidia"`. Port default `8787`→`8788`. Env prefix `KEYHIVE_`→`ATLAS_`. Service name `atlas-proxy`. Logging tag `atlas`. |
| `bin/atlas` | `bin/keyhive` (1528 lines) | **clean rewrite, ~150 lines.** Only `start|stop|restart|status|logs`. No scanner, no web, no fallback, no doctor, no `ai` dump. Talks to `127.0.0.1:8788`. |
| `systemd/atlas-proxy.service` | `systemd/keyhive-proxy.service` | new unit. Port 8788. `WorkingDirectory=/root/claude/atlas`. NVIDIA env only. No `KEYHIVE_*`, no fallback vars, no HF base URL. |
| `setup/install.sh` | `setup/install.sh` | strip node/npm/apt-for-node. Keep venv + pip + handoff. |
| `setup/installer.py` | `setup/installer.py` | strip node runtime, gmail, agentmail, hcaptcha, scheduler/web units, package.json. Keep venv check, systemd install (atlas-proxy only), CLI symlink, Claude env wiring (`ANTHROPIC_BASE_URL`→`:8788`). |
| `setup/requirements.txt` | `setup/requirements.txt` | `httpx`, `python-dotenv`, `fastapi`, `uvicorn[standard]`, `rich`. No npm comments. |
| `README.md` | new | Atlas NVIDIA-only docs. |

---

## 4. Files NOT Needed (left in KeyHive, never copied)

- `proxy/hf_client.py` — HF router client
- `proxy/key_store.py` — HF key pool with health tracking
- `proxy/fallback/manager.py` — `FallbackManager` dual-provider router
- `proxy/fallback/__init__.py` — package marker (flattened away)
- `scripts/*` — all 8 (HF/browser/captcha/scheduler)
- `telegram/*` — entire bot + bridge
- `profiles/*` — 21k Chrome profiles
- `legacy/*`, `logs/*`, `assets/*`, `node_modules/`, `package.json`
- `systemd/api-maker-scheduler.service`, `systemd/keyhive-web.service`
- All `.env` HF/captcha/telegram keys

---

## 5. Dependency List

**Python (`setup/requirements.txt`):**
- `httpx` — NVIDIA upstream client
- `python-dotenv` — `.env` loading
- `fastapi` — HTTP framework
- `uvicorn[standard]` — ASGI server
- `rich` — installer output only

**Nothing else.** No Node, no npm, no Playwright, no aiogram, no captcha libs.
Atlas's `proxy/` imports only those 4 (+ stdlib).

**Env (`~/claude/atlas/.env`, created by installer):**
- `ATLAS_PROXY_HOST=127.0.0.1`
- `ATLAS_PROXY_PORT=8788`
- `ATLAS_KEYS_FILE=<atlas>/data/keys.txt`
- `ATLAS_NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1/chat/completions`
- `ATLAS_NVIDIA_MODEL=moonshotai/kimi-k2.6`
- `ATLAS_PROXY_RELOAD_SECONDS=5`
- `ATLAS_PROXY_REQUEST_TIMEOUT=300`
- `ATLAS_PROXY_MAX_RETRIES=2`
- `ATLAS_PROXY_MAX_KEY_FAILOVERS=3`
- `ATLAS_PROXY_DEBUG=0`

No HF, no telegram, no captcha, no gmail keys. Ever.

---

## 6. Migration / Testing Plan

Atlas is greenfield — no migration of running state. KeyHive's 43M-request
`proxy_stats.json` stays in KeyHive; Atlas gets a fresh empty one.

### Build order
1. Scaffold dirs ✅
2. Write verbatim modules (`openai_compat`, `stats`, `system_prompt`, `nvidia_client`, `__init__`)
3. Write `atlas_proxy.py` (the rewrite)
4. Write `bin/atlas`, `systemd/atlas-proxy.service`, `setup/*`
5. Seed `data/keys.txt` (real NVIDIA keys), `data/system_prompt_override.txt`, empty `data/proxy_stats.json`
6. Write `README.md`

### Verification (do NOT touch 8787)
1. `cd ~/claude/atlas && python -m proxy.atlas_proxy` boots on 8788
2. `curl 127.0.0.1:8788/health` → `{"service":"atlas-proxy",...}`
3. `curl 127.0.0.1:8788/v1/models` → NVIDIA model, `owned_by: nvidia`
4. `curl 127.0.0.1:8788/stats` → fresh counters
5. `curl -X POST 127.0.0.1:8788/v1/chat/completions` (non-stream) with a real key → 200
6. Same with `"stream": true` → SSE
7. `curl -X POST 127.0.0.1:8788/v1/messages` (Anthropic) → 200, Anthropic-shaped response (confirms the bug fix)
8. Confirm KeyHive still listening on 8787, untouched

### Independence guarantees
- Separate venv (`~/claude/atlas/.venv`)
- Separate data dir, separate stats file, separate keys file
- Separate systemd unit name (`atlas-proxy.service`)
- Separate env prefix (`ATLAS_*` not `KEYHIVE_*`)
- No path in Atlas references `~/api_maker`
