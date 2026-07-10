# KeyHive Architecture Audit & NVIDIA-Only Cleanup Plan

> Source repo: `~/api_maker` (legacy, **untouched**)
> Target repo: `~/claude/keyhive/` (fresh build)
> Mandate: extract the working NVIDIA proxy, drop everything HF/browser/captcha/telegram/scheduler.
> New proxy **must run on a different port** than the existing `8787`.

---

## 1. Current Architecture Summary

`~/api_maker` is a **multi-system HF-key-acquisition platform** with a bolt-on OpenAI/Anthropic-compatible proxy. The proxy itself is a FastAPI + uvicorn + httpx service, but it's architected **HF-primary, NVIDIA-fallback** — NVIDIA is emergency backup, not the main path.

### What's actually running (the proxy, the only thing worth keeping)

```
proxy/keyhive_proxy.py        ← FastAPI app + entry point (python -m proxy.keyhive_proxy)
  ├─ proxy/hf_client.py            HFClient  → router.huggingface.co  (PRIMARY, delete)
  ├─ proxy/key_store.py            HF key pool w/ health tracking    (HF-only, delete)
  ├─ proxy/fallback/nvidia_client.py   NvidiaClient → integrate.api.nvidia.com  (KEEP)
  ├─ proxy/fallback/key_store.py       NvidiaKeyStore round-robin pool           (KEEP)
  ├─ proxy/fallback/manager.py         FallbackManager HF↔NVDA router           (delete)
  ├─ proxy/openai_compat.py        OpenAI↔Anthropic↔router JSON translation     (KEEP)
  ├─ proxy/stats.py                per-request stats → data/proxy_stats.json    (KEEP)
  └─ proxy/system_prompt.py        system-reminder strip + persona override     (KEEP)
```

### What's dead weight around it

| Area | What it is | Status |
|---|---|---|
| `scripts/*.js` (4 files) | Playwright browser automation for HF key-gen + hCaptcha cookie refresh | Dead |
| `scripts/*.py` (4 files) | scheduler.py (Chrome cron), burner_email.py, count_keys.py, run_stats.py | Dead |
| `telegram/` | Full aiogram bot + bridge (auth, media, TTS, ffmpeg) | Dead |
| `profiles/microsoft/` | Chrome browser profiles for hCaptcha accounts (`hcaptchaacc*@hotmail.com`) | Dead |
| `legacy/hf_data/` | `.bak` of hc_cookie.json + ip_proxys.txt | Dead |
| `data/hc_cookie.json`, `data/ip_proxys.*`, `data/run_stats.*`, `data/.last_key_count` | HF scanner runtime state | Dead |
| `package.json` + `node_modules/` | `patchright`, `playwright`, `dotenv`, `socks` — all browser automation | Dead |
| `systemd/api-maker-scheduler.service` | runs `scripts/scheduler.py` with `DISPLAY=:1` (Chrome) | Dead |
| `systemd/keyhive-web.service` | runs `web_ui.back_end.app:app` — **dir doesn't exist** | Dead (broken) |
| `proxy/nvidia/` | empty dir, stale `.pyc` only | Dead bytecode |
| `proxy/hf_client.py`, `proxy/key_store.py`, `proxy/fallback/manager.py` | HF client + HF key store + dual-provider router | Dead (after refactor) |

### Plugin/tool cruft (not real project files — scattered everywhere)

`.remember/`, `.plugin-config/`, `.pair-programming-session.md`, `.session-summary.md`, `.pytest_cache/` appear at repo root and inside `proxy/`, `data/`, `scripts/`, `telegram/` (5 locations). The `data/.remember/logs/autonomous/` dir alone has 70+ save logs. Also `data/.claude/settings.local.json` — a stray tool config. All junk.

### Scale notes

- `profiles/` is **21,801 files** — Chrome browser profiles for hCaptcha accounts (google + microsoft). Largest single chunk of dead weight.
- `telegram/` is 37 files; `telegram_main_bridge.py` only exists as `.orig` (live file already deleted, backup remains). Fully dead.
- `logs/` holds `fail_hf_flow.png`, `keyhive-scanner.log`, `vpn_rotate.log` — scanner/vpn logs. Dead.

### The port

**Current: `8787` on `127.0.0.1`** — confirmed three ways:
- `systemd/keyhive-proxy.service`: `Environment=KEYHIVE_PROXY_PORT=8787`
- `bin/keyhive`: `PROXY_URL="${KEYHIVE_PROXY_URL:-http://127.0.0.1:8787}"`
- `proxy/keyhive_proxy.py:47`: `PORT = int(os.getenv("KEYHIVE_PROXY_PORT", "8787"))`

Also: `keyhive-web.service` binds `0.0.0.0:8080`, `api-maker-scheduler.service` has no port (it's a Chrome cron).

### The critical latent bug (found during audit)

**`/v1/messages` (the Anthropic-protocol endpoint) is hardcoded to HF** — `keyhive_proxy.py:379`:
```python
response = await handle_non_stream(DEFAULT_MODEL, payload, rid, "hf", started)
```
`DEFAULT_MODEL` is the HF model. This endpoint **never falls back to NVIDIA**. Since the proxy's entire purpose is letting Claude Code (Anthropic protocol) reach NVIDIA, this endpoint is currently HF-only by design. The NVIDIA-only refactor **must** repoint `/v1/messages` at NVIDIA or the proxy is useless for its intended consumer.

### Secondary findings

- `FallbackManager` has config drift: systemd sets `KEYHIVE_FALLBACK_EXIT_AT=10` but the constructor pins `exit_at=1` — the env var is ignored. Moot once deleted, but evidence the fallback layer was already half-broken.
- `hf_client.py` exports `retry_after_seconds()` that nothing imports — dead helper.
- `bin/keyhive` is a 1528-line bash CLI that references stale paths (`proxy/server.py`, `web/server.js` — neither exists). Half of it is scanner/HF control, half is proxy control.
- `installer.py:425` curls `claude.ai/install.sh` and sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` — **this is the Claude Code → proxy wiring, the whole point of the project.** Keep, but the hardcoded `8787` must change to the new port.
- `.env` has a leaked credential on a bare line (`concerneddesign380@agentmail.to/Lorenzo25!`) — flagged, not touching `api_maker`.

---

## 2. Proposed Simplified Architecture

```
~/claude/keyhive/
├── README.md                      ← rewritten, NVIDIA-only
├── bin/
│   └── keyhive                    ← slimmed: proxy control only, no scanner
├── proxy/
│   ├── __init__.py
│   ├── keyhive_proxy.py           ← refactored: NVIDIA sole provider, no HF branches
│   ├── nvidia_client.py           ← from proxy/fallback/nvidia_client.py (flattened)
│   ├── nvidia_key_store.py        ← from proxy/fallback/key_store.py (flattened)
│   ├── openai_compat.py           ← unchanged
│   ├── stats.py                   ← provider_hf counter vestigial but harmless
│   └── system_prompt.py           ← unchanged
├── setup/
│   ├── install.sh                 ← stripped: venv + pip only, no node/npm
│   ├── installer.py               ← stripped: proxy unit + claude code wiring only
│   └── requirements.txt           ← httpx, python-dotenv, fastapi, uvicorn[standard], rich
└── systemd/
    └── keyhive-proxy.service      ← port changed, HF env vars dropped
```

**Port:** `8788` (one above current, avoids collision with the still-running `api_maker` on 8787 — both can coexist during migration).

**Entry point unchanged:** `python -m proxy.keyhive_proxy` → uvicorn → FastAPI app.

**Provider model:** single provider, no router. `handle_non_stream`/`handle_stream` call NVIDIA directly. `FallbackManager` deleted entirely — with one provider there's nothing to arbitrate.

---

## 3. File-by-File Removal List

### Top-level dirs/files — delete entirely
- `scripts/` (all 8 files — 4 JS browser automation, 4 Python scanner/burner/stats)
- `telegram/` (entire bot + bridge)
- `profiles/` (Chrome browser profiles)
- `legacy/` (`.bak` files only, empty subdirs)
- `node_modules/`
- `package.json`, `package-lock.json`
- `assets/` (header image — not needed for a proxy service; optional keep for README)
- `systemd/api-maker-scheduler.service`
- `systemd/keyhive-web.service`

### `proxy/` — delete
- `proxy/hf_client.py` (HF upstream client)
- `proxy/key_store.py` (HF key pool — `KeyState`/`KeyStore`)
- `proxy/fallback/manager.py` (dual-provider router)
- `proxy/fallback/__init__.py` (package marker, gone with flattening)
- `proxy/nvidia/` (empty dir + stale `.pyc`)
- `proxy/tests/` (empty dir)

### `data/` — delete these, keep runtime files
- `data/hc_cookie.json`, `data/ip_proxys.json`, `data/ip_proxys.txt`
- `data/run_stats.json`, `data/run_stats.lock`, `data/.last_key_count`
- `data/keys.txt` (HF keys — gone with HF)
- All plugin cruft: `data/.remember/`, `data/.plugin-config/`, `data/.pair-programming-session.md`, `data/.session-summary.md`, `data/.claude/`
- **Keep:** `data/nvda_fallback_keys.txt` (rename to `data/keys.txt`), `data/proxy_stats.json`, `data/system_prompt_override.txt`, `data/.gitkeep`

### Plugin cruft — delete everywhere
- `.remember/`, `.plugin-config/`, `.pair-programming-session.md`, `.session-summary.md`, `.pytest_cache/` at root and in every subdirectory

---

## 4. File-by-File Keep List

| File | Source | Treatment |
|---|---|---|
| `proxy/__init__.py` | `proxy/__init__.py` | copy as-is |
| `proxy/keyhive_proxy.py` | `proxy/keyhive_proxy.py` | **heavy refactor** (see §5) |
| `proxy/nvidia_client.py` | `proxy/fallback/nvidia_client.py` | copy, flatten import path |
| `proxy/nvidia_key_store.py` | `proxy/fallback/key_store.py` | copy, flatten import path |
| `proxy/openai_compat.py` | `proxy/openai_compat.py` | copy as-is (provider-agnostic) |
| `proxy/stats.py` | `proxy/stats.py` | copy as-is (`provider_hf` counter dead but harmless) |
| `proxy/system_prompt.py` | `proxy/system_prompt.py` | copy as-is |
| `bin/keyhive` | `bin/keyhive` | **slim** — keep proxy control half, drop scanner half |
| `setup/install.sh` | `setup/install.sh` | **strip** — drop node/npm bootstrap |
| `setup/installer.py` | `setup/installer.py` | **strip** — drop node/gmail/hcaptcha/scheduler branches |
| `setup/requirements.txt` | `setup/requirements.txt` | clean (drop npm comments) |
| `systemd/keyhive-proxy.service` | `systemd/keyhive-proxy.service` | **edit** — port + env vars |
| `README.md` | new | rewrite NVIDIA-only |

---

## 5. Required Refactors

### `proxy/keyhive_proxy.py` — the big one

**Imports to remove:**
```python
from proxy.fallback.key_store import NvidiaKeyStore   # → from proxy.nvidia_key_store import NvidiaKeyStore
from proxy.fallback.manager import FallbackManager     # DELETE
from proxy.fallback.nvidia_client import NvidiaClient  # → from proxy.nvidia_client import NvidiaClient
from proxy.hf_client import HFClient                   # DELETE
from proxy.key_store import KeyState, KeyStore         # DELETE
```

**Globals to remove/change:**
- Delete `DEFAULT_PROVIDER`, `FALLBACK_PROVIDER`, `HF_BASE_URL`, `DEFAULT_MODEL` (HF model)
- Keep `NVIDIA_MODEL`, `NVIDIA_BASE_URL`, `NVIDIA_KEYS_FILE`
- Change `PORT` default `8787` → `8788`

**Objects to remove:**
- `hf_client = HFClient(...)` (line 102) — delete
- `key_store = KeyStore(...)` (HF pool) — delete
- `fallback_manager = FallbackManager(...)` (line 105) — delete
- `refresh_provider_mode()` wrapper — delete
- `choose_key_or_503()` — delete (HF key selection)

**Handler refactor — promote NVIDIA from fallback to sole path:**
- `handle_non_stream` / `handle_stream`: currently HF-primary with NVIDIA as a branch reached only on HF exhaustion/402/429/401/403. **Rewrite to call NVIDIA directly** (the `handle_nvidia_non_stream`/`handle_nvidia_stream` functions already exist and are self-contained — promote them to the main path, delete the HF try-loop).
- Or: rename `handle_nvidia_non_stream` → `handle_non_stream` and delete the old HF-primary version. Cleaner.

**Endpoint fixes:**
- `/v1/messages` line 379: `handle_non_stream(DEFAULT_MODEL, ..., "hf", ...)` → `handle_non_stream(NVIDIA_MODEL, ..., "nvidia", ...)`. **This is the bug fix that makes the proxy actually useful.**
- `/v1/models`: `"owned_by": "huggingface"` → `"owned_by": "nvidia"`, list `NVIDIA_MODEL` not `DEFAULT_MODEL`.
- `/health`, `/stats`: drop `current_provider` HF/NVIDIA fields or hardcode to `"nvidia"`.
- Error messages: replace "Hugging Face keys exhausted and NVIDIA fallback unavailable" with NVIDIA-only wording.

**`stats.py` interaction:** `record_success(provider, ...)` / `record_failure(provider)` — pass `"nvidia"` everywhere. The `provider_hf` bucket stays in the schema but never increments. Harmless. Optional: collapse the schema later.

### `proxy/nvidia_key_store.py` (from `fallback/key_store.py`)
- Flatten import path only. No logic change. **Decision point:** the HF `KeyStore` had rich health tracking (`fail_key`, `cooldown_key`, `exhaust_key`, `invalidate_key`) that `NvidiaKeyStore` lacks (it just rotates, never removes keys). For a production multi-key NVIDIA proxy, you may want to port that health-tracking onto `NvidiaKeyStore` so a burned `nvapi-` key gets cooled down instead of retried every request. **Flag for user decision — see §8.**

### `proxy/nvidia_client.py` (from `fallback/nvidia_client.py`)
- Flatten import path only. No logic change. `is_valid_key()` already checks `nvapi-` prefix.

### `bin/keyhive`
- 1528 lines → ~400. Keep: `proxy start|stop|restart|status|logs|stats|test`. Delete: scanner subcommands, `fallback` display, `doctor` HF/playwright checks, Gmail/agentmail/hcaptcha prompts, all `hf_keys.js`/`hc_cookie.js` references.
- Fix stale path refs (`proxy/server.py`, `web/server.js` don't exist).
- Update `PROXY_URL` default `8787` → `8788`.

### `setup/installer.py`
- Delete: `install_node_runtime()`, `install_web_ui_runtime()`, `ensure_package_json()`, `prompt_agentmail()`, `prompt_gmail()`, `ensure_runtime_files()` hc_cookie line, scheduler unit from `install_systemd_units()`.
- Keep: `render_systemd_unit()`, `install_systemd_units()` (proxy unit only), `install_cli()`, `install_claude_code()` (update `ANTHROPIC_BASE_URL` to `:8788`), `configure_proxy_env_defaults()`.
- `rich` stays in requirements (installer uses it).

### `setup/install.sh`
- Drop `ensure_node_command()`, the `node`/`npm` apt installs. Keep venv + pip + handoff to installer.

### `systemd/keyhive-proxy.service`
- `KEYHIVE_PROXY_PORT=8787` → `8788`
- Delete: `KEYHIVE_PROXY_DEFAULT_PROVIDER=hf`, `KEYHIVE_PROXY_FALLBACK_PROVIDER=nvidia`, `KEYHIVE_FALLBACK_ENABLED`, `KEYHIVE_FALLBACK_PROVIDER`, `KEYHIVE_FALLBACK_ENTER_AT`, `KEYHIVE_FALLBACK_EXIT_AT`, `KEYHIVE_HF_BASE_URL`, `KEYHIVE_PROXY_DEFAULT_MODEL` (HF model)
- Keep: `KEYHIVE_PROXY_HOST`, `KEYHIVE_PROXY_PORT`, `KEYHIVE_KEYS_FILE` (now points to NVIDIA keys), `KEYHIVE_NVIDIA_BASE_URL`, `KEYHIVE_PROXY_NVIDIA_MODEL`, `KEYHIVE_PROXY_RELOAD_SECONDS`, `KEYHIVE_PROXY_MAX_KEY_FAILOVERS`, `KEYHIVE_PROXY_REQUEST_TIMEOUT`, `KEYHIVE_PROXY_MAX_RETRIES`, `KEYHIVE_PROXY_DEBUG`
- Update `ExecStart` stays `@PYTHON_BIN@ -m proxy.keyhive_proxy`

---

## 6. Potential Breaking Changes

1. **`/v1/messages` was HF-only** — after refactor it goes to NVIDIA. This *fixes* a latent bug but **changes behavior**: any client currently relying on the Anthropic endpoint hitting HF will now hit NVIDIA. Since HF is being removed entirely, this is intended, but flag it.
2. **`/v1/models` model list changes** — `DEFAULT_MODEL` (HF) disappears, only `NVIDIA_MODEL` remains. Clients pinning the HF model name will break.
3. **`data/keys.txt` semantics change** — was HF tokens, becomes `nvapi-` keys. Rename `nvda_fallback_keys.txt` → `keys.txt` to keep the `KEYHIVE_KEYS_FILE` path consistent, or keep separate names. **Decision point.**
4. **`stats.py` schema** — `provider_hf` bucket stops incrementing. Any dashboard reading `provider_hf` gets zeros. Harmless but visible.
5. **Port change 8787 → 8788** — `installer.py:425` hardcodes `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` for Claude Code wiring. **Must update to 8788** or Claude Code keeps talking to the old api_maker proxy.
6. **`bin/keyhive` scanner subcommands vanish** — any operator scripts calling `keyhive start` (scanner) or `keyhive runs=N` will break. Proxy subcommands (`proxy start`, etc.) unaffected.
7. **No `KEYHIVE_FALLBACK_*` env vars** — any operator scripts or docs referencing fallback knobs will 404 silently (code does `os.getenv` with defaults).
8. **`NvidiaKeyStore` lacks health tracking** — if an `nvapi-` key is rate-limited (429) or invalid (401), the current code has no `fail_key`/`cooldown_key` equivalent for NVIDIA — it just rotates. The HF path had this. Without porting it, a dead key gets retried every rotation. **See §8.**

---

## 7. Migration Plan (ordered, non-destructive to `api_maker`)

### Phase 0 — Setup (no code changes)
0.1. Create `~/claude/keyhive/` dir structure.
0.2. `git init` in `~/claude` (already a repo — create `keyhive/` subdirectory or build at root; **decision point**: root vs subdir).

### Phase 1 — Copy clean files (verbatim)
1.1. `proxy/__init__.py`, `proxy/openai_compat.py`, `proxy/stats.py`, `proxy/system_prompt.py` — copy as-is.
1.2. `proxy/fallback/nvidia_client.py` → `proxy/nvidia_client.py` (fix internal imports — none, it's self-contained).
1.3. `proxy/fallback/key_store.py` → `proxy/nvidia_key_store.py` (self-contained).
1.4. `setup/requirements.txt` — copy, strip npm comments.

### Phase 2 — Refactor `keyhive_proxy.py`
2.1. Copy `proxy/keyhive_proxy.py` to new repo.
2.2. Remove HF imports, `hf_client`, `key_store`, `fallback_manager`, `refresh_provider_mode`, `choose_key_or_503`.
2.3. Promote `handle_nvidia_non_stream`/`handle_nvidia_stream` to the main handler path (rename or inline).
2.4. Fix `/v1/messages` → route to NVIDIA with `NVIDIA_MODEL`.
2.5. Fix `/v1/models` `owned_by` + model list.
2.6. Update error messages, `/health`, `/stats` provider fields.
2.7. Change `PORT` default `8787` → `8788`.
2.8. Update `NvidiaKeyStore`/`NvidiaClient` import paths.

### Phase 3 — Supporting files
3.1. `systemd/keyhive-proxy.service` — port 8788, drop HF/fallback env vars.
3.2. `setup/installer.py` — strip node/gmail/hcaptcha/scheduler branches, update `ANTHROPIC_BASE_URL` to 8788, drop scheduler + web units from `install_systemd_units()`.
3.3. `setup/install.sh` — strip node/npm bootstrap.
3.4. `bin/keyhive` — slim to proxy control only, update `PROXY_URL` to 8788, fix stale paths.
3.5. `README.md` — rewrite NVIDIA-only.

### Phase 4 — Data + config
4.1. Copy `data/nvda_fallback_keys.txt` → `data/keys.txt` (NVIDIA keys).
4.2. Copy `data/system_prompt_override.txt`, `data/proxy_stats.json` (or let it regenerate).
4.3. Create `.env` with only NVIDIA-proxy keys: `NVDA_KEY`, `KEYHIVE_PROXY_HOST`, `KEYHIVE_PROXY_PORT=8788`, `KEYHIVE_KEYS_FILE`, `KEYHIVE_NVIDIA_BASE_URL`, `KEYHIVE_PROXY_NVIDIA_MODEL`.

### Phase 5 — Verify (do NOT start on 8787)
5.1. `python -m proxy.keyhive_proxy --port 8788` — boots, `/health` returns 200.
5.2. `curl /v1/chat/completions` with an `nvapi-` key — non-stream + stream.
5.3. `curl /v1/messages` (Anthropic) — **confirms the bug fix**.
5.4. `curl /v1/models` — lists NVIDIA model only.
5.5. Confirm `api_maker` on 8787 is untouched and still running.

### Phase 6 — Cutover (only after Phase 5 green)
6.1. `systemctl stop keyhive-proxy` (the old 8787 one) — or leave running during overlap.
6.2. Install new unit, `systemctl daemon-reload`, `systemctl start keyhive-proxy` (8788).
6.3. Update Claude Code `ANTHROPIC_BASE_URL` → `:8788`.

---

## 9. README Rewrite Scope (current heading structure)

Current `README.md` sections with verdicts for NVIDIA-only:

```
#  KeyHive                                    KEEP — rewrite for NVIDIA-only
## What It Does                               REWRITE — currently describes HF scanner + NVIDIA fallback
## Repository Structure                       REWRITE — drop scanner/telegram/profiles/web_ui
## Main Files                                 REWRITE — drop HF entries
## Dependencies                                REWRITE — drop Node/playwright/patchright
## Install                                    REWRITE — drop npm install, drop scheduler
## Environment                                REWRITE — drop HF/telegram/captcha/gmail keys
## Setup Flow                                 DROP — HF scanner onboarding
## `keyhive` CLI                              REWRITE — keep proxy/logs/diag, drop scanner
## Scanner                                    DROP — HF browser automation
## Proxy                                      KEEP — this is the whole point
## Web UI                                     DROP — status dashboard for scanner+proxy
## Systemd                                    REWRITE — keep only keyhive-proxy.service
## Logs                                       REWRITE — drop scanner/vpn logs
## Data                                       REWRITE — keep only nvda_fallback_keys.txt + proxy_stats.json
## Profiles                                   DROP — browser profiles
## Troubleshooting                            PRUNE — keep proxy-only items
## Uninstall / Reset                          REWRITE — proxy-only
## Safety Notes                               KEEP — prune HF refs
```

Drop entirely: **Setup Flow, Scanner, Web UI, Logs (scanner/vpn), Profiles, Dependencies (Node), Environment (HF/telegram/gmail/captcha)**.
Keep/rewrite: **Proxy, Systemd (proxy service), Data (nvda keys + proxy_stats), `keyhive` CLI (proxy/logs/diag subset)**.

---

## 8. Open Decisions for the User

1. **Repo layout** — build `keyhive/` as a subdirectory of `~/claude`, or at `~/claude` root? (Current `~/claude` has `heart/`, `tests/`, `data/` already.)
2. **Port** — `8788` (recommended, adjacent, coexists during migration) or another?
3. **`NvidiaKeyStore` health tracking** — port the HF `KeyStore`'s `fail_key`/`cooldown_key`/`exhaust_key` logic onto NVIDIA keys so a rate-limited `nvapi-` key cools down instead of retrying every rotation? **Recommended: yes** for a production multi-key proxy. Current NVIDIA store just rotates blindly.
4. **`data/keys.txt` naming** — rename `nvda_fallback_keys.txt` → `keys.txt` (cleaner, matches `KEYHIVE_KEYS_FILE` default) or keep the fallback name?
5. **`assets/header.png`** — keep for README or drop?
6. **`stats.py` schema** — collapse `provider_hf`/`provider_nvidia` to a single counter now, or leave vestigial?
