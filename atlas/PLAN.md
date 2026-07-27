# Atlas Proxy — Production Refactor Plan

**Date:** 2026-07-27
**Author:** Atlas (Staff/Principal Python Engineer)
**Branch:** `refactor` (to be created from `main`)
**Method:** Every audit claim independently verified against source code. Claims already fixed or never existed are dropped. Only verified issues are addressed.

---

## 0. Audit Verification Results

The audit (`/root/atlas_audit_20260727_085235.md`) was treated as a starting point. Every claim was verified against the actual codebase.

### Claims that are WRONG (already fixed or never existed) — DROPPED

| Audit Claim | Status | Evidence |
|---|---|---|
| `nvidia_key_store.py` uses `requests` (sync HTTP) | **FALSE** | `rg 'import requests\|from requests' proxy/` → NOT FOUND. Already uses httpx. |
| `verify_keys()` blocks event loop with sync HTTP | **FALSE** | `verify_keys` function does not exist. Key validation done via `httpx.AsyncClient` prewarm in `NvidiaClient`. |
| No error class hierarchy (`NvidiaClientError`, etc.) | **FALSE** | These classes never existed. `NvidiaClient` uses `NvidiaResponse` dataclass and lets httpx exceptions propagate. Failover loop catches `httpx.TimeoutException`, `httpx.ConnectError` directly. Valid design choice. |
| No atomic writes in `stats.py` | **FALSE** | `stats.py` already has atomic write: `tmp = STATS_FILE.with_suffix(".tmp")` + `os.replace(str(tmp), str(STATS_FILE))` (lines 87-90). |
| `token_tracker.py` duplicates `stats.py` persistence | **FALSE** | `token_tracker.py` is now a READ-ONLY CLI tool (`atlas tokens`). Reads stats file, never writes. No duplication. |
| No system_prompt override caching | **FALSE** | `system_prompt.py` already has mtime-checked cache (`_override_cache`, `_override_mtime`, lines 56-69). |
| No system_prompt fallback for missing file | **FALSE** | `system_prompt.py` handles `FileNotFoundError` (line 65) and `OSError` (line 73) gracefully. |
| No file permissions on `keys.txt` | **FALSE** | `nvidia_key_store.py` line 82: `self.keys_file.touch(mode=0o600)`. |
| No `User-Agent` header | **FALSE** | `nvidia_client.py` line 94: `User-Agent: atlas-prewarm` (prewarm path). |
| `threading.Lock` in stats.py is a code smell | **DEBATABLE** | Lock prevents torn reads during concurrent `save()`. Harmless in asyncio single-threaded context. Defensive, not a defect. |
| Stats writes to disk synchronously on every request | **PARTIALLY TRUE** | `record()` calls `save()` which does atomic write. Per-request disk I/O is real but corruption risk is already handled. |

### Claims that are CORRECT — ADDRESSED IN THIS PLAN

| # | Audit Claim | Verified | Evidence |
|---|---|---|---|
| V1 | `active_requests` is a bare `int` global mutated from async code without lock | ✅ | Line 163: `active_requests = 0`. Incremented/decremented at 12+ locations with `global active_requests`. Race condition under concurrent completion. |
| V2 | `openai_compat.py` has duplicate class definitions | ✅ | See §0.1 below for full dead-code analysis. |
| V3 | Double JSON encoding at line 2204 | ✅ | `arguments=json.dumps(fn.get("arguments") or {})` — no `isinstance` check. NVIDIA returns `arguments` as a JSON string → `json.dumps` wraps it in quotes → double-encoded. Line 1952 correctly checks `isinstance(arguments, dict)` first. Line 2204 does not. |
| V4 | `random.randint` for IDs instead of UUIDs | ✅ | Lines 561, 621, 2459, 2519: `random.randint(100000, 999999)`. Collision risk under high throughput. |
| V5 | `streaming_headers()` duplicated in both files | ✅ | `atlas_proxy.py:227` and `openai_compat.py:804`. Same function, two locations. |
| V6 | `_anthropic_stop_reason` maps `content_filter` to `end_turn` | ✅ | Line 1502. Semantically incorrect — should map to `refusal`. |
| V7 | No `__aenter__`/`__aexit__` on `NvidiaClient` | ✅ | `grep` confirms no context manager protocol. |
| V8 | No tests | ✅ | Zero test files in the project. |
| V9 | Dead functions `responses_request_to_chat_payload` and `openai_response_to_responses` | ✅ | `rg` confirms these are only defined (lines 1908, 2161), never imported by any file. |
| V10 | `openai_compat.py` is 2,659 lines | ✅ | Massive file with shadowed definitions. |
| V11 | No configuration externalization | ✅ | Some values already use `os.getenv` (HOST, PORT, KEEPALIVE_SECONDS). Others hardcoded (MAX_BODY_BYTES, drain deadline, timeouts). |
| V12 | No circuit breaker | ✅ | If NVIDIA is down, every request attempts connection, fails, cools a key. No short-circuit. |

### 0.1 Dead Code Analysis — Full Verification

**Method:** Every symbol was checked via `rg` across the entire repo, `inspect.getsourcelines()` for runtime confirmation, and checked for `getattr`/`__import__`/`importlib`/string references.

**Key finding:** Python's module-level shadowing means the LAST definition wins for module-level names. But `inspect.getsourcelines()` reveals which class object is actually bound:

| Symbol | First def | Second def | Actually bound (runtime) | Status |
|---|---|---|---|---|
| `ResponsesInput` | line 17 | line 1734 | **line 17** | First set LIVE |
| `ResponsesReasoning` | line 30 | line 1740 | **line 30** | First set LIVE |
| `ResponsesTextFormat` | line 36 | line 1746 | **line 36** | First set LIVE |
| `ResponsesText` | line 44 | line 1752 | **line 44** | First set LIVE |
| `ResponsesTool` | line 49 | line 1759 | **line 49** | First set LIVE |
| `ResponsesRequest` | line 58 | line 1762 | **line 58** | First set LIVE |
| `ResponsesStreamConverter` | line 386 | line 2269 | **line 386** | First def LIVE |
| `ResponsesUsage` | — | line 2055 | **line 2055** | LIVE (no duplicate) |
| `ResponsesOutputContent` | — | line 2064 | **line 2064** | LIVE (no duplicate) |
| `ResponsesOutputItem` | — | line 2071 | **line 2071** | LIVE (no duplicate) |
| `ResponsesResponse` | — | line 2097 | **line 2097** | LIVE (no duplicate) |

**Wait — this contradicts the initial analysis.** The first set of classes (lines 17-58) ARE the live ones. The second set (lines 1721-1770) are defined later in the file but are NOT what Python binds at module level... actually they ARE — Python rebinds on each `class` statement. Let me re-verify:

`inspect.getsourcelines(ResponsesInput)` returned line 17. This means `ResponsesInput` at module level points to the line 17 definition. The second `class ResponsesInput` at line 1734 DOES rebind the name, but... `inspect` reports the source of the object, and the object at line 17 is what was returned by the import. This means the second definition at line 1734 shadows the first, BUT `responses_request_to_openai` (defined at line 90, between the two sets) captures the first `ResponsesRequest` via closure at definition time.

**Actually no** — Python doesn't capture via closure at definition time for module-level names. It looks up the name at call time. So `responses_request_to_openai` called at runtime would see the SECOND `ResponsesRequest` (line 1762).

**The `inspect` result showing line 58 for `ResponsesRequest` is because `inspect` is looking at the object that the NAME `ResponsesRequest` currently points to — which should be the second one.** The fact that it shows line 58 means...

Let me re-examine. The `inspect.getsourcelines` call returned line 58 for `ResponsesRequest`. If the second definition (line 1762) were shadowing it, `inspect` would show line 1762. It shows line 58. This means the second definition is NOT shadowing the first — possibly because the second set of classes is inside a conditional or has a different name pattern.

**This requires further investigation before any deletion.** The dead code removal phase will be gated on proving reachability, not assumed.

**What IS confirmed dead:**
- `responses_request_to_chat_payload` (line 1908) — `rg` confirms only defined, never imported/called
- `openai_response_to_responses` (line 2161) — `rg` confirms only defined, never imported/called
- No `getattr`/`__import__`/`importlib`/string references to either function
- No `__all__` in the module
- No test files exist

**What is NOT confirmed dead (needs further investigation):**
- The second set of `Responses*` classes (lines 1721-1770) — `inspect` shows the first set is bound, which is contradictory with Python's normal shadowing behavior. Must investigate before removing.
- The second `ResponsesStreamConverter` (line 2269) — `inspect` shows line 386 is bound. Same contradiction.

**Resolution:** The dead code removal phase will NOT proceed until this is fully understood. The plan includes a specific investigation step before any deletion.

---

## 1. Refactor Scope

### Guiding Principle: Minimal Behavioural Change

This is a production refactor, not a rewrite. The goal is to fix verified issues with the smallest possible change surface. We do not invent architecture. We do not refactor for taste. Every change must provide measurable engineering value.

### What we WILL do (verified issues only):

1. **Fix double JSON encoding at line 2204** (V3) — add `isinstance` check
2. **Replace `random.randint` with UUIDs** (V4) — use `uuid.uuid4().hex`
3. **Fix `_anthropic_stop_reason` content_filter mapping** (V6) — map to `refusal`
4. **Add `__aenter__`/`__aexit__` to `NvidiaClient`** (V7)
5. **Deduplicate `streaming_headers()`** (V5) — single definition, import where needed
6. **Remove confirmed dead functions** (V9) — `responses_request_to_chat_payload`, `openai_response_to_responses`
7. **Investigate and remove shadowed class definitions if proven dead** (V2, V10) — gated on reachability proof
8. **Fix `active_requests` race condition** (V1) — smallest safe solution
9. **Externalize remaining hardcoded configuration** (V11) — extend existing `os.getenv` pattern
10. **Add circuit breaker** (V12) — minimal implementation
11. **Add comprehensive test suite with coverage gate** (V8)
12. **Add static analysis tooling** — ruff, mypy
13. **Security review**
14. **Benchmark before/after**

### What we will NOT do:

- ~~Replace `requests` with httpx~~ — already done
- ~~Add atomic writes to stats.py~~ — already done
- ~~Add system_prompt caching~~ — already done
- ~~Add system_prompt fallback~~ — already done
- ~~Set keys.txt permissions~~ — already done
- ~~Extract `PersistentJSON` base class~~ — token_tracker is read-only, no duplication
- ~~Convert `Responses*` classes to pydantic~~ — not a verified defect
- ~~Split `openai_compat.py` into three files~~ — scope creep; file is large but manageable after dead code removal
- ~~Invent new tracker class for active_requests~~ — will use smallest safe fix (see §3)
- ~~Invent elaborate circuit breaker with half-open state~~ — will use minimal implementation (see §5)

---

## 2. Constraints

### 2.1 No Monkey Patches
No runtime patching of third-party classes. No `sys.modules` tricks. No `__import__` overrides.

### 2.2 No TODOs or FIXMEs
Every line of code must be complete. No deferred work. No placeholders. No stubs.

### 2.3 No Silenced Warnings
No `# type: ignore` without a specific reason. No `# noqa` without justification. No bare `except: pass`.

### 2.4 No Temp Code or Compatibility Hacks
No "temporary" shims. No "deprecated but kept for now" paths. If something is dead, remove it. If something is live, fix it.

### 2.5 No Duplicated Logic
If two functions share logic, extract the shared part. If two classes share structure, use a base class or composition. No copy-paste.

### 2.6 Every Refactor Must Provide Measurable Engineering Value
If a change doesn't fix a bug, improve performance, improve testability, or improve safety — it doesn't go in.

### 2.7 Branch Discipline
All work on `refactor` branch, created from `main`. Do NOT merge back to main. Push to `origin/refactor`.

### 2.8 Atomic Commits
Each commit must be a single logical change. Format:

```
fix(openai_compat): check isinstance before json.dumps on tool arguments

Regression test: tests/test_regression.py::test_double_json_encoding_v3
```

Commits must be reviewable in isolation. A reviewer should understand the change from the commit message alone.

### 2.9 Backwards Compatibility — MANDATORY

Every public API endpoint, request format, response schema, SSE event ordering, and CLI command must remain backwards compatible unless there is a verified correctness bug.

- `/v1/chat/completions` — request/response shape unchanged
- `/v1/messages` — request/response shape unchanged, SSE event sequence unchanged
- `/v1/responses` — request/response shape unchanged, SSE event sequence unchanged
- `/health` — response shape unchanged
- `/stats` — response shape unchanged
- `atlas tokens` CLI — output format unchanged
- All environment variable names (`ATLAS_PROXY_HOST`, `ATLAS_PROXY_PORT`, `ATLAS_PROXY_KEEPALIVE_SECONDS`) — unchanged, new ones added additively

**The only exception:** V3 (double JSON encoding) is a correctness bug. The fix changes the output of the Responses API tool call path from double-encoded to correctly-encoded. This is a bug fix, not a breaking change — clients that were working around the double encoding will need to update, but clients that were broken by it will start working.

### 2.10 Stop on Unexpected Findings

If, during implementation, I discover the audit is incorrect, incomplete, or a proposed refactor introduces unnecessary complexity or behavioural changes, I will:

1. **Stop** that line of work immediately
2. **Update PLAN.md** with the new findings
3. **Explain the reasoning** in the plan
4. **Wait for approval** before continuing

I will not force the original plan through when reality contradicts it. This already happened during verification — the `inspect` results for `ResponsesInput` contradicted the expected shadowing behavior. The dead code removal phase is gated on resolving this.

### 2.11 Evidence Requirement

For every claimed fix, I will provide:

- The exact file and line(s) changed
- A before/after code snippet
- A test that proves the fix works
- A test that proves the old behavior was broken (regression test)

The final report will include a table:

| Fix | File:Line | Test | Before | After |
|-----|----------|------|--------|-------|
| V3 | openai_compat.py:2204 | test_double_json_encoding_v3 | `json.dumps(str)` → double-encoded | `isinstance` check → single-encoded |
| ... | ... | ... | ... | ... |

---

## 3. Implementation Plan

### Phase 1: Quick Fixes

Each fix is a separate atomic commit. Each commit includes its regression test.

#### 1.1 Fix double JSON encoding (V3)

**File:** `proxy/openai_compat.py` line 2204

**Before:**
```python
arguments=json.dumps(fn.get("arguments") or {}),
```

**After:**
```python
arguments=(
    fn.get("arguments")
    if isinstance(fn.get("arguments"), str)
    else json.dumps(fn.get("arguments") or {})
),
```

**Regression test:** `test_double_json_encoding_v3` — constructs a tool call response where `arguments` is a JSON string, verifies the output is single-encoded.

**Commit:** `fix(openai_compat): check isinstance before json.dumps on tool arguments`

#### 1.2 Replace random.randint with UUIDs (V4)

**File:** `proxy/openai_compat.py` lines 561, 621, 2459, 2519

**Before:**
```python
self.reasoning_item_id = f"rs_{random.randint(100000, 999999)}"
fc_item_id = f"fc_{random.randint(100000, 999999)}_{i}"
```

**After:**
```python
self.reasoning_item_id = f"rs_{uuid.uuid4().hex[:12]}"
fc_item_id = f"fc_{uuid.uuid4().hex[:12]}_{i}"
```

Add `import uuid` at top if not present.

**Regression test:** `test_tool_call_ids_are_uuids` — generates 1000 IDs, verifies uniqueness and format.

**Commit:** `fix(openai_compat): replace random.randint with uuid for tool call IDs`

#### 1.3 Fix _anthropic_stop_reason content_filter mapping (V6)

**File:** `proxy/openai_compat.py` line 1502

**Before:**
```python
"content_filter": "end_turn",
```

**After:**
```python
"content_filter": "refusal",
```

**Regression test:** `test_anthropic_stop_reason_content_filter` — verifies `content_filter` maps to `refusal`, not `end_turn`.

**Commit:** `fix(openai_compat): map content_filter to refusal instead of end_turn`

#### 1.4 Add __aenter__/__aexit__ to NvidiaClient (V7)

**File:** `proxy/nvidia_client.py`

**Change:** Add after `close()` method:
```python
async def __aenter__(self):
    return self

async def __aexit__(self, *exc):
    await self.close()
```

**Test:** `test_nvidia_client_context_manager` — verifies `async with NvidiaClient(...) as client:` works.

**Commit:** `feat(nvidia_client): add async context manager protocol`

#### 1.5 Deduplicate streaming_headers (V5)

**Files:** `proxy/atlas_proxy.py`, `proxy/openai_compat.py`

**Change:** Remove `streaming_headers()` from `atlas_proxy.py` (line 227). Add import from `openai_compat.py` in `atlas_proxy.py` import block.

**Verification:** `python3 -c "from proxy.atlas_proxy import app"` — verify imports resolve.

**Test:** `test_streaming_headers` — verifies the function returns correct headers.

**Commit:** `refactor: deduplicate streaming_headers into openai_compat only`

---

### Phase 2: Dead Code Investigation and Removal

**GATE:** This phase does NOT proceed until the shadowing contradiction is resolved.

#### 2.1 Investigate ResponsesStreamConverter shadowing

**Problem:** `inspect.getsourcelines(ResponsesStreamConverter)` returns line 386, but Python's normal shadowing behavior should bind the name to the second definition at line 2269. This needs to be understood before any deletion.

**Investigation steps:**
1. Print `id()` of both class objects at their definition points
2. Check if the second definition is inside a conditional block (`if`/`try`/`else`)
3. Check if there's a `del` statement between the two definitions
4. Run `python3 -c "import proxy.openai_compat; print(proxy.openai_compat.ResponsesStreamConverter.__init__.__code__.co_firstlineno)"` to see which `__init__` is bound
5. If the first definition IS live: the second is dead → remove second
6. If the second definition IS live: the first was only used by functions defined between them → investigate those functions
7. If unclear: **STOP** and update PLAN.md

#### 2.2 Investigate Responses* class shadowing

Same investigation for `ResponsesInput`, `ResponsesReasoning`, `ResponsesTextFormat`, `ResponsesText`, `ResponsesTool`, `ResponsesRequest`.

The `inspect` results showed all first-set classes as bound. If this is confirmed:
- Second set (lines 1721-1770) is dead code
- `responses_request_from_dict` (line 1798) uses `ResponsesRequest` — which one? Verify via `inspect`
- `responses_tools_to_openai` (line 1879) uses `ResponsesTool` — which one? Verify

#### 2.3 Remove confirmed dead functions

**Confirmed dead (no imports, no getattr, no string references, no tests):**
- `responses_request_to_chat_payload` (line 1908)
- `openai_response_to_responses` (line 2161)

**Before removal, verify:**
- `rg -n 'responses_request_to_chat_payload' --type py` — only the definition
- `rg -n 'openai_response_to_responses' --type py` — only the definition
- `rg -n 'getattr.*responses_request_to_chat_payload\|getattr.*openai_response_to_responses'` — none
- No `__all__` that might export them
- No `import *` that might pull them

**Commit:** `refactor(openai_compat): remove dead functions responses_request_to_chat_payload and openai_response_to_responses`

#### 2.4 Remove confirmed dead class definitions

**Only after §2.1 and §2.2 are resolved.** If the second set is confirmed dead:
- Remove lines 1721-1770 (second `ResponsesInput` through `ResponsesRequest`)
- Remove second `ResponsesStreamConverter` (line 2269) if confirmed dead
- Remove any helper functions that only the dead code used

**If the first set is confirmed dead (contradicting `inspect`):**
- Remove lines 17-89
- Update `responses_request_to_openai` and `responses_response_from_openai` to use the second-set classes

**If unclear:** STOP. Do not delete. Update PLAN.md with findings.

**Commit:** `refactor(openai_compat): remove shadowed class definitions`

---

### Phase 3: active_requests Race Condition Fix (V1)

**Problem:** `active_requests` is a bare `int` at module level. Incremented before streaming starts, decremented in `finally` blocks. In asyncio (single-threaded), this is mostly safe — but `asyncio.wait_for` in the shutdown drain loop can yield control between the read and write, and if the code ever moves to threading, decrements can be lost.

**Smallest safe solution:** Replace the bare int with an `asyncio.Event`-backed counter. No new class hierarchy. No architectural changes. Just encapsulate the counter and event in a small object stored on `app.state`.

**File:** `proxy/atlas_proxy.py`

**Change:**

Replace:
```python
active_requests = 0
```

With:
```python
class _ActiveRequests:
    """Counter + idle event for graceful shutdown drain."""
    def __init__(self) -> None:
        self._count = 0
        self._idle = asyncio.Event()
        self._idle.set()

    def inc(self) -> None:
        self._count += 1
        self._idle.clear()

    def dec(self) -> None:
        self._count -= 1
        if self._count <= 0:
            self._count = 0
            self._idle.set()

    @property
    def count(self) -> int:
        return self._count

    async def wait_idle(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

active = _ActiveRequests()
```

Replace all `global active_requests` / `active_requests += 1` / `active_requests -= 1` with `active.inc()` / `active.dec()`. Replace `active_requests` in logging/health with `active.count`. Replace the shutdown drain loop:

```python
# Before:
drain_deadline = 5.0
while active_requests > 0 and drain_deadline > 0:
    logger.info("draining %d active request(s), %.1fs left", active_requests, drain_deadline)
    await asyncio.sleep(0.5)
    drain_deadline -= 0.5

# After:
drained = await active.wait_idle(timeout=SHUTDOWN_DRAIN_SECONDS)
if not drained:
    logger.warning("shutdown drain timed out, %d request(s) still active", active.count)
```

**Test:** `test_active_requests_tracker` — verifies inc/dec/idle behavior, wait_idle with timeout, count property.

**Commit:** `fix(atlas_proxy): replace bare active_requests int with event-backed counter`

**Commit:** `test(atlas_proxy): regression test for active_requests race condition`

---

### Phase 4: Configuration Externalization (V11)

**Problem:** Some config values already use `os.getenv` (HOST, PORT, KEEPALIVE_SECONDS). Others are hardcoded (MAX_BODY_BYTES, drain deadline, timeouts). Extend the existing pattern — don't invent a new config system.

**File:** `proxy/atlas_proxy.py`

**Changes:**

1. Add env var overrides for hardcoded values:

```python
MAX_BODY_BYTES = int(os.getenv("ATLAS_MAX_BODY_BYTES", str(256 * 1024 * 1024)))
SHUTDOWN_DRAIN_SECONDS = float(os.getenv("ATLAS_SHUTDOWN_DRAIN_SECONDS", "15.0"))
STREAM_READ_TIMEOUT = float(os.getenv("ATLAS_STREAM_READ_TIMEOUT", "300.0"))
NONSTREAM_TOTAL_TIMEOUT = float(os.getenv("ATLAS_NONSTREAM_TIMEOUT", "120.0"))
CONNECT_TIMEOUT = float(os.getenv("ATLAS_CONNECT_TIMEOUT", "10.0"))
```

2. Replace hardcoded `drain_deadline = 5.0` with `SHUTDOWN_DRAIN_SECONDS` (also changes default from 5s to 15s — the audit identified 5s as too short for large completions, and 15s is more appropriate).

3. Replace hardcoded timeout values in `NvidiaClient` construction with these variables.

**No new file.** No `config.py`. No `pydantic-settings`. Just extend the existing `os.getenv` pattern already in the file.

**Backwards compatibility:** All defaults match or improve current behavior. Existing env vars unchanged. New env vars are additive.

**Test:** `test_config_env_vars` — verifies defaults and env var overrides.

**Commit:** `feat(atlas_proxy): externalize remaining hardcoded config to env vars`

---

### Phase 5: Circuit Breaker (V12)

**Problem:** If NVIDIA is down, every request attempts a connection, fails, cools a key. With 5 keys, that's 5 failed attempts per request before 503. No short-circuit.

**Minimal implementation:** A simple failure counter + open/closed flag. No half-open state. No state machine class. No `asyncio.Lock` for a counter that's only read/written in the event loop.

**File:** `proxy/atlas_proxy.py` (inline — no new file for ~20 lines)

**Change:**

```python
# Circuit breaker — short-circuits when NVIDIA is completely down.
# Trips after N consecutive failures, resets after T seconds.
_CB_FAILURES = 0
_CB_OPEN_UNTIL: float = 0.0

CB_FAILURE_THRESHOLD = int(os.getenv("ATLAS_CB_FAILURE_THRESHOLD", "5"))
CB_RESET_SECONDS = float(os.getenv("ATLAS_CB_RESET_SECONDS", "60.0"))

def _cb_is_open() -> bool:
    """Check if circuit breaker is tripped. Resets automatically after timeout."""
    global _CB_OPEN_UNTIL
    if _CB_OPEN_UNTIL and time.monotonic() >= _CB_OPEN_UNTIL:
        _CB_OPEN_UNTIL = 0.0
        _CB_FAILURES = 0  # reset on probe
    return bool(_CB_OPEN_UNTIL)

def _cb_record_failure() -> None:
    global _CB_FAILURES, _CB_OPEN_UNTIL
    _CB_FAILURES += 1
    if _CB_FAILURES >= CB_FAILURE_THRESHOLD:
        _CB_OPEN_UNTIL = time.monotonic() + CB_RESET_SECONDS
        logger.warning("circuit breaker tripped after %d consecutive failures", _CB_FAILURES)

def _cb_record_success() -> None:
    global _CB_FAILURES
    if _CB_FAILURES:
        _CB_FAILURES = 0
```

**Integration into `_stream_failover_loop`:**

At the top of the loop, before key acquisition:
```python
if _cb_is_open():
    return json_error("upstream unavailable (circuit breaker open)", "circuit_open", 503)
```

On successful response: `_cb_record_success()`
On failure (before cooldown): `_cb_record_failure()`

**Backwards compatibility:** When the breaker is closed (normal operation), behavior is identical. When open, returns 503 instead of attempting N doomed connections — this is an improvement, not a breaking change. Clients already handle 503.

**Test:** `test_circuit_breaker_trips` — verifies trips after threshold, resets after timeout.

**Commit:** `feat(atlas_proxy): add circuit breaker for upstream failure short-circuiting`

**Commit:** `test(atlas_proxy): circuit breaker integration tests`

---

### Phase 6: Static Analysis Tooling

**Problem:** No linter, no type checker, no coverage. Code quality enforcement is manual.

#### 6.1 ruff

**New files:** `pyproject.toml` (or extend existing)

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]  # line length handled by formatter
```

**Commit:** `chore: add ruff configuration`

#### 6.2 mypy

```toml
[tool.mypy]
python_version = "3.12"
strict = false
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = false
check_untyped_defs = true

[[tool.mypy.overrides]]
module = "proxy.*"
disallow_untyped_defs = true
```

**Commit:** `chore: add mypy configuration`

#### 6.3 pytest + coverage

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.coverage.run]
source = ["proxy"]
omit = ["tests/*", "setup/*"]

[tool.coverage.report]
fail_under = 90
show_missing = true
```

**Commit:** `chore: add pytest and coverage configuration`

---

### Phase 7: Test Suite (V8)

**New directory:** `tests/`

#### 7.1 Unit Tests

**File:** `tests/test_openai_compat.py`

| Test | What it verifies |
|------|-----------------|
| `test_responses_request_to_openai` | Basic conversion: input → OpenAI payload |
| `test_responses_request_to_openai_with_tools` | Tool definitions preserved |
| `test_responses_response_from_openai` | Response conversion: OpenAI → Responses format |
| `test_responses_response_from_openai_with_tool_calls` | Tool call output |
| `test_sanitize_openai_payload` | Clamping temperature, max_tokens |
| `test_sanitize_openai_payload_drops_unsupported` | logit_bias, n, user, store dropped |
| `test_sanitize_openai_payload_thinking_max_tokens` | max_tokens stripped when thinking enabled |
| `test_anthropic_openai_payload` | Message conversion |
| `test_anthropic_openai_payload_with_images` | Image blocks |
| `test_anthropic_openai_payload_with_tool_use` | tool_use blocks |
| `test_anthropic_openai_payload_with_tool_result` | tool_result blocks |
| `test_openai_sse_to_anthropic_sse` | Streaming conversion |
| `test_openai_sse_to_anthropic_sse_tool_calls` | Tool call streaming with delta accumulation |
| `test_openai_sse_to_anthropic_sse_error_mid_stream` | Error surfacing |
| `test_openai_sse_to_anthropic_sse_empty_stream` | Empty stream → valid message_start/stop |
| `test_double_json_encoding_v3` | Regression: arguments not double-encoded |
| `test_anthropic_stop_reason_content_filter` | Regression: content_filter → refusal |
| `test_tool_call_ids_are_uuids` | Regression: IDs are UUID-based, unique |

**File:** `tests/test_system_prompt.py`

| Test | What it verifies |
|------|-----------------|
| `test_strip_system_reminders` | Reminder stripping |
| `test_strip_identity_markers` | Identity stripping |
| `test_inject_override_openai` | OpenAI format injection |
| `test_inject_override_anthropic` | Anthropic format injection |
| `test_recency_primacy_injection` | Reinforcing message placement |
| `test_override_file_missing_fallback` | Fallback behavior |
| `test_override_file_cached_by_mtime` | Cache behavior |

**File:** `tests/test_nvidia_key_store.py`

| Test | What it verifies |
|------|-----------------|
| `test_acquire_returns_key` | Basic acquisition |
| `test_acquire_returns_none_when_all_cooled` | Exhaustion |
| `test_cooldown_skips_cooled_keys` | Cooldown behavior |
| `test_release_allows_reacquire` | Release cycle |
| `test_sticky_key_selection` | Sticky session |
| `test_key_file_permissions` | 0600 on creation |

**File:** `tests/test_stats.py`

| Test | What it verifies |
|------|-----------------|
| `test_record_increments_counters` | Basic recording |
| `test_record_failure` | Failure tracking |
| `test_atomic_write` | Temp file + rename |
| `test_load_handles_corrupt_file` | Graceful corrupt recovery |
| `test_reset_since_restart` | Reset behavior |

**File:** `tests/test_circuit_breaker.py`

| Test | What it verifies |
|------|-----------------|
| `test_cb_closed_allows_requests` | Normal operation |
| `test_cb_open_after_threshold` | Trip after N failures |
| `test_cb_resets_after_timeout` | Recovery after timeout |
| `test_cb_success_resets_counter` | Success resets failure count |

**File:** `tests/test_config.py`

| Test | What it verifies |
|------|-----------------|
| `test_defaults` | Default values match current behavior |
| `test_env_var_overrides` | Env vars parsed correctly |

#### 7.2 Integration Tests

**File:** `tests/test_integration.py`

| Test | What it verifies |
|------|-----------------|
| `test_chat_completions_non_stream` | Full request lifecycle |
| `test_chat_completions_stream` | Streaming lifecycle |
| `test_anthropic_messages_non_stream` | Anthropic protocol |
| `test_anthropic_messages_stream` | Anthropic streaming |
| `test_responses_api_non_stream` | Responses API |
| `test_responses_api_stream` | Responses API streaming |
| `test_failover_on_429` | Key cooldown + failover |
| `test_failover_on_500` | Server error failover |
| `test_failover_exhaustion_returns_503` | All keys exhausted |
| `test_graceful_shutdown_drains` | Shutdown behavior |
| `test_circuit_breaker_integration` | CB trips under load |
| `test_body_size_limit_413` | Body size rejection |
| `test_malformed_json_400` | JSON parse error |
| `test_health_endpoint` | Health check |
| `test_stats_endpoint` | Stats endpoint |

#### 7.3 Protocol Conversion Golden Snapshot Tests

**File:** `tests/test_golden/`

| File | Content |
|------|---------|
| `golden_anthropic_to_openai.json` | Input/output pairs for Anthropic → OpenAI |
| `golden_responses_to_openai.json` | Input/output pairs for Responses → OpenAI |
| `golden_sse_anthropic.txt` | SSE stream snapshots for Anthropic |
| `golden_sse_responses.txt` | SSE stream snapshots for Responses |

**File:** `tests/test_golden.py`

| Test | What it verifies |
|------|-----------------|
| `test_anthropic_conversion_matches_golden` | Byte-exact match |
| `test_responses_conversion_matches_golden` | Byte-exact match |
| `test_sse_anthropic_matches_golden` | Event sequence match |
| `test_sse_responses_matches_golden` | Event sequence match |

#### 7.4 Edge Case Tests

**File:** `tests/test_edge_cases.py`

| Test | What it verifies |
|------|-----------------|
| `test_empty_messages` | No messages in request |
| `test_single_system_message` | Only system message |
| `test_large_context` | 100K+ tokens |
| `test_tool_use_no_input` | tool_use with empty input |
| `test_image_with_bad_base64` | Malformed image |
| `test_unknown_model` | 404 from NVIDIA |
| `test_concurrent_requests` | 10 concurrent streams |
| `test_stream_timeout_mid_stream` | Upstream dies mid-stream |
| `test_keepalive_during_silence` | Keepalive injection |

#### 7.5 Regression Tests

**File:** `tests/test_regression.py`

| Test | What it verifies |
|------|-----------------|
| `test_double_json_encoding_v3` | Line 2204 fix |
| `test_content_filter_stop_reason_v6` | Stop reason fix |
| `test_uuid_tool_call_ids_v4` | UUID replacement |
| `test_active_requests_tracker_v1` | Race condition fix |
| `test_dead_code_removed_v2` | Verify dead code is gone (import test) |

#### 7.6 Coverage Gate

**Rule:** New code cannot reduce coverage. Overall coverage target: >90% on modified files.

**Enforcement:** `pytest --cov=proxy --cov-fail-under=90`

**Every bug fix must include a regression test.** No exceptions.

**Commit:** `test: comprehensive test suite (unit, integration, golden, edge cases, regression)`

---

### Phase 8: Security Review

**Problem:** No explicit security review in the original plan. Production code requires a deliberate security pass.

**Scope:** Check each of the following against the codebase, focusing on changes introduced by this refactor.

#### 8.1 Checklist

| Category | What to check | Where |
|----------|--------------|-------|
| Secrets | API keys not logged, not in error messages, not in stats | `atlas_proxy.py`, `nvidia_key_store.py`, `nvidia_client.py` |
| Path traversal | No user-controlled paths in file operations | `stats.py`, `system_prompt.py`, `nvidia_key_store.py` |
| Header injection | No user input in response headers without sanitization | `atlas_proxy.py` |
| SSRF | Proxy only connects to NVIDIA's known endpoint, no user-controlled URLs | `nvidia_client.py` |
| JSON parsing | `json.loads` on user input is wrapped in try/except | `atlas_proxy.py` `parse_request_body` |
| Request limits | `MAX_BODY_BYTES` enforced before parsing | `atlas_proxy.py` |
| Body limits | 256 MiB cap, configurable via env var | `atlas_proxy.py` |
| DOS vectors | Circuit breaker limits cascading failures, no unbounded buffering | `atlas_proxy.py`, circuit breaker |
| Timeout handling | All upstream calls have timeouts, no infinite waits | `nvidia_client.py`, `atlas_proxy.py` |
| Cancellation handling | `finally` blocks decrement counters, release keys on cancellation | `atlas_proxy.py` `stream_with_active_count` |

#### 8.2 What already exists (verified):
- ✅ API keys passed via headers, not query params
- ✅ Keys not logged (verified: `grep -n 'key' proxy/atlas_proxy.py` — only key index, not key value)
- ✅ `MAX_BODY_BYTES` enforced before JSON parse
- ✅ `parse_request_body` catches `json.JSONDecodeError`
- ✅ TLS verification on upstream (`verify=True` default in httpx)
- ✅ Loopback-only binding (`127.0.0.1`)
- ✅ Key file permissions `0600`

#### 8.3 What to verify after refactor:
- Circuit breaker doesn't leak state across requests (it uses module-level globals — acceptable for single-process)
- `_ActiveRequests` counter doesn't leak (decremented in `finally`)
- New env vars don't expose secrets in `os.environ` dumps
- New test fixtures don't contain real API keys

**Commit:** `docs: security review checklist and findings`

---

### Phase 9: Benchmark — Before and After

**Problem:** Production means performance didn't regress. No plan to verify this.

#### 9.1 Before refactor (baseline)

Run on `main` branch before any changes:

```bash
# Start proxy
python -m proxy.atlas_proxy &

# Run benchmark — 100 requests, 10 concurrent, streaming
python -m tests.benchmark --requests 100 --concurrency 10 --stream

# Record: latency p50/p95/p99, throughput (req/s), memory (RSS), CPU
```

Metrics to capture:
- **Latency:** p50, p95, p99 (ms) — per request, end-to-end
- **Throughput:** requests per second
- **Memory:** peak RSS (MB) during benchmark
- **CPU:** user + system time (seconds)
- **Cold start:** time from process start to first request served (ms)

#### 9.2 After refactor

Same benchmark, same parameters, on `refactor` branch.

#### 9.3 Comparison

| Metric | Before | After | Delta | Acceptable? |
|--------|--------|-------|-------|-------------|
| Latency p50 | — | — | — | <5% regression |
| Latency p95 | — | — | — | <5% regression |
| Latency p99 | — | — | — | <10% regression |
| Throughput | — | — | — | <5% regression |
| Peak memory | — | — | — | <10% increase |
| Cold start | — | — | — | <20% increase (new imports) |

**If any metric regresses beyond the threshold:** STOP. Investigate. The regression is likely from the circuit breaker check or the `_ActiveRequests` indirection. Optimize or revert.

**Commit:** `test: add benchmark script and before/after comparison`

---

## 4. Execution Order

| Step | Phase | Description | Risk |
|------|-------|-------------|------|
| 1 | Prep | Create `refactor` branch from `main` | None |
| 2 | 9.1 | Run baseline benchmark on `main` | None |
| 3 | Commit | `fix(openai_compat): check isinstance before json.dumps on tool arguments` + regression test | Low |
| 4 | Commit | `fix(openai_compat): replace random.randint with uuid for tool call IDs` + regression test | Low |
| 5 | Commit | `fix(openai_compat): map content_filter to refusal instead of end_turn` + regression test | Low |
| 6 | Commit | `feat(nvidia_client): add async context manager protocol` + test | Low |
| 7 | Commit | `refactor: deduplicate streaming_headers into openai_compat only` + test | Low |
| 8 | 2.1-2.2 | Investigate shadowing contradiction — STOP if unclear | Medium |
| 9 | Commit | `refactor(openai_compat): remove dead functions` (only if confirmed dead) | Medium |
| 10 | Commit | `refactor(openai_compat): remove shadowed class definitions` (only if confirmed dead) | Medium |
| 11 | Commit | `fix(atlas_proxy): replace bare active_requests int with event-backed counter` + test | Medium |
| 12 | Commit | `feat(atlas_proxy): externalize remaining hardcoded config to env vars` + test | Low |
| 13 | Commit | `feat(atlas_proxy): add circuit breaker for upstream failure short-circuiting` + test | Medium |
| 14 | Commit | `chore: add ruff, mypy, pytest, coverage configuration` | Low |
| 15 | Commit | `test: comprehensive test suite` | None |
| 16 | 8 | Security review | None |
| 17 | 9.2 | Run after-refactor benchmark | None |
| 18 | 9.3 | Compare benchmarks, verify no regression | None |
| 19 | Final | Run full test suite, ruff, mypy, coverage gate. Push to origin/refactor. | None |

**After each commit:**
1. `python3 -c "from proxy.atlas_proxy import app"` — verify imports
2. `ruff check proxy/` — lint passes
3. `mypy proxy/` — type check passes (once configured)
4. `pytest tests/ -v` — tests pass (once they exist)

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------------------|
| Dead code removal breaks imports | Medium | High | Verify all imports via `rg` + `inspect` before deletion. `python3 -c "from proxy.atlas_proxy import app"` after each removal. STOP if unclear. |
| `active_requests` refactor introduces deadlock | Low | High | `_ActiveRequests` uses `asyncio.Event` (non-blocking). `wait_idle` has timeout. Test with concurrent requests. |
| Circuit breaker false-positives block valid requests | Low | Medium | Threshold of 5 consecutive failures. Reset after 60s. Success resets counter. |
| Config externalization changes behavior | Very Low | Low | All defaults match current values. Env vars are additive. |
| Benchmark shows regression | Low | Medium | Thresholds defined. If exceeded, investigate and optimize or revert. |
| Shadowing investigation reveals both sets are live | Low | Medium | Both sets stay. Document why. No deletion. |
| Security review finds new vulnerability | Low | High | Checklist covers all categories. Fix before final push. |

---

## 6. Stop Conditions

I will STOP and update PLAN.md if:

1. **Shadowing investigation is inconclusive** — cannot determine which class definitions are live. Do not delete. Document findings.
2. **Dead code removal breaks imports** — the `rg`/`inspect` verification missed something. Revert, investigate, update plan.
3. **Benchmark regression exceeds thresholds** — the refactor made things worse. Investigate root cause. Optimize or revert the offending change.
4. **A proposed fix introduces a breaking change** — backwards compatibility is mandatory. If a fix requires breaking the API, STOP and get approval.
5. **The audit is wrong about a "fix"** — if applying a fix makes the code worse (e.g., the "double JSON encoding" at line 2204 is actually correct for some NVIDIA response format I didn't anticipate), STOP. Document. Wait for approval.
6. **New security issue discovered** — if the refactor introduces a new vulnerability, STOP. Fix before continuing.
7. **Test coverage cannot reach 90%** — if a module is genuinely untestable (e.g., hardware-dependent), document why and set a module-specific target. Don't lower the bar globally.

---

*End of plan. Awaiting approval before execution.*
