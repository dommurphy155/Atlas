"""prettylog — production-grade request lifecycle logging for the Atlas proxy.

PARALLEL IMPLEMENTATION (logging-only). Nothing here changes proxy, request,
provider, key-pool, retry, routing, or streaming behaviour. This module is
imported alongside proxy.logger until cutover; it never touches request state.

Architecture
------------
• Root-handler formatter replacement  — cleans up EVERY existing log line
  (short rid, no empty fields, coloured level) with zero call-site changes.
• RequestTrace — per-request lifecycle recorder (start → upstream → TTFT →
  chunks → usage → finish). Keyed by the full 16-hex rid; displays 6 hex chars.
• instrument_aiter() — transparent async-iterator wrapper that counts chunks /
  measures TTFT without altering yielded bytes, order, or timing semantics.
• keys_health() — compact/degraded key-pool rendering for the health loop.
• Thresholds via env: ATLAS_TTFT_GOOD_MS / _BAD_MS, ATLAS_TOTAL_GOOD_S /
  _BAD_S, ATLAS_TPS_GOOD / _TPS_BAD. Colour: ATLAS_LOG_COLOR=auto|always|never.

Concurrency: traces mutate only inside the running event loop; the registry
uses a threading.Lock because uvicorn workers / background tasks may touch it
from different tasks. Emission goes through the stdlib logging machinery
(single handler, GIL-protected queueing) — no extra threads, no blocking I/O
beyond what the existing stdout StreamHandler already does.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration (read once at import)
# ---------------------------------------------------------------------------
COLOR_MODE = os.environ.get("ATLAS_LOG_COLOR", "auto").strip().lower()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

TTFT_GOOD_MS = float(os.environ.get("ATLAS_TTFT_GOOD_MS", 1500))
TTFT_BAD_MS = float(os.environ.get("ATLAS_TTFT_BAD_MS", 4000))
TOTAL_GOOD_S = float(os.environ.get("ATLAS_TOTAL_GOOD_S", 10))
TOTAL_BAD_S = float(os.environ.get("ATLAS_TOTAL_BAD_S", 30))
TPS_GOOD = float(os.environ.get("ATLAS_TPS_GOOD", 100))   # tokens/sec
TPS_BAD = float(os.environ.get("ATLAS_TPS_BAD", 30))

VERBOSE = os.environ.get("ATLAS_LOG_VERBOSE", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
def _use_color() -> bool:
    if COLOR_MODE == "always":
        return True
    if COLOR_MODE == "never":
        return False
    return sys.stdout.isatty() or bool(os.environ.get("JOURNAL_STREAM"))


USE_COLOR = _use_color()

C_RESET = "\033[0m" if USE_COLOR else ""
C_DIM = "\033[2m" if USE_COLOR else ""
C_GREEN = "\033[38;5;46m" if USE_COLOR else ""
C_ORANGE = "\033[38;5;208m" if USE_COLOR else ""
C_RED = "\033[38;5;196m" if USE_COLOR else ""
C_BLUE = "\033[38;5;75m" if USE_COLOR else ""
C_GREY = "\033[38;5;245m" if USE_COLOR else ""
C_WHITE = "\033[97m" if USE_COLOR else ""

DOT_GOOD = f"{C_GREEN}●{C_RESET}"
DOT_WARN = f"{C_ORANGE}●{C_RESET}"
DOT_BAD = f"{C_RED}●{C_RESET}"
DOT_INFO = f"{C_BLUE}●{C_RESET}"

# ---------------------------------------------------------------------------
# Custom levels
# ---------------------------------------------------------------------------
SUCCESS = 25  # between INFO(20) and WARNING(30)
logging.addLevelName(SUCCESS, "SUCCESS")

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def dur(seconds: Optional[float]) -> str:
    """Human duration: 842ms / 7.34s / 12.47s."""
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 100:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def dur_ms(milliseconds: Optional[float]) -> str:
    if milliseconds is None:
        return "—"
    return dur(milliseconds / 1000.0)


def tok(n: Optional[int]) -> str:
    return f"{n:,}" if n is not None else "—"


def latency_class(ms: float) -> str:
    """GOOD / MEDIUM / BAD for a millisecond latency against TTFT thresholds."""
    if ms < TTFT_GOOD_MS:
        return "good"
    if ms < TTFT_BAD_MS:
        return "medium"
    return "bad"


def span_class(seconds: float) -> str:
    if seconds < TOTAL_GOOD_S:
        return "good"
    if seconds < TOTAL_BAD_S:
        return "medium"
    return "bad"


def tps_class(tps: float) -> str:
    if tps >= TPS_GOOD:
        return "good"
    if tps >= TPS_BAD:
        return "medium"
    return "bad"


_CLASS_MARK = {"good": ("✓", C_GREEN), "medium": ("⚠", C_ORANGE), "bad": ("✗", C_RED)}
_CLASS_DOT = {"good": DOT_GOOD, "medium": DOT_WARN, "bad": DOT_BAD}


def mark_for(cls: str) -> Tuple[str, str]:
    return _CLASS_MARK.get(cls, ("·", ""))


def dot_for(cls: str) -> str:
    return _CLASS_DOT.get(cls, DOT_INFO)


def paint(cls: str, text: str) -> str:
    _, colour = _CLASS_MARK.get(cls, ("", ""))
    return f"{colour}{text}{C_RESET}" if colour else text


def status_mark(status: int) -> Tuple[str, str]:
    """(glyph, colour-class) for an HTTP status."""
    if status < 300:
        return "✓", "good"
    if status < 500:
        return "⚠", "medium"
    return "✗", "bad"


_PROVIDER_SHORT = {
    "openai": "OPENAI",
    "anthropic": "ANTHROPIC",
    "huggingface": "HF",
    "openrouter": "OPENROUTER",
}


def provider_tag(provider: str) -> str:
    return _PROVIDER_SHORT.get((provider or "").lower(), (provider or "?").upper())


def short_rid(rid: str) -> str:
    return (rid or "")[:6]


# ---------------------------------------------------------------------------
# Root formatter — cleans up ALL existing log output with no call-site edits
# ---------------------------------------------------------------------------
_RE_RID = re.compile(r"\breq=([0-9a-f]{6})[0-9a-f]{6,10}\b")
_RE_EMPTY_PAYLOAD_PATH = re.compile(r"\s*payload_path=(?=\s|$)")
# Redact API-key material that legacy call sites may embed (keypool retire logs).
_RE_KEY = re.compile(r"\b((?:hf|sk|sk-or-v1|github)[-_][A-Za-z0-9]{3})[A-Za-z0-9_-]{3,}\b")

_LEVEL_STYLE = {
    "DEBUG": (C_GREY, "DEBUG"),
    "INFO": ("", "INFO"),
    "SUCCESS": (C_GREEN, "SUCCESS"),
    "WARNING": (C_ORANGE, "WARN"),
    "ERROR": (C_RED, "ERROR"),
    "CRITICAL": (C_RED, "FATAL"),
}


class PrettyFormatter(logging.Formatter):
    """Compact terminal-first formatter.

    HH:MM:SS  LEVEL  message
      • level token padded to 7, coloured
      • req=<16hex> compacted to 6 hex
      • empty `payload_path=` fields removed
      • exceptions rendered as a short tail (full trace only at DEBUG)
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        colour, label = _LEVEL_STYLE.get(record.levelname, ("", record.levelname))
        try:
            msg = record.getMessage()
        except Exception:
            msg = record.msg
        msg = _RE_KEY.sub(r"\1…", _RE_EMPTY_PAYLOAD_PATH.sub("", _RE_RID.sub(r"req=\1", msg)))
        lvl = f"{colour}{label:<7}{C_RESET}" if colour else f"{label:<7}"
        line = f"{C_DIM}{ts}{C_RESET}  {lvl} {msg}"
        if record.exc_info:
            if record.levelno >= logging.ERROR or LOG_LEVEL == "DEBUG":
                tail = self.formatException(record.exc_info).splitlines()[-3:]
                line += "\n" + "\n".join(f"{C_GREY}{l}{C_RESET}" for l in tail)
        return line


def attach(formatter_cls: type = PrettyFormatter) -> None:
    """Swap the formatter onto the existing root handler(s).

    Safe to call repeatedly. Does NOT add handlers or alter levels/routes —
    output destination stays exactly what proxy.logger already configured.
    """
    root = logging.getLogger()
    for h in root.handlers:
        h.setFormatter(formatter_cls())


# ---------------------------------------------------------------------------
# File mirror — tees every log line into a rotating plain-text file
# ---------------------------------------------------------------------------
LOG_FILE = Path(
    os.environ.get(
        "ATLAS_LOG_FILE",
        str(Path(__file__).resolve().parent / "logs" / "atlas-proxy.log"),
    )
)
LOG_FILE_MAX_BYTES = int(os.environ.get("ATLAS_LOG_FILE_MAX_BYTES", str(10 * 1024 * 1024)))
LOG_FILE_BACKUPS = int(os.environ.get("ATLAS_LOG_FILE_BACKUPS", "3"))


class _PlainFormatter(logging.Formatter):
    """ANSI-free twin of PrettyFormatter for file output."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        try:
            msg = record.getMessage()
        except Exception:
            msg = record.msg
        msg = _RE_KEY.sub(r"\1…", _RE_EMPTY_PAYLOAD_PATH.sub("", _RE_RID.sub(r"req=\1", msg)))
        line = f"{ts}  {record.levelname:<7} {msg}"
        if record.exc_info and (record.levelno >= logging.ERROR or LOG_LEVEL == "DEBUG"):
            tail = self.formatException(record.exc_info).splitlines()[-3:]
            line += "\n" + "\n".join(tail)
        return line


def mirror_file(path: Optional[Path] = None) -> bool:
    """Attach a rotating plain-text file handler alongside stdout.

    Adds one destination; does not touch existing handlers, levels or routes.
    Safe to call repeatedly. Returns True when the handler is active.
    """
    target = Path(path) if path else LOG_FILE
    root = logging.getLogger()
    want = os.path.abspath(str(target))
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and h.baseFilename == want:
            return True  # already mirroring to this file
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            target,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUPS,
        )
        fh.setFormatter(_PlainFormatter())
        root.addHandler(fh)
        return True
    except OSError:
        return False  # unwritable path — journald still carries everything


# ---------------------------------------------------------------------------
# Request lifecycle tracing
# ---------------------------------------------------------------------------
class RequestTrace:
    """Per-request telemetry recorder. Purely observational."""

    __slots__ = (
        "rid", "provider", "model", "endpoint", "stream", "key_idx",
        "payload_path", "t_start", "t_sent", "t_upstream", "t_first_byte",
        "chunks", "bytes_out", "prompt_tokens", "completion_tokens",
        "total_tokens", "_finished", "_lock",
    )

    def __init__(self, rid: str) -> None:
        self.rid = rid
        self.provider = ""
        self.model = ""
        self.endpoint = ""
        self.stream = False
        self.key_idx: Optional[int] = None
        self.payload_path = ""
        self.t_start = time.perf_counter()
        self.t_sent: Optional[float] = None
        self.t_upstream: Optional[float] = None
        self.t_first_byte: Optional[float] = None
        self.chunks = 0
        self.bytes_out = 0
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self._finished = False
        self._lock = threading.Lock()

    # -- phases -------------------------------------------------------------
    def start(
        self,
        provider: str,
        model: Any,
        endpoint: str,
        stream: bool,
        payload_path: str = "",
    ) -> None:
        self.provider = provider or ""
        self.model = str(model) if model is not None else "?"
        self.endpoint = endpoint or ""
        self.stream = bool(stream)
        self.payload_path = payload_path or ""
        tag = provider_tag(self.provider)
        bits = [
            f"{C_BLUE}▶{C_RESET} {short_rid(self.rid)}",
            tag,
            self.model,
            "STREAM" if self.stream else "",
        ]
        if self.payload_path:
            bits.append(f"{C_GREY}payload={os.path.basename(self.payload_path)}{C_RESET}")
        _emit(logging.INFO, " ".join(b for b in bits if b))

    def sent(self, key_idx: Optional[int] = None) -> None:
        """Request bytes handed to the OS socket (start of upstream phase)."""
        with self._lock:
            if key_idx is not None:
                self.key_idx = key_idx
            if self.t_sent is None:
                self.t_sent = time.perf_counter()

    def upstream(self, key_idx: int, status: int) -> None:
        """Upstream accepted the request (response headers received)."""
        with self._lock:
            self.key_idx = key_idx
            # headers-in marks the end of the connect phase; keep the earlier
            # sent() timestamp when available so upstream_s measures transfer,
            # not just connection setup.
            if self.t_upstream is None:
                self.t_upstream = time.perf_counter()

    def observe_chunk(self, data: Any = b"") -> None:
        """Called per streamed frame/chunk. Never alters the chunk itself."""
        with self._lock:
            self.chunks += 1
            if isinstance(data, (bytes, bytearray)):
                self.bytes_out += len(data)
            if self.t_first_byte is None:
                self.t_first_byte = time.perf_counter()

    def usage(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

    @property
    def ttft_s(self) -> Optional[float]:
        if self.t_first_byte is None:
            return None
        return self.t_first_byte - self.t_start

    @property
    def upstream_s(self) -> Optional[float]:
        """Connect + transfer time: request-sent → last byte/first-token."""
        ref = self.t_first_byte or (time.perf_counter() if not self._finished else None)
        base = self.t_sent or self.t_upstream
        if ref is None or base is None:
            return None
        return ref - base

    @property
    def total_s(self) -> float:
        return time.perf_counter() - self.t_start

    # -- completion ----------------------------------------------------------
    def finish(
        self,
        status: int,
        *,
        error: str = "",
        phase: str = "",
    ) -> None:
        if self._finished:
            return
        with self._lock:
            self._finished = True
        glyph, cls = status_mark(status)
        rid = short_rid(self.rid)
        total = self.total_s
        ttft = self.ttft_s
        up = self.upstream_s

        head = f"{paint(cls, glyph)} {rid} {paint(cls, str(status))} {provider_tag(self.provider)}/{self.model}"

        # throughput when we genuinely have output tokens + stream timing
        tps: Optional[float] = None
        out_tok = self.completion_tokens
        if out_tok and self.stream and ttft is not None and total > ttft:
            tps = out_tok / max(total - ttft, 1e-6)

        metrics = []
        if ttft is not None:
            metrics.append(("ttft", dur(ttft), latency_class(ttft * 1000)))
        if up is not None:
            metrics.append(("upstream", dur(up), span_class(up)))

        bad_total = span_class(total) == "bad"
        medium_total = span_class(total) == "medium"

        if self.stream:
            if self.chunks:
                metrics.append(("chunks", str(self.chunks), ""))
            if self.bytes_out:
                metrics.append(("bytes", tok(self.bytes_out), ""))
        else:
            metrics.append(("duration", dur(total), span_class(total)))

        parts = [" ".join(f"{k}={paint(c, v)}" if c else f"{k}={v}" for k, v, c in metrics)]
        if self.completion_tokens is not None:
            parts.append(f"out={tok(self.completion_tokens)}tok")
        elif self.total_tokens is not None:
            parts.append(f"tokens={tok(self.total_tokens)}")
        elif not self.stream:
            parts.append(f"{C_GREY}usage=unavailable{C_RESET}")
        if tps is not None:
            parts.append(f"{paint(tps_class(tps), f'{tps:.0f}tok/s')}")
        if error:
            parts.append(f"{C_RED}error={error[:120]}{C_RESET}")
        if phase:
            parts.append(f"{C_GREY}phase={phase}{C_RESET}")

        expand = bool(error or bad_total or VERBOSE or LOG_LEVEL == "DEBUG")
        if expand:
            body = "\n".join(
                f"          {k:<10}{paint(c, v) if c else v}" for k, v, c in metrics
            )
            usage_lines = self._usage_block(out_tok, tps, status_ok=status < 400)
            _emit(
                logging.ERROR if error else (logging.WARNING if bad_total else logging.INFO),
                f"{head}\n{body}\n{usage_lines}",
            )
        else:
            _emit(SUCCESS if status < 400 else logging.WARNING, f"{head}  " + "  ".join(parts))

    def fail(
        self,
        *,
        status: int,
        provider: str = "",
        model: str = "",
        phase: str,
        error: str,
    ) -> None:
        """Structured error block (connection failures, timeouts, upstream 5xx)."""
        if self._finished:
            return
        with self._lock:
            self._finished = True
        glyph, cls = status_mark(status)
        rid = short_rid(self.rid)
        rows = [
            ("status", str(status), cls),
            ("provider", provider_tag(provider or self.provider), ""),
            ("model", model or self.model, ""),
            ("phase", phase, ""),
            ("duration", dur(self.total_s), span_class(self.total_s)),
            ("error", error[:200], "bad"),
        ]
        body = "\n".join(
            f"          {k:<10}{paint(c, v) if c else v}" for k, v, c in rows
        )
        _emit(logging.ERROR, f"{C_RED}✗{C_RESET} REQ {rid}\n{body}")

    def _usage_block(self, out_tok: Optional[int], tps: Optional[float], status_ok: bool) -> str:
        if self.prompt_tokens is None and self.completion_tokens is None and self.total_tokens is None:
            return "          usage     unavailable"
        lines = []
        if self.prompt_tokens is not None:
            lines.append(f"input       {tok(self.prompt_tokens)}")
        if self.completion_tokens is not None:
            lines.append(f"output      {tok(self.completion_tokens)}")
        if self.total_tokens is not None:
            lines.append(f"total       {tok(self.total_tokens)}")
        if tps is not None:
            lines.append(f"throughput  {tps:.0f} tok/s")
        return "TOKENS\n" + "\n".join(f"          {l}" for l in lines)


# Registry -------------------------------------------------------------------
_traces: Dict[str, RequestTrace] = {}
_registry_lock = threading.Lock()


def trace(rid: str) -> RequestTrace:
    """Get-or-create the trace for a request id (full 16-hex rid)."""
    with _registry_lock:
        t = _traces.get(rid)
        if t is None:
            t = RequestTrace(rid)
            # bound memory: keep at most the most recent 512 requests
            if len(_traces) > 512:
                for k in sorted(_traces)[: len(_traces) - 512]:
                    _traces.pop(k, None)
            _traces[rid] = t
        return t


def drop_trace(rid: str) -> None:
    with _registry_lock:
        _traces.pop(rid, None)


# ---------------------------------------------------------------------------
# Transparent stream instrumentation
# ---------------------------------------------------------------------------
async def instrument_aiter(
    source: AsyncIterator[bytes],
    tr: RequestTrace,
) -> AsyncIterator[bytes]:
    """Pass-through async iterator: counts frames and TTFT, yields untouched.

    Yield order, framing, back-pressure and error propagation are identical
    to iterating `source` directly — this wrapper adds observation only.
    """
    try:
        async for chunk in source:
            tr.observe_chunk(chunk)
            yield chunk
    finally:
        pass


# ---------------------------------------------------------------------------
# Key-pool health rendering
# ---------------------------------------------------------------------------
DEGRADED_RATIO = 0.9  # healthy/total below this ⇒ ⚠ DEGRADED


def keys_health(stats: Dict[str, Any], *, logger: Optional[logging.Logger] = None) -> None:
    total = stats.get("total", 0)
    healthy = stats.get("healthy", 0)
    cooling = stats.get("cooling", 0)
    suspended = stats.get("suspended", 0)
    inflight = stats.get("in_flight", 0)

    ratio_ok = total == 0 or (healthy / total) >= DEGRADED_RATIO
    clean = ratio_ok and suspended == 0

    if clean:
        dot = DOT_GOOD if total and healthy == total else DOT_WARN
        _emit(
            logging.INFO,
            f"KEYS {dot} total={total} healthy={healthy} cooling={cooling}"
            f" suspended={suspended} inflight={inflight}",
        )
        return

    dot = DOT_WARN if healthy > 0 else DOT_BAD
    body = "\n".join([
        f"          total       {total}",
        f"          healthy     {healthy}  {'🟢' if ratio_ok else '🔴'}",
        f"          cooling     {cooling}",
        f"          suspended   {suspended}",
        f"          in-flight   {inflight}",
    ])
    _emit(logging.WARNING, f"KEYS ⚠ DEGRADED {dot}\n{body}")


# ---------------------------------------------------------------------------
# Internal emission
# ---------------------------------------------------------------------------
_req_logger = logging.getLogger("atlas.req")


def _emit(level: int, message: str) -> None:
    if _req_logger.isEnabledFor(level):
        _req_logger.log(level, message)


__all__ = [
    "SUCCESS",
    "attach",
    "PrettyFormatter",
    "RequestTrace",
    "trace",
    "drop_trace",
    "instrument_aiter",
    "keys_health",
    "dur",
    "dur_ms",
    "tok",
    "latency_class",
    "span_class",
    "tps_class",
    "status_mark",
    "provider_tag",
    "short_rid",
]
