"""Atlas model discovery, ranking and caching layer.

Reusable abstraction over the three upstream model catalogues:
OpenRouter, NVIDIA, HuggingFace.  All three are PUBLIC APIs — no
auth required.  Each provider has a parser, a discovery entry point,
and ranking is shared via `rank()` + `filter_relevant()`.

Public surface:
    Model          — dataclass holding a single model's metadata
    rank()         — rank models for coding/reasoning/agentic workloads
    filter_relevant() — drop embeddings/audio/speech/moderation/rerank
    openrouter_models()  — async, returns list[Model] for OpenRouter
    nvidia_models()      — async, returns list[Model] for NVIDIA
    huggingface_models() — async, returns list[Model] for HF
    PROVIDER_MENU  — list of ProviderMenu entries for the picker
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import httpx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Model:
    """A single model returned by a provider catalogue."""
    id: str
    name: str
    provider: str                              # "openrouter" | "nvidia" | "huggingface"
    pricing: dict = field(default_factory=dict)   # raw pricing dict (strings → float)
    context_length: int = 0
    modalities: list[str] = field(default_factory=list)  # ["text"], ["text","image"], etc.
    tasks: list[str] = field(default_factory=list)       # ["conversational"], ["coding"], etc.
    architecture: str = ""
    benchmarks: dict = field(default_factory=dict)
    score_hint: float = 0.0                    # provider-specific signal (popularity, leaderboard rank)
    raw: dict = field(default_factory=dict)

    @property
    def prompt_price(self) -> float:
        return _to_float(self.pricing.get("prompt"))

    @property
    def completion_price(self) -> float:
        return _to_float(self.pricing.get("completion"))

    @property
    def is_free(self) -> bool:
        """Verifies both input AND output price are literally $0 (not just ':free' slug)."""
        return self.prompt_price == 0.0 and self.completion_price == 0.0

    @property
    def is_text_only(self) -> bool:
        return set(self.modalities) <= {"text"}

    @property
    def is_llm(self) -> bool:
        """Genuine text-generation LLM — not embeddings, image, audio, moderation, speech, rerank."""
        return "text" in self.modalities and not (
            set(self.modalities) & {"embeddings", "audio", "speech", "moderation", "rerank"}
        )


def _to_float(v) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

CACHE_DIR_NAME = ".cache"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def _cache_path(repo_root: Path, provider: str) -> Path:
    return repo_root / CACHE_DIR_NAME / f"{provider}_models.json"


def _cache_meta_path(repo_root: Path, provider: str) -> Path:
    return repo_root / CACHE_DIR_NAME / f"{provider}_models.meta.json"


def load_cached(repo_root: Path, provider: str, ttl: int = CACHE_TTL_SECONDS) -> list[Model] | None:
    path = _cache_path(repo_root, provider)
    meta_path = _cache_meta_path(repo_root, provider)
    if not path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if time.time() - meta.get("fetched_at", 0) > ttl:
            return None
        return [Model(**m) for m in json.loads(path.read_text())]
    except Exception:
        return None


def save_cache(repo_root: Path, provider: str, models: list[Model]) -> None:
    cache_dir = repo_root / CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(repo_root, provider)
    meta_path = _cache_meta_path(repo_root, provider)
    path.write_text(json.dumps([m.__dict__ for m in models], indent=2))
    meta_path.write_text(json.dumps({"fetched_at": int(time.time()), "provider": provider}, indent=2))


# ---------------------------------------------------------------------------
# Ranking + filtering
# ---------------------------------------------------------------------------

# Static workload weights for ranking (best-first).
_RANK_WEIGHTS = {
    "modalities_text_only": 20,
    "task_conversational": 15,
    "task_coding": 40,
    "task_agentic": 35,
    "task_reasoning": 30,
    "context_length": 0.0005,
    "free_bonus": 30,             # tie-breaker: free models slightly preferred
    "score_hint": 50.0,           # provider signal (leaderboard rank) — scales by value
}


def rank(models: list[Model]) -> list[Model]:
    """Rank models for coding/reasoning/agentic workloads (best first)."""

    def score(m: Model) -> float:
        s = 0.0
        if m.is_text_only:
            s += _RANK_WEIGHTS["modalities_text_only"]
        for t in m.tasks:
            s += _RANK_WEIGHTS.get(f"task_{t}", 0)
        s += m.context_length * _RANK_WEIGHTS["context_length"]
        if m.is_free:
            s += _RANK_WEIGHTS["free_bonus"]
        s += m.score_hint * _RANK_WEIGHTS["score_hint"]
        return s

    return sorted(models, key=score, reverse=True)


def filter_relevant(models: list[Model]) -> list[Model]:
    """Drop models that are clearly irrelevant for a coding agent:
    embeddings / audio / speech / moderation / rerank / transcription.

    Keep text LLMs (including ones that also handle image INPUT, as long
    as the primary output modality is text — e.g. vision-capable chat models).
    """
    BAD = {"embeddings", "audio", "speech", "moderation", "rerank", "transcription"}
    result = []
    for m in models:
        if not m.modalities:
            continue
        if BAD & set(m.modalities):
            continue
        if "text" not in m.modalities:
            continue
        result.append(m)
    return result


def filter_free_only(models: Iterable[Model]) -> list[Model]:
    """Filter to models whose prompt AND completion price are exactly $0."""
    return [m for m in models if m.is_free]


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Sort options that put the best models first.
OPENROUTER_SORTS = [
    "agentic-high-to-low",
    "coding-high-to-low",
    "intelligence-high-to-low",
    "top-weekly",
    "most-popular",
]


async def _fetch_openrouter_raw(sort: str, limit: int = 200) -> list[dict]:
    """Fetch a single page from OpenRouter sorted by `sort`."""
    url = f"{OPENROUTER_URL}/models?output_modalities=text&sort={sort}&limit={limit}"
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json().get("data", [])


async def openrouter_models(repo_root: Path, ttl: int = CACHE_TTL_SECONDS,
                             free_only: bool = True) -> list[Model]:
    """Fetch OpenRouter models — public, no key.

    Strategy: pull 200 models with each sort, merge by id, take the best
    `score_hint` (popularity-weighted). Then filter to text LLMs, optionally
    to free-only, and rank.
    """
    cached = load_cached(repo_root, "openrouter", ttl)
    if cached is not None:
        return _finalize_openrouter(cached, free_only=free_only)

    by_id: dict[str, Model] = {}
    for sort in OPENROUTER_SORTS:
        try:
            rows = await _fetch_openrouter_raw(sort)
        except httpx.HTTPError:
            continue
        for rank_idx, item in enumerate(rows):
            mid = item.get("id")
            if not mid:
                continue
            # Lower rank index = better.  Convert to a positive hint:
            # best model gets 1.0, worst gets 0.0
            hint = max(0.0, 1.0 - (rank_idx / max(len(rows), 1)))
            existing = by_id.get(mid)
            if existing is None or hint > existing.score_hint:
                by_id[mid] = _parse_openrouter(item, score_hint=hint)

    models = list(by_id.values())
    models = filter_relevant(models)
    if free_only:
        models = filter_free_only(models)
    models = rank(models)

    save_cache(repo_root, "openrouter", models)
    return models


def _finalize_openrouter(models: list[Model], free_only: bool) -> list[Model]:
    if free_only:
        models = filter_free_only(models)
    return rank(models)


def _parse_openrouter(item: dict, score_hint: float = 0.0) -> Model:
    arch = item.get("architecture", {}) or {}
    # OpenRouter nests modalities inside architecture.input_modalities
    modalities = list(arch.get("input_modalities", []) or [])
    if not modalities:
        # Fall back: parse from architecture.modality string ("text+image+file->text")
        mod_str = arch.get("modality", "")
        if mod_str:
            lhs = mod_str.split("->")[0]
            modalities = [m.strip() for m in lhs.split("+") if m.strip()]

    return Model(
        id=item.get("id", ""),
        name=item.get("name", item.get("id", "")),
        provider="openrouter",
        pricing=item.get("pricing", {}) or {},
        context_length=int(item.get("context_length", 0) or 0),
        modalities=modalities,
        tasks=_infer_or_tasks(item),
        architecture=arch.get("modality", arch.get("architecture", "")),
        benchmarks=item.get("benchmarks", {}) or {},
        score_hint=score_hint,
        raw=item,
    )


def _infer_or_tasks(item: dict) -> list[str]:
    tasks = set()
    blob = " ".join([
        str(item.get("description", "")),
        str(item.get("name", "")),
        str(item.get("id", "")),
        str(item.get("canonical_slug", "")),
    ]).lower()
    if any(k in blob for k in ("coding", "code", "programming", "developer", "codex")):
        tasks.add("coding")
    if any(k in blob for k in ("agentic", "agent", "tool use", "tool calling")):
        tasks.add("agentic")
    if any(k in blob for k in ("reasoning", "reason", "deep", "math", "logic", "thinking")):
        tasks.add("reasoning")
    if "instruct" in blob or "chat" in blob:
        tasks.add("conversational")
    if not tasks:
        tasks.add("conversational")
    return sorted(tasks)


# ---------------------------------------------------------------------------
# NVIDIA
# ---------------------------------------------------------------------------

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"


async def _fetch_nvidia_raw() -> list[dict]:
    """Fetch NVIDIA NIM models — public, no key required."""
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(f"{NVIDIA_URL}/models")
        r.raise_for_status()
        return r.json().get("data", [])


async def nvidia_models(repo_root: Path, ttl: int = CACHE_TTL_SECONDS) -> list[Model]:
    """Fetch NVIDIA NIM models.  No auth required.

    NVIDIA's catalogue returns minimal {id, object, created, owned_by} entries —
    no pricing, no metadata, no modality info.  We infer model capabilities
    from the model id (e.g. 'llama-3.1-70b-instruct' → LLM; 'nvidia/nv-embed' → embed).
    """
    cached = load_cached(repo_root, "nvidia", ttl)
    if cached is not None:
        return rank(cached)

    try:
        rows = await _fetch_nvidia_raw()
    except httpx.HTTPError:
        return []

    models = [_parse_nvidia(row) for row in rows]
    models = filter_relevant(models)
    models = rank(models)
    save_cache(repo_root, "nvidia", models)
    return models


_NVIDIA_BAD_SLUGS = {
    "embed", "embedding", "rerank", "asr", "tts", "speech", "parakeet",
    "canary", "whisper", "vila", "cosmos", "guard", "nvclip", "clip",
    "mistral-nemo", "nemotron-parse", "qwen2.5-coder",  # variants
}


def _parse_nvidia(item: dict) -> Model:
    mid = item.get("id", "")
    name = mid.split("/")[-1] if "/" in mid else mid
    lower = mid.lower()
    # Infer capabilities from the slug.
    modalities: list[str] = []
    tasks: list[str] = []
    is_llm = True
    if any(b in lower for b in _NVIDIA_BAD_SLUGS):
        is_llm = False
    if "embed" in lower:
        modalities = ["embeddings"]
    elif "rerank" in lower:
        modalities = ["rerank"]
    elif "asr" in lower or "parakeet" in lower or "canary" in lower or "whisper" in lower:
        modalities = ["audio"]
        is_llm = False
    elif "tts" in lower or "speech" in lower:
        modalities = ["speech"]
        is_llm = False
    elif "vision" in lower or "vila" in lower or "cosmos" in lower:
        modalities = ["text", "image"]
    else:
        modalities = ["text"]

    if is_llm:
        if "coder" in lower or "code" in lower:
            tasks.append("coding")
        if "reasoning" in lower or "r1" in lower.split("/")[-1] or "thinking" in lower:
            tasks.append("reasoning")
        if "instruct" in lower or "chat" in lower or is_llm:
            tasks.append("conversational")
        if not tasks:
            tasks = ["conversational"]
    return Model(
        id=mid,
        name=name,
        provider="nvidia",
        pricing={},
        context_length=0,
        modalities=modalities,
        tasks=tasks,
        architecture="",
        benchmarks={},
        score_hint=0.0,
        raw=item,
    )


# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

HF_URL = "https://huggingface.co/api"
# Curated list of inference providers to query (each returns its own catalogue).
HF_PROVIDERS = ["deepinfra", "together", "fireworks-ai", "replicate", "nebius"]


async def _fetch_hf_raw(provider: str, limit: int = 200) -> list[dict]:
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(f"{HF_URL}/models?inference_provider={provider}&limit={limit}")
        r.raise_for_status()
        return r.json()


async def huggingface_models(repo_root: Path, ttl: int = CACHE_TTL_SECONDS,
                              providers: list[str] | None = None) -> list[Model]:
    """Fetch HuggingFace models across curated inference providers.

    Public API, no key needed for the public catalogue.
    """
    cached = load_cached(repo_root, "huggingface", ttl)
    if cached is not None:
        return rank(cached)

    if providers is None:
        providers = HF_PROVIDERS

    by_id: dict[str, Model] = {}
    for provider in providers:
        try:
            rows = await _fetch_hf_raw(provider)
        except httpx.HTTPError:
            continue
        for rank_idx, item in enumerate(rows):
            mid = item.get("id")
            if not mid:
                continue
            hint = max(0.0, 1.0 - (rank_idx / max(len(rows), 1)))
            existing = by_id.get(mid)
            if existing is None or hint > existing.score_hint:
                parsed = _parse_hf(item, provider)
                # Replace score_hint with provider signal
                object.__setattr__(parsed, "score_hint", hint)
                by_id[mid] = parsed

    models = list(by_id.values())
    models = filter_relevant(models)
    models = rank(models)
    save_cache(repo_root, "huggingface", models)
    return models


_HF_BAD_PIPELINES = {
    "feature-extraction", "sentence-similarity", "text-to-image",
    "text-to-video", "image-to-text", "image-to-image", "image-to-video",
    "audio-to-audio", "audio-classification", "text-to-speech",
    "automatic-speech-recognition", "text-classification",
    "token-classification", "translation", "summarization",
    "fill-mask", "zero-shot-classification", "reinforcement-learning",
    "robotics", "tabular-classification", "tabular-regression",
    "voice-activity-detection", "depth-estimation", "object-detection",
    "image-classification", "image-segmentation", "keypoint-detection",
}


def _parse_hf(item: dict, provider: str) -> Model:
    mid = item.get("id", "")
    name = mid.split("/")[-1] if "/" in mid else mid
    pipeline = item.get("pipeline_tag", "") or ""
    tags = item.get("tags", []) or []

    # Determine modalities from pipeline_tag + tags.
    if pipeline in _HF_BAD_PIPELINES:
        if "image" in pipeline:
            modalities = ["text", "image"]   # multimodal
        else:
            modalities = ["text"]            # filtered out by filter_relevant
    elif pipeline in ("text-generation", "conversational", "text2text-generation"):
        if "image" in tags or "multimodal" in tags:
            modalities = ["text", "image"]
        else:
            modalities = ["text"]
    else:
        modalities = ["text"]

    tasks = []
    blob = " ".join([mid, name, pipeline] + tags).lower()
    if "code" in blob or "coder" in blob:
        tasks.append("coding")
    if "instruct" in blob or "chat" in blob or pipeline == "conversational":
        tasks.append("conversational")
    if "reason" in blob or "thinking" in blob or "r1" in blob.split("/")[-1]:
        tasks.append("reasoning")
    if "agent" in blob or "tool" in blob:
        tasks.append("agentic")
    if not tasks:
        tasks = ["conversational"]

    return Model(
        id=mid,
        name=name,
        provider=provider,
        pricing={},
        context_length=0,
        modalities=modalities,
        tasks=tasks,
        architecture="",
        benchmarks={},
        score_hint=0.0,
        raw=item,
    )


# ---------------------------------------------------------------------------
# Provider menu
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderMenu:
    key: str           # "openrouter" | "nvidia" | "huggingface"
    label: str         # "OpenRouter"
    description: str


PROVIDER_MENU = [
    ProviderMenu("openrouter", "OpenRouter", "400+ models, free tier, agentic leaderboard"),
    ProviderMenu("nvidia", "NVIDIA", "NIM inference endpoints, no key required"),
    ProviderMenu("huggingface", "Hugging Face", "Inference Providers — frontier models, free tier"),
]
