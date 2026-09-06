"""Tests for the model discovery/ranking/caching layer and the switch command."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the repo is on the path
REPO_ROOT = Path("/root/atlas_proxy")
sys.path.insert(0, str(REPO_ROOT / "atlas"))

from bin.models import (
    Model,
    rank,
    filter_relevant,
    filter_free_only,
    load_cached,
    save_cache,
    PROVIDER_MENU,
)
from bin.switch import (
    _save_selected,
    _load_favourites,
    _save_favourites,
    _filter_models,
    _ctx_str,
    _short_name,
)


# ---- Model helpers ----------------------------------------------------------

def make_model(provider="openrouter", prompt=0.0, completion=0.0,
               modalities=None, tasks=None, model_id="test/model",
               context_length=0):
    return Model(
        id=model_id,
        name=model_id,
        provider=provider,
        pricing={"prompt": prompt, "completion": completion},
        modalities=["text"] if modalities is None else modalities,
        tasks=["conversational"] if tasks is None else tasks,
        context_length=context_length,
    )


class TestModel:
    def test_free_model_detects_zero_prices(self):
        m = make_model(prompt=0.0, completion=0.0)
        assert m.is_free is True

    def test_non_free_model_detects_prices(self):
        m = make_model(prompt=0.002, completion=0.008)
        assert m.is_free is False

    def test_free_only_if_both_zero(self):
        m = make_model(prompt=0.0, completion=0.001)
        assert m.is_free is False

    def test_pricing_string_to_float(self):
        """OpenRouter sends pricing as strings — Model should handle that."""
        m = Model(id="x", name="x", provider="openrouter",
                  pricing={"prompt": "0", "completion": "0"},
                  modalities=["text"], tasks=["conversational"])
        assert m.prompt_price == 0.0
        assert m.is_free is True

    def test_text_only_is_llm(self):
        m = make_model(modalities=["text"], tasks=["conversational"])
        assert m.is_llm is True
        assert m.is_text_only is True

    def test_embeddings_not_llm(self):
        m = make_model(modalities=["embeddings"])
        assert m.is_llm is False

    def test_image_modality_not_text_only(self):
        m = make_model(modalities=["text", "image"])
        assert m.is_text_only is False
        # But is_llm should still be True (text in modalities, no bad set)
        assert m.is_llm is True


class TestFilterRelevant:
    def test_drops_embeddings(self):
        models = [
            make_model(model_id="a", modalities=["embeddings"]),
            make_model(model_id="b", modalities=["text"]),
        ]
        assert [m.id for m in filter_relevant(models)] == ["b"]

    def test_drops_audio_and_speech(self):
        models = [
            make_model(model_id="a", modalities=["audio"]),
            make_model(model_id="b", modalities=["speech"]),
        ]
        assert filter_relevant(models) == []

    def test_drops_moderation(self):
        models = [make_model(model_id="a", modalities=["moderation"])]
        assert filter_relevant(models) == []

    def test_drops_rerank(self):
        models = [make_model(model_id="a", modalities=["rerank"])]
        assert filter_relevant(models) == []

    def test_keeps_text_llm(self):
        models = [make_model(model_id="a", tasks=["conversational", "coding"])]
        assert len(filter_relevant(models)) == 1

    def test_keeps_vision_capable_chat_model(self):
        """A model with text+image input is still useful for chat."""
        models = [make_model(model_id="a", modalities=["text", "image"])]
        assert len(filter_relevant(models)) == 1

    def test_empty_modalities_dropped(self):
        models = [make_model(model_id="a", modalities=[])]
        assert filter_relevant(models) == []

    def test_no_text_modality_dropped(self):
        """If 'text' is not in modalities, drop it (image-only etc)."""
        models = [make_model(model_id="a", modalities=["image"])]
        assert filter_relevant(models) == []


class TestFilterFreeOnly:
    def test_keeps_only_zero_priced(self):
        models = [
            make_model(model_id="free", prompt=0.0, completion=0.0),
            make_model(model_id="paid", prompt=0.01, completion=0.01),
        ]
        result = filter_free_only(models)
        assert [m.id for m in result] == ["free"]

    def test_handles_string_prices(self):
        m = Model(id="x", name="x", provider="openrouter",
                  pricing={"prompt": "0", "completion": "0"},
                  modalities=["text"], tasks=["conversational"])
        assert filter_free_only([m]) == [m]


class TestRank:
    def test_coding_model_ranks_above_conversational(self):
        coding = make_model(model_id="coding", tasks=["conversational", "coding"])
        chat = make_model(model_id="chat", tasks=["conversational"])
        ranked = rank([chat, coding])
        assert ranked[0].id == "coding"

    def test_agentic_ranks_high(self):
        agent = make_model(model_id="agent", tasks=["conversational", "agentic"])
        chat = make_model(model_id="chat", tasks=["conversational"])
        ranked = rank([chat, agent])
        assert ranked[0].id == "agent"

    def test_text_only_ranks_above_multimodal(self):
        text = make_model(model_id="text", modalities=["text"])
        multimodal = make_model(model_id="multi", modalities=["text", "image"])
        ranked = rank([multimodal, text])
        assert ranked[0].id == "text"

    def test_returns_new_list_does_not_mutate(self):
        models = [make_model(model_id="a"), make_model(model_id="b")]
        ranked = rank(models)
        assert ranked is not models
        assert len(ranked) == len(models)

    def test_free_bonus(self):
        # Free with same task profile as paid → free wins.
        free = make_model(model_id="free", prompt=0.0, completion=0.0,
                          tasks=["conversational", "coding", "agentic"])
        paid = make_model(model_id="paid", prompt=0.01, completion=0.03,
                          tasks=["conversational", "coding", "agentic"])
        ranked = rank([paid, free])
        assert ranked[0].id == "free"

    def test_score_hint_boosts(self):
        # Same task profile, but a high score_hint (provider popularity) wins.
        popular = Model(id="x/popular", name="x/popular", provider="openrouter",
                        pricing={"prompt": 0, "completion": 0},
                        modalities=["text"], tasks=["conversational"],
                        score_hint=0.9)
        unpopular = Model(id="x/unpop", name="x/unpop", provider="openrouter",
                          pricing={"prompt": 0, "completion": 0},
                          modalities=["text"], tasks=["conversational"],
                          score_hint=0.1)
        ranked = rank([unpopular, popular])
        assert ranked[0].id == "x/popular"

    def test_long_context_ranks_above_short(self):
        short = make_model(model_id="short", context_length=4096, tasks=["conversational"])
        long = make_model(model_id="long", context_length=1_000_000, tasks=["conversational"])
        ranked = rank([short, long])
        assert ranked[0].id == "long"


# ---- Caching ----------------------------------------------------------------

class TestCache:
    def test_load_cached_returns_none_when_no_cache(self, tmp_path):
        assert load_cached(tmp_path, "openrouter") is None

    def test_save_and_load_cached(self, tmp_path):
        models = [make_model(model_id="a"), make_model(model_id="b")]
        save_cache(tmp_path, "openrouter", models)
        result = load_cached(tmp_path, "openrouter")
        assert result is not None
        assert [m.id for m in result] == ["a", "b"]

    def test_load_cached_expired(self, tmp_path):
        models = [make_model(model_id="a")]
        save_cache(tmp_path, "openrouter", models)
        meta_path = tmp_path / ".cache" / "openrouter_models.meta.json"
        meta = json.loads(meta_path.read_text())
        meta["fetched_at"] = 0
        meta_path.write_text(json.dumps(meta))
        assert load_cached(tmp_path, "openrouter", ttl=1) is None

    def test_save_cache_creates_cache_dir(self, tmp_path):
        save_cache(tmp_path, "openrouter", [make_model()])
        assert (tmp_path / ".cache").is_dir()


# ---- Provider menu ----------------------------------------------------------

class TestProviderMenu:
    def test_three_providers(self):
        assert len(PROVIDER_MENU) == 3

    def test_provider_labels(self):
        labels = [p.label for p in PROVIDER_MENU]
        assert labels == ["OpenRouter", "NVIDIA", "Hugging Face"]

    def test_provider_keys(self):
        keys = [p.key for p in PROVIDER_MENU]
        assert keys == ["openrouter", "nvidia", "huggingface"]


# ---- Switch helpers ---------------------------------------------------------

class TestSaveSelected:
    def test_saves_provider_and_model(self, tmp_path, monkeypatch):
        # Switch module computes RUNTIME_PROVIDER_FILE at import time,
        # so we monkeypatch the constant directly.
        import bin.switch as switch_mod
        runtime_file = tmp_path / "data" / "proxy_data" / "runtime_provider.json"
        monkeypatch.setattr(switch_mod, "RUNTIME_PROVIDER_FILE", runtime_file)
        _save_selected("openrouter", "z-ai/glm-5.2:free")
        assert runtime_file.exists()
        data = json.loads(runtime_file.read_text())
        assert data["provider"] == "openrouter"
        assert data["model"] == "z-ai/glm-5.2:free"

    def test_creates_dirs_if_missing(self, tmp_path, monkeypatch):
        import bin.switch as switch_mod
        runtime_file = tmp_path / "data" / "proxy_data" / "runtime_provider.json"
        monkeypatch.setattr(switch_mod, "RUNTIME_PROVIDER_FILE", runtime_file)
        _save_selected("nvidia", "meta/llama-3.1-70b-instruct")
        assert runtime_file.exists()

    def test_preserves_existing_keys(self, tmp_path, monkeypatch):
        """If runtime_provider.json already has fields, we don't blow them away."""
        import bin.switch as switch_mod
        runtime_file = tmp_path / "data" / "proxy_data" / "runtime_provider.json"
        runtime_file.parent.mkdir(parents=True, exist_ok=True)
        runtime_file.write_text(json.dumps({
            "provider": "old-provider",
            "model": "old-model",
            "custom_field": "preserved",
        }))
        monkeypatch.setattr(switch_mod, "RUNTIME_PROVIDER_FILE", runtime_file)
        _save_selected("openrouter", "new-model")
        data = json.loads(runtime_file.read_text())
        assert data["provider"] == "openrouter"
        assert data["model"] == "new-model"
        assert data["custom_field"] == "preserved"


class TestFavourites:
    def test_load_empty_when_no_file(self, tmp_path, monkeypatch):
        import bin.switch as switch_mod
        monkeypatch.setattr(switch_mod, "PREFS_FILE", tmp_path / ".atlas_preferences.json")
        assert _load_favourites() == set()

    def test_save_and_load(self, tmp_path, monkeypatch):
        import bin.switch as switch_mod
        monkeypatch.setattr(switch_mod, "PREFS_FILE", tmp_path / ".atlas_preferences.json")
        _save_favourites({"a", "b"})
        assert _load_favourites() == {"a", "b"}

    def test_toggle_removes(self, tmp_path, monkeypatch):
        import bin.switch as switch_mod
        monkeypatch.setattr(switch_mod, "PREFS_FILE", tmp_path / ".atlas_preferences.json")
        _save_favourites({"a"})
        favs = _load_favourites()
        favs.discard("a")
        _save_favourites(favs)
        assert _load_favourites() == set()


# ---- _filter_models (the picker uses this) ---------------------------------

class TestFilterModels:
    def test_no_query_returns_all(self):
        models = [make_model(model_id="a"), make_model(model_id="b")]
        assert _filter_models(models, "") == models

    def test_substring_match(self):
        models = [
            make_model(model_id="openai/gpt-4"),
            make_model(model_id="z-ai/glm-5.2:free"),
        ]
        result = _filter_models(models, "glm")
        assert [m.id for m in result] == ["z-ai/glm-5.2:free"]

    def test_task_match(self):
        models = [
            make_model(model_id="a", tasks=["conversational", "coding"]),
            make_model(model_id="b", tasks=["conversational"]),
        ]
        result = _filter_models(models, "coding")
        assert [m.id for m in result] == ["a"]


class TestCtxStr:
    def test_unknown_context(self):
        m = make_model()
        assert _ctx_str(m) == "—"

    def test_small_context(self):
        m = make_model(context_length=4096)
        assert _ctx_str(m) == "4k"

    def test_million_context(self):
        m = make_model(context_length=1_048_000)
        assert _ctx_str(m) == "1M"


class TestShortName:
    def test_strips_provider_prefix(self):
        assert _short_name("z-ai/glm-5.2:free") == "glm-5.2:free"

    def test_no_prefix(self):
        assert _short_name("gpt-4") == "gpt-4"
