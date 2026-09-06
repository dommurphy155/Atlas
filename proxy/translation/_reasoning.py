"""Reasoning / thinking-block normalisation and emission.

Three helpers, all pure:

  collect_reasoning_hints:    read a request body and gather every known
                              reasoning/thinking alias into a single
                              internal "hints" dict.
  emit_reasoning_for_openai_chat:  turn hints into (reasoning, effort)
                              suitable for the OpenAI chat completions
                              request body.
  emit_thinking_for_anthropic:  turn hints into (thinking, output_config)
                              suitable for the Anthropic messages request
                              body.

Colocating these avoids a tangle of cross-imports between the two
body-prep functions. Both body-prep helpers depend on this module.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ._helpers import extract_budget, normalize_effort

# Re-bind for legacy import path
_normalize_effort = normalize_effort
_extract_budget = extract_budget


def collect_reasoning_hints(body: Dict[str, Any]) -> Dict[str, Any]:
    """Gather every known reasoning/thinking alias into a single internal form.

    Returns a dict that may contain:
      mode: "none" | "effort" | "budget" | "adaptive"
      effort: str
      budget_tokens: int
      exclude: bool
      display: str (Anthropic)
    """
    hints: Dict[str, Any] = {}

    # 1. Explicit reasoning object (OpenRouter / OpenAI style)
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        if "effort" in reasoning:
            e = normalize_effort(reasoning["effort"])
            if e:
                hints["effort"] = e
                hints["mode"] = "effort" if e != "none" else "none"
        if "max_tokens" in reasoning:
            b = extract_budget(reasoning)
            if b:
                hints["budget_tokens"] = b
                hints.setdefault("mode", "budget")
        if reasoning.get("exclude") is True or reasoning.get("exclude") == "true":
            hints["exclude"] = True
        if reasoning.get("enabled") is False:
            hints["mode"] = "none"
        elif reasoning.get("enabled") is True and "mode" not in hints:
            hints["mode"] = "adaptive"

    # 2. Top-level reasoning_effort (OpenAI / OpenRouter shorthand)
    if "reasoning_effort" in body:
        e = normalize_effort(body["reasoning_effort"])
        if e:
            hints["effort"] = e
            hints["mode"] = "effort" if e != "none" else "none"

    # 3. Anthropic thinking object
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        ttype = thinking.get("type")
        if ttype == "disabled":
            hints["mode"] = "none"
        elif ttype == "enabled":
            b = extract_budget(thinking)
            if b:
                hints["budget_tokens"] = max(b, 1024)  # Anthropic minimum
                hints["mode"] = "budget"
            else:
                hints["mode"] = "adaptive"
        elif ttype == "adaptive":
            hints["mode"] = "adaptive"
        if "display" in thinking:
            hints["display"] = thinking["display"]
    elif isinstance(thinking, bool):
        hints["mode"] = "adaptive" if thinking else "none"
    elif isinstance(thinking, str):
        t = thinking.lower()
        if t in ("enabled", "on", "true"):
            hints["mode"] = "adaptive"
        elif t in ("disabled", "off", "false", "none"):
            hints["mode"] = "none"

    # 4. Top-level budget aliases
    for key in ("budget_tokens", "thinking_budget", "reasoning_budget"):
        if key in body:
            b = extract_budget(body[key])
            if b:
                hints["budget_tokens"] = max(b, 1024)
                hints.setdefault("mode", "budget")

    # 5. include_reasoning (deprecated OpenRouter alias -> exclude=False)
    if body.get("include_reasoning") is True:
        hints["exclude"] = False
    elif body.get("include_reasoning") is False:
        hints["exclude"] = True

    # 6. verbosity -> effort mapping for Anthropic (OpenRouter documents this)
    verbosity = body.get("verbosity")
    if verbosity and "effort" not in hints:
        e = normalize_effort(verbosity)
        if e and e != "none":
            hints["effort"] = e
            hints.setdefault("mode", "effort")

    # 7. output_config.effort (already Anthropic-shaped)
    oc = body.get("output_config")
    if isinstance(oc, dict) and "effort" in oc and "effort" not in hints:
        e = normalize_effort(oc["effort"])
        if e:
            hints["effort"] = e
            hints.setdefault("mode", "effort")

    return hints


def emit_reasoning_for_openai_chat(
    hints: Dict[str, Any],
) -> Tuple[Optional[Dict], Optional[str]]:
    """Produce (reasoning object, reasoning_effort shorthand) for chat path."""
    if not hints or hints.get("mode") == "none":
        # Explicitly disable when requested
        if hints.get("mode") == "none":
            return {"effort": "none"}, "none"
        return None, None

    reasoning: Dict[str, Any] = {}
    effort = hints.get("effort")
    budget = hints.get("budget_tokens")

    if effort:
        reasoning["effort"] = effort
    elif budget:
        reasoning["max_tokens"] = budget
    else:
        # adaptive / enabled without specifics -> high effort
        reasoning["effort"] = "high"

    if hints.get("exclude") is True:
        reasoning["exclude"] = True

    # Prefer the shorthand when only effort is present (cleaner for OpenRouter)
    if list(reasoning.keys()) == ["effort"]:
        return reasoning, reasoning["effort"]
    return reasoning, None


def emit_thinking_for_anthropic(
    hints: Dict[str, Any], max_tokens: Optional[int] = None
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Produce (thinking object, output_config) for Anthropic messages path."""
    if not hints:
        return None, None

    mode = hints.get("mode")
    if mode == "none":
        return {"type": "disabled"}, None

    thinking: Dict[str, Any] = {}
    output_config: Optional[Dict] = None

    budget = hints.get("budget_tokens")
    effort = hints.get("effort")

    if mode == "budget" and budget:
        # Clamp budget relative to max_tokens when possible
        if max_tokens and budget >= max_tokens:
            budget = max(1024, max_tokens - 1)
        thinking = {"type": "enabled", "budget_tokens": max(budget, 1024)}
    else:
        # Prefer adaptive for modern models; effort goes into output_config
        thinking = {"type": "adaptive"}
        if effort and effort != "none":
            output_config = {"effort": effort}
        elif mode == "adaptive" and not effort:
            pass  # pure adaptive
        else:
            # fallback: if we only had a vague "enabled", use high effort
            output_config = {"effort": effort or "high"}

    if hints.get("display"):
        thinking["display"] = hints["display"]

    return thinking, output_config


# Legacy underscore aliases for callers that imported these under their
# private names from the inlined translation.py.
_collect_reasoning_hints = collect_reasoning_hints
_emit_reasoning_for_openai_chat = emit_reasoning_for_openai_chat
_emit_thinking_for_anthropic = emit_thinking_for_anthropic
