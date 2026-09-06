"""Tool and tool_choice conversion between Anthropic and OpenAI shapes.

Pure conversion logic — no I/O, no side effects, no upstream calls.
Both directions live here because they share the same edge-case table
(custom/freeform tools, server-tool passthrough, cache_control stashing).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# OpenRouter server-tool type names that pass through unchanged in both
# directions. Anything starting with "openrouter:" is left alone by
# anthropic_tools_to_openai and openai_tools_to_anthropic.
_OPENROUTER_SERVER_TOOLS = (
    "openrouter:web_search", "openrouter:datetime",
    "openrouter:image_generation", "openrouter:web_fetch",
    "openrouter:apply_patch", "openrouter:shell", "openrouter:fusion",
)


def anthropic_tools_to_openai(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """Anthropic {name, description, input_schema} -> OpenAI function tools."""
    if not tools:
        return tools
    out: List[Dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Already OpenAI chat-shaped: {"type": "function", "function": {...}}
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(t)
            continue
        # OpenRouter server tools passthrough (never have a "function" key)
        if t.get("type") in _OPENROUTER_SERVER_TOOLS:
            out.append(t)
            continue
        # Codex custom/freeform tools are not OpenAI function tools.
        # OpenRouter needs a JSON-callable representation, so adapt every
        # custom/freeform tool generically. The original tool type is preserved
        # request-locally by routes.py and restored on the Responses API side.
        #
        # A freeform/custom grammar has no JSON parameter schema, therefore the
        # adapter exposes one string field, `input`, containing the raw tool
        # input. This is an internal transport representation only.
        if t.get("type") in ("freeform", "custom"):
            name = t.get("name")
            if not name:
                continue
            original_desc = t.get("description") or ""
            desc = (
                original_desc
                + ("\n\n" if original_desc else "")
                + "This is a custom/freeform tool. "
                  "Arguments MUST be a JSON object with a single string field "
                  "`input` containing the complete raw tool input."
            )
            out.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Complete raw input for the custom/freeform tool.",
                            },
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            })
            continue
        # Hosted custom/freeform tool forms may arrive as a bare tool type with
        # no `name`. Preserve the known hosted form as a generic custom tool.
        if t.get("type") == "apply_patch" and "name" not in t:
            out.append({
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": (
                        "Custom/freeform tool. Arguments MUST be a JSON object "
                        "with a single string field `input` containing the complete "
                        "raw tool input."
                    ),
                    "strict": False,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                        },
                        "required": ["input"],
                        "additionalProperties": False,
                    },
                },
            })
            continue
        # Anthropic custom / server tool passthrough when it already has type
        if "type" in t and t["type"] not in (None, "function") and "name" not in t:
            out.append(t)
            continue
        name = t.get("name") or ""
        desc = t.get("description")
        schema = t.get("input_schema") or t.get("parameters") or {
            "type": "object",
            "properties": {},
        }
        fn: Dict[str, Any] = {"name": name, "parameters": schema}
        if desc is not None:
            fn["description"] = desc
        if "strict" in t:
            fn["strict"] = t["strict"]
        # Preserve cache_control for round-trip back to Anthropic format.
        # OpenAI chat/completions doesn't support it natively; we stash it
        # on the function dict under a private key.
        if "cache_control" in t:
            fn["_anthropic_cache_control"] = t["cache_control"]
        out.append({"type": "function", "function": fn})
    return out


def openai_tools_to_anthropic(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """OpenAI function tools -> Anthropic {name, description, input_schema}."""
    if not tools:
        return tools
    out: List[Dict] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Already Anthropic-shaped
        if "input_schema" in t and "name" in t and "function" not in t:
            out.append(t)
            continue
        # OpenRouter server tools – leave as-is; Anthropic path may not understand them
        ttype = t.get("type")
        if isinstance(ttype, str) and ttype.startswith("openrouter:"):
            out.append(t)
            continue
        fn = t.get("function") or t
        name = fn.get("name") or t.get("name") or ""
        desc = fn.get("description")
        params = fn.get("parameters") or fn.get("input_schema") or {
            "type": "object",
            "properties": {},
        }
        tool: Dict[str, Any] = {"name": name, "input_schema": params}
        if desc is not None:
            tool["description"] = desc
        if "strict" in fn:
            tool["strict"] = fn["strict"]
        # Re-emit cache_control preserved from anthropic_tools_to_openai
        if "_anthropic_cache_control" in fn:
            tool["cache_control"] = fn["_anthropic_cache_control"]
        out.append(tool)
    return out


def convert_tool_choice_anthropic_to_openai(tc: Any) -> Any:
    if tc is None:
        return None
    if isinstance(tc, str):
        return {"auto": "auto", "any": "required", "none": "none", "required": "required"}.get(
            tc, tc
        )
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "auto":
            return "auto"
        if t == "any":
            return "required"
        if t == "none":
            return "none"
        if t == "tool" and "name" in tc:
            return {"type": "function", "function": {"name": tc["name"]}}
        if t == "function":
            return tc
        # Preserve disable_parallel_tool_use → parallel_tool_calls is handled elsewhere
    return tc


def convert_tool_choice_openai_to_anthropic(
    tc: Any, *, parallel_tool_calls: Optional[bool] = None
) -> Any:
    if tc is None:
        return None
    disable_parallel = parallel_tool_calls is False

    if tc == "auto":
        out: Dict[str, Any] = {"type": "auto"}
        if disable_parallel:
            out["disable_parallel_tool_use"] = True
        return out
    if tc == "required":
        out = {"type": "any"}
        if disable_parallel:
            out["disable_parallel_tool_use"] = True
        return out
    if tc == "none":
        return {"type": "none"}
    if isinstance(tc, dict):
        if tc.get("type") == "function":
            name = (tc.get("function") or {}).get("name")
            if name:
                out = {"type": "tool", "name": name}
                if disable_parallel:
                    out["disable_parallel_tool_use"] = True
                return out
        if tc.get("type") in ("auto", "any", "none", "tool"):
            out = dict(tc)
            if disable_parallel and "disable_parallel_tool_use" not in out:
                out["disable_parallel_tool_use"] = True
            return out
        if "type" in tc:
            return tc
    return tc
