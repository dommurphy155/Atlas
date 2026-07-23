"""
Tool call parser - Ported from Ollama's tools/tools.go with improvements.

This module implements a state machine for parsing tool calls from streaming
model responses. It handles:
- Tag detection (custom tags, {, [, etc.)
- Tool name extraction
- JSON argument parsing
- Partial tag handling
- Multi-tool calls
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from src.core.types import ToolCall, ToolDefinition


class ToolParserState(Enum):
    """States for the tool parser state machine."""
    LOOKING_FOR_TAG = auto()
    TOOL_CALLING = auto()
    DONE = auto()


@dataclass
class ToolParser:
    """
    Stateful parser for extracting tool calls from streaming responses.

    Similar to Ollama's Parser, this maintains state across chunks to properly
    sequence tool call detection, argument parsing, and content extraction.
    """

    tools: list[ToolDefinition]
    tag: str = "{"
    state: ToolParserState = ToolParserState.LOOKING_FOR_TAG
    buffer: str = ""
    tool_call_index: int = 0

    def __init__(self, tools: list[ToolDefinition], tag: str = "{"):
        self.tools = tools
        self.tag = tag

    def get_buffer(self) -> str:
        """Get current buffer contents."""
        return self.buffer

    def add(self, content: str) -> tuple[list[ToolCall], str]:
        """
        Process input string to parse tool calls and content.

        Returns:
            tuple: (list of tool calls found, remaining content to output)
        """
        if self.state == ToolParserState.DONE:
            return [], content

        self.buffer += content

        if self.state == ToolParserState.LOOKING_FOR_TAG:
            content = self._handle_looking_for_tag()

            if not content:
                return [], ""

        # Process any complete tool calls
        tool_calls = []
        while True:
            tool_call = self._parse_tool_call()
            if tool_call is None:
                break
            tool_calls.append(tool_call)

        if self._done():
            self.state = ToolParserState.DONE
            remaining = self.buffer
            self.buffer = ""
            return tool_calls, remaining

        return tool_calls, content if content else ""

    def _handle_looking_for_tag(self) -> str:
        """Handle the LOOKING_FOR_TAG state."""
        tag_index, found = self._find_tag()

        if tag_index == -1:
            # No tag found, output everything
            content = self.buffer
            self.buffer = ""
            return content

        # Tag found - output content before tag
        content = self.buffer[:tag_index]
        self.buffer = self.buffer[tag_index:]

        # For { or [ tags, check if we have content before them
        if self.tag in ("{", "["):
            if content.strip():
                # There's content before the tag - we're done
                self.state = ToolParserState.DONE
                return content + self.buffer

        if not found:
            # Partial tag, need more data
            return ""

        self.state = ToolParserState.TOOL_CALLING
        return content

    def _find_tag(self) -> tuple[int, bool]:
        """
        Find the tool call tag in the buffer.

        Returns:
            tuple: (index of tag, whether tag is complete)
        """
        # First check for complete tag
        tag_index = self.buffer.find(self.tag)
        if tag_index != -1:
            return tag_index, True

        # Check for partial tag overlap
        max_check = min(len(self.buffer), len(self.tag))
        for i in range(max_check, 0, -1):
            if self.buffer.endswith(self.tag[:i]):
                # Partial match at end
                return len(self.buffer) - i, False

        return -1, False

    def _parse_tool_call(self) -> Optional[ToolCall]:
        """Parse the next complete tool call from buffer."""
        # Find the tool
        tool, end_pos = self._find_tool()
        if tool is None:
            return None

        # Find arguments
        args, args_end = self._find_arguments(tool.name)
        if args is None:
            return None

        # Create tool call
        tc = ToolCall(
            id=f"call_{self.tool_call_index}",
            name=tool.name,
            arguments=args,
            index=self.tool_call_index,
        )
        self.tool_call_index += 1

        # Remove parsed content from buffer
        self.buffer = self.buffer[args_end:]
        self.state = ToolParserState.LOOKING_FOR_TAG

        return tc

    def _find_tool(self) -> tuple[Optional[ToolDefinition], int]:
        """Find the first matching tool in the buffer."""
        if not self.buffer:
            return None, 0

        # Check if buffer ends with partial tool name
        # This prevents matching "get" when seeing "get_weather"
        longest_name = ""
        for tool in self.tools:
            if len(tool.name) > len(longest_name):
                longest_name = tool.name

        # Check for partial tool name at end
        for i in range(1, min(len(self.buffer), len(longest_name)) + 1):
            tail = self.buffer[-i:]
            for tool in self.tools:
                name = tool.name
                if len(tail) < len(name) and name.startswith(tail):
                    # Partial tool name match
                    return None, 0

        # Find first occurrence of any tool name
        best_match: Optional[ToolDefinition] = None
        best_start = -1
        best_end = -1

        for tool in self.tools:
            name = tool.name
            pos = self.buffer.find(name)
            if pos == -1:
                continue

            # Skip if we have a better match
            if best_start != -1:
                if pos > best_start:
                    continue
                if pos == best_start and len(name) <= len(best_match.name if best_match else ""):
                    continue

            best_match = tool
            best_start = pos
            best_end = pos + len(name)

        if best_match is not None:
            return best_match, best_end

        return None, 0

    def _find_arguments(self, tool_name: str) -> tuple[Optional[dict], int]:
        """
        Find JSON arguments for the tool call.

        Returns:
            tuple: (parsed arguments dict, end position in buffer)
        """
        if not self.buffer:
            return None, 0

        # Find the opening brace
        start = -1
        brace_count = 0
        in_string = False
        escaped = False

        for i, char in enumerate(self.buffer):
            if escaped:
                escaped = False
                continue

            if char == '\\':
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start != -1:
                    # Found complete JSON
                    json_str = self.buffer[start:i+1]
                    try:
                        args = json.loads(json_str)
                        if isinstance(args, dict):
                            return args, i + 1
                    except json.JSONDecodeError:
                        # Invalid JSON, continue looking
                        start = -1
                        brace_count = 0

        return None, 0

    def _done(self) -> bool:
        """Check if parsing is done."""
        if self.tag in ("{", "["):
            # For these tags, check for closing bracket
            brace_count = 0
            in_string = False
            escaped = False

            for char in self.buffer:
                if escaped:
                    escaped = False
                    continue

                if char == '\\':
                    escaped = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if char == self.tag:
                    brace_count += 1
                elif char == ('}' if self.tag == "{" else ']'):
                    brace_count -= 1
                    if brace_count == 0:
                        return True

        return False

    def content(self) -> str:
        """Get any remaining content that should be sent to user."""
        if self.tool_call_index > 0:
            return ""

        if self.tag in ("{", "["):
            return self.buffer

        return ""


def create_tool_parser(tools: list[ToolDefinition], tag: str = "{") -> ToolParser:
    """Create a new tool parser with given tools and tag."""
    return ToolParser(tools=tools, tag=tag)
