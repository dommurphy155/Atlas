"""
Thinking parser - Ported from Ollama's thinking/parser.go with improvements.

This module implements a state machine for parsing thinking/reasoning blocks
from streaming model responses. It handles:
- Opening/closing tag detection
- Partial tag handling
- Whitespace management
- Content vs thinking separation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


class ThinkingState(Enum):
    """States for the thinking parser state machine."""
    LOOKING_FOR_OPENING = auto()
    THINKING_STARTED_EATING_WHITESPACE = auto()
    THINKING = auto()
    THINKING_DONE_EATING_WHITESPACE = auto()
    THINKING_DONE = auto()


@dataclass
class ThinkingParser:
    """
    Stateful parser for extracting thinking/reasoning blocks from streaming responses.

    Similar to Ollama's thinking parser, this maintains state across chunks to properly
    handle thinking blocks with their opening and closing tags.
    """

    opening_tag: str = "<thinking>"
    closing_tag: str = "</thinking>"
    state: ThinkingState = ThinkingState.LOOKING_FOR_OPENING
    buffer: str = ""

    def __init__(self, opening_tag: str = "<thinking>", closing_tag: str = "</thinking>"):
        self.opening_tag = opening_tag
        self.closing_tag = closing_tag

    def add_content(self, content: str) -> Tuple[str, str]:
        """
        Process input content to extract thinking and regular content.

        Args:
            content: New content to process

        Returns:
            Tuple of (thinking_content, remaining_content)
        """
        self.buffer += content

        thinking_parts = []
        remaining_parts = []

        keep_looping = True
        while keep_looping:
            thinking, remaining, keep_looping = self._eat()
            if thinking:
                thinking_parts.append(thinking)
            if remaining:
                remaining_parts.append(remaining)

        return "".join(thinking_parts), "".join(remaining_parts)

    def _eat(self) -> Tuple[str, str, bool]:
        """
        Process the current state and buffer.

        Returns:
            Tuple of (thinking_content, remaining_content, should_continue)
        """
        if self.state == ThinkingState.LOOKING_FOR_OPENING:
            return self._handle_looking_for_opening()

        elif self.state == ThinkingState.THINKING_STARTED_EATING_WHITESPACE:
            return self._handle_thinking_started_eating_whitespace()

        elif self.state == ThinkingState.THINKING:
            return self._handle_thinking()

        elif self.state == ThinkingState.THINKING_DONE_EATING_WHITESPACE:
            return self._handle_thinking_done_eating_whitespace()

        elif self.state == ThinkingState.THINKING_DONE:
            return self._handle_thinking_done()

        return "", "", False

    def _handle_looking_for_opening(self) -> Tuple[str, str, bool]:
        """Handle the LOOKING_FOR_OPENING state."""
        # Find non-whitespace content
        content = self.buffer.lstrip()

        if content.startswith(self.opening_tag):
            # Found opening tag
            # Get everything after the opening tag
            after_tag = content[len(self.opening_tag):]
            # Strip leading whitespace after tag
            after_tag = after_tag.lstrip()

            self.buffer = after_tag

            if not after_tag:
                # No content after tag yet
                self.state = ThinkingState.THINKING_STARTED_EATING_WHITESPACE
            else:
                self.state = ThinkingState.THINKING

            return "", "", True

        elif self.opening_tag.startswith(content):
            # Partial opening tag - need more data
            # Keep everything that could be part of the tag
            partial_match = content
            self.buffer = partial_match
            return "", "", False

        elif not content:
            # Only whitespace so far
            self.buffer = ""
            return "", "", False

        else:
            # No thinking tag found - content is regular output
            self.state = ThinkingState.THINKING_DONE
            result = self.buffer
            self.buffer = ""
            return "", result, False

    def _handle_thinking_started_eating_whitespace(self) -> Tuple[str, str, bool]:
        """Handle the THINKING_STARTED_EATING_WHITESPACE state."""
        content = self.buffer.lstrip()

        if not content:
            # Still only whitespace
            self.buffer = ""
            return "", "", False

        # Got non-whitespace - now in thinking state
        self.buffer = content
        self.state = ThinkingState.THINKING
        return "", "", True

    def _handle_thinking(self) -> Tuple[str, str, bool]:
        """Handle the THINKING state - looking for closing tag."""
        content = self.buffer

        # Check for closing tag
        if self.closing_tag in content:
            # Found closing tag
            parts = content.split(self.closing_tag, 1)
            thinking = parts[0]
            remaining = parts[1] if len(parts) > 1 else ""

            # Strip leading whitespace from remaining
            remaining = remaining.lstrip()

            self.buffer = remaining

            if not remaining:
                self.state = ThinkingState.THINKING_DONE_EATING_WHITESPACE
            else:
                self.state = ThinkingState.THINKING_DONE

            return thinking, "", True

        # Check for partial closing tag
        overlap = self._overlap(content, self.closing_tag)
        if overlap > 0:
            # Partial closing tag - need more data
            thinking = content[:-overlap]
            self.buffer = content[-overlap:]
            return thinking, "", False

        # No closing tag found - all is thinking
        self.buffer = ""
        return content, "", False

    def _handle_thinking_done_eating_whitespace(self) -> Tuple[str, str, bool]:
        """Handle the THINKING_DONE_EATING_WHITESPACE state."""
        content = self.buffer.lstrip()

        if not content:
            # Still only whitespace
            self.buffer = ""
            return "", "", False

        # Got non-whitespace - we're done
        self.state = ThinkingState.THINKING_DONE
        result = self.buffer
        self.buffer = ""
        return "", result, False

    def _handle_thinking_done(self) -> Tuple[str, str, bool]:
        """Handle the THINKING_DONE state."""
        result = self.buffer
        self.buffer = ""
        return "", result, False

    def _overlap(self, s: str, delim: str) -> int:
        """
        Find the longest overlap between the end of s and the start of delim.

        This is used for detecting partial tags in streaming data.
        """
        max_check = min(len(delim), len(s))
        for i in range(max_check, 0, -1):
            if s.endswith(delim[:i]):
                return i
        return 0

    def has_thinking(self) -> bool:
        """Check if parser has processed any thinking content."""
        return self.state in (
            ThinkingState.THINKING,
            ThinkingState.THINKING_STARTED_EATING_WHITESPACE,
            ThinkingState.THINKING_DONE_EATING_WHITESPACE,
            ThinkingState.THINKING_DONE,
        )

    def is_complete(self) -> bool:
        """Check if parsing is complete."""
        return self.state == ThinkingState.THINKING_DONE


def create_thinking_parser(
    opening_tag: str = "<thinking>",
    closing_tag: str = "</thinking>"
) -> ThinkingParser:
    """Create a new thinking parser with custom tags."""
    return ThinkingParser(opening_tag=opening_tag, closing_tag=closing_tag)
