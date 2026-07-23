"""Protocol adapter tests."""
import pytest
from src.protocols.base import ProtocolType, get_adapter
import src.core.types as types


class TestOpenAIAdapter:
    """Tests for OpenAI adapter."""

    def test_parse_request_basic(self):
        """Test basic request parsing."""
        adapter = get_adapter(ProtocolType.OPENAI)

        body = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
        }

        req = adapter.parse_request(body)

        assert req.model == "gpt-4"
        assert len(req.messages) == 2
        # Check content is string for simple messages
        assert req.messages[0].content == "You are helpful."
        assert req.messages[1].content == "Hello"
        # Options should have the values
        assert req.options.temperature == 0.7
        assert req.options.max_tokens == 100

    def test_parse_request_with_tools(self):
        """Test request parsing with tools."""
        adapter = get_adapter(ProtocolType.OPENAI)

        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"}
                            },
                            "required": ["location"],
                        },
                    },
                }
            ],
        }

        req = adapter.parse_request(body)

        assert req.options.tools is not None
        assert len(req.options.tools) == 1
        assert req.options.tools[0].name == "get_weather"

    def test_format_response(self):
        """Test response formatting."""
        adapter = get_adapter(ProtocolType.OPENAI)

        response = types.Response(
            id="chatcmpl-123",
            model="gpt-4",
            content=[types.ContentBlock(type="text", text="Hello!")],
            usage=types.Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason=types.FinishReason.STOP,
        )

        result = adapter.format_response(response)

        assert result["id"] == "chatcmpl-123"
        assert result["model"] == "gpt-4"
        # Content might be list or string
        content = result["choices"][0]["message"]["content"]
        assert "Hello!" in str(content)
        assert result["usage"]["prompt_tokens"] == 10


class TestAnthropicAdapter:
    """Tests for Anthropic adapter."""

    def test_parse_request_basic(self):
        """Test basic Anthropic request parsing."""
        adapter = get_adapter(ProtocolType.ANTHROPIC)

        body = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"},
            ],
        }

        req = adapter.parse_request(body)

        assert req.model == "claude-3-opus-20240229"
        assert req.options.max_tokens == 1024
        assert len(req.messages) == 1

    def test_parse_system_message(self):
        """Test system message handling."""
        adapter = get_adapter(ProtocolType.ANTHROPIC)

        body = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 1024,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }

        req = adapter.parse_request(body)

        # System should be in the request.system field
        assert req.system == "You are a helpful assistant."

    def test_format_response(self):
        """Test Anthropic response formatting."""
        adapter = get_adapter(ProtocolType.ANTHROPIC)

        response = types.Response(
            id="msg_123",
            model="claude-3-opus",
            content=[types.ContentBlock(type="text", text="Hello!")],
            usage=types.Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            finish_reason=types.FinishReason.STOP,
        )

        result = adapter.format_response(response)

        assert result["id"] == "msg_123"
        assert result["type"] == "message"
        # Content handling varies
        assert "content" in result
