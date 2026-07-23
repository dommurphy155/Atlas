# Atlas Proxy v2

Industrial-grade AI protocol proxy combining the best ideas from Atlas and Ollama.

## Features

- **Multi-Protocol Support**: OpenAI, Anthropic, Claude Code
- **Provider Agnostic**: NVIDIA, Anthropic, OpenAI, and more
- **Tool Calling**: Full state-machine parser for tool calls
- **Thinking Blocks**: Reasoning model support
- **Streaming**: SSE with keepalive
- **Key Rotation**: Automatic API key rotation with cooldown

## Architecture

```
src/
├── core/          # Canonical types, errors
├── protocols/     # OpenAI, Anthropic adapters
├── providers/     # Provider abstraction
├── parsing/       # Tool & thinking parsers
├── streaming/     # SSE streaming engine
├── config/        # Configuration
├── logging/       # Logging & metrics
└── server.py      # FastAPI server
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment
export ATLAS_NVIDIA_MODEL=deepseek-ai/deepseek-v4-pro

# Run server
python -m src.server
```

## Configuration

See `src/config/__init__.py` for all options.

## License

MIT
