"""Main FastAPI server."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from src.config import get_config
from src.logging import logger, stats
from src.protocols.base import ProtocolType, get_adapter
from src.providers.registry import get_registry, register_provider
from src.providers.base import ProviderConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize providers
    config = get_config()
    # TODO: Initialize providers from config
    yield
    # Cleanup
    reg = get_registry()
    await reg.close_all()


app = FastAPI(title="Atlas Proxy v2", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "atlas-proxy-v2"}


@app.get("/stats")
async def stats_endpoint():
    return stats.get()


@app.get("/v1/models")
async def models():
    # Return available models
    return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI chat completions endpoint."""
    body = await request.json()
    adapter = get_adapter(ProtocolType.OPENAI)

    # Parse request
    req = adapter.parse_request(body)

    # TODO: Route to provider
    return JSONResponse({"error": "Not implemented"})


@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic messages endpoint."""
    body = await request.json()
    adapter = get_adapter(ProtocolType.ANTHROPIC)

    # Parse request
    req = adapter.parse_request(body)

    # TODO: Route to provider
    return JSONResponse({"error": "Not implemented"})


def main():
    import uvicorn
    config = get_config()
    uvicorn.run(
        "src.server:app",
        host=config.server.host,
        port=config.server.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
