"""Main FastAPI server."""

import uuid
from typing import Optional
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager

from src.config import get_config
from src.logging import logger, stats
from src.protocols.base import ProtocolType, get_adapter
from src.providers.registry import get_registry, register_provider
from src.providers.base import ProviderConfig, Provider
from src.streaming.sse import format_sse_event
from src.core.errors import (
    ProxyError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ProviderError,
)
import src.core.types as types


# Provider implementations
class NVIDIAProvider(Provider):
    """NVIDIA NIM provider."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client = None

    @property
    def name(self) -> str:
        return "nvidia"

    @property
    def supported_models(self) -> list[str]:
        return ["*"]  # Dynamic - any model can be forwarded

    async def complete(self, request: types.Request) -> types.Response:
        import httpx
        # Build NVIDIA NIM API URL
        base_url = self.config.base_url or "https://integrate.api.nvidia.com/v1"
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        # Build payload matching internal request format
        payload = {
            "model": request.model,
            "messages": self._convert_messages(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": False,
        }

        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)

        if request.tool_choice:
            payload["tool_choice"] = request.tool_choice.model_dump()

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

            if resp.status_code == 401:
                raise AuthenticationError("Invalid NVIDIA API key")
            if resp.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            if resp.status_code >= 400:
                raise ProviderError(f"NVIDIA API error: {resp.text}")

            data = resp.json()
            return self._convert_response(data)

    async def stream(self, request: types.Request):
        import httpx
        base_url = self.config.base_url or "https://integrate.api.nvidia.com/v1"
        url = f"{base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model,
            "messages": self._convert_messages(request.messages),
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": True,
        }

        if request.tools:
            payload["tools"] = self._convert_tools(request.tools)

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    if resp.status_code == 401:
                        raise AuthenticationError("Invalid NVIDIA API key")
                    if resp.status_code == 429:
                        raise RateLimitError("Rate limit exceeded")
                    text = await resp.aread()
                    raise ProviderError(f"NVIDIA API error: {text.decode()}")

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            yield {"type": "done"}
                            continue
                        try:
                            import json
                            chunk = json.loads(data)
                            yield self._convert_chunk(chunk)
                        except Exception:
                            continue

    def _convert_messages(self, messages: list[types.Message]) -> list[dict]:
        result = []
        for msg in messages:
            result.append({
                "role": msg.role,
                "content": self._convert_content(msg.content),
            })
        return result

    def _convert_content(self, content: list[types.ContentBlock]) -> str:
        parts = []
        for block in content:
            if block.type == "text":
                parts.append(block.text)
            elif block.type == "image":
                # Handle image URLs
                if block.source:
                    if block.source.get("type") == "base64":
                        parts.append(f"data:{block.source.get('media_type', 'image/png')};base64,{block.source.get('data', '')}")
                    else:
                        parts.append(block.source.get("url", ""))
        return "\n".join(parts)

    def _convert_tools(self, tools: list[types.ToolDefinition]) -> list[dict]:
        result = []
        for tool in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        return result

    def _convert_response(self, data: dict) -> types.Response:
        msg = data["choices"][0]["message"]

        content = []
        if "content" in msg:
            content.append(types.ContentBlock(type="text", text=msg["content"]))

        tool_calls = None
        if "tool_calls" in msg:
            tool_calls = []
            for tc in msg["tool_calls"]:
                tool_calls.append(types.ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ))

        usage = types.Usage(
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            total_tokens=data.get("usage", {}).get("total_tokens", 0),
        )

        finish_reason = None
        if data["choices"][0].get("finish_reason"):
            fr = data["choices"][0]["finish_reason"]
            if fr == "stop":
                finish_reason = types.FinishReason.STOP
            elif fr == "length":
                finish_reason = types.FinishReason.LENGTH
            elif fr == "tool_calls":
                finish_reason = types.FinishReason.TOOL_CALLS

        return types.Response(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            model=data.get("model", ""),
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )

    def _convert_chunk(self, chunk: dict) -> dict:
        if not chunk.get("choices"):
            return {"type": "keepalive"}

        choice = chunk["choices"][0]
        delta = choice.get("delta", {})

        result = {"type": "chunk", "content": delta.get("content", "")}

        if "tool_calls" in delta:
            result["tool_calls"] = delta["tool_calls"]

        if choice.get("finish_reason"):
            fr = choice["finish_reason"]
            if fr == "stop":
                result["finish_reason"] = "stop"
            elif fr == "length":
                result["finish_reason"] = "length"
            elif fr == "tool_calls":
                result["finish_reason"] = "tool_calls"

        if "usage" in chunk:
            result["usage"] = chunk["usage"]

        return result

    async def close(self):
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    reg = get_registry()

    # Register NVIDIA provider if configured
    if config.nvidia_api_key:
        nvidia_config = ProviderConfig(
            name="nvidia",
            api_key=config.nvidia_api_key,
            base_url=config.nvidia_base_url,
        )
        register_provider(NVIDIAProvider(nvidia_config))
        logger.info("Registered NVIDIA provider")

    yield

    # Cleanup
    await reg.close_all()


app = FastAPI(title="Atlas Proxy v2", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "atlas-proxy-v2"}


@app.get("/stats")
async def stats_endpoint():
    return stats.get()


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(None)):
    """List available models."""
    config = get_config()

    # Simple auth check
    if config.api_keys:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        token = authorization.replace("Bearer ", "")
        if token not in config.api_keys:
            raise HTTPException(status_code=401, detail="Invalid authorization")

    # Return configured models
    model_data = []

    # NVIDIA models
    if config.nvidia_api_key:
        model_data.append({
            "id": config.nvidia_model or "meta/llama-3.1-70b-instruct",
            "object": "model",
            "created": 1700000000,
            "owned_by": "nvidia",
        })

    return {"object": "list", "data": model_data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    """OpenAI chat completions endpoint."""
    config = get_config()

    # Auth check
    if config.api_keys:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        token = authorization.replace("Bearer ", "")
        if token not in config.api_keys:
            raise HTTPException(status_code=401, detail="Invalid authorization")

    body = await request.json()
    adapter = get_adapter(ProtocolType.OPENAI)

    try:
        # Parse request
        req = adapter.parse_request(body)

        # Get provider
        reg = get_registry()
        provider = reg.get("nvidia")

        if not provider:
            raise HTTPException(status_code=503, detail="No provider available")

        # Check if streaming
        stream = body.get("stream", False)

        if stream:
            async def generate():
                try:
                    async for event in provider.stream(req):
                        if event.get("type") == "keepalive":
                            yield format_sse_event("", event_type="ping")
                            continue

                        if event.get("type") == "done":
                            yield format_sse_event("", event_type="done")
                            break

                        # Convert to OpenAI format
                        chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": 1700000000,
                            "model": req.model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": event.get("finish_reason"),
                            }],
                        }

                        if event.get("content"):
                            chunk_data["choices"][0]["delta"]["content"] = event["content"]

                        if event.get("tool_calls"):
                            chunk_data["choices"][0]["delta"]["tool_calls"] = event["tool_calls"]

                        if event.get("usage"):
                            chunk_data["usage"] = event["usage"]

                        yield format_sse_event(str(chunk_data))
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield format_sse_event(f'{{"error": "{{"message": "{str(e)}", "type": "server_error"}}"}}')

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            # Non-streaming
            resp = await provider.complete(req)

            # Convert to OpenAI format
            result = adapter.format_response(resp)
            return JSONResponse(result)

    except ProxyError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/messages")
async def messages(request: Request, authorization: Optional[str] = Header(None)):
    """Anthropic messages endpoint."""
    config = get_config()

    # Auth check
    if config.api_keys:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing authorization")
        token = authorization.replace("Bearer ", "")
        if token not in config.api_keys:
            raise HTTPException(status_code=401, detail="Invalid authorization")

    body = await request.json()
    adapter = get_adapter(ProtocolType.ANTHROPIC)

    try:
        req = adapter.parse_request(body)

        reg = get_registry()
        provider = reg.get("nvidia")

        if not provider:
            raise HTTPException(status_code=503, detail="No provider available")

        stream = body.get("stream", False)

        if stream:
            async def generate():
                try:
                    async for event in provider.stream(req):
                        if event.get("type") == "keepalive":
                            yield format_sse_event("", event_type="ping")
                            continue

                        if event.get("type") == "done":
                            yield format_sse_event("", event_type="message_delta")
                            yield format_sse_event('{"type": "message_stop"}', event_type="message_stop")
                            break

                        # Convert to Anthropic format
                        if event.get("content"):
                            yield format_sse_event(
                                f'{{"type": "content_block_delta", "delta": {{"type": "text_delta", "text": {repr(event["content"])}}}}}'
                            )

                        if event.get("tool_calls"):
                            for tc in event["tool_calls"]:
                                yield format_sse_event(
                                    f'{{"type": "content_block_delta", "delta": {{"type": "input_json_delta", "partial_json": {repr(tc.get("arguments", ""))}}}}}'
                                )

                        if event.get("usage"):
                            yield format_sse_event(
                                f'{{"type": "message_delta", "usage": {{"output_tokens": {event["usage"].get("completion_tokens", 0)}}}}}',
                                event_type="message_delta"
                            )
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    yield format_sse_event(f'{{"error": {{"type": "server_error", "message": "{str(e)}"}}}}')

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            resp = await provider.complete(req)
            result = adapter.format_response(resp)
            return JSONResponse(result)

    except ProxyError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
