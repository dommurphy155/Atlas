"""Server tests."""
import pytest
from httpx import AsyncClient, ASGITransport
from src.server import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_health():
    """Test health endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "atlas-proxy-v2"


@pytest.mark.asyncio
async def test_stats():
    """Test stats endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests" in data
        assert "successes" in data


@pytest.mark.asyncio
async def test_models_no_auth():
    """Test models endpoint without auth."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/models")
        # Should work with no API keys configured
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"


@pytest.mark.asyncio
async def test_chat_completions_missing_body():
    """Test chat completions with missing body."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json={})
        # Should fail with no provider
        assert resp.status_code in (400, 503)
