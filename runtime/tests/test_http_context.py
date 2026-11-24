import ssl
import httpx
import pytest
from orchestrator.http import verified_context, create_replay_http_client


@pytest.mark.asyncio
async def test_context_reused_without_disabling_verification_or_changing_sdk_defaults(
    monkeypatch,
):
    verified_context.cache_clear()
    original = httpx.create_ssl_context
    calls = []

    def context(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(httpx, "create_ssl_context", context)
    # Count certificate-store loads separately from HTTPX accepting an existing context.
    loaded = []
    original_ssl = ssl.create_default_context

    def load(*args, **kwargs):
        loaded.append(True)
        return original_ssl(*args, **kwargs)

    monkeypatch.setattr(ssl, "create_default_context", load)
    timeout = httpx.Timeout(5, read=7)
    first = create_replay_http_client(headers={"x-unit": "trace"}, timeout=timeout)
    second = create_replay_http_client()
    try:
        assert len(loaded) == 1
        ctx = verified_context(None, None)
        assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
        assert first.follow_redirects and second.follow_redirects
        assert first.timeout == timeout and second.timeout == httpx.Timeout(30)
        assert first.headers["x-unit"] == "trace"
    finally:
        await first.aclose()
        await second.aclose()
        verified_context.cache_clear()
