"""Preserve MCP HTTP defaults while reusing the verified certificate store."""

from functools import lru_cache
import os
import httpx


@lru_cache(maxsize=1)
def verified_context(cert_file, cert_directory):
    # HTTPX itself implements its default CA/environment policy.
    return httpx.create_ssl_context(verify=True, trust_env=True)


def create_replay_http_client(headers=None, timeout=None, auth=None):
    context = verified_context(
        os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR")
    )
    return httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout if timeout is not None else httpx.Timeout(30),
        headers=headers,
        auth=auth,
        verify=context,
    )
