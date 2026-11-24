"""Concurrent official MCP client with bounded retries, pagination and partial results."""

import asyncio
import json
import logging
import random
import time
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from orchestrator.http import create_replay_http_client

LOG = logging.getLogger(__name__)
DEFAULT_URLS = {
    s: f"http://{s}:8000/mcp" for s in ("kubernetes", "sentry", "cloudwatch", "grafana")
}


class SourceFailure(Exception):
    def __init__(self, code, retryable=False):
        super().__init__(code)
        self.code = code
        self.retryable = retryable


def classify_failure(exc):
    if isinstance(exc, BaseExceptionGroup):
        failures = [classify_failure(child) for child in exc.exceptions]
        return next((f for f in failures if not f.retryable), failures[0])
    if isinstance(exc, SourceFailure):
        return exc
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return SourceFailure("TIMEOUT", True)
    if isinstance(exc, (ConnectionError, OSError, httpx.TransportError)):
        return SourceFailure("CONNECTION_ERROR", True)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return SourceFailure(f"HTTP_{status}", status == 429 or status >= 500)
    return SourceFailure("CLIENT_ERROR", False)


class Orchestrator:
    def __init__(self, urls=None, timeout=5.0, retries=2, client_factory=None):
        self.urls = urls or DEFAULT_URLS
        self.timeout = timeout
        self.retries = retries
        self.client_factory = client_factory
        self.rng = random.Random(1411)

    async def _call(self, url, args, timing=None):
        timing = timing if timing is not None else {}
        timing.update(phase="client_setup", pages=[])

        def factory(**kwargs):
            started = time.perf_counter()
            cpu = time.process_time()
            if args.get("trace_id"):
                kwargs["headers"] = {
                    **(kwargs.get("headers") or {}),
                    "x-mozaic-trace-id": args["trace_id"],
                }
            try:
                return create_replay_http_client(**kwargs)
            finally:
                timing["client_factory_seconds"] = time.perf_counter() - started
                timing["client_factory_cpu_seconds"] = time.process_time() - cpu

        async with streamablehttp_client(
            url,
            timeout=self.timeout,
            sse_read_timeout=self.timeout,
            httpx_client_factory=factory,
        ) as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                timing["phase"] = "initialize"
                started = time.perf_counter()
                try:
                    await session.initialize()
                finally:
                    timing["initialize_seconds"] = time.perf_counter() - started
                offset = 0
                rows = []
                degraded = False
                ingestion = None
                while True:
                    timing["phase"] = "page"
                    page = dict(offset=offset)
                    timing["pages"].append(page)
                    started = time.perf_counter()
                    try:
                        result = await session.call_tool(
                            "query_window", {**args, "offset": offset, "limit": 500}
                        )
                    finally:
                        page["roundtrip_seconds"] = time.perf_counter() - started
                    if result.isError:
                        raise SourceFailure("MCP_TOOL_ERROR", False)
                    data = result.structuredContent or json.loads(
                        result.content[0].text
                    )
                    page["server_timing"] = data.get("server_timing")
                    page["records"] = len(data.get("records", []))
                    degraded = degraded or data.get("degraded", False)
                    ingestion = data.get("ingestion", ingestion)
                    if not data["ok"]:
                        if data["error"]["code"] == "NOT_FOUND":
                            return dict(
                                records=[], degraded=degraded, ingestion=ingestion
                            )
                        raise SourceFailure(
                            data["error"]["code"], data["error"].get("retryable", False)
                        )
                    rows.extend(data["records"])
                    next_offset = data["next_offset"]
                    if next_offset is not None and (
                        not data["records"] or next_offset <= offset
                    ):
                        raise SourceFailure("INVALID_PAGINATION", False)
                    offset = next_offset
                    if offset is None:
                        return dict(
                            records=rows, degraded=degraded, ingestion=ingestion
                        )

    async def _source(self, source, url, args):
        start = time.perf_counter()
        failures = []
        attempt_timings = []
        for attempt in range(self.retries + 1):
            try:
                timing = dict(attempt=attempt + 1)
                attempt_timings.append(timing)
                traced = (
                    {**args, "trace_id": f"{args['trace_id']}:{source}:{attempt + 1}"}
                    if args.get("trace_id")
                    else args
                )
                call = (
                    self.client_factory(url, traced)
                    if self.client_factory
                    else self._call(url, traced, timing)
                )
                timing["trace_id"] = traced.get("trace_id")
                rows = await asyncio.wait_for(call, timeout=self.timeout)
                timing["phase"] = "complete"
                quality = rows if isinstance(rows, dict) else dict(records=rows)
                return dict(
                    source=source,
                    ok=True,
                    records=quality["records"],
                    degraded=quality.get("degraded", False),
                    ingestion=quality.get("ingestion"),
                    attempts=attempt + 1,
                    failures=failures,
                    attempt_timings=attempt_timings,
                    seconds=time.perf_counter() - start,
                )
            except Exception as exc:
                # Exception messages are not logged: HTTP libraries can include credentials.
                failure = classify_failure(exc)
                code = failure.code
                failures.append(code)
                LOG.warning(
                    "source=%s attempt=%d failure=%s", source, attempt + 1, code
                )
                if not failure.retryable:
                    break
                if attempt < self.retries:
                    await asyncio.sleep(0.1 * 2**attempt + self.rng.uniform(0, 0.05))
        return dict(
            source=source,
            ok=False,
            records=[],
            attempts=attempt + 1,
            attempt_timings=attempt_timings,
            error=dict(code=failure.code, retryable=failure.retryable, causes=failures),
            seconds=time.perf_counter() - start,
        )

    async def fetch(self, start, end, service=None, faults=None, trace_id=None):
        args = dict(start=start, end=end)
        if trace_id is not None:
            args["trace_id"] = trace_id
        if service is not None:
            args["service"] = service
        before = time.perf_counter()
        sources = await asyncio.gather(
            *(
                self._source(s, url, {**args, **(faults or {}).get(s, {})})
                for s, url in self.urls.items()
            )
        )
        return dict(
            records=sum([s["records"] for s in sources], []),
            sources=[{k: v for k, v in s.items() if k != "records"} for s in sources],
            degraded=any(not s["ok"] or s.get("degraded", False) for s in sources),
            query_seconds=time.perf_counter() - before,
        )
