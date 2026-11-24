"""Opt-in page and HTTP timing; never record request or response payloads."""

from contextvars import ContextVar
from functools import wraps
import inspect
import json
import os
from pathlib import Path
import time
import uuid

HTTP_TIMING = ContextVar("mozaic_http_timing", default=None)
MAX_LOG_BYTES = 50_000_000


class TimingSink:
    def __init__(self, source):
        directory = os.environ.get("MOZAIC_MCP_TIMING_DIR")
        self.path = Path(directory) / f"{source}.jsonl" if directory else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event):
        if self.path and (
            not self.path.exists() or self.path.stat().st_size < MAX_LOG_BYTES
        ):
            with self.path.open("a") as handle:
                handle.write(json.dumps(event, separators=(",", ":")) + "\n")


def timed_page(method):
    signature = inspect.signature(method)

    @wraps(method)
    def call(self, *args, **kwargs):
        trace_id = kwargs.pop("trace_id", None)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        begin = time.perf_counter()
        data = None
        try:
            data = method(self, *args, **kwargs)
            context = HTTP_TIMING.get()
            data["server_timing"] = dict(
                filter_seconds=time.perf_counter() - begin,
                request_id=context["request_id"] if context else None,
            )
            return data
        finally:
            self.timing.write(
                dict(
                    kind="page",
                    source=self.source,
                    trace_id=trace_id,
                    method=method.__name__,
                    offset=bound.arguments.get("offset", 0),
                    limit=bound.arguments.get("limit", 500),
                    request_id=(HTTP_TIMING.get() or {}).get("request_id"),
                    filter_seconds=time.perf_counter() - begin,
                    records=len(data.get("records", [])) if data else 0,
                    total=data.get("total") if data else None,
                    ok=data.get("ok") if data else False,
                )
            )

    return call


def install_serialization_timer():
    from mcp.types import JSONRPCMessage

    original = JSONRPCMessage.model_dump_json

    @wraps(original)
    def timed_dump(self, *args, **kwargs):
        before = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            context = HTTP_TIMING.get()
            if context is not None:
                context["serialize_seconds"] += time.perf_counter() - before

    JSONRPCMessage.model_dump_json = timed_dump


class TimingMiddleware:
    def __init__(self, app, source):
        self.app = app
        self.source = source
        self.sink = TimingSink(source)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        begin = time.perf_counter()
        event = dict(
            kind="http",
            trace_id=dict(scope.get("headers", []))
            .get(b"x-mozaic-trace-id", b"")
            .decode("ascii", errors="replace")
            or None,
            source=self.source,
            request_id=uuid.uuid4().hex,
            serialize_seconds=0.0,
            send_seconds=0.0,
            response_bytes=0,
            status=None,
            response_start_seconds=None,
        )
        token = HTTP_TIMING.set(event)

        async def measured_send(message):
            if message["type"] == "http.response.start":
                event["status"] = message["status"]
                event["response_start_seconds"] = time.perf_counter() - begin
            if message["type"] == "http.response.body":
                event["response_bytes"] += len(message.get("body", b""))
            started = time.perf_counter()
            try:
                await send(message)
            finally:
                event["send_seconds"] += time.perf_counter() - started

        try:
            await self.app(scope, receive, measured_send)
        finally:
            event["http_seconds"] = time.perf_counter() - begin
            self.sink.write(event)
            HTTP_TIMING.reset(token)
