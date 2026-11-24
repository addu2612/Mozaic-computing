"""One official MCP Streamable HTTP replay service per source."""

import argparse
import asyncio
from datetime import datetime
import json
import logging
from bisect import bisect_left
from pathlib import Path
from typing import Any
from mcp.server.fastmcp import FastMCP
from mozaic.schema import project
from mcp_servers.timing import (
    TimingSink,
    timed_page,
    TimingMiddleware,
    install_serialization_timer,
)

REJECTED_LINE_SAMPLE_LIMIT = 20


def error(code, message, retryable=False):
    return dict(ok=False, error=dict(code=code, message=message, retryable=retryable))


class ReplaySource:
    source = ""

    def __init__(self, directory="data/dev"):
        self.timing = TimingSink(self.source)
        validated = []
        self.invalid_record_count = 0
        self.rejected_line_sample = []
        path = Path(directory) / f"{self.source}.jsonl"
        with path.open("rb") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    raw = json.loads(line.decode("utf-8"))
                    canonical = project(raw)
                    if canonical.source != self.source:
                        raise ValueError("source mismatch")
                    public = {
                        k: v
                        for k, v in raw.items()
                        if k not in ("is_anomaly", "anomaly_type", "incident_id")
                    }
                    public.update(
                        service=canonical.service, namespace=canonical.namespace
                    )
                    validated.append((canonical.timestamp, canonical.record_id, public))
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    IndexError,
                    OverflowError,
                    RecursionError,
                ) as exc:
                    self.invalid_record_count += 1
                    if len(self.rejected_line_sample) < REJECTED_LINE_SAMPLE_LIMIT:
                        self.rejected_line_sample.append(
                            dict(line=number, error=type(exc).__name__)
                        )
                        logging.getLogger(__name__).warning(
                            "source=%s rejected_line=%d error=%s",
                            self.source,
                            number,
                            type(exc).__name__,
                        )
        validated.sort(key=lambda item: (item[0], item[1]))
        self.timestamps = [item[0] for item in validated]
        self.records = [item[2] for item in validated]
        if self.invalid_record_count > len(self.rejected_line_sample):
            logging.getLogger(__name__).warning(
                "source=%s rejected_records=%d further rejection warnings suppressed",
                self.source,
                self.invalid_record_count,
            )

    def _metadata(self, include_rejected_sample=False):
        metadata = dict(
            source=self.source,
            degraded=bool(self.invalid_record_count),
            ingestion=dict(
                valid_records=len(self.records),
                invalid_records=self.invalid_record_count,
            ),
        )
        if include_rejected_sample:
            metadata["ingestion"]["rejected_line_sample"] = self.rejected_line_sample
            metadata["ingestion"]["rejected_line_sample_limit"] = (
                REJECTED_LINE_SAMPLE_LIMIT
            )
            metadata["ingestion"]["rejected_lines_truncated"] = (
                self.invalid_record_count > len(self.rejected_line_sample)
            )
        return metadata

    def _page(self, found, offset, limit):
        if (
            type(offset) is not int
            or type(limit) is not int
            or offset < 0
            or not 1 <= limit <= 1000
        ):
            return error(
                "INVALID_ARGUMENT", "offset >= 0 and 1 <= limit <= 1000 required"
            )
        if not found:
            return {
                **error("NOT_FOUND", "No replay records match the filters"),
                **self._metadata(),
            }
        page = found[offset : offset + limit]
        return dict(
            ok=True,
            records=page,
            total=len(found),
            next_offset=offset + len(page) if offset + len(page) < len(found) else None,
            **self._metadata(),
        )

    @timed_page
    def query(
        self, start, end, service=None, namespace=None, offset=0, limit=500, kind=None
    ):
        try:
            a = datetime.fromisoformat(start.replace("Z", "+00:00"))
            b = datetime.fromisoformat(end.replace("Z", "+00:00"))
            if a.tzinfo is None or b.tzinfo is None or a >= b:
                raise ValueError("invalid window")
        except (ValueError, TypeError, AttributeError):
            return error("INVALID_ARGUMENT", "timezone-aware start < end required")
        left = bisect_left(self.timestamps, a)
        right = bisect_left(self.timestamps, b)
        found = [
            r
            for r in self.records[left:right]
            if (service is None or r["service"] == service)
            and (namespace is None or r["namespace"] == namespace)
            and (
                kind is None
                or (kind == "events" and r["payload"].get("kind") == "Event")
                or (kind == "logs" and "log" in r["payload"])
            )
        ]
        return self._page(found, offset, limit)

    @timed_page
    def lookup(self, field, value, offset=0, limit=500):
        found = [r for r in self.records if str(r["payload"].get(field, "")) == value]
        return self._page(found, offset, limit)


class KubernetesReplay(ReplaySource):
    source = "kubernetes"


class SentryReplay(ReplaySource):
    source = "sentry"


class CloudWatchReplay(ReplaySource):
    source = "cloudwatch"


class GrafanaReplay(ReplaySource):
    source = "grafana"


class PublicLogsReplay(ReplaySource):
    source = "rca_logs"


class PublicMetricsReplay(ReplaySource):
    source = "rca_metrics"


class PublicTracesReplay(ReplaySource):
    source = "rca_traces"


CLASSES = {
    c.source: c
    for c in (
        KubernetesReplay,
        SentryReplay,
        CloudWatchReplay,
        GrafanaReplay,
        PublicLogsReplay,
        PublicMetricsReplay,
        PublicTracesReplay,
    )
}


def create_server(source, directory="data/dev", port=8000):
    replay = CLASSES[source](directory)
    server = FastMCP(
        f"MOZAIC {source} replay",
        host="0.0.0.0",
        port=port,
        json_response=True,
        stateless_http=True,
        log_level="WARNING",
    )

    @server.tool()
    async def query_window(
        start: str,
        end: str,
        service: str | None = None,
        namespace: str | None = None,
        offset: int = 0,
        limit: int = 500,
        delay_seconds: float = 0.0,
        fail: bool = False,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Read a paginated UTC incident window. Delay/fail flags are explicit fault injection."""
        if not 0 <= delay_seconds <= 10:
            return error(
                "INVALID_ARGUMENT", "delay_seconds must be between zero and ten"
            )
        if fail:
            return error("UNAVAILABLE", "Injected replay failure", True)
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        return replay.query(
            start, end, service, namespace, offset, limit, trace_id=trace_id
        )

    @server.tool()
    def capabilities() -> dict[str, Any]:
        """Report replay scope honestly; no upstream cloud connection is made."""
        return dict(
            ok=True,
            source=source,
            mode="offline-replay",
            records=len(replay.records),
            live_cloud=False,
            ingestion=replay._metadata(include_rejected_sample=True)["ingestion"],
            degraded=bool(replay.invalid_record_count),
        )

    if source == "kubernetes":

        @server.tool()
        def pod_events(
            start: str,
            end: str,
            namespace: str = "production",
            offset: int = 0,
            limit: int = 500,
        ) -> dict[str, Any]:
            """Read events with pagination applied after event and namespace filtering."""
            return replay.query(
                start,
                end,
                namespace=namespace,
                offset=offset,
                limit=limit,
                kind="events",
            )

        @server.tool()
        def pod_logs(
            start: str, end: str, service: str, offset: int = 0, limit: int = 500
        ) -> dict[str, Any]:
            """Read pod logs with pagination applied after service and log filtering."""
            return replay.query(
                start, end, service=service, offset=offset, limit=limit, kind="logs"
            )
    elif source == "sentry":

        @server.tool()
        def issue_context(
            event_id: str, offset: int = 0, limit: int = 500
        ) -> dict[str, Any]:
            """Look up a generated error event, including its stack and metadata."""
            return replay.lookup("event_id", event_id, offset, limit)
    elif source == "cloudwatch":

        @server.tool()
        def metric_lookup(
            metric_name: str, offset: int = 0, limit: int = 500
        ) -> dict[str, Any]:
            """Find generated datapoints by CloudWatch metric name."""
            return replay.lookup("MetricName", metric_name, offset, limit)
    elif source == "grafana":

        @server.tool()
        def dashboard_panels(
            dashboard: str, offset: int = 0, limit: int = 500
        ) -> dict[str, Any]:
            """Find generated panel payloads by dashboard name."""
            return replay.lookup("dashboard", dashboard, offset, limit)

    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=CLASSES, required=True)
    parser.add_argument("--data", default="data/dev")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    install_serialization_timer()
    server = create_server(args.source, args.data, args.port)
    app = server.streamable_http_app()
    app.add_middleware(TimingMiddleware, source=args.source)
    uvicorn.run(
        app,
        host=server.settings.host,
        port=server.settings.port,
        log_level=server.settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
