"""Strict cluster-label schema and separate parse/semantic assessments."""

import json
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ClusterLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=120)
    incident_type: Literal[
        "memory_leak",
        "deployment_failure",
        "database_connection_pool_exhaustion",
        "unknown",
    ]
    summary: str = Field(min_length=1, max_length=300)


def assess(response, majority):
    try:
        json.loads(response)
    except (ValueError, TypeError):
        return dict(
            json_valid=False,
            parsed=False,
            correct=False,
            value=None,
            validation_errors=["invalid_json"],
        )
    try:
        value = ClusterLabel.model_validate_json(response)
        return dict(
            json_valid=True,
            parsed=True,
            correct=value.incident_type == majority,
            value=value.model_dump(),
            validation_errors=[],
        )
    except Exception as exc:
        return dict(
            json_valid=True,
            parsed=False,
            correct=False,
            value=None,
            validation_errors=[e["type"] for e in exc.errors()]
            if hasattr(exc, "errors")
            else [type(exc).__name__],
        )
