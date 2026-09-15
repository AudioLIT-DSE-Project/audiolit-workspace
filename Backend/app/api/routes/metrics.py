"""Metrics synthesis API (LIT-189).

Exposes the latencies recorded on the request path (via
``metrics_synthesis.collector``) as a single synthesized report scored against
the SRS 3.4.1 budgets, plus an endpoint for instrumented callers to submit a
latency/frame sample.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...infrastructure import metrics_synthesis as metrics

router = APIRouter()


class LatencySampleIn(BaseModel):
    operation: str = Field(
        ..., description="One of the SRS 3.4.1 operations, e.g. 'cached_request'."
    )
    latency_ms: float = Field(..., gt=0)
    note: str | None = None


class FrameSampleIn(BaseModel):
    frame_ms: float = Field(..., gt=0)


@router.get("/metrics/synthesis")
async def get_metrics_synthesis():
    """The synthesized latency/FPS report for this process, scored vs SRS 3.4.1."""
    return metrics.collector.summarize()


@router.post("/metrics/samples/latency")
async def record_latency_sample(sample: LatencySampleIn):
    """Record one latency observation into the process collector.

    Cluster dashboards and the E2E suite can POST real measurements here so
    that `/metrics/synthesis` reflects observed behaviour alongside the CI
    benchmarks.
    """
    if sample.operation not in metrics.SRS_PERFORMANCE_TARGETS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown operation '{sample.operation}'. Valid operations: "
                + ", ".join(sorted(metrics.SRS_PERFORMANCE_TARGETS))
            ),
        )
    metrics.collector.record_latency(sample.operation, sample.latency_ms)
    return {"accepted": True, "operation": sample.operation, "latency_ms": sample.latency_ms}


@router.post("/metrics/samples/frame")
async def record_frame_sample(sample: FrameSampleIn):
    """Record one canvas frame render latency into the FPS synthesis (FR12)."""
    metrics.collector.record_frame(sample.frame_ms)
    return {"accepted": True, "frame_ms": sample.frame_ms}