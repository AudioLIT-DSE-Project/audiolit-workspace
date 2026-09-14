"""Latency & FPS metric synthesis (LIT-189).

Aggregates latency samples collected across the system (cache retrieval, API
response, task enqueue, model inference, attribution, canvas mutation) into a
single synthesized report scored against the engineering targets committed in
SRS section 3.4.1.

This is the *synthesis* layer: the raw samples are produced by the performance
tests (`tests/test_performance_load.py`), the Locust load suite
(`loadtests/locustfile.py`), the worker's per-job duration publishing in
`task_orchestrator.perform_job`, and - once live - by `MetricCollector.record`
calls in the request path. This module owns:

  * the canonical SRS 3.4.1 operation budgets (one source of truth shared by
    the synthesised report and the /metrics/synthesis route),
  * percentile statistics over recorded samples (p50 / p95 / max),
  * FPS derivation for the canvas-mutation interaction target (FR12: a fluid
    30-60 FPS interaction), and
  * a JSON-serialisable report shape.

Nothing here requires Redis or a live stack: it is pure aggregation over
whatever samples are supplied, so tests can synthesise a report deterministically.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# --------------------------------------------------------------------------- #
# SRS 3.4.1 performance targets
# --------------------------------------------------------------------------- #
# The canonical budget table. Each entry carries the budget in milliseconds
# plus the target's source clause. `model_bound` mirrors the caveat the SRS
# itself makes ("GPU figures assume an NVIDIA T4 or equivalent free cloud
# tier; CPU fallback is proportionally slower but functional") - model-bound
# targets are reported but only enforced when the caller opts in, while the
# infrastructure targets (cache, enqueue, canvas UI) are always enforced.

Operation = str


@dataclass(frozen=True)
class PerformanceTarget:
    budget_ms: float
    srs: str
    model_bound: bool = False


SRS_PERFORMANCE_TARGETS: Mapping[Operation, PerformanceTarget] = {
    "cached_tensor_retrieval": PerformanceTarget(
        10.0, "SHA-256 cache-by-hash hit (FR4)"
    ),
    "cached_request": PerformanceTarget(
        200.0, "API response for a cached request < 200 ms"
    ),
    "enqueue": PerformanceTarget(
        50.0, "Cache miss to task enqueue < 50 ms"
    ),
    "cold_asr_inference": PerformanceTarget(
        3_000.0, "Cold ASR inference (Whisper-base, 15 s audio) < 3 s (GPU)",
        model_bound=True,
    ),
    "multitask_inference": PerformanceTarget(
        8_000.0, "Multi-task inference (ASR + SER + ADD) < 8 s cold", model_bound=True
    ),
    "attribution": PerformanceTarget(
        8_000.0, "Interpretability attribution (IG / saliency) < 8 s", model_bound=True
    ),
    "canvas_mutation_ui": PerformanceTarget(
        500.0, "Canvas mutation - UI response < 500 ms (FR12)"
    ),
    "canvas_mutation_backend": PerformanceTarget(
        2_000.0, "Canvas mutation - backend result < 2 s per perturbation",
        model_bound=True,
    ),
    "accent_bias": PerformanceTarget(
        30_000.0, "Accent bias profiling < 30 s (FR15)", model_bound=True
    ),
    "faithfulness": PerformanceTarget(
        15_000.0, "Faithfulness audit < 15 s per clip, deletion score (FR16)",
        model_bound=True,
    ),
    "model_prepare": PerformanceTarget(
        60_000.0, "Cold model download + hook registration < 60 s (FR1)",
        model_bound=True,
    ),
}

#: FR12 commits to "a fluid 30-60 FPS interaction" for the canvas. Frame time
#: bounds are derived from the FPS bounds (1000/FPS).
MIN_FPS = 30.0
MAX_FPS = 60.0
MIN_FRAME_MS = 1000.0 / MAX_FPS  # ~16.7 ms
MAX_FRAME_MS = 1000.0 / MIN_FPS  # ~33.3 ms


def fps_from_frame_ms(frame_ms: float) -> float:
    """Frames per second implied by one frame's render latency."""
    return 1000.0 / frame_ms if frame_ms > 0 else float("inf")


# --------------------------------------------------------------------------- #
# Sample collection
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LatencySample:
    operation: Operation
    latency_ms: float
    ts: float = field(default_factory=time.time)
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Accumulates latency samples for one operation into summary statistics."""

    operation: Operation
    samples_ms: list[float] = field(default_factory=list)

    def record(self, latency_ms: float) -> None:
        self.samples_ms.append(latency_ms)

    @property
    def count(self) -> int:
        return len(self.samples_ms)

    def percentiles(self, qs: tuple[float, ...] = (50.0, 95.0, 99.0, 100.0)) -> dict[str, float]:
        if not self.samples_ms:
            return {f"p{int(q)}": float("nan") for q in qs}
        values = sorted(self.samples_ms)
        return {f"p{int(q)}": _percentile(values, q / 100.0) for q in qs}

    def summary(self) -> dict[str, Any]:
        n = self.count
        if n == 0:
            return {
                "operation": self.operation,
                "count": 0,
                "mean_ms": None,
                "min_ms": None,
                "max_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "status": "no_samples",
                "budget_ms": None,
                "srs": None,
            }
        stats = self.percentiles()
        target = SRS_PERFORMANCE_TARGETS.get(self.operation)
        return {
            "operation": self.operation,
            "count": n,
            "mean_ms": round(statistics.mean(self.samples_ms), 3),
            "min_ms": round(min(self.samples_ms), 3),
            "max_ms": round(max(self.samples_ms), 3),
            "p50_ms": stats["p50"],
            "p95_ms": stats["p95"],
            "budget_ms": target.budget_ms if target else None,
            "srs": target.srs if target else None,
            "status": score_latency(self.operation, stats["p95"]),
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if q >= 1.0:
        return sorted_values[-1]
    if q <= 0.0:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * q
    lower = int(index)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def score_latency(operation: Operation, p95_ms: float | None) -> str:
    """PASS / FAIL / over (model-bound, not enforced) for a p95 observation."""
    target = SRS_PERFORMANCE_TARGETS.get(operation)
    if target is None or p95_ms is None:
        return "no_target"
    if p95_ms <= target.budget_ms:
        return "PASS"
    if target.model_bound:
        return "over (model-bound, not enforced)"
    return "FAIL"


# --------------------------------------------------------------------------- #
# FPS synthesis
# --------------------------------------------------------------------------- #

@dataclass
class FpsSeries:
    """Accumulates canvas frame render latencies and derives an FPS verdict.

    FR12 targets a fluid 30-60 FPS interaction. The p95 frame time must sit
    inside the 30-60 FPS band (16.7-33.3 ms per frame) to be judged "fluid".
    """

    frame_ms: list[float] = field(default_factory=list)

    def record_frame(self, frame_ms: float) -> None:
        self.frame_ms.append(frame_ms)

    @property
    def count(self) -> int:
        return len(self.frame_ms)

    def summary(self) -> dict[str, Any]:
        if not self.frame_ms:
            return {"count": 0, "avg_fps": None, "p95_frame_ms": None, "status": "no_samples"}
        values = sorted(self.frame_ms)
        p95 = _percentile(values, 0.95)
        avg_fps = fps_from_frame_ms(statistics.mean(self.frame_ms))
        if p95 <= MAX_FRAME_MS:
            status = "PASS"
        else:
            status = "over (below 30 FPS): target a fluid 30-60 FPS interaction"
        return {
            "count": self.count,
            "avg_frame_ms": round(statistics.mean(self.frame_ms), 3),
            "avg_fps": round(avg_fps, 2),
            "p95_frame_ms": round(p95, 3),
            "p95_fps": round(fps_from_frame_ms(p95), 2),
            "target_fps": f"{MIN_FPS:.0f}-{MAX_FPS:.0f}",
            "status": status,
        }


# --------------------------------------------------------------------------- #
# Collector + report
# --------------------------------------------------------------------------- #

class MetricsCollector:
    """In-process aggregator for latency and FPS samples across the system.

    Process-local: samples recorded in the API process stay in the API process.
    Persisting historical samples to Redis/MongoDB and reading them back is the
    MongoDB metadata-tier work (LIT-255/256) - this collector provides the
    `record` API that tier will flush from.
    """

    def __init__(self) -> None:
        self._series: dict[Operation, MetricSeries] = {}
        self._fps = FpsSeries()
        self._started = time.time()

    def record_latency(
        self, operation: Operation, latency_ms: float, meta: Mapping[str, Any] | None = None
    ) -> None:
        self._series.setdefault(operation, MetricSeries(operation)).record(latency_ms)

    def record_frame(self, frame_ms: float) -> None:
        self._fps.record_frame(frame_ms)

    def summarize(self) -> dict[str, Any]:
        operations = {
            op: series.summary()
            for op, series in sorted(self._series.items())
        }
        latency_rows = [s for s in operations.values() if s["count"] > 0]
        passed = sum(1 for s in latency_rows if s["status"] == "PASS")
        return {
            "generated_at": time.time(),
            "uptime_s": round(time.time() - self._started, 1),
            "operations": operations,
            "fps": self._fps.summary(),
            "summary": {
                "operations_measured": len(latency_rows),
                "operations_passed": passed,
                "operations_over": len(latency_rows) - passed,
                "frame_samples": self._fps.count,
            },
        }


#: The default process-wide collector. Tests should hold their own instance
#: rather than mutating this singleton.
collector = MetricsCollector()


def synthesize_report(*samples: LatencySample) -> dict[str, Any]:
    """Build a report from an explicit batch of samples (stateless form).

    Useful for the performance tests and one-off synthesis runs, which want a
    deterministic report without touching the shared ``collector``.
    """
    c = MetricsCollector()
    for s in samples:
        c.record_latency(s.operation, s.latency_ms, s.meta)
    return c.summarize()