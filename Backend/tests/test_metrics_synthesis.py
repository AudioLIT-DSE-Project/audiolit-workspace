"""Latency & FPS metric synthesis tests (LIT-189)."""

import pytest
from app.infrastructure import metrics_synthesis as metrics


class TestScoreLatency:
    def test_pass_within_budget(self):
        assert metrics.score_latency("cached_request", 150.0) == "PASS"

    def test_fail_infrastructure_target(self):
        assert metrics.score_latency("cached_request", 250.0) == "FAIL"

    def test_model_bound_over_reported_not_failed(self):
        assert metrics.score_latency("cold_asr_inference", 10_000.0) == (
            "over (model-bound, not enforced)"
        )

    def test_unknown_operation(self):
        assert metrics.score_latency("does_not_exist", 1.0) == "no_target"


class TestFpsSynthesis:
    def test_fps_from_frame_ms(self):
        assert metrics.fps_from_frame_ms(20.0) == 50.0

    def test_fluid_interaction_within_band(self):
        series = metrics.FpsSeries()
        for ms in (16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 33.0):
            series.record_frame(ms)
        report = series.summary()
        assert report["status"] == "PASS"
        assert 30.0 <= report["avg_fps"] <= 60.0

    def test_slow_interaction_fails(self):
        series = metrics.FpsSeries()
        # Mostly well inside budget, but a p95 above 33.3 ms means the tail is
        # not fluid, which is exactly what p95 is there to catch.
        for ms in (16.0, 17.0, 18.0, 19.0, 20.0, 60.0, 65.0, 70.0, 75.0, 80.0):
            series.record_frame(ms)
        report = series.summary()
        assert report["p95_frame_ms"] > metrics.MAX_FRAME_MS
        assert report["status"].startswith("over")

    def test_no_samples(self):
        report = metrics.FpsSeries().summary()
        assert report["status"] == "no_samples"


class TestSynthesizeReport:
    def test_empty_report(self):
        report = metrics.synthesize_report()
        assert report["summary"]["operations_measured"] == 0
        assert report["fps"]["status"] == "no_samples"

    def test_scored_against_srs(self):
        report = metrics.synthesize_report(
            metrics.LatencySample("cached_request", 120.0),
            metrics.LatencySample("cached_request", 90.0),
            metrics.LatencySample("enqueue", 40.0),
            metrics.LatencySample("attribution", 7_500.0),
        )
        ops = report["operations"]
        assert ops["cached_request"]["status"] == "PASS"
        assert ops["cached_request"]["p95_ms"] <= 200.0
        assert ops["enqueue"]["status"] == "PASS"
        assert ops["attribution"]["status"] == "PASS"
        assert report["summary"]["operations_measured"] == 3
        assert report["summary"]["operations_passed"] == 3

    def test_percentiles(self):
        samples = [metrics.LatencySample("enqueue", ms) for ms in range(1, 101)]
        report = metrics.synthesize_report(*samples)
        enqueue = report["operations"]["enqueue"]
        # Linear-interpolation percentile: the median of 1..100 is 50.5
        # ((50 + 51) / 2), and p95 is the value at rank 0.95*(n-1) = 94.05.
        assert enqueue["p50_ms"] == pytest.approx(50.5, abs=0.2)
        assert enqueue["p95_ms"] == pytest.approx(95.05, abs=0.2)
        assert enqueue["max_ms"] == 100.0

    def test_fps_key_present(self):
        report = metrics.synthesize_report()
        assert "fps" in report


class TestCollector:
    def test_records_and_summarizes(self):
        c = metrics.MetricsCollector()
        c.record_latency("cached_request", 100.0)
        c.record_latency("cached_request", 150.0)
        c.record_frame(20.0)
        report = c.summarize()
        assert report["operations"]["cached_request"]["count"] == 2
        assert report["fps"]["count"] == 1
        assert report["summary"]["frame_samples"] == 1

    def test_isolated_from_global_collector(self):
        c = metrics.MetricsCollector()
        c.record_latency("cached_tensor_retrieval", 5.0)
        report = c.summarize()
        assert report["operations"]["cached_tensor_retrieval"]["count"] == 1


@pytest.mark.asyncio
class TestMetricsRoute:
    async def test_get_synthesis(self, client):
        resp = await client.get("/metrics/synthesis")
        assert resp.status_code == 200
        body = resp.json()
        assert "operations" in body
        assert "fps" in body

    async def test_record_latency_and_reports(self, client):
        r1 = await client.post(
            "/metrics/samples/latency",
            json={"operation": "cached_request", "latency_ms": 100.0},
        )
        assert r1.status_code == 200

        r2 = await client.post("/metrics/samples/frame", json={"frame_ms": 20.0})
        assert r2.status_code == 200

        resp = await client.get("/metrics/synthesis")
        assert resp.status_code == 200
        body = resp.json()
        assert body["operations"]["cached_request"]["count"] >= 1
        assert body["fps"]["count"] >= 1

    async def test_unknown_operation_rejected(self, client):
        resp = await client.post(
            "/metrics/samples/latency", json={"operation": "nope", "latency_ms": 5.0}
        )
        assert resp.status_code == 400