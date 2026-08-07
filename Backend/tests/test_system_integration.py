"""End-to-end system integration / demo-path smoke tests (LIT-132).

Exercises the two MVP demo paths that get shown at the mid-evaluation:

  1. Multi-task inference: one audio payload fans out to ASR + SER + deepfake
     and fans back in as a single unified response (LIT-150 orchestrator wiring
     LIT-128 ADD + LIT-206 SER + Whisper ASR).
  2. XAI attribution: all three attribution methods run — Integrated Gradients
     (LIT-126), Grad-CAM (LIT-148), and LIME/SHAP occlusion (LIT-130).

Models are mocked (no downloads), and the fan-in uses the deterministic
drain-children-then-aggregate path — so these run fast and, importantly, without
the SimpleWorker burst-drain lockups the DoD calls out ("no terminal lockups
during demo paths"). This is the automated slice of LIT-132; the manual
walkthrough / stability-run / presentation prep are the team's.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fakeredis import FakeServer, FakeStrictRedis
from rq import Queue, SimpleWorker

from app.orchestration import multitask_orchestrator_service as mt
from app.domain.saliency_service import (
    attribution_timeline,
    compute_grad_cam,
    integrated_gradients,
    occlusion_attribution,
)


def _drain(conn, *queue_names) -> None:
    SimpleWorker([Queue(n, connection=conn) for n in queue_names], connection=conn).work(burst=True)


@pytest.fixture
def fake_conn(monkeypatch):
    conn = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(mt, "get_redis_connection", lambda: conn)
    yield conn
    conn.flushall()


class TestMultiTaskDemoPath:
    def test_one_clip_returns_asr_ser_and_deepfake_together(self, fake_conn):
        ser = {"predicted_emotion": "happy", "probabilities": {"happy": 0.8, "sad": 0.2}, "confidence": 0.8}
        add = {"predicted_label": "spoof", "synthetic_probability": 0.9, "confidence": 0.9,
               "probabilities": {"bona-fide": 0.1, "spoof": 0.9}}

        with patch.object(mt, "transcribe_whisper_base", return_value="the quick brown fox"), \
             patch.object(mt, "predict_ser", return_value=ser), \
             patch.object(mt, "predict_deepfake", return_value=add):
            result = mt.enqueue_multitask("clip.wav")
            _drain(fake_conn, mt.ASR_QUEUE_NAME, mt.SER_QUEUE_NAME, mt.ADD_QUEUE_NAME)

        agg = mt.aggregate_multitask(
            [result["asr_job_id"], result["ser_job_id"], result["add_job_id"]]
        )

        assert agg["failed"] == {}
        by_task = {r["task_name"]: r for r in agg["succeeded"].values()}
        assert set(by_task) == {"asr", "ser", "add"}
        # ASR transcript
        assert by_task["asr"]["transcript"] == "the quick brown fox"
        # SER emotion class + full distribution + confidence (LIT-206)
        assert by_task["ser"]["predicted_emotion"] == "happy"
        assert by_task["ser"]["confidence"] == 0.8
        assert set(by_task["ser"]["probabilities"]) == {"happy", "sad"}
        # Deepfake binary score (LIT-128)
        assert by_task["add"]["predicted_label"] == "spoof"
        assert by_task["add"]["synthetic_probability"] == 0.9

    def test_unified_response_is_json_serializable(self, fake_conn):
        # The demo delivers the response as JSON — no tensors/ndarrays may leak.
        ser = {"predicted_emotion": "neutral", "probabilities": {"neutral": 1.0}, "confidence": 1.0}
        add = {"predicted_label": "bona-fide", "synthetic_probability": 0.05, "confidence": 0.95,
               "probabilities": {"bona-fide": 0.95, "spoof": 0.05}}
        with patch.object(mt, "transcribe_whisper_base", return_value="hi"), \
             patch.object(mt, "predict_ser", return_value=ser), \
             patch.object(mt, "predict_deepfake", return_value=add):
            result = mt.enqueue_multitask("clip.wav")
            _drain(fake_conn, mt.ASR_QUEUE_NAME, mt.SER_QUEUE_NAME, mt.ADD_QUEUE_NAME)
        agg = mt.aggregate_multitask(
            [result["asr_job_id"], result["ser_job_id"], result["add_job_id"]]
        )
        json.dumps(agg)  # raises if anything isn't JSON-safe

    # (Graceful degradation when one child fails is covered deterministically by
    # test_multitask_orchestrator.test_asr_failure_does_not_lose_ser_result; it's
    # kept out of this suite because draining a *failing* child through a
    # SimpleWorker burst is exactly the flaky path we don't want in a
    # "no-lockup" demo check.)


class _TinyConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        return self.fc(torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1))


class TestXaiDemoPath:
    def test_all_three_attribution_methods_run(self):
        # Integrated Gradients (LIT-126) -> ms-aligned timeline
        w = torch.tensor([[1.0, 2.0, 3.0]])
        ig = integrated_gradients(lambda x: (x * w).sum(dim=1), torch.tensor([[1.0, 1.0, 1.0]]))
        timeline = attribution_timeline(ig, sample_rate=16000)
        assert len(timeline) == 3
        assert all(0.0 <= e["weight"] <= 1.0 for e in timeline)

        # Grad-CAM (LIT-148)
        cam = compute_grad_cam(_TinyConvNet(), torch.randn(1, 1, 8, 8))
        assert cam.shape == (8, 8)
        assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6

        # LIME/SHAP occlusion (LIT-130) over a spectrogram
        spec = np.random.default_rng(0).random((16, 16)).astype(np.float32)
        imp = occlusion_attribution(lambda s: float(s[:4, :4].mean()), spec, 4, 4)
        assert imp.shape == (4, 4)
