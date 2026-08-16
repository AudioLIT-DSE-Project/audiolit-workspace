"""Endpoint tests for acoustic.py (LIT-231, FR10).

Pure DSP, no model load, so these run the real
`acoustic_profiler_service.extract_acoustic_profile` end-to-end against a
synthetic WAV file rather than mocking it -- fast enough (a few hundred ms of
audio) not to need mocking, and it's the route's file-resolution/HTTP-status
behaviour under test here, not the DSP itself (already covered by
test_acoustic_profiler_service.py).
"""
from __future__ import annotations

import pytest


class TestAcousticProfile:
    @pytest.mark.asyncio
    async def test_profile_by_file_path_returns_aligned_timeline(self, client, sample_audio_file):
        r = await client.post("/acoustic/profile", json={"file_path": str(sample_audio_file)})

        assert r.status_code == 200
        body = r.json()
        assert body["sample_rate"] == 16000
        assert body["duration_s"] == pytest.approx(5.0, abs=0.1)
        assert len(body["timeline"]) > 0
        first = body["timeline"][0]
        assert set(first.keys()) == {"t_ms", "f0_hz", "rms"}

    @pytest.mark.asyncio
    async def test_missing_file_returns_404(self, client):
        r = await client.post("/acoustic/profile", json={"file_path": "/no/such/file.wav"})
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_missing_reference_returns_400(self, client):
        r = await client.post("/acoustic/profile", json={})
        assert r.status_code == 400
