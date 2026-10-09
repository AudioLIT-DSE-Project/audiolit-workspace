"""SR1's duration cap and SR4/C4's retention sweep on the upload route.

Both were requirements with no implementation rather than requirements with a
broken implementation, which is why nothing failed: the size cap was enforced
and the duration was measured, reported to the UI, and then never compared
against anything; and uploaded audio was only ever removed by an explicit
DELETE the browser had to remember to send.

The upload route had no tests at all, so these are also its first.

Requests go through httpx's AsyncClient rather than TestClient, matching
tests/test_security.py: TestClient drives the app on its own portal event loop,
which binds the session middleware's async redis pool to that loop and then
breaks conftest's fakeredis teardown on the pytest-asyncio loop.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import soundfile as sf
from httpx import AsyncClient

import app.api.routes.upload as upload
from app.main import app


def _wav_bytes(seconds: float, sample_rate: int = 8000) -> bytes:
    """A decodable mono clip of a given length, small even when long."""
    buf = io.BytesIO()
    samples = np.zeros(int(seconds * sample_rate), dtype="float32")
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


async def _post_clip(seconds: float, name: str = "clip.wav"):
    async with AsyncClient(app=app, base_url="http://test") as client:
        return await client.post(
            "/upload",
            files={"file": (name, _wav_bytes(seconds), "audio/wav")},
            data={"model": "whisper-base"},
        )


def _aged_file(directory, name: str, age_seconds: float):
    path = directory / name
    path.write_bytes(b"x")
    stamp = time.time() - age_seconds
    os.utime(path, (stamp, stamp))
    return path


class TestDurationCap:
    async def test_a_clip_over_the_cap_is_rejected(self, monkeypatch, tmp_path):
        # The cap is lowered rather than generating a 15-minute clip: what is
        # under test is the comparison, not librosa's duration arithmetic.
        monkeypatch.setattr(upload, "MAX_UPLOAD_DURATION_SECONDS", 1.0)
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)

        response = await _post_clip(3.0, "long.wav")

        assert response.status_code == 413, (
            f"a clip over the duration cap returned {response.status_code}; "
            "SR1 requires it to be rejected before hashing"
        )
        assert "minutes" in response.json()["detail"]
        assert not list(tmp_path.iterdir()), (
            "the rejected clip was left on disk, so the cap bounds what is "
            "analysed but not what is stored"
        )

    async def test_a_clip_under_the_cap_is_accepted(self, monkeypatch, tmp_path):
        monkeypatch.setattr(upload, "MAX_UPLOAD_DURATION_SECONDS", 60.0)
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)

        response = await _post_clip(2.0, "short.wav")

        assert response.status_code == 200, response.text

    async def test_a_zero_cap_disables_the_check(self, monkeypatch, tmp_path):
        # The env override exists so a local operator can lift the limit; 0
        # must mean "no limit", not "reject everything".
        monkeypatch.setattr(upload, "MAX_UPLOAD_DURATION_SECONDS", 0.0)
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)

        response = await _post_clip(2.0, "any.wav")

        assert response.status_code == 200, response.text


class TestDecodeRejection:
    """An undecodable body must be reported as such, not as a server error.

    The handler for this path logged before raising, and `logger` was never
    defined in the module, so the handler itself raised NameError. The outer
    `except Exception` caught that and returned
    `500 name 'logger' is not defined` - an internal error for what is
    actually a bad request, with the real reason hidden from the user and the
    422 branch unreachable. Found by sending a non-audio body, which no test
    had done.
    """

    async def test_a_non_audio_body_is_rejected_as_unprocessable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)

        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/upload",
                files={"file": ("broken.wav", b"not audio at all" * 50, "audio/wav")},
                data={"model": "whisper-base"},
            )

        assert response.status_code == 422, (
            f"an undecodable upload returned {response.status_code}: "
            f"{response.text[:200]}"
        )
        assert "decoded" in response.json()["detail"]
        assert not list(tmp_path.iterdir()), "the undecodable file was left on disk"


class TestRetentionSweep:
    def test_files_past_the_window_are_removed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(upload, "UPLOAD_RETENTION_SECONDS", 3600.0)

        stale = _aged_file(tmp_path, "stale.wav", 7200)
        fresh = _aged_file(tmp_path, "fresh.wav", 60)

        removed = upload.purge_expired_uploads()

        assert removed == 1
        assert not stale.exists()
        assert fresh.exists(), (
            "the sweep deleted a file inside the retention window, which would "
            "pull audio out from under an analysis still running on it"
        )

    def test_a_zero_window_keeps_everything(self, monkeypatch, tmp_path):
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(upload, "UPLOAD_RETENTION_SECONDS", 0.0)

        old = _aged_file(tmp_path, "ancient.wav", 10 ** 7)

        assert upload.purge_expired_uploads() == 0
        assert old.exists()

    def test_a_missing_upload_directory_is_not_an_error(self, monkeypatch, tmp_path):
        # The directory is created at import time, but a deployment that wipes
        # its volume between restarts must not 500 on the next upload.
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path / "gone")
        monkeypatch.setattr(upload, "UPLOAD_RETENTION_SECONDS", 3600.0)

        assert upload.purge_expired_uploads() == 0

    async def test_uploading_triggers_the_sweep(self, monkeypatch, tmp_path):
        """The sweep is what makes retention automatic; if the handler stops
        calling it, files accumulate again with no visible symptom."""
        monkeypatch.setattr(upload, "UPLOAD_DIR", tmp_path)
        monkeypatch.setattr(upload, "UPLOAD_RETENTION_SECONDS", 3600.0)
        monkeypatch.setattr(upload, "MAX_UPLOAD_DURATION_SECONDS", 60.0)

        stale = _aged_file(tmp_path, "stale.wav", 7200)

        response = await _post_clip(1.0, "new.wav")

        assert response.status_code == 200, response.text
        assert not stale.exists(), "the upload handler did not purge expired files"
