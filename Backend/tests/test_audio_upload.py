import io
import tempfile
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def make_wav_bytes(duration_seconds: float = 0.05, framerate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(framerate)
        frame_count = int(duration_seconds * framerate)
        writer.writeframes(b"\x00\x00" * frame_count)
    return buffer.getvalue()


def test_upload_audio_wav_streams_successfully() -> None:
    wav_bytes = make_wav_bytes(duration_seconds=0.05)
    response = client.post(
        "/api/audio/upload",
        files={"file": ("sample.wav", wav_bytes, "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["transaction_token"].startswith("aud_")
    assert payload["file_path"].endswith(".tmp")
    assert payload["status"] == "uploaded"

    audio_config = payload["audio_config"]
    assert audio_config["format"] == "wav"
    assert audio_config["sample_rate_hz"] == 8000
    assert audio_config["channels"] == 1
    assert audio_config["duration_sec"] == pytest.approx(0.05, rel=0.1)
    assert audio_config["total_bytes"] == len(wav_bytes)

    stored_file = Path(payload["file_path"])
    assert stored_file.exists()
    assert stored_file.is_file()
    assert str(UPLOAD_DIR := Path("/tmp/audiolit/uploads")) in str(stored_file)
