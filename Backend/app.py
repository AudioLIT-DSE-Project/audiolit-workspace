import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import librosa
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

UPLOAD_DIR = Path("/tmp/audiolit/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_DURATION_SECONDS = 15 * 60
CHUNK_SIZE = 8 * 1024
ALLOWED_EXTENSIONS = {".wav", ".mp3"}
ALLOWED_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg"}


class AudioConfig(BaseModel):
    format: str
    sample_rate_hz: int
    channels: int
    duration_sec: float
    total_bytes: int


class UploadResponse(BaseModel):
    transaction_token: str
    file_path: str
    audio_config: AudioConfig
    status: str = "uploaded"


class AudioMetadataExtractionError(Exception):
    pass


app = FastAPI(title="AudioLIT Upload Service")


def _validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported file extension. Only .wav and .mp3 are allowed.")
    return extension


def _validate_content_type(content_type: Optional[str]) -> None:
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported MIME type. Only audio/wav and audio/mpeg are allowed.")


def _validate_magic_bytes(chunk: bytes, extension: str) -> None:
    if extension == ".wav":
        if len(chunk) < 12 or not chunk.startswith(b"RIFF") or chunk[8:12] != b"WAVE":
            raise HTTPException(status_code=422, detail="Invalid WAV header.")
        return

    if extension == ".mp3":
        if chunk.startswith(b"ID3"):
            return
        if len(chunk) >= 2 and chunk[0] == 0xFF and (chunk[1] & 0xE0) == 0xE0:
            return
        raise HTTPException(status_code=422, detail="Invalid MP3 header.")


def _infer_channels(audio_data) -> int:
    if audio_data.ndim == 1:
        return 1
    return audio_data.shape[0]


async def _stream_to_disk(upload_file: UploadFile, path: Path, extension: str) -> int:
    total_bytes = 0
    header_checked = False

    async with upload_file:
        async with aiofiles.open(path, "wb") as output_file:
            async for chunk in upload_file.stream():
                if not chunk:
                    continue
                if not header_checked:
                    _validate_magic_bytes(chunk, extension)
                    header_checked = True
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Uploaded payload exceeds 100MB limit.")
                await output_file.write(chunk)

    if not header_checked:
        raise HTTPException(status_code=422, detail="Empty or invalid file payload.")

    return total_bytes


def _load_audio_metadata(path: Path) -> tuple[int, int, float]:
    info = sf.info(str(path))
    duration = librosa.get_duration(filename=str(path))
    return info.samplerate, info.channels, duration


async def _extract_audio_metadata(path: Path, extension: str) -> AudioConfig:
    try:
        sample_rate, channels, duration = await asyncio.to_thread(_load_audio_metadata, path)
    except Exception as exc:
        raise AudioMetadataExtractionError("Audio metadata extraction failed.") from exc

    if duration <= 0:
        raise AudioMetadataExtractionError("Audio duration could not be determined.")

    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(status_code=422, detail="Audio duration exceeds 15 minutes limit.")

    return AudioConfig(
        format=extension.lstrip("."),
        sample_rate_hz=sample_rate,
        channels=channels,
        duration_sec=duration,
        total_bytes=path.stat().st_size,
    )


@app.post("/api/audio/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)) -> UploadResponse:
    extension = _validate_extension(file.filename)
    _validate_content_type(file.content_type)

    transaction_token = f"aud_{uuid.uuid4().hex}"
    temp_file = tempfile.NamedTemporaryFile(prefix=transaction_token + "_", suffix=".tmp", dir=UPLOAD_DIR, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        await _stream_to_disk(file, temp_path, extension)
        audio_config = await _extract_audio_metadata(temp_path, extension)
    except HTTPException:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    except AudioMetadataExtractionError as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Audio upload failed.") from exc

    return UploadResponse(
        transaction_token=transaction_token,
        file_path=str(temp_path),
        audio_config=audio_config,
        status="uploaded",
    )
