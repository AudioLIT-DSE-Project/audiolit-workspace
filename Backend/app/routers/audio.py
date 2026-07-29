import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import aiofiles
import librosa
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.responses import AudioConfig, UploadResponse

router = APIRouter()
UPLOAD_DIR = Path("/tmp/audiolit/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024
MAX_DURATION_SECONDS = 15 * 60
ALLOWED_EXTENSIONS = {".wav", ".mp3"}
ALLOWED_MIME_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg"}
CHUNK_SIZE = 8 * 1024


def _validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=422, detail="Unsupported file extension. Only .wav and .mp3 are allowed.")
    return extension


def _validate_content_type(content_type: Optional[str]) -> None:
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported MIME type. Only audio/wav and audio/mpeg are allowed.")


def _validate_magic_bytes(path: Path, extension: str) -> None:
    with path.open("rb") as file_handle:
        header = file_handle.read(12)

    if extension == ".wav":
        if len(header) < 12 or not header.startswith(b"RIFF") or header[8:12] != b"WAVE":
            raise HTTPException(status_code=422, detail="Invalid WAV header.")
        return

    if extension == ".mp3":
        if header.startswith(b"ID3"):
            return
        if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
            return
        raise HTTPException(status_code=422, detail="Invalid MP3 header.")


def _load_audio_metadata(path: Path) -> tuple[int, int, float]:
    info = sf.info(str(path))
    duration = librosa.get_duration(filename=str(path))
    return info.samplerate, info.channels, duration


async def _extract_audio_metadata(path: Path, extension: str) -> AudioConfig:
    try:
        sample_rate, channels, duration = await asyncio.to_thread(_load_audio_metadata, path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Audio metadata extraction failed.") from exc

    if duration <= 0:
        raise HTTPException(status_code=422, detail="Audio duration could not be determined.")
    if duration > MAX_DURATION_SECONDS:
        raise HTTPException(status_code=413, detail="Audio duration exceeds 15 minutes limit.")

    return AudioConfig(
        format=extension.lstrip("."),
        sample_rate_hz=sample_rate,
        channels=channels,
        duration_sec=duration,
        total_bytes=path.stat().st_size,
    )


async def _write_upload_to_disk(upload_file: UploadFile, destination: Path) -> int:
    total_bytes = 0
    async with upload_file:
        async with aiofiles.open(destination, "wb") as out_file:
            async for chunk in upload_file.stream():
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Uploaded payload exceeds 100MB limit.")
                await out_file.write(chunk)
    return total_bytes


@router.post("/audio/upload", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)) -> UploadResponse:
    extension = _validate_extension(file.filename)
    _validate_content_type(file.content_type)

    transaction_token = f"aud_{uuid.uuid4().hex}"
    temp_file = tempfile.NamedTemporaryFile(prefix=transaction_token + "_", suffix=".tmp", dir=UPLOAD_DIR, delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        await _write_upload_to_disk(file, temp_path)
        _validate_magic_bytes(temp_path, extension)
        audio_config = await _extract_audio_metadata(temp_path, extension)
    except HTTPException:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Audio upload failed.")

    return UploadResponse(
        transaction_token=transaction_token,
        file_path=str(temp_path),
        audio_config=audio_config,
        status="uploaded",
    )
