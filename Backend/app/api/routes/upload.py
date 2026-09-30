from fastapi import APIRouter, UploadFile, File, Form,HTTPException
from fastapi.responses import JSONResponse, FileResponse
import logging
import os
import shutil
import time
from pathlib import Path
import uuid
import librosa
import soundfile as sf
import requests

# The decode-failure handler below logs before raising. Without this the
# handler itself raised NameError, which the outer `except Exception` turned
# into a 500 "name 'logger' is not defined" - so every undecodable upload
# reported an internal error instead of the 422 it was written to return.
logger = logging.getLogger(__name__)

router = APIRouter()

# Ensure uploads directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# LIT-160: previously unbounded - shutil.copyfileobj wrote a request body to
# disk in full before anything checked its size, so a multi-GB upload was
# fully buffered before being rejected (if ever). Enforced by counting bytes
# during the streamed write below rather than trusting a Content-Length
# header, which can be absent or spoofed.
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("AUDIOLIT_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100MB
UPLOAD_CHUNK_SIZE = 1024 * 1024

# SR1 - the size cap above was enforced but the duration cap in the same
# requirement was not: duration was measured and reported, never checked. A
# 40-minute 8 kHz mono clip is well under 100 MB and used to be accepted, then
# fanned out to five workers holding the whole decoded array in memory.
MAX_UPLOAD_DURATION_SECONDS = float(os.getenv("AUDIOLIT_MAX_UPLOAD_SECONDS", str(15 * 60)))  # 15 min

# SR4 / constraint C4 - uploaded audio is transient. Only an explicit
# DELETE /upload/{file_id} removed it before, so anything the browser never
# deleted (a closed tab, a failed request) stayed on disk indefinitely. The
# sweep below runs on upload and at startup; set to 0 to keep files.
UPLOAD_RETENTION_SECONDS = float(os.getenv("AUDIOLIT_UPLOAD_RETENTION_SECONDS", str(24 * 60 * 60)))  # 24 h


def purge_expired_uploads(retention_seconds: float | None = None) -> int:
    """Delete uploads older than the retention window. Returns the count.

    Best-effort by design: a file another request is mid-read on may fail to
    unlink on Windows, and that must not fail the upload that triggered the
    sweep.
    """
    window = UPLOAD_RETENTION_SECONDS if retention_seconds is None else retention_seconds
    if window <= 0:
        return 0
    cutoff = time.time() - window
    removed = 0
    try:
        entries = list(UPLOAD_DIR.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed

@router.get("/upload/test")
async def test_upload_endpoint():
    """Test endpoint to verify upload service is working"""
    return {"status": "Upload service is working", "upload_dir": str(UPLOAD_DIR.absolute())}

@router.post("/upload")
async def upload_audio_file(file: UploadFile = File(...), model: str = Form("whisper-base")):
    """
    Upload an audio file and return the file path for processing
    """
    purge_expired_uploads()

    # Validate file type (allow audio/* and video/webm commonly emitted by browser MediaRecorder)
    if file.content_type and not (file.content_type.startswith('audio/') or file.content_type in ['video/webm', 'application/octet-stream']):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")
    
    # Validate file extension
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.webm', '.ogg', '.aac', '.opus']
    file_extension = Path(file.filename).suffix.lower() if file.filename else ''
    if not file_extension or file_extension not in allowed_extensions:
        # Default fallback for blob uploads without explicit extension
        file_extension = '.webm' if 'webm' in (file.content_type or '') else '.wav'
    
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        # Stream to disk with a hard cap instead of shutil.copyfileobj's
        # unbounded copy - abort as soon as the cap is crossed rather than
        # after the whole body has already been written.
        bytes_written = 0
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_SIZE_BYTES:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum upload size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
                    )
                buffer.write(chunk)

        # Get audio metadata; reject files librosa can't decode instead of
        # silently accepting them with duration=0, which was indistinguishable
        # from an actual zero-length clip.
        try:
            try:
                audio_data, sample_rate = sf.read(file_path)
                duration = float(len(audio_data)) / float(sample_rate) if sample_rate > 0 else 0.0
            except Exception:
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                duration = float(librosa.get_duration(y=audio_data, sr=sample_rate))
            file_size = file_path.stat().st_size
        except Exception as decode_err:
            logger.error(f"Audio decoding failure on {file_path}: {decode_err}")
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail=f"File could not be decoded as audio. It may be corrupted or in an unsupported format.",
            )

        if MAX_UPLOAD_DURATION_SECONDS > 0 and duration > MAX_UPLOAD_DURATION_SECONDS:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Audio is {duration / 60:.1f} minutes long; the maximum is "
                    f"{MAX_UPLOAD_DURATION_SECONDS / 60:.0f} minutes."
                ),
            )

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename or unique_filename,
                "file_path": str(file_path),
                "file_id": unique_filename,
                "duration": duration,
                "sample_rate": sample_rate,
                "size": file_size,
                "prediction": None
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@router.delete("/upload/{file_id}")
async def delete_uploaded_file(file_id: str):
    """
    Delete an uploaded file
    """
    file_path = UPLOAD_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return JSONResponse(
            status_code=200,
            content={"message": "File deleted successfully"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@router.get("/upload/file/{file_id}")
@router.head("/upload/file/{file_id}")
async def serve_audio_file(file_id: str):
    """
    Serve an uploaded audio file for playback
    """
    file_path = UPLOAD_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine the correct media type based on file extension
    file_extension = file_path.suffix.lower()
    media_type_map = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac'
    }
    media_type = media_type_map.get(file_extension, 'audio/*')
    
    # LIT-223: remove the route-level Access-Control-Allow-Origin: "*" here -
    # CORS is the app's CORSMiddleware's job, with a restricted origin allow-list.
    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'public, max-age=3600',
            'Content-Disposition': f'inline; filename="{file_id}"'
        }
    )

@router.get("/upload/metadata/{file_id}")
async def get_audio_metadata(file_id: str):
    """
    Get metadata for an uploaded audio file
    """
    file_path = UPLOAD_DIR / file_id
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        file_size = file_path.stat().st_size
        
        return JSONResponse(
            status_code=200,
            content={
                "file_id": file_id,
                "duration": duration,
                "sample_rate": sample_rate,
                "size": file_size,
                "channels": 1 if len(audio_data.shape) == 1 else audio_data.shape[0]
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read audio metadata: {str(e)}")

@router.get("/upload/list")
async def list_uploaded_files():
    """
    List all uploaded files
    """
    try:
        files = []
        for file_path in UPLOAD_DIR.iterdir():
            if file_path.is_file():
                files.append({
                    "file_id": file_path.name,
                    "filename": file_path.name,
                    "size": file_path.stat().st_size,
                    "created_at": file_path.stat().st_ctime
                })
        
        return JSONResponse(
            status_code=200,
            content={"files": files}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")
