from fastapi import APIRouter, UploadFile, File, Form,HTTPException
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
from pathlib import Path
import uuid
import librosa
import soundfile as sf
import requests
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

@router.get("/upload/test")
async def test_upload_endpoint():
    """Test endpoint to verify upload service is working"""
    return {"status": "Upload service is working", "upload_dir": str(UPLOAD_DIR.absolute())}

@router.post("/upload")
async def upload_audio_file(file: UploadFile = File(...),model: str = Form(...)):
    """
    Upload an audio file and return the file path for processing
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are allowed.")
    
    # Validate file extension
    allowed_extensions = ['.wav', '.mp3', '.m4a', '.flac']
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file extension. Allowed: {', '.join(allowed_extensions)}")
    
    try:
        # Generate unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

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
            audio_data, sample_rate = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sample_rate)
            file_size = file_path.stat().st_size
        except Exception:
            file_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail="File could not be decoded as audio. It may be corrupted or in an unsupported format.",
            )

        # FR3.2 / SAD §3.6.2: no model inference on the request path. This route
        # used to await a full forward pass and an embedding extraction before
        # responding, which made upload latency a function of model speed and
        # made it the most visible violation of the async architecture the SAD
        # describes.
        #
        # Nothing is lost: the client dispatches the real multi-task job to
        # POST /api/inference/multitask immediately after upload and follows it
        # over the WebSocket channel, and embeddings are computed on demand by
        # POST /inferences/embeddings. `prediction` stays in the response shape,
        # as null, so existing callers keep parsing.
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully",
                "filename": file.filename,
                "file_path": str(file_path),
                "file_id": unique_filename,
                "duration": duration,
                "sample_rate": sample_rate,
                "size": file_size,
                "prediction": None
            }
        )

    except HTTPException:
        # Preserve the specific status/detail raised above (413 oversized,
        # 422 undecodable) instead of letting the generic handler below
        # flatten it into a 500.
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
@router.options("/upload/file/{file_id}")
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
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            'Accept-Ranges': 'bytes',
            'Cache-Control': 'public, max-age=3600',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range, Accept-Encoding',
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
