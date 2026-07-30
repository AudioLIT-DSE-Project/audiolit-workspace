from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    timestamp: str


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
    status: str


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
    path: str
    timestamp: str
