from datetime import datetime, timezone

from fastapi import APIRouter

from app.schemas.responses import HealthResponse

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="audiolit-api",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
