from datetime import datetime, timezone

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.responses import ErrorResponse


async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ErrorResponse(
        error_code="VALIDATION_ERROR",
        detail="Validation failed for the request.",
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    error_code = "NOT_FOUND" if exc.status_code == 404 else "VALIDATION_ERROR" if exc.status_code == 422 else "INTERNAL_ERROR"
    payload = ErrorResponse(
        error_code=error_code,
        detail=str(exc.detail),
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    payload = ErrorResponse(
        error_code="INTERNAL_ERROR",
        detail="The request could not be processed.",
        path=str(request.url.path),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return JSONResponse(status_code=500, content=payload.model_dump())
