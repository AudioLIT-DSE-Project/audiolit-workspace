from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from app.middleware.error_handler import (
    http_exception_handler,
    request_validation_error_handler,
    unexpected_exception_handler,
)
from app.routers.audio import router as audio_router
from app.routers.base import router as base_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AudioLIT API",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)

    app.include_router(base_router, prefix="/api")
    app.include_router(audio_router, prefix="/api")

    return app


app = create_app()
