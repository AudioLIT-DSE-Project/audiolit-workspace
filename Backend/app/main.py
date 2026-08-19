import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .infrastructure.session import SessionMiddleware
from .infrastructure import dataset_ingestion
from .infrastructure.settings import settings
from .api.routes import inference as inference_routes
from .api.routes import session as session_routes, results as results_routes, inferences as inferences_routes, upload as upload_routes, health as health_routes
from .api.routes import datasets as datasets_routes, saliency as saliency_routes, perturbations as perturbations_routes, dataset_management as dataset_management_routes, debug as debug_routes
from .api.routes import tasks as tasks_routes
from .api.routes import models as models_routes, acoustic as acoustic_routes, evaluation as evaluation_routes

logger = logging.getLogger(__name__)
app = FastAPI(title="LIT for Voice – API")


@app.on_event("startup")
async def _warn_if_dataset_footprint_over_limit() -> None:
    """FR2.2 — surface it in the logs, not just via GET /datasets/footprint,
    the moment the provisioned corpora exceed the ~100 GB working bound."""
    try:
        usage = dataset_ingestion.measure_footprint()
        total_gb = sum(usage.values()) / (1024 ** 3)
        if total_gb > settings.DATASET_FOOTPRINT_LIMIT_GB:
            logger.warning(
                "Dataset working footprint is %.1f GB, over the configured %.1f GB "
                "limit (FR2.2). Per-corpus usage: %s",
                total_gb, settings.DATASET_FOOTPRINT_LIMIT_GB, usage,
            )
    except Exception:
        logger.warning("Could not measure dataset footprint at startup", exc_info=True)

# Configure CORS origins - default to common development origins if not set
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
if allowed_origins_env:
    origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
else:
    # Default development origins
    origins = [
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://localhost:8081",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081",
        "http://127.0.0.1:5173"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(SessionMiddleware)
app.include_router(tasks_routes.router)
app.include_router(inference_routes.router)
app.include_router(session_routes.router, tags=["Session"])
app.include_router(results_routes.router, tags=["Results"])
app.include_router(inferences_routes.router, tags=["Inferences"])
app.include_router(upload_routes.router, tags=["Upload"])
app.include_router(dataset_management_routes.router, prefix="/upload", tags=["Dataset Management"])
app.include_router(datasets_routes.router, tags=["Datasets"])
app.include_router(saliency_routes.router, tags=["Saliency"])
app.include_router(perturbations_routes.router, tags=["Perturbations"])
app.include_router(health_routes.router, tags=["Health"])
app.include_router(debug_routes.router, tags=["Debug"])
app.include_router(models_routes.router, tags=["Models"])
app.include_router(acoustic_routes.router, tags=["Acoustic"])
app.include_router(evaluation_routes.router, tags=["Evaluation"])
