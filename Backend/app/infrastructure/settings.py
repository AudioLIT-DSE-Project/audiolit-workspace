from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # use "none" on cross-site + https
    COOKIE_DOMAIN: str | None = None

    # FR2.2 — active dataset working footprint bound (~100 GB across all
    # seven corpora per the SRS) and the per-request row cap that keeps
    # `/{dataset}/metadata` from materializing an entire large corpus.
    DATASET_FOOTPRINT_LIMIT_GB: float = 100.0
    DATASET_METADATA_ROW_CAP: int = 2000

settings = Settings()
