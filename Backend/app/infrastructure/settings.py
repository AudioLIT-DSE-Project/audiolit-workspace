from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"

    # LIT-259 — structured task logs. "json" makes every audiolit.* log record a
    # single JSON line ({"ts","level","logger","event",...extra}); "text" falls
    # back to the default human-readable formatting for local dev.
    LOG_FORMAT: str = "json"
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

    # SRS §3.10 / SAD §9 — durable MongoDB metadata tier. Records must survive
    # in MongoDB; nothing here ever touches audio bytes (constraint C4, SR4).
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "audiolit"
    #: Ordinary analysis records expire after a day; bias reports are retained
    #: permanently (SAD §9).
    MONGO_ANALYSIS_TTL_HOURS: int = 24
    #: How long to wait for a Mongo server before declaring it unavailable and
    #: degrading writes to no-ops (SRS §3.3.1 graceful degradation).
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 1500

settings = Settings()
