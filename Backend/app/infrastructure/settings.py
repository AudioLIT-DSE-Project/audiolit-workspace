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

    # SRS §3.10 / SAD §9 — durable MongoDB metadata tier. Records must survive
    # in MongoDB; nothing here ever touches audio bytes (constraint C4, SR4).
    # Empty by default = the tier is "configured off": `get_metadata_store()`
    # returns None and every write-through is a logged no-op, so local
    # development and CI never need a MongoDB server (SAD §11.1 degradation).
    # Set MONGO_URL (e.g. mongodb://127.0.0.1:27017) to enable the tier.
    MONGO_URL: str = ""
    MONGO_DB_NAME: str = "audiolit"
    #: Ordinary analysis records expire after a day; bias reports are retained
    #: permanently (SAD §9).
    MONGO_ANALYSIS_TTL_HOURS: int = 24
    #: How long to wait for a Mongo server before declaring it unavailable and
    #: degrading writes to no-ops (SRS §3.3.1 graceful degradation).
    MONGO_SERVER_SELECTION_TIMEOUT_MS: int = 1500

settings = Settings()
