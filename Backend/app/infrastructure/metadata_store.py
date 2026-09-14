"""MongoDB 6.0+ metadata tier (SRS §3.10, SAD §9, constraints C4/SR4).

AudioLIT keeps two stores with separate lifetimes (SAD §9): Redis holds the
temporary, recomputable cache and job queue; MongoDB holds **durable records**
that must survive - models, analysis results, and bias reports. MongoDB stores
metadata only, never audio bytes: audio records carry a file-path reference
and analysis records carry a Redis tensor key rather than the tensor itself.

Collections (SRS §3.10, one per document family):

* ``models``           - model_id, name, architecture, revision, weight_digest,
                         hf_model_id (reproducibility: revision + weight digest)
* ``audio_samples``    - sample_id, filename, duration, sample_rate,
                         file_path_reference, uploaded_at  (file-path *only*)
* ``analysis_results`` - analysis_id, sample_id, model_id, task, prediction,
                         redis_tensor_key, created_at  (TTL 24h; tensor key only)
* ``bias_reports``     - report_id, model_id, cohort, WER, disparity_metrics,
                         created_at  (retained permanently, beyond the TTL window)

Durability semantics (SAD §9): bias reports are kept permanently; ordinary
analysis records expire after 24 hours via a TTL index on ``created_at``.

Graceful degradation (SRS §3.3.1): MongoDB is a supporting record store, not a
request-path dependency. If it is unreachable, every write is a no-op (the
durable record is skipped, not fatal) and reads return empty/None, so models,
inference, and the cache keep working. ``available`` reports the current state
for dashboards and health checks. Detailed degradation behaviour is verified
in LIT-258's test tier.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Mapping, Optional

from .settings import settings

logger = logging.getLogger("audiolit.metadata_store")

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import (
        ConnectionFailure,
        PyMongoError,
        ServerSelectionTimeoutError,
    )
    from pymongo.database import Database

    _PYMONGO_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency declared in requirements.txt
    ASCENDING = DESCENDING = "x"
    MongoClient = None  # type: ignore[assignment,misc]
    ConnectionFailure = PyMongoError = ServerSelectionTimeoutError = Exception
    logging.getLogger("audiolit.metadata_store").warning(
        "pymongo is not installed; MongoDB metadata tier is unavailable."
    )
    _PYMONGO_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Document schemas (SRS §3.10)
# --------------------------------------------------------------------------- #

# JSON-schema-ish validation documents for Mongo's `create_collection(validator=)`.
MODELS_SCHEMA: Mapping[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["model_id", "name", "architecture", "revision"],
        "properties": {
            "model_id": {"bsonType": "string"},
            "name": {"bsonType": "string"},
            "architecture": {"bsonType": "string"},
            "revision": {"bsonType": "string"},
            "weight_digest": {"bsonType": "string"},
            "hf_model_id": {"bsonType": "string"},
        },
    }
}

AUDIO_SAMPLES_SCHEMA: Mapping[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["sample_id", "file_path_reference"],
        "properties": {
            "sample_id": {"bsonType": "string"},
            "filename": {"bsonType": "string"},
            "duration": {"bsonType": "number"},
            "sample_rate": {"bsonType": "number"},
            "file_path_reference": {"bsonType": "string"},
            "uploaded_at": {"bsonType": "object"},
        },
    }
}

ANALYSIS_RESULTS_SCHEMA: Mapping[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["analysis_id", "task", "created_at"],
        "properties": {
            "analysis_id": {"bsonType": "string"},
            "sample_id": {"bsonType": "string"},
            "model_id": {"bsonType": "string"},
            "task": {"bsonType": "string"},
            "prediction": {"bsonType": "object"},
            "redis_tensor_key": {"bsonType": "string"},
            "created_at": {"bsonType": "object"},
        },
    }
}

BIAS_REPORTS_SCHEMA: Mapping[str, Any] = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["report_id", "model_id", "cohort", "created_at"],
        "properties": {
            "report_id": {"bsonType": "string"},
            "model_id": {"bsonType": "string"},
            "cohort": {"bsonType": "string"},
            "WER": {"bsonType": "number"},
            "disparity_metrics": {"bsonType": "object"},
            "created_at": {"bsonType": "object"},
        },
    }
}

#: The four collections of the tier, name -> validator.
COLLECTIONS: Mapping[str, Mapping[str, Any]] = {
    "models": MODELS_SCHEMA,
    "audio_samples": AUDIO_SAMPLES_SCHEMA,
    "analysis_results": ANALYSIS_RESULTS_SCHEMA,
    "bias_reports": BIAS_REPORTS_SCHEMA,
}

#: Compound indexes on frequently queried fields (SRS §3.10 "indexing and
#: retention"); sample/model identifiers span the cross-collection joins in
#: the figure. There is deliberately no index on any field derived from audio.
INDEXES: Mapping[str, list[tuple[Any, Any]]] = {
    "models": [("model_id", ASCENDING)],
    "audio_samples": [("sample_id", ASCENDING)],
    # Compound (sample_id, model_id) supports "analysis history for this
    # sample" and "all analyses of this model on this sample".
    "analysis_results": [
        ("analysis_id", ASCENDING),
        ("sample_id", ASCENDING),
        ("model_id", ASCENDING),
        [("sample_id", ASCENDING), ("model_id", ASCENDING)],
    ],
    # Compound (model_id, cohort) supports "all reports for one model/cohort".
    "bias_reports": [
        ("report_id", ASCENDING),
        ("model_id", ASCENDING),
        ("cohort", ASCENDING),
        [("model_id", ASCENDING), ("cohort", ASCENDING)],
    ],
}


# --------------------------------------------------------------------------- #
# Metadata store
# --------------------------------------------------------------------------- #

class MetadataStore:
    """Durable record store backed by MongoDB 6.0+.

    Lazy client: the client is built on first write/read, so importing this
    module never requires a live server. Writes degrade to logged no-ops when
    the server is unreachable (SRS §3.3.1); no code path raises on MongoDB
    being down.
    """

    def __init__(self, client: Any | None = None, db: Any | None = None) -> None:
        """Inject an explicit client/db (tests use mongomock) or stay lazy."""
        self._client = client
        self._db = db

    # -- connection -------------------------------------------------------- #

    @property
    def available(self) -> bool:
        """True if the client can reach MongoDB right now.

        ``serverSelectionTimeoutMS`` bounds the probe so a dead server cannot
        hang a worker or a health check (SRS §3.3.1).
        """
        if not _PYMONGO_AVAILABLE:
            return False
        try:
            return self._get_db().client.admin.command("ping") is not None
        except (ConnectionFailure, ServerSelectionTimeoutError, PyMongoError):
            return False

    def _get_db(self) -> Any:
        if self._db is not None:
            return self._db
        if self._client is None:
            self._client = MongoClient(
                settings.MONGO_URL,
                serverSelectionTimeoutMS=settings.MONGO_SERVER_SELECTION_TIMEOUT_MS,
            )
        self._db = self._client[settings.MONGO_DB_NAME]
        return self._db

    def ensure_schema(self) -> None:
        """Provision collections with document validation + indexes.

        Mongo's ``create_collection(validator=...)`` is used for validation, a
        TTL index gives transient analysis records their 24-hour expiry
        (SRS §3.10). Bias reports get no TTL index, so they survive beyond the
        analysis window (retained permanently, SAD §9).
        """
        db = self._get_db()
        for name, validator in COLLECTIONS.items():
            if name not in db.list_collection_names():
                try:
                    db.create_collection(name, validator=validator)
                except NotImplementedError:
                    # mongomock (and some drivers) reject schema validators.
                    logger.debug("metadata.create_collection(no-validator): %s", name)
                    db.create_collection(name)
        self._ensure_indexes(db)

    def _ensure_indexes(self, db: Any) -> None:
        ttl_seconds = settings.MONGO_ANALYSIS_TTL_HOURS * 3600
        per_collection = {
            "models": ("model_id", 1),
            "audio_samples": ("sample_id", 1),
            "analysis_results": ("analysis_id", 1),
            "bias_reports": ("report_id", 1),
        }
        for name, (id_field, direction) in per_collection.items():
            db[name].create_index([(id_field, direction)], unique=True)
        # Cross-collection compound indexes (SRS §3.10).
        db["analysis_results"].create_index([("sample_id", 1), ("model_id", 1)])
        db["bias_reports"].create_index([("model_id", 1), ("cohort", 1)])
        # TTL: transient analyses expire; bias reports intentionally lack one.
        db["analysis_results"].create_index(
            [("created_at", 1)], expireAfterSeconds=ttl_seconds
        )

    def _collection(self, name: str) -> Any:
        return self._get_db()[name]

    # -- models ------------------------------------------------------------ #

    def upsert_model(self, model: Mapping[str, Any]) -> bool:
        """Create or refresh a model record; returns True on write, False on
        degraded (MongoDB unavailable)."""
        if not self._try_preflight():
            return False
        try:
            self._collection("models").update_one(
                {"model_id": model["model_id"]},
                {"$set": dict(model)},
                upsert=True,
            )
            return True
        except PyMongoError as exc:
            logger.warning("metadata.models.upsert_failed: %s", exc)
            return False

    def get_model(self, model_id: str) -> Optional[dict[str, Any]]:
        try:
            return self._collection("models").find_one({"model_id": model_id})
        except PyMongoError as exc:
            logger.warning("metadata.models.read_failed: %s", exc)
            return None

    def list_models(self) -> list[dict[str, Any]]:
        try:
            return list(self._collection("models").find())
        except PyMongoError as exc:
            logger.warning("metadata.models.list_failed: %s", exc)
            return []

    # -- audio samples ----------------------------------------------------- #

    def upsert_audio_sample(self, sample: Mapping[str, Any]) -> bool:
        """Record a sample's *metadata*; ``file_path_reference`` only, never
        audio bytes (SRS §3.10 / C4)."""
        if not self._try_preflight():
            return False
        try:
            self._collection("audio_samples").update_one(
                {"sample_id": sample["sample_id"]},
                {"$set": dict(sample)},
                upsert=True,
            )
            return True
        except PyMongoError as exc:
            logger.warning("metadata.samples.upsert_failed: %s", exc)
            return False

    def get_audio_sample(self, sample_id: str) -> Optional[dict[str, Any]]:
        try:
            return self._collection("audio_samples").find_one({"sample_id": sample_id})
        except PyMongoError as exc:
            logger.warning("metadata.samples.read_failed: %s", exc)
            return None

    # -- analysis results -------------------------------------------------- #

    def insert_analysis(self, analysis: Mapping[str, Any]) -> bool:
        """Persist one analysis record (redis_tensor_key, not the tensor)."""
        if not self._try_preflight():
            return False
        try:
            if "created_at" not in analysis:
                analysis = {**analysis, "created_at": time.time()}
            self._collection("analysis_results").insert_one(dict(analysis))
            return True
        except PyMongoError as exc:
            logger.warning("metadata.analysis.insert_failed: %s", exc)
            return False

    def list_analyses_for_sample(self, sample_id: str) -> list[dict[str, Any]]:
        try:
            return list(
                self._collection("analysis_results").find({"sample_id": sample_id})
            )
        except PyMongoError as exc:
            logger.warning("metadata.analysis.list_failed: %s", exc)
            return []

    # -- bias reports ------------------------------------------------------ #

    def insert_bias_report(self, report: Mapping[str, Any]) -> bool:
        """Persist one bias report; kept permanently (no TTL index, SAD §9)."""
        if not self._try_preflight():
            return False
        try:
            if "created_at" not in report:
                report = {**report, "created_at": time.time()}
            self._collection("bias_reports").insert_one(dict(report))
            return True
        except PyMongoError as exc:
            logger.warning("metadata.bias.insert_failed: %s", exc)
            return False

    def list_bias_reports(
        self, model_id: str | None = None, cohort: str | None = None
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if model_id:
            query["model_id"] = model_id
        if cohort:
            query["cohort"] = cohort
        try:
            return list(self._collection("bias_reports").find(query))
        except PyMongoError as exc:
            logger.warning("metadata.bias.list_failed: %s", exc)
            return []

    def drop_all(self) -> None:
        """Test/cleanup helper - removes every collection in the tier."""
        if not _PYMONGO_AVAILABLE:
            return
        db = self._get_db()
        for name in COLLECTIONS:
            db.drop_collection(name)

    # -- helpers ----------------------------------------------------------- #

    def _try_preflight(self) -> bool:
        """Returns True when writes can proceed; False when degraded.

        A quick ping keeps a totally dead server from burning the
        serverSelectionTimeout on every write, and a Mongo unavailable during
        a burst of analysis records does not spam warnings per record.
        """
        if not _PYMONGO_AVAILABLE:
            return False
        try:
            self._get_db().client.admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, PyMongoError) as exc:
            logger.info("metadata.degraded: %s", exc)
            return False


#: The default process-wide store. Tests inject their own instance/connect.
metadata_store = MetadataStore()


def get_metadata_store() -> MetadataStore:
    """Module-level accessor so routes/workers can swap the store in tests by
    rebinding the module attribute (see the LIT-229 pattern note)."""
    return metadata_store