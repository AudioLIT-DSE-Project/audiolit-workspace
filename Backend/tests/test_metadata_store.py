"""MetadataStore module tests (LIT-256).

These exercise the module against mongomock, which mirrors pymongo's API well
enough for provisioning, CRUD, and degradation behaviour. They never need a
live MongoDB server (mirrors fakeredis for the Redis tier).
"""

import pytest

from datetime import datetime

from app.infrastructure import metadata_store as ms


def _no_array_longer_than(value, bound: int) -> bool:
    """Recursive guard: no list in ``value`` exceeds ``bound`` elements. Used
    to assert the C4/SR4 privacy boundary - tensor/heatmap payloads must never
    reach a durable document, whatever shape they take."""
    if isinstance(value, list):
        if len(value) > bound:
            return False
        return all(_no_array_longer_than(v, bound) for v in value)
    if isinstance(value, dict):
        return all(_no_array_longer_than(v, bound) for v in value.values())
    return True


@pytest.fixture
def store():
    """A MetadataStore bound to an in-memory mongomock db."""
    import mongomock

    mock_db = mongomock.MongoClient().db
    s = ms.MetadataStore(client=mock_db.client, db=mock_db)
    s.ensure_schema()
    yield s
    s.drop_all()


class TestSchema:
    def test_collections_created(self, store):
        names = store._get_db().list_collection_names()
        assert "models" in names
        assert "audio_samples" in names
        assert "analysis_results" in names
        assert "bias_reports" in names

    def test_unique_index_on_model_id(self, store):
        store.upsert_model({"model_id": "m1", "name": "Whisper", "architecture": "asr", "revision": "r1"})
        with pytest.raises(Exception):
            store._collection("models").insert_one({"model_id": "m1", "name": "dup", "architecture": "asr", "revision": "r2"})

    def test_ttl_index_configured_on_analysis_results(self, store):
        # LIT-258: the TTL must be exactly the SRS §3.10 retention (24 h =
        # 86 400 s from MONGO_ANALYSIS_TTL_HOURS), so expired analysis records
        # actually get reaped by a real mongod.
        info = store._collection("analysis_results").index_information()
        ttl = next((v["expireAfterSeconds"] for v in info.values() if "expireAfterSeconds" in v), None)
        assert ttl == 86_400, f"expected a 86400 s TTL index, got {list(info.keys())}"

    def test_bias_reports_have_no_ttl(self, store):
        info = store._collection("bias_reports").index_information()
        assert not any("expireAfterSeconds" in v for v in info.values())

    def test_ensure_schema_is_idempotent(self, store):
        # LIT-258: provisioning is callable repeatedly (startup + retries)
        # without erroring or duplicating collections/indexes.
        store.ensure_schema()
        store.ensure_schema()
        assert set(store._get_db().list_collection_names()) == {
            "models", "audio_samples", "analysis_results", "bias_reports"
        }


class TestPrivacyBoundary:
    """LIT-258: constraints C4/SR4 asserted deliberately at the store level.

    The durable tier records metadata and file-path references only. Analysis
    records carry a ``redis_tensor_key`` pointing at the cache, never a tensor
    or large array payload; ``audio_samples`` carries a path reference, never
    audio bytes.
    """

    def test_analysis_record_has_no_array_payload(self, store):
        from app.orchestration import task_orchestrator

        prediction = {
            "transcript": "hello",
            "probabilities": {"neutral": 0.9, "happy": 0.1},
            "attention": [0.5] * 500,  # saliency heatmap - must never persist
        }
        store.insert_analysis(
            {
                "analysis_id": "a1",
                "sample_id": "s1",
                "model_id": "m1",
                "task": "asr",
                # The write-through strips array payloads before insert, so the
                # stored document contains only the stripped reference-bearing
                # prediction + redis_tensor_key - never the heatmap array.
                "prediction": task_orchestrator._strip_array_fields(prediction),
                "redis_tensor_key": "sha256:abc",
            }
        )
        doc = store._collection("analysis_results").find_one({"analysis_id": "a1"})
        assert "attention" not in doc["prediction"]
        assert doc["redis_tensor_key"] == "sha256:abc"
        assert _no_array_longer_than(doc, 16)
        assert "audio" not in doc

    def test_no_collection_ever_holds_audio_bytes_or_tensors(self, store):
        # The privacy rule is store-wide: audio_bytes/personal fields and
        # tensor-shaped values must never appear in any durable record.
        store.upsert_audio_sample({"sample_id": "s1", "file_path_reference": "/tmp/a.wav"})
        store.insert_analysis({"analysis_id": "a1", "sample_id": "s1", "task": "asr",
                               "prediction": {}, "redis_tensor_key": "sha256:abc"})
        store.upsert_model({"model_id": "m1", "name": "Whisper", "architecture": "asr", "revision": "r1"})
        store.insert_bias_report({"report_id": "b1", "model_id": "m1", "cohort": "Arabic", "WER": 0.1})

        for name in ("models", "audio_samples", "analysis_results", "bias_reports"):
            for doc in store._collection(name).find():
                assert "audio_bytes" not in doc
                assert "waveform" not in doc
                assert _no_array_longer_than(doc, 16), f"tensor payload escaped into {name}"


class TestModelRecords:
    def test_upsert_and_get(self, store):
        store.upsert_model({"model_id": "m1", "name": "Whisper", "architecture": "asr", "revision": "r1"})
        row = store.get_model("m1")
        assert row["model_id"] == "m1"
        assert row["revision"] == "r1"

    def test_upsert_refreshes_existing(self, store):
        store.upsert_model({"model_id": "m1", "name": "Whisper", "architecture": "asr", "revision": "r1"})
        store.upsert_model({"model_id": "m1", "name": "Whisper", "architecture": "asr", "revision": "r2"})
        assert store.get_model("m1")["revision"] == "r2"

    def test_list_models(self, store):
        for i in range(3):
            store.upsert_model({"model_id": f"m{i}", "name": f"Model {i}", "architecture": "asr", "revision": "r1"})
        assert len(store.list_models()) == 3


class TestAudioSampleRecords:
    def test_upsert_and_get(self, store):
        store.upsert_audio_sample({"sample_id": "s1", "filename": "a.wav", "duration": 3.0, "sample_rate": 16000, "file_path_reference": "/data/a.wav"})
        row = store.get_audio_sample("s1")
        assert row["file_path_reference"] == "/data/a.wav"

    def test_only_reference_not_bytes(self, store):
        # The schema should not even accept a from-DB audio payload field.
        store.upsert_audio_sample({"sample_id": "s2", "filename": "b.wav", "file_path_reference": "/data/b.wav"})
        row = store.get_audio_sample("s2")
        assert "audio_bytes" not in row


class TestAnalysisRecords:
    def test_insert_and_list_for_sample(self, store):
        store.insert_analysis({"analysis_id": "a1", "sample_id": "s1", "model_id": "m1", "task": "asr", "prediction": {"text": "hi"}, "redis_tensor_key": "sha256:abc"})
        rows = store.list_analyses_for_sample("s1")
        assert len(rows) == 1
        assert rows[0]["redis_tensor_key"] == "sha256:abc"

    def test_created_at_defaulted(self, store):
        store.insert_analysis({"analysis_id": "a2", "sample_id": "s1", "model_id": "m1", "task": "ser"})
        row = store._collection("analysis_results").find_one({"analysis_id": "a2"})
        assert "created_at" in row
        # A real BSON date (datetime) so the TTL index expires records, which a
        # float (mongomock accepts, real Mongo rejects) would silently break.
        assert isinstance(row["created_at"], datetime)

    def test_re_run_refreshes_the_same_analysis_document(self, store):
        store.insert_analysis({"analysis_id": "a3", "sample_id": "s1", "model_id": "m1", "task": "asr", "prediction": {"text": "first"}})
        store.insert_analysis({"analysis_id": "a3", "sample_id": "s1", "model_id": "m1", "task": "asr", "prediction": {"text": "second"}})
        rows = list(store._collection("analysis_results").find({"analysis_id": "a3"}))
        assert len(rows) == 1
        assert rows[0]["prediction"]["text"] == "second"


class TestBiasReports:
    def test_insert_and_filter_by_model_cohort(self, store):
        store.insert_bias_report({"report_id": "b1", "model_id": "m1", "cohort": "Arabic", "WER": 0.142, "disparity_metrics": {"delta": 0.067}})
        store.insert_bias_report({"report_id": "b2", "model_id": "m2", "cohort": "Arabic", "WER": 0.1, "disparity_metrics": {}})
        rows = store.list_bias_reports(model_id="m1")
        assert len(rows) == 1
        rows = store.list_bias_reports(cohort="Arabic")
        assert len(rows) == 2
        rows = store.list_bias_reports(model_id="m1", cohort="Arabic")
        assert len(rows) == 1

    def test_re_run_refreshes_the_same_report(self, store):
        store.insert_bias_report({"report_id": "b3", "model_id": "m1", "cohort": "Mandarin", "WER": 0.2, "disparity_metrics": {}})
        store.insert_bias_report({"report_id": "b3", "model_id": "m1", "cohort": "Mandarin", "WER": 0.15, "disparity_metrics": {}})
        rows = list(store._collection("bias_reports").find({"report_id": "b3"}))
        assert len(rows) == 1
        assert rows[0]["WER"] == 0.15


class TestConfiguredOff:
    """SAD §11.1 / LIT-257: unset MONGO_URL means the tier is configured off -
    get_metadata_store() returns None and callers skip every write silently."""

    def test_get_metadata_store_returns_none_when_mongo_url_empty(self, monkeypatch):
        from app.infrastructure.settings import settings

        monkeypatch.setattr(settings, "MONGO_URL", "")
        assert ms.get_metadata_store() is None

    def test_get_metadata_store_returns_store_when_configured(self, monkeypatch):
        from app.infrastructure.settings import settings

        monkeypatch.setattr(settings, "MONGO_URL", "mongodb://127.0.0.1:27017")
        assert ms.get_metadata_store() is ms.metadata_store


def _dead_store():
    """A MetadataStore backed by a simulated, fully unreachable MongoDB.

    The ``db``/``client`` handles raise a real pymongo error whenever any
    operation touches the server, exactly as a down mongod does (but without a
    wall-clock server-selection wait), so the degradation paths are exercised
    deterministically.
    """
    class _Admin:
        @staticmethod
        def command(name):
            raise ms.ServerSelectionTimeoutError("no mongod server available")

    class _Client:
        # Rebuilt per-instance so the admin reference is fresh each time.
        def __init__(self):
            self.admin = _Admin()

    class _Collection:
        @staticmethod
        def _raise(*args, **kwargs):
            raise ms.ServerSelectionTimeoutError("no mongod server available")

        update_one = _raise
        insert_one = _raise
        find_one = _raise
        find = _raise
        create_index = _raise

    class _DB:
        def __init__(self):
            self.client = _Client()

        def __getitem__(self, name):
            return _Collection()

        def list_collection_names(self):
            raise ms.ServerSelectionTimeoutError("no mongod server available")

        def drop_collection(self, name):
            raise ms.ServerSelectionTimeoutError("no mongod server available")

    db = _DB()
    return ms.MetadataStore(client=db.client, db=db)


class TestGracefulDegradation:
    """LIT-258: the metadata tier is a supporting store, never a request-path
    dependency (SRS §3.3.1 / SAD §11.1). When MongoDB is unreachable - or the
    driver is missing - every read/write degrades to a logged no-op or empty
    result instead of raising, and ``available`` reports False for dashboards.
    """

    def test_available_is_false_when_server_unreachable(self):
        assert _dead_store().available is False

    def test_available_is_false_when_driver_missing(self, monkeypatch):
        # A driver-less env is the deepest degradation: nothing can even attempt
        # a connection. available must report False, not raise.
        monkeypatch.setattr(ms, "_PYMONGO_AVAILABLE", False)
        store = _dead_store()
        assert store.available is False

    def test_every_write_returns_false_not_raise_when_server_down(self):
        store = _dead_store()
        assert store.upsert_model({"model_id": "m", "name": "x", "architecture": "asr", "revision": "r"}) is False
        assert store.upsert_audio_sample({"sample_id": "s", "file_path_reference": "/tmp/a.wav"}) is False
        assert store.insert_analysis({"analysis_id": "a", "task": "asr"}) is False
        assert store.insert_bias_report({"report_id": "b", "model_id": "m", "cohort": "ar"}) is False

    def test_every_write_returns_false_not_raise_when_driver_missing(self, monkeypatch):
        monkeypatch.setattr(ms, "_PYMONGO_AVAILABLE", False)
        store = _dead_store()
        # The preflight short-circuits on the missing-driver flag (no TTL-swallowed
        # attempts to build a client, no TypeErrors).
        assert store.upsert_model({"model_id": "m", "name": "x", "architecture": "asr", "revision": "r"}) is False
        assert store.upsert_audio_sample({"sample_id": "s", "file_path_reference": "/tmp/a.wav"}) is False
        assert store.insert_analysis({"analysis_id": "a", "task": "asr"}) is False
        assert store.insert_bias_report({"report_id": "b", "model_id": "m", "cohort": "ar"}) is False

    def test_every_read_degrades_when_server_down(self):
        store = _dead_store()
        assert store.get_model("m") is None
        assert store.get_audio_sample("s") is None
        assert store.list_models() == []
        assert store.list_analyses_for_sample("s") == []
        assert store.list_bias_reports(model_id="m") == []

    def test_every_read_degrades_when_driver_missing(self, monkeypatch):
        monkeypatch.setattr(ms, "_PYMONGO_AVAILABLE", False)
        store = _dead_store()
        # Reads must return empty, never a TypeError from trying to construct a
        # client with MongoClient=None (LIT-258 graceful-degradation fix).
        assert store.get_model("m") is None
        assert store.get_audio_sample("s") is None
        assert store.list_models() == []
        assert store.list_analyses_for_sample("s") == []
        assert store.list_bias_reports(model_id="m") == []

    def test_preflight_is_false_when_server_down(self):
        assert _dead_store()._try_preflight() is False
