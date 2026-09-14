"""MetadataStore module tests (LIT-256).

These exercise the module against mongomock, which mirrors pymongo's API well
enough for provisioning, CRUD, and degradation behaviour. They never need a
live MongoDB server (mirrors fakeredis for the Redis tier).
"""

import pytest

from app.infrastructure import metadata_store as ms


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
        info = store._collection("analysis_results").index_information()
        any_ttl = any("expireAfterSeconds" in v for v in info.values())
        assert any_ttl, f"expected a TTL index, got {list(info.keys())}"

    def test_bias_reports_have_no_ttl(self, store):
        info = store._collection("bias_reports").index_information()
        assert not any("expireAfterSeconds" in v for v in info.values())


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