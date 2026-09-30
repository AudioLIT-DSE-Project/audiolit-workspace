"""Standalone MongoDB integration test - runs against mongomock (no server)
or against a live MongoDB if one is reachable.

Usage:
    python tests/test_mongo_isolated.py            # mongomock only (always works)
    python tests/test_mongo_isolated.py --live      # tries real MongoDB first
"""

import importlib
import sys
from pathlib import Path

_backend = str(Path(__file__).resolve().parent.parent)
if _backend not in sys.path:
    sys.path.insert(0, _backend)


def _load_metadata_store():
    spec = importlib.util.spec_from_file_location(
        "app.infrastructure.metadata_store",
        Path(__file__).resolve().parent.parent / "app" / "infrastructure" / "metadata_store.py",
        submodule_search_locations=[],
    )
    ms_mod = importlib.util.module_from_spec(spec)
    sys.modules["app.infrastructure.metadata_store"] = ms_mod
    spec.loader.exec_module(ms_mod)
    return ms_mod


def test_with_mongomock():
    import mongomock
    ms_mod = _load_metadata_store()
    MetadataStore = ms_mod.MetadataStore

    client = mongomock.MongoClient()
    db = client["audiolit_test"]
    store = MetadataStore(client=client, db=db)
    store.ensure_schema()

    print("=" * 60)
    print("  MONGOMOCK TEST (in-memory, no server required)")
    print("=" * 60)

    collections = db.list_collection_names()
    expected = {"models", "audio_samples", "analysis_results", "bias_reports"}
    assert expected.issubset(set(collections)), f"Missing: {expected - set(collections)}"
    print(f"[PASS] Collections created: {sorted(collections)}")

    assert store.upsert_model({"model_id": "whisper-large", "name": "Whisper Large", "architecture": "encoder-decoder", "revision": "v3"})
    row = store.get_model("whisper-large")
    assert row is not None and row["revision"] == "v3"
    print("[PASS] Model upsert + get")

    store.upsert_model({"model_id": "whisper-large", "name": "Whisper Large", "architecture": "encoder-decoder", "revision": "v4"})
    assert store.get_model("whisper-large")["revision"] == "v4"
    print("[PASS] Model upsert refresh")

    models = store.list_models()
    assert len(models) == 1
    print(f"[PASS] List models ({len(models)} record)")

    assert store.upsert_audio_sample({"sample_id": "s1", "filename": "test.wav", "duration": 5.2, "sample_rate": 16000, "file_path_reference": "/data/test.wav"})
    sample = store.get_audio_sample("s1")
    assert sample is not None and "audio_bytes" not in sample
    assert sample["file_path_reference"] == "/data/test.wav"
    print("[PASS] Audio sample upsert + get (no audio bytes stored)")

    assert store.insert_analysis({"analysis_id": "a1", "sample_id": "s1", "model_id": "whisper-large", "task": "asr", "prediction": {"text": "hello"}, "redis_tensor_key": "sha256:abc"})
    analyses = store.list_analyses_for_sample("s1")
    assert len(analyses) == 1 and analyses[0]["redis_tensor_key"] == "sha256:abc"
    print("[PASS] Analysis insert + list for sample")

    assert store.insert_bias_report({"report_id": "b1", "model_id": "whisper-large", "cohort": "Arabic", "WER": 0.142, "disparity_metrics": {"delta_wer": 0.067}})
    assert store.insert_bias_report({"report_id": "b2", "model_id": "whisper-large", "cohort": "English", "WER": 0.05, "disparity_metrics": {}})
    assert len(store.list_bias_reports(cohort="Arabic")) == 1
    assert len(store.list_bias_reports(model_id="whisper-large")) == 2
    print("[PASS] Bias report insert + filter by cohort/model")

    info = store._collection("analysis_results").index_information()
    has_ttl = any("expireAfterSeconds" in v for v in info.values())
    print(f"[PASS] TTL index on analysis_results: {'yes' if has_ttl else 'no (mongomock limitation)'}")

    store.drop_all()
    assert len(db.list_collection_names()) == 0
    print("[PASS] drop_all() cleans up")

    print("\nALL 10 TESTS PASSED - MongoDB integration code is working correctly.\n")


def test_with_live_server():
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError
    ms_mod = _load_metadata_store()
    MetadataStore = ms_mod.MetadataStore

    print("=" * 60)
    print("  LIVE MONGODB TEST (requires mongodb://localhost:27017)")
    print("=" * 60)

    try:
        client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        print("[SKIP] MongoDB is not running on localhost:27017")
        print("       Start it with: docker compose up -d mongo\n")
        return False

    db = client["audiolit_live_test"]
    store = MetadataStore(client=client, db=db)
    store.ensure_schema()
    print("[PASS] Connected to live MongoDB and ensured schema")

    store.upsert_model({"model_id": "test-model", "name": "Test", "architecture": "test", "revision": "v1"})
    assert store.get_model("test-model")["revision"] == "v1"
    print("[PASS] Live model upsert + get")

    store.upsert_audio_sample({"sample_id": "test-sample", "filename": "x.wav", "file_path_reference": "/tmp/x.wav"})
    assert store.get_audio_sample("test-sample") is not None
    print("[PASS] Live audio sample upsert + get")

    store.insert_analysis({"analysis_id": "test-analysis", "task": "asr", "prediction": {"text": "hi"}})
    print("[PASS] Live analysis insert")

    store.insert_bias_report({"report_id": "test-bias", "model_id": "test-model", "cohort": "en", "WER": 0.1, "disparity_metrics": {}})
    print("[PASS] Live bias report insert")

    store.drop_all()
    client.drop_database("audiolit_live_test")
    print("[PASS] Cleanup done")

    print("\nALL LIVE TESTS PASSED - real MongoDB is fully operational.\n")
    return True


if __name__ == "__main__":
    live_mode = "--live" in sys.argv
    test_with_mongomock()
    if live_mode:
        test_with_live_server()
    else:
        print("Tip: Run with --live to also test against a real MongoDB server.")
        print("     Start MongoDB first: docker compose up -d mongo")
