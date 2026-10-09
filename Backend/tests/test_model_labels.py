"""Class names for custom checkpoints published without `id2label`.

Such a checkpoint reports `LABEL_0 .. LABEL_n`, which is what the dataset table
showed in its "Predicted Emotion" column. The names are entered by the user in
the Custom Model dialog; these cover the store, the route, the point where the
loaded SER model picks them up, and the cache keys that must stop serving
results labelled with the placeholders.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api.routes import models as models_routes
from app.domain.model_registry_service import LoadedModel
from app.infrastructure import cache_keys as ck
from app.infrastructure import model_labels

CUSTOM = "myorg/custom-ser"
NAMES = ["angry", "happy", "neutral"]


def _model(labels, architectures=("Wav2Vec2ForSequenceClassification",)):
    config = SimpleNamespace(
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
        architectures=list(architectures),
    )
    return SimpleNamespace(config=config)


def _loaded(model) -> LoadedModel:
    return LoadedModel(
        model_id=CUSTOM, revision="abc123", family="wav2vec2",
        weights_sha256="deadbeef", model=model, available_layers=[],
    )


class TestStore:
    def test_round_trip_and_removal(self):
        assert model_labels.get_label_override(CUSTOM) is None
        model_labels.set_label_override(CUSTOM, NAMES)
        assert model_labels.get_label_override(CUSTOM) == NAMES
        model_labels.set_label_override(CUSTOM, None)
        assert model_labels.get_label_override(CUSTOM) is None

    def test_corrupt_file_degrades_to_no_names(self):
        model_labels.labels_path().write_text("{not json")
        assert model_labels.get_label_override(CUSTOM) is None

    def test_placeholders_are_recognised(self):
        assert model_labels.are_placeholders(["LABEL_0", "LABEL_1"])
        assert not model_labels.are_placeholders(["LABEL_0", "happy"])
        assert not model_labels.are_placeholders([])

    @pytest.mark.parametrize("bad", [
        ["angry", "happy"],            # wrong count
        ["angry", " ", "neutral"],     # blank
        ["angry", "Angry", "neutral"],  # duplicate once normalised
        ["angry", "x" * 41, "neutral"],
    ])
    def test_invalid_names_are_rejected(self, bad):
        with pytest.raises(ValueError):
            model_labels.normalise_labels(bad, 3)

    def test_names_are_trimmed_and_lower_cased(self):
        assert model_labels.normalise_labels([" Angry", "HAPPY ", "neutral"], 3) == NAMES


class TestCacheKeys:
    def test_entering_names_moves_the_ser_keys(self):
        h = ("0" * 32,)
        before = set(ck.ser_keys(h, CUSTOM))
        model_labels.set_label_override(CUSTOM, NAMES)
        after = set(ck.ser_keys(h, CUSTOM))
        assert not (before & after), "a prediction cached as LABEL_n would still be served"

    def test_default_model_keys_are_untouched(self):
        h = ("0" * 32,)
        model_labels.set_label_override(ck.DEFAULT_SER_MODEL, NAMES)
        assert ("wav2vec2", f"wav2vec2_detailed_{h[0]}") in ck.ser_keys(h)


class TestLoadedModelPicksUpNames:
    @pytest.fixture
    def ml(self, monkeypatch):
        import app.domain.model_loader_service as ml

        class _FE:
            @classmethod
            def from_pretrained(cls, mid, **kw):
                return cls()

        self.model = _model(["LABEL_0", "LABEL_1", "LABEL_2"])
        monkeypatch.setattr(ml, "Wav2Vec2FeatureExtractor", _FE)
        monkeypatch.setattr(ml._model_registry, "get", lambda mid, **kw: _loaded(self.model))
        monkeypatch.setattr(ml, "_emo_model_cache", {})
        monkeypatch.setattr(ml, "_emo_checkpoint_labels", {})
        return ml

    def test_names_entered_after_the_model_is_cached_apply(self, ml):
        _, model, _ = ml.ensure_emo_model_loaded(CUSTOM)
        assert model.config.id2label[1] == "LABEL_1"

        model_labels.set_label_override(CUSTOM, NAMES)
        _, model, _ = ml.ensure_emo_model_loaded(CUSTOM)
        assert model.config.id2label == {0: "angry", 1: "happy", 2: "neutral"}
        assert model.config.label2id == {"angry": 0, "happy": 1, "neutral": 2}

    def test_removing_names_restores_the_checkpoint_labels(self, ml):
        model_labels.set_label_override(CUSTOM, NAMES)
        ml.ensure_emo_model_loaded(CUSTOM)
        model_labels.set_label_override(CUSTOM, None)
        _, model, _ = ml.ensure_emo_model_loaded(CUSTOM)
        assert model.config.id2label == {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}

    def test_a_stored_list_of_the_wrong_length_is_ignored(self, ml):
        model_labels.set_label_override(CUSTOM, ["angry", "happy"])
        _, model, _ = ml.ensure_emo_model_loaded(CUSTOM)
        assert model.config.id2label[0] == "LABEL_0"


class TestRoutes:
    @pytest.fixture
    def placeholder_model(self, monkeypatch):
        loaded = _loaded(_model(["LABEL_0", "LABEL_1", "LABEL_2"]))
        monkeypatch.setattr(models_routes.registry, "get", lambda model_id, revision="main": loaded)

    @pytest.mark.asyncio
    async def test_resolve_flags_placeholder_labels(self, client, placeholder_model):
        body = (await client.post("/models/resolve", json={"model_id": CUSTOM})).json()
        assert body["labels"] == ["LABEL_0", "LABEL_1", "LABEL_2"]
        assert body["labels_are_placeholders"] is True

    @pytest.mark.asyncio
    async def test_resolve_reports_no_labels_for_a_non_classifier(self, client, monkeypatch):
        # transformers gives every config a default two-entry id2label.
        loaded = _loaded(_model(["LABEL_0", "LABEL_1"], architectures=("WhisperForConditionalGeneration",)))
        monkeypatch.setattr(models_routes.registry, "get", lambda model_id, revision="main": loaded)
        body = (await client.post("/models/resolve", json={"model_id": CUSTOM})).json()
        assert body["labels"] == []
        assert body["labels_are_placeholders"] is False

    @pytest.mark.asyncio
    async def test_saved_names_come_back_on_the_next_resolve(self, client, placeholder_model):
        r = await client.put("/models/labels", json={"model_id": CUSTOM, "labels": ["Angry", "Happy", "Neutral"]})
        assert r.status_code == 200
        assert r.json()["labels"] == NAMES

        body = (await client.post("/models/resolve", json={"model_id": CUSTOM})).json()
        assert body["labels"] == NAMES
        assert body["labels_are_placeholders"] is True  # still true of the checkpoint

    @pytest.mark.asyncio
    async def test_wrong_number_of_names_is_422(self, client, placeholder_model):
        r = await client.put("/models/labels", json={"model_id": CUSTOM, "labels": ["angry"]})
        assert r.status_code == 422
        assert model_labels.get_label_override(CUSTOM) is None

    @pytest.mark.asyncio
    async def test_empty_list_removes_the_names(self, client, placeholder_model):
        model_labels.set_label_override(CUSTOM, NAMES)
        r = await client.put("/models/labels", json={"model_id": CUSTOM, "labels": []})
        assert r.status_code == 200
        assert r.json()["labels"] == ["LABEL_0", "LABEL_1", "LABEL_2"]
        assert model_labels.get_label_override(CUSTOM) is None
