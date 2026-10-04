"""Group-wise emotion-accuracy bias for the SER corpora (FR15; CREMA-D, ESD).

L2-ARCTIC is profiled by word error rate per accent. The emotion corpora have
no accent label; what they have is the speaker's race/sex/ethnicity (CREMA-D)
or language (ESD), and an emotion label to score against. These tests cover the
loader fields that grouping depends on, the scoring and ranking, the background
task, and the route's choice of diagnostic per corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fakeredis import FakeServer, FakeStrictRedis

from app.api.routes import evaluation as evaluation_routes
from app.domain import emotion_bias_runner
from app.domain.emotion_bias_runner import (
    resolve_group_field,
    run_emotion_bias_diagnostic,
)
from app.infrastructure import rq_connection
from app.infrastructure.dataset_ingestion import CremaDLoader, ESDLoader
from app.orchestration import task_orchestrator
from app.orchestration.task_orchestrator import emotion_bias_task


def _wav(path: Path) -> None:
    """A short, non-silent clip: the cohort loader rejects silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    sf.write(path, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)


@pytest.fixture
def crema_sample_root(tmp_path: Path) -> Path:
    """The flat provisioned layout: clips at the root, demographics in the
    sample catalog, and no VideoDemographics.csv."""
    root = tmp_path / "crema_d"
    rows = [
        ("1001_DFA_ANG_XX", "1001", "Male", "Caucasian", "Not Hispanic"),
        ("1001_IEO_SAD_XX", "1001", "Male", "Caucasian", "Not Hispanic"),
        ("1002_TIE_SAD_XX", "1002", "Female", "African American", "Not Hispanic"),
        ("1002_IOM_HAP_XX", "1002", "Female", "African American", "Not Hispanic"),
        ("1003_ITH_SAD_XX", "1003", "Female", "Asian", "Hispanic"),
    ]
    for stem, *_ in rows:
        _wav(root / f"{stem}.wav")
    lines = ["filename,actor_id,emotion,intensity,sentence,stimulus_number,age,sex,race,ethnicity,dataset"]
    for stem, actor, sex, race, ethnicity in rows:
        _, sentence, emotion, intensity = stem.split("_")
        lines.append(f"{stem}.wav,{actor},{emotion},{intensity},{sentence},1,30,{sex},{race},{ethnicity},CREMA-D")
    (root / CremaDLoader.SAMPLE_CATALOG_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def esd_sample(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "esd"
    rows = [
        ("0003_000101", "0003", "Happy"),
        ("0003_000102", "0003", "Sad"),
        ("0010_000103", "0010", "Sad"),
        ("0011_000104", "0011", "Sad"),
        ("0020_000105", "0020", "Angry"),
    ]
    lines = ["filename,sample_id,speaker_id,emotion,emotion_cn,transcription,dataset"]
    for sample_id, speaker, emotion in rows:
        _wav(root / f"{sample_id}.wav")
        lines.append(f"{sample_id}.wav,{sample_id},{speaker},{emotion},{emotion},text,ESD")
    catalog = root / "esd_test_100_metadata.csv"
    catalog.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")
    return catalog, root


@pytest.fixture
def broker(monkeypatch):
    fake = FakeStrictRedis(server=FakeServer())
    monkeypatch.setattr(rq_connection, "_CONNECTION", fake)
    task_orchestrator.reset_queue_cache()
    yield fake
    task_orchestrator.reset_queue_cache()
    fake.flushall()


class TestLoaderGroupFields:
    def test_crema_d_reads_demographics_from_the_sample_catalog(self, crema_sample_root: Path):
        by_id = {s.sample_id: s for s in CremaDLoader(crema_sample_root).iter_metadata()}
        assert by_id["1001_DFA_ANG_XX"].demographic["race"] == "Caucasian"
        assert by_id["1002_TIE_SAD_XX"].demographic["sex"] == "Female"
        assert by_id["1003_ITH_SAD_XX"].demographic["ethnicity"] == "Hispanic"

    def test_crema_d_prefers_the_corpus_demographics_file(self, crema_sample_root: Path):
        (crema_sample_root / CremaDLoader.DEMOGRAPHICS_FILE).write_text(
            "ActorID,Age,Sex,Race,Ethnicity\n1001,51,Male,Asian,Not Hispanic\n", encoding="utf-8"
        )
        by_id = {s.sample_id: s for s in CremaDLoader(crema_sample_root).iter_metadata()}
        assert by_id["1001_DFA_ANG_XX"].demographic["race"] == "Asian"

    def test_esd_language_comes_from_the_speaker_id(self, esd_sample):
        catalog, root = esd_sample
        by_id = {s.sample_id: s for s in ESDLoader(catalog, root).iter_metadata()}
        assert by_id["0003_000101"].language == "zh"
        assert by_id["0010_000103"].demographic["language"] == "Mandarin"
        assert by_id["0011_000104"].language == "en"
        assert by_id["0020_000105"].demographic["language"] == "English"
        # The emotion label is still normalised.
        assert by_id["0003_000101"].label == "happy"


class TestGroupField:
    def test_defaults_to_the_corpus_first_field(self):
        assert resolve_group_field("crema-d", None) == "race"
        assert resolve_group_field("ESD", None) == "language"

    def test_accepts_another_offered_field(self):
        assert resolve_group_field("crema-d", "sex") == "sex"

    def test_rejects_a_field_the_corpus_does_not_have(self):
        with pytest.raises(ValueError, match="race, sex, ethnicity"):
            resolve_group_field("crema-d", "language")

    def test_rejects_a_corpus_without_group_labels(self):
        with pytest.raises(ValueError, match="not available"):
            resolve_group_field("common-voice", None)


class TestRunEmotionBiasDiagnostic:
    def test_scores_each_group_and_ranks_the_worst_first(self, crema_sample_root: Path):
        # A model that always answers "sad": right on 1 of 2 Caucasian clips,
        # 1 of 2 African American clips, and the one Asian clip.
        report = run_emotion_bias_diagnostic(
            lambda path: "sad", "crema-d", model_id="m", root_dir=crema_sample_root
        )
        assert report.group_by == "race"
        by_group = {c.group: c for c in report.cohorts}
        assert by_group["Asian"].accuracy == 1.0
        assert by_group["Caucasian"].accuracy == 0.5
        assert by_group["African American"].correct_count == 1
        # Lowest accuracy first; the best group is last.
        assert report.cohorts[-1].group == "Asian"
        assert [c.accuracy for c in report.cohorts] == sorted(c.accuracy for c in report.cohorts)

    def test_can_group_by_another_field(self, crema_sample_root: Path):
        report = run_emotion_bias_diagnostic(
            lambda path: "sad", "crema-d", group_by="sex", root_dir=crema_sample_root
        )
        assert {c.group for c in report.cohorts} == {"Male", "Female"}

    def test_esd_groups_by_language(self, esd_sample):
        catalog, root = esd_sample
        report = run_emotion_bias_diagnostic(
            lambda path: "sad", "esd", catalog_path=catalog, audio_base_dir=root
        )
        by_group = {c.group: c for c in report.cohorts}
        assert by_group["Mandarin"].sample_count == 3
        assert by_group["Mandarin"].accuracy == pytest.approx(2 / 3)
        assert by_group["English"].accuracy == 0.5

    def test_prediction_is_compared_case_insensitively(self, crema_sample_root: Path):
        report = run_emotion_bias_diagnostic(
            lambda path: " SAD ", "crema-d", group_by="sex", root_dir=crema_sample_root
        )
        assert any(r.correct for r in report.sample_results)

    def test_report_is_json_ready(self, crema_sample_root: Path):
        import json

        payload = run_emotion_bias_diagnostic(
            lambda path: None, "crema-d", root_dir=crema_sample_root
        ).to_json_dict()
        json.dumps(payload)
        assert payload["metric"] == "emotion_accuracy"
        assert all(c["accuracy"] == 0.0 for c in payload["cohorts"])

    def test_samples_per_cohort_bounds_each_group(self, crema_sample_root: Path):
        report = run_emotion_bias_diagnostic(
            lambda path: "sad", "crema-d", samples_per_cohort=1, root_dir=crema_sample_root
        )
        assert all(c.sample_count == 1 for c in report.cohorts)


class TestSerPredictor:
    def test_builtin_model_key_selects_the_default_checkpoint(self, monkeypatch):
        from app.domain import model_loader_service

        seen = []

        def _fake_predict_ser(audio_path, model_id=None):
            seen.append(model_id)
            return {"predicted_emotion": "happy"}

        monkeypatch.setattr(model_loader_service, "predict_ser", _fake_predict_ser)
        assert emotion_bias_runner.make_ser_predictor("wav2vec2")("clip.wav") == "happy"
        assert emotion_bias_runner.make_ser_predictor("org/custom-ser")("clip.wav") == "happy"
        assert seen == [None, "org/custom-ser"]


class TestEmotionBiasTask:
    def test_runs_the_diagnostic_and_returns_its_json(self, broker, monkeypatch):
        captured = {}

        class _FakeReport:
            corpus, group_by, cohorts = "crema-d", "race", []

            def to_json_dict(self):
                return {"corpus": "crema-d", "metric": "emotion_accuracy", "cohorts": []}

        def _fake_run(predict, corpus, model_id, group_by, samples_per_cohort):
            captured.update(corpus=corpus, model_id=model_id, group_by=group_by, n=samples_per_cohort)
            return _FakeReport()

        monkeypatch.setattr(emotion_bias_runner, "make_ser_predictor", lambda model_id: (lambda p: "sad"))
        monkeypatch.setattr(emotion_bias_runner, "run_emotion_bias_diagnostic", _fake_run)

        result = emotion_bias_task("wav2vec2", "crema-d", "race", 25)
        assert result == {"corpus": "crema-d", "metric": "emotion_accuracy", "cohorts": []}
        assert captured == {"corpus": "crema-d", "model_id": "wav2vec2", "group_by": "race", "n": 25}

    def test_enqueues_on_the_ser_queue(self, broker):
        result = task_orchestrator.enqueue_emotion_bias("wav2vec2", "esd", "language", 25)
        assert result.family_jobs == {"emotion_bias": result.job_id}
        assert task_orchestrator.get_queue(task_orchestrator.WorkerFamily.SER).job_ids == [result.job_id]


class TestBiasRoute:
    @pytest.fixture
    def enqueued(self, monkeypatch):
        calls = []

        class _Result:
            def __init__(self, kind):
                self.kind = kind

            def as_response(self):
                return {
                    "job_id": "job-1",
                    "websocket_url": "/api/ws/tasks/job-1",
                    "schema_version": "1.0",
                    "family_jobs": {self.kind: "job-1"},
                    "cache_key": None,
                }

        def _accent(model_id, corpus="l2-arctic", samples_per_cohort=None):
            calls.append(("accent", model_id, corpus, None, samples_per_cohort))
            return _Result("accent_bias")

        def _emotion(model_id, corpus, group_by=None, samples_per_cohort=None):
            calls.append(("emotion", model_id, corpus, group_by, samples_per_cohort))
            return _Result("emotion_bias")

        monkeypatch.setattr(evaluation_routes, "enqueue_accent_bias", _accent)
        monkeypatch.setattr(evaluation_routes, "enqueue_emotion_bias", _emotion)
        return calls

    @pytest.mark.asyncio
    async def test_l2_arctic_still_runs_the_accent_diagnostic(self, client, enqueued):
        r = await client.post(
            "/evaluation/accent-bias", json={"model_id": "openai/whisper-base", "corpus": "l2-arctic"}
        )
        assert r.status_code == 200
        assert enqueued == [("accent", "openai/whisper-base", "l2-arctic", None, 10)]

    @pytest.mark.asyncio
    async def test_crema_d_runs_emotion_bias_grouped_by_race_by_default(self, client, enqueued):
        r = await client.post(
            "/evaluation/accent-bias",
            json={"model_id": "wav2vec2", "corpus": "crema-d", "samples_per_cohort": 25},
        )
        assert r.status_code == 200
        assert r.json()["family_jobs"] == {"emotion_bias": "job-1"}
        assert enqueued == [("emotion", "wav2vec2", "crema-d", "race", 25)]

    @pytest.mark.asyncio
    async def test_esd_groups_by_language(self, client, enqueued):
        r = await client.post("/evaluation/accent-bias", json={"model_id": "wav2vec2", "corpus": "esd"})
        assert r.status_code == 200
        assert enqueued == [("emotion", "wav2vec2", "esd", "language", 10)]

    @pytest.mark.asyncio
    async def test_a_field_the_corpus_lacks_is_a_400(self, client, enqueued):
        r = await client.post(
            "/evaluation/accent-bias",
            json={"model_id": "wav2vec2", "corpus": "esd", "group_by": "race"},
        )
        assert r.status_code == 400
        assert "language" in r.json()["detail"]
        assert enqueued == []

    @pytest.mark.asyncio
    async def test_any_other_corpus_is_refused(self, client, enqueued):
        for corpus in ("common-voice", "ravdess", "asvspoof-2021"):
            r = await client.post(
                "/evaluation/accent-bias", json={"model_id": "openai/whisper-base", "corpus": corpus}
            )
            assert r.status_code == 400
            assert "L2-ARCTIC, CREMA-D and ESD" in r.json()["detail"]
        assert enqueued == []
