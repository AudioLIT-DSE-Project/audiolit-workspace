"""Tests for the CREMA-D and RAVDESS SER corpus loaders (LIT-208, FR2/FR6).

Synthetic corpus trees; no real audio downloaded (both corpora are research-use
licensed and `Backend/data/` is gitignored).

The label vocabulary is the load-bearing part. These clips exist to measure the
SER classifier against known ground truth (LIT-224's outstanding DoD item), so a
loader that emitted "fear" where the model says "fearful" would score every fear
clip wrong while looking perfectly healthy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.infrastructure.dataset_ingestion import (
    EMOTION_LABELS,
    TARGET_SAMPLE_RATE,
    CremaDLoader,
    RavdessLoader,
    TaskFamily,
    demo_clips_by_emotion,
    get_loader,
    list_supported_corpora,
)


def _wav(path: Path, sr: int = 22_050, seconds: float = 0.2) -> None:
    """Write a short tone at a non-target rate so standardisation is exercised."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)


@pytest.fixture
def crema_root(tmp_path: Path) -> Path:
    root = tmp_path / "crema_d"
    audio = root / "AudioWAV"
    for stem in ("1001_DFA_ANG_XX", "1001_IEO_HAP_HI", "1002_TIE_SAD_LO",
                 "1002_IOM_NEU_XX", "1003_ITH_FEA_MD", "1003_IWW_DIS_XX"):
        _wav(audio / f"{stem}.wav")
    (root / "VideoDemographics.csv").write_text(
        "ActorID,Age,Sex,Race,Ethnicity\n"
        "1001,51,Male,Caucasian,Not Hispanic\n"
        "1002,21,Female,African American,Not Hispanic\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def ravdess_root(tmp_path: Path) -> Path:
    root = tmp_path / "ravdess"
    # actor 01 (odd -> male), actor 02 (even -> female)
    _wav(root / "Actor_01" / "03-01-05-01-02-01-01.wav")  # speech, angry
    _wav(root / "Actor_01" / "03-01-01-01-01-01-01.wav")  # speech, neutral
    _wav(root / "Actor_02" / "03-01-04-02-01-01-02.wav")  # speech, sad, strong
    _wav(root / "Actor_02" / "03-02-03-01-01-01-02.wav")  # SONG, happy
    return root


class TestCremaD:
    def test_parses_label_speaker_and_intensity_from_filename(self, crema_root: Path):
        by_id = {m.sample_id: m for m in CremaDLoader(crema_root)}
        angry = by_id["1001_DFA_ANG_XX"]
        assert angry.label == "angry"
        assert angry.speaker_id == "1001"
        assert angry.task_family is TaskFamily.SER
        assert angry.extra["sentence_code"] == "DFA"
        assert by_id["1001_IEO_HAP_HI"].extra["intensity"] == "high"
        assert by_id["1002_TIE_SAD_LO"].extra["intensity"] == "low"

    def test_all_six_crema_emotions_map_into_the_canonical_vocabulary(self, crema_root: Path):
        labels = {m.label for m in CremaDLoader(crema_root)}
        assert labels == {"angry", "happy", "sad", "neutral", "fearful", "disgust"}
        assert labels <= set(EMOTION_LABELS)

    def test_demographics_are_attached_per_actor(self, crema_root: Path):
        by_id = {m.sample_id: m for m in CremaDLoader(crema_root)}
        assert by_id["1001_DFA_ANG_XX"].demographic["sex"] == "Male"
        assert by_id["1002_TIE_SAD_LO"].demographic["age"] == "21"

    def test_missing_demographics_file_is_tolerated(self, crema_root: Path):
        (crema_root / "VideoDemographics.csv").unlink()
        samples = list(CremaDLoader(crema_root))
        assert len(samples) == 6
        # intensity still present; the census fields are simply absent
        assert "sex" not in samples[0].demographic
        assert samples[0].demographic["intensity"]

    def test_unparseable_filenames_are_skipped_not_fabricated(self, crema_root: Path):
        _wav(crema_root / "AudioWAV" / "not-a-crema-name.wav")
        _wav(crema_root / "AudioWAV" / "1004_DFA_ZZZ_XX.wav")  # unknown emotion code
        assert len(list(CremaDLoader(crema_root))) == 6

    def test_missing_root_raises_rather_than_yielding_nothing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(CremaDLoader(tmp_path / "absent"))

    def test_flat_layout_without_audiowav_subfolder_is_tolerated(self, tmp_path: Path):
        """Real-world distributions sometimes get extracted flat (LIT-235) -
        mirrors RavdessLoader's existing flat-layout tolerance."""
        root = tmp_path / "crema_d_flat"
        _wav(root / "1001_DFA_ANG_XX.wav")
        samples = list(CremaDLoader(root))
        assert [m.label for m in samples] == ["angry"]

    def test_audio_is_standardised_to_16k_mono(self, crema_root: Path):
        loader = CremaDLoader(crema_root)
        meta = next(iter(loader))
        audio, sr = loader.load_sample_audio(meta)
        assert sr == TARGET_SAMPLE_RATE
        assert audio.ndim == 1


class TestRavdess:
    def test_parses_emotion_intensity_and_statement(self, ravdess_root: Path):
        by_id = {m.sample_id: m for m in RavdessLoader(ravdess_root)}
        angry = by_id["03-01-05-01-02-01-01"]
        assert angry.label == "angry"
        assert angry.speaker_id == "actor_01"
        assert angry.extra["statement"] == "Dogs are sitting by the door"
        assert by_id["03-01-04-02-01-01-02"].demographic["intensity"] == "strong"

    def test_actor_parity_gives_gender(self, ravdess_root: Path):
        by_id = {m.sample_id: m for m in RavdessLoader(ravdess_root)}
        assert by_id["03-01-05-01-02-01-01"].demographic["sex"] == "male"    # actor 01
        assert by_id["03-01-04-02-01-01-02"].demographic["sex"] == "female"  # actor 02

    def test_song_channel_is_excluded_by_default(self, ravdess_root: Path):
        """Song would otherwise be scored as if it were speech in an SER demo."""
        ids = {m.sample_id for m in RavdessLoader(ravdess_root)}
        assert "03-02-03-01-01-01-02" not in ids
        assert len(ids) == 3

    def test_song_channel_can_be_opted_in(self, ravdess_root: Path):
        samples = list(RavdessLoader(ravdess_root, include_song=True))
        channels = {m.extra["vocal_channel"] for m in samples}
        assert channels == {"speech", "song"}
        assert len(samples) == 4

    def test_flat_layout_is_tolerated(self, tmp_path: Path):
        root = tmp_path / "ravdess_flat"
        _wav(root / "03-01-06-01-01-01-03.wav")
        samples = list(RavdessLoader(root))
        assert [m.label for m in samples] == ["fearful"]

    def test_labels_are_in_the_canonical_vocabulary(self, ravdess_root: Path):
        labels = {m.label for m in RavdessLoader(ravdess_root, include_song=True)}
        assert labels <= set(EMOTION_LABELS)

    def test_unparseable_filenames_are_skipped(self, ravdess_root: Path):
        _wav(ravdess_root / "Actor_01" / "03-01-99-01-01-01-01.wav")  # bad emotion code
        _wav(ravdess_root / "Actor_01" / "short-name.wav")
        assert len(list(RavdessLoader(ravdess_root))) == 3

    def test_missing_root_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(RavdessLoader(tmp_path / "absent"))


try:  # pragma: no cover - depends on whether LIT-224 (PR #43) has merged yet
    from app.domain.model_loader_service import _EMO_PINNED_LABELS
except ImportError:  # pragma: no cover
    _EMO_PINNED_LABELS = None


class TestLabelVocabularyMatchesTheClassifier:
    """The reason this corpus exists is to score the SER model, so the two label
    sets have to line up. Guards the specific mismatch that would be invisible:
    "fear"/"surprise" (corpus convention) vs "fearful"/"surprised" (model)."""

    def test_canonical_labels_use_the_model_spellings(self):
        # Asserted against literals so this holds independently of LIT-224.
        assert "fearful" in EMOTION_LABELS and "fear" not in EMOTION_LABELS
        assert "surprised" in EMOTION_LABELS and "surprise" not in EMOTION_LABELS

    @pytest.mark.skipif(
        _EMO_PINNED_LABELS is None,
        reason="SER checkpoint labels land with LIT-224 (PR #43); cross-check activates on merge",
    )
    def test_every_classifier_label_is_representable(self):
        assert set(_EMO_PINNED_LABELS) <= set(EMOTION_LABELS), (
            "the SER checkpoint can predict a class no corpus label can express"
        )

    @pytest.mark.skipif(
        _EMO_PINNED_LABELS is None,
        reason="SER checkpoint labels land with LIT-224 (PR #43); cross-check activates on merge",
    )
    def test_calm_is_corpus_only_and_known_to_be_unscoreable(self):
        """RAVDESS has 'calm'; the classifier has no such class. Kept as real
        ground truth, but an accuracy harness must exclude it deliberately."""
        assert "calm" in EMOTION_LABELS
        assert "calm" not in _EMO_PINNED_LABELS


class TestRegistryWiring:
    def test_both_corpora_resolve_through_get_loader(self, crema_root: Path, ravdess_root: Path):
        assert isinstance(get_loader("crema-d", root_dir=crema_root), CremaDLoader)
        assert isinstance(get_loader("ravdess", root_dir=ravdess_root), RavdessLoader)

    def test_esd_still_reports_itself_as_unimplemented(self):
        """Out of LIT-208's scope; must say so rather than return empty data."""
        with pytest.raises(NotImplementedError, match="LIT-208"):
            get_loader("esd")

    def test_corpus_list_is_unchanged(self):
        assert set(list_supported_corpora()) == {
            "common-voice", "librispeech", "crema-d",
            "ravdess", "esd", "l2-arctic", "asvspoof-2021",
        }


class TestDemoClips:
    def test_returns_up_to_n_per_emotion(self, crema_root: Path):
        picked = demo_clips_by_emotion(CremaDLoader(crema_root), per_class=1)
        assert set(picked) == {"angry", "happy", "sad", "neutral", "fearful", "disgust"}
        assert all(len(v) == 1 for v in picked.values())

    def test_is_deterministic_for_a_seed(self, crema_root: Path):
        a = demo_clips_by_emotion(CremaDLoader(crema_root), per_class=1, seed=7)
        b = demo_clips_by_emotion(CremaDLoader(crema_root), per_class=1, seed=7)
        assert {k: [m.sample_id for m in v] for k, v in a.items()} == {
            k: [m.sample_id for m in v] for k, v in b.items()
        }

    def test_does_not_invent_clips_when_a_class_is_short(self, crema_root: Path):
        picked = demo_clips_by_emotion(CremaDLoader(crema_root), per_class=5)
        # only one clip per emotion exists in the fixture
        assert all(len(v) == 1 for v in picked.values())

    def test_rejects_negative_per_class(self, crema_root: Path):
        with pytest.raises(ValueError):
            demo_clips_by_emotion(CremaDLoader(crema_root), per_class=-1)
