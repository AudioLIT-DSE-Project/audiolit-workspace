"""
One-time local data-prep for the L2-ARCTIC accent-bias diagnostic (LIT-170).

`L2ArcticLoader` (app/infrastructure/dataset_ingestion.py) reads each
utterance's ground truth from `<root>/<SPEAKER>/transcript/<utt>.txt`, but a
locally checked-out L2-ARCTIC subset may only ship `wav/` audio with the
ground-truth text sitting in `l2_arctic_subset_metadata.csv`'s `statement_en`
column instead -- which no loader reads. Without the per-utterance .txt
files, every sample is silently skipped ("no ground-truth transcript") and
the group-wise WER diagnostic (run_accent_bias_diagnostic /
accent_bias_runner.py) scores nothing.

This materializes those `transcript/<utt>.txt` files from the CSV so the
diagnostic can run for real. Local data prep only -- `Backend/data/` is
gitignored, so this needs to be re-run by anyone who (re-)provisions the
L2-ARCTIC subset locally; it does not touch loader or app code.

Usage: python -m scripts.prepare_l2arctic_transcripts
"""
from __future__ import annotations

import csv

from app.infrastructure.dataset_ingestion import DATA_DIR

L2ARCTIC_DIR = DATA_DIR / "l2arctic"
CSV_PATH = L2ARCTIC_DIR / "l2_arctic_subset_metadata.csv"


def prepare_transcripts() -> dict:
    written = 0
    skipped_existing = 0
    missing_wav = 0
    missing_text = 0

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            speaker = row["speaker_id"].strip()
            sample_id = row["sample_id"].strip()
            text = row["statement_en"].strip()

            if not speaker or not sample_id:
                continue
            if not text:
                missing_text += 1
                continue

            wav_path = L2ARCTIC_DIR / speaker / "wav" / f"{sample_id}.wav"
            if not wav_path.exists():
                missing_wav += 1
                continue

            transcript_path = L2ARCTIC_DIR / speaker / "transcript" / f"{sample_id}.txt"
            if transcript_path.exists():
                skipped_existing += 1
                continue

            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(text, encoding="utf-8")
            written += 1

    return {
        "written": written,
        "skipped_existing": skipped_existing,
        "missing_wav": missing_wav,
        "missing_text": missing_text,
    }


def main() -> None:
    if not CSV_PATH.exists():
        print(f"No local L2-ARCTIC metadata CSV at {CSV_PATH} -- nothing to do.")
        return
    result = prepare_transcripts()
    print(
        f"written={result['written']} skipped_existing={result['skipped_existing']} "
        f"missing_wav={result['missing_wav']} missing_text={result['missing_text']}"
    )


if __name__ == "__main__":
    main()
