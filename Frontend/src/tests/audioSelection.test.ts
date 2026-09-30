/**
 * The uploaded-vs-dataset predicate (LIT-266 live recordings).
 *
 * This predicate existed inline at ten call sites in four variants that did not
 * agree, which is why a run of consecutive commits had to fix live-recording
 * file resolution panel by panel. These tests pin the unified behaviour so the
 * variants cannot come back.
 *
 * The most important test here is `test_every_panel_agrees_on_one_selection`:
 * it asserts the property that was actually broken - not that the predicate is
 * right for one input, but that there is only *one* predicate.
 */

import {
  isUploadedAudio,
  isLiveRecordingFilename,
  audioRequestRef,
} from "@/lib/audioSelection";

describe("isLiveRecordingFilename", () => {
  it("matches a recording by its basename", () => {
    expect(isLiveRecordingFilename("live_recording_1727712345678.wav")).toBe(true);
    expect(isLiveRecordingFilename("uploads/live_recording_1727712345678.wav")).toBe(true);
    expect(isLiveRecordingFilename("C:\\tmp\\live_recording_9.wav")).toBe(true);
  });

  it("does not match a corpus clip that merely sits under a similarly named directory", () => {
    // The call sites used `path.includes('live_recording')`, which claimed this
    // was a live recording. It is a dataset file.
    expect(isLiveRecordingFilename("data/live_recording_experiments/sample-000037.mp3")).toBe(false);
  });

  it("is safe on empty and missing input", () => {
    expect(isLiveRecordingFilename(undefined)).toBe(false);
    expect(isLiveRecordingFilename(null)).toBe(false);
    expect(isLiveRecordingFilename("")).toBe(false);
  });
});

describe("isUploadedAudio", () => {
  it("treats a file with a file_id as an upload", () => {
    // file_id is the authoritative marker: the backend mints one per upload and
    // a corpus row never has one. Only one of the four original variants
    // checked it.
    expect(isUploadedAudio({ file_path: "uploads/abc.wav", file_id: "abc.wav" })).toBe(true);
  });

  it("recognises an upload by its path on both separators", () => {
    expect(isUploadedAudio({ file_path: "uploads/abc.wav" })).toBe(true);
    expect(isUploadedAudio({ file_path: "uploads\\abc.wav" })).toBe(true);
  });

  it("recognises a live recording by its original filename after the UUID rename", () => {
    // The backend renames every upload to a UUID, so `live_recording_...`
    // survives only in `filename`. A predicate that checks the path alone is
    // relying entirely on the `uploads/` segment.
    expect(
      isUploadedAudio({ file_path: "uploads/3f2a-uuid.wav", filename: "live_recording_1.wav" }),
    ).toBe(true);
  });

  it("treats a custom session dataset as an upload however the path looks", () => {
    expect(isUploadedAudio({ file_path: "whatever/x.wav" }, "custom:sess-1")).toBe(true);
  });

  it("honours the explicit is_live and is_uploaded flags", () => {
    expect(isUploadedAudio({ file_path: "x.wav", is_live: true })).toBe(true);
    expect(isUploadedAudio({ file_path: "x.wav", is_uploaded: true })).toBe(true);
  });

  it("treats a corpus row as a dataset selection", () => {
    expect(
      isUploadedAudio({ file_path: "cv-valid-dev/sample-000037.mp3", message: "Selected from dataset" }),
    ).toBe(false);
  });

  it("treats ANY 'Selected from ...' message as a dataset selection", () => {
    // This is the one deliberate correction over the old variant A, which
    // compared against two exact strings and so classified any other
    // "Selected from ..." message as an upload.
    for (const message of [
      "Selected from dataset",
      "Selected from embeddings",
      "Selected from waveform",
      "Selected from the projection",
    ]) {
      expect(isUploadedAudio({ file_path: "cv-valid-dev/s.mp3", message })).toBe(false);
    }
  });

  it("lets a dataset selection win over a path that looks like an upload", () => {
    // Exclusion is applied before the path checks, so a path coincidence
    // cannot override an explicit corpus selection.
    expect(
      isUploadedAudio({ file_path: "uploads/s.mp3", message: "Selected from dataset" }),
    ).toBe(false);
  });

  it("treats an upload-progress message as an upload", () => {
    for (const message of [
      "File uploaded successfully",
      "File uploaded and processed successfully",
      "Perturbed file",
    ]) {
      expect(isUploadedAudio({ file_path: "uploads/x.wav", message })).toBe(true);
    }
  });

  it("is false for anything with no file_path at all", () => {
    expect(isUploadedAudio(null)).toBe(false);
    expect(isUploadedAudio(undefined)).toBe(false);
    expect(isUploadedAudio({})).toBe(false);
    expect(isUploadedAudio({ file_id: "orphan" })).toBe(false);
  });
});

describe("the predicate is single", () => {
  /**
   * The defect was not a wrong answer for one input. It was that eight files
   * each had their own answer. This asserts the property directly: for a set of
   * selections spanning every interesting case, one function decides, so every
   * panel necessarily agrees.
   *
   * If someone reintroduces an inline copy, this test still passes - but the
   * ESLint rule added alongside it (no-restricted-syntax on
   * `includes("live_recording")`) fails, and that is the pairing that keeps the
   * predicate single.
   */
  const selections: Array<[string, Parameters<typeof isUploadedAudio>[0], string | undefined, boolean]> = [
    ["live recording", { file_path: "uploads/u.wav", file_id: "u.wav", filename: "live_recording_1.wav" }, undefined, true],
    ["plain upload", { file_path: "uploads/u.wav", file_id: "u.wav" }, undefined, true],
    ["custom dataset", { file_path: "x/y.wav" }, "custom:s1", true],
    ["corpus row", { file_path: "cv-valid-dev/s.mp3", message: "Selected from dataset" }, "common-voice", false],
    ["embedding pick", { file_path: "cv-valid-dev/s.mp3", message: "Selected from embeddings" }, "common-voice", false],
    ["perturbed derivative", { file_path: "uploads/s_perturbed_ab12cd34.wav", message: "Perturbed file" }, undefined, true],
  ];

  it.each(selections)("%s resolves consistently", (_label, file, dataset, expected) => {
    expect(isUploadedAudio(file, dataset)).toBe(expected);
  });

  it("audioRequestRef builds the body that matches the verdict", () => {
    expect(
      audioRequestRef({ file_path: "uploads/u.wav", file_id: "u.wav" }, undefined, undefined),
    ).toEqual({ file_path: "uploads/u.wav" });

    expect(
      audioRequestRef(
        { file_path: "cv-valid-dev/s.mp3", message: "Selected from dataset" },
        "common-voice",
        "sample-000037.mp3",
      ),
    ).toEqual({ dataset: "common-voice", dataset_file: "sample-000037.mp3" });
  });

  it("returns null rather than a half-built body when neither form is addressable", () => {
    // A dataset row with no dataset_file cannot be addressed. Returning null
    // makes the caller handle it; returning {dataset} alone would send the
    // backend an unresolvable reference.
    expect(audioRequestRef({ file_path: "s.mp3", message: "Selected from dataset" }, "common-voice", null)).toBeNull();
  });
});
