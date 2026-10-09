/**
 * One predicate for "is the selected clip an uploaded/live file, or a dataset row?"
 *
 * Every panel has to answer this, because the answer decides which request body
 * it sends: an uploaded clip is addressed by `file_path`, a dataset row by
 * `dataset` + `dataset_file`. Get it wrong and the backend resolves a different
 * file, or none.
 *
 * It was implemented inline at ten call sites across eight files, in **four
 * variants that did not agree**:
 *
 *   A (MainLayout, PredictionPanel x2, DeepfakeForensicPanel,
 *      AttentionVisualization)
 *       ... || (message && message !== "Selected from dataset"
 *                       && message !== "Selected from embeddings")
 *
 *   B (PerturbationTools x3, FaithfulnessAuditPanel)
 *       ... || message === "Perturbed file"
 *           || message === "File uploaded successfully"
 *           || message === "File uploaded and processed successfully"
 *           || (message && !message.includes("Selected from"))
 *
 *   C (SaliencyVisualization)
 *       adds `'file_id' in selectedFile` and `startsWith('uploads/')`, and
 *       applies the "Selected from" exclusion as a trailing AND rather than
 *       inside the OR - different precedence, different result.
 *
 *   D (AudioDataTable) - structurally different again: `file_id` / `is_live` /
 *      `is_uploaded` flags, a filename prefix, and a cross-reference against
 *      the uploaded-files list. That one answers a different question (should
 *      this *row* show live-clip affordances) and keeps its own implementation;
 *      see `isLiveRecordingFilename` below, which it can share.
 *
 * A and B disagree whenever the message is a "Selected from ..." string that is
 * not one of A's two exact spellings: A calls it an upload, B calls it a
 * dataset row. So the Prediction panel and the Perturbation panel could send
 * *different request shapes for the same selected clip*. That is the run of
 * consecutive "properly resolve live audio recording file references" fixes in
 * the git log - each panel was corrected on its own, because the predicate had
 * no single home.
 *
 * This module is that home. Behaviour is the union of A, B and C, with one
 * deliberate correction: any `Selected from ...` message now means a dataset
 * selection at every call site (B's and C's reading). Under A, a message such
 * as "Selected from waveform" was classified as an upload, which is wrong.
 */

/** The shape every caller actually relies on. Deliberately loose: callers pass
 *  several different file types and some carry `message` only at runtime. */
export interface SelectableAudio {
  file_path?: string;
  file_id?: string;
  filename?: string;
  message?: string;
  is_live?: boolean;
  is_uploaded?: boolean;
}

/** Messages the UI attaches when the user picked an existing corpus row. */
const DATASET_SELECTION_PREFIX = "Selected from";

/** What `AudioUploadRecorderModal` names a recording before upload. */
const LIVE_RECORDING_PREFIX = "live_recording_";

/**
 * True if `name` is one of our live recordings.
 *
 * Checks the **basename**, not the whole path. A corpus clip that merely has
 * `live_recording` somewhere in its directory path is not a live recording, and
 * the substring test the call sites used would have claimed it was.
 */
export const isLiveRecordingFilename = (name?: string | null): boolean => {
  if (!name) return false;
  const base = name.split("/").pop()?.split("\\").pop() ?? name;
  return base.startsWith(LIVE_RECORDING_PREFIX);
};

/**
 * True if the selection is an uploaded or live-recorded clip, addressed by
 * `file_path`; false if it is a corpus row, addressed by `dataset` +
 * `dataset_file`.
 *
 * `dataset` is the currently selected dataset id, because a `custom:` dataset is
 * a session upload however its path looks.
 *
 * Note on `file_path`: the backend renames every upload to a UUID, so a live
 * recording's stored path is `uploads/<uuid>.wav` and the
 * `live_recording_...` name survives only in `filename`. Both are checked -
 * checking the path alone (as the call sites did) relies entirely on the
 * `uploads/` segment and would break if the upload directory were ever
 * relocated or made absolute.
 */
export const isUploadedAudio = (
  file: SelectableAudio | null | undefined,
  dataset?: string | null,
): boolean => {
  if (!file || typeof file !== "object" || !file.file_path) return false;

  const message = typeof file.message === "string" ? file.message : "";

  // An explicit corpus selection is never an upload, whatever the path looks
  // like. Applied first so it cannot be overridden by a path coincidence.
  if (message.startsWith(DATASET_SELECTION_PREFIX)) return false;

  return Boolean(
    file.file_id ||
      file.is_live ||
      file.is_uploaded ||
      dataset?.startsWith("custom:") ||
      file.file_path.includes("uploads/") ||
      file.file_path.includes("uploads\\") ||
      isLiveRecordingFilename(file.file_path) ||
      isLiveRecordingFilename(file.filename) ||
      message !== "",
  );
};

/**
 * The request body fragment that addresses `file` for the backend.
 *
 * Callers were each assembling this by hand next to their own copy of the
 * predicate, which is the other half of the same drift: a site could classify
 * correctly and still build the wrong body.
 */
export const audioRequestRef = (
  file: SelectableAudio | null | undefined,
  dataset?: string | null,
  datasetFile?: string | null,
): { file_path: string } | { dataset: string; dataset_file: string } | null => {
  if (isUploadedAudio(file, dataset)) {
    return { file_path: file!.file_path as string };
  }
  if (dataset && datasetFile) {
    return { dataset, dataset_file: datasetFile };
  }
  return null;
};
