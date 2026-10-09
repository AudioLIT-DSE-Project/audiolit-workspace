export type ModelTaskFamily = "ASR" | "SER" | "DEEPFAKE";

// Explicit family lookup for the built-in dropdown keys - avoids the substring
// heuristic below misclassifying "wav2vec2-add" (contains "wav2vec2") as SER.
const BUILTIN_MODEL_TASK_FAMILY: Record<string, ModelTaskFamily> = {
  "whisper-base": "ASR",
  wav2vec2: "SER",
  "melody-machine": "DEEPFAKE",
  "wav2vec2-add": "DEEPFAKE",
};

export const getModelTaskFamily = (modelName: string): ModelTaskFamily => {
  if (BUILTIN_MODEL_TASK_FAMILY[modelName]) return BUILTIN_MODEL_TASK_FAMILY[modelName];
  const m = modelName.toLowerCase();
  // Check DEEPFAKE substrings first: a custom-resolved wav2vec2 checkpoint
  // fine-tuned for deepfake detection would otherwise match "wav2vec2" below.
  if (m.includes("asvspoof") || m.includes("deepfake") || m.includes("spoof") || m.includes("fake")) return "DEEPFAKE";
  if (m.includes("wav2vec2") || m.includes("ser") || m.includes("emotion")) return "SER";
  return "ASR";
};
