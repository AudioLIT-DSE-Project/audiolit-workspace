import { useEffect, useState } from "react";
import { Info, X } from "lucide-react";

interface DatasetLicenseNoticeProps {
  /** Registry name of the selected corpus, e.g. "ravdess". */
  datasetName: string;
  /** Human-friendly label, falls back to datasetName if not given. */
  datasetLabel?: string;
  license: string;
}

/**
 * FR2.3 / SAD C5 — a licence notice for non-commercial benchmark corpora
 * (RAVDESS, L2-ARCTIC, ESD, ASVspoof 2021 DF). Dismissible per dataset: the
 * dismissal is keyed by `datasetName` so switching to a different
 * non-commercial corpus shows the notice again rather than staying hidden
 * from an unrelated dataset's dismissal.
 */
export const DatasetLicenseNotice = ({
  datasetName,
  datasetLabel,
  license,
}: DatasetLicenseNoticeProps) => {
  const [dismissedFor, setDismissedFor] = useState<string | null>(null);

  // Re-arm the notice whenever the selected dataset changes.
  useEffect(() => {
    setDismissedFor((prev) => (prev === datasetName ? prev : null));
  }, [datasetName]);

  if (dismissedFor === datasetName) return null;

  return (
    <div className="bg-amber-500/10 border-b border-amber-500/20 px-3 py-1.5 flex items-center justify-between text-xs text-amber-700 dark:text-amber-400 gap-2">
      <div className="flex items-center gap-2 truncate">
        <Info className="h-3.5 w-3.5 shrink-0" />
        <span className="truncate">
          <span className="font-medium">{datasetLabel || datasetName}</span> is a
          non-commercial / research-use corpus — licence: {license}.
        </span>
      </div>
      <button
        type="button"
        aria-label="Dismiss licence notice"
        className="shrink-0 rounded p-0.5 hover:bg-amber-500/20 transition-colors"
        onClick={() => setDismissedFor(datasetName)}
      >
        <X className="h-3 w-3" />
      </button>
    </div>
  );
};
