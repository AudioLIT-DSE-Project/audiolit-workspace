import { AlertTriangle, CircleSlash } from "lucide-react";

/**
 * FR17.1 — makes fabricated model output visibly distinguishable from measured.
 *
 * The backend contract (LIT-238) tags every XAI payload `measured`, `fallback`
 * or `unavailable`. Until this rendered, a synthesised attention pattern looked
 * exactly like a genuine one on screen, which is the whole defect FR17 exists
 * to remove — the flag existed, but only in the network tab.
 *
 * `measured` renders nothing on purpose: a badge on every correct result trains
 * people to ignore badges.
 */
export type Provenance = "measured" | "fallback" | "unavailable";

interface ProvenanceBadgeProps {
  provenance?: Provenance | string | null;
  reason?: string | null;
  className?: string;
}

export const ProvenanceBadge = ({ provenance, reason, className = "" }: ProvenanceBadgeProps) => {
  if (!provenance || provenance === "measured") return null;

  const isFallback = provenance === "fallback";
  const label = isFallback ? "Not model output" : "Unavailable";
  const Icon = isFallback ? AlertTriangle : CircleSlash;
  const tone = isFallback
    ? "bg-amber-100 text-amber-900 border-amber-300"
    : "bg-muted text-muted-foreground border-border";

  return (
    <span
      role="status"
      title={reason || undefined}
      data-provenance={provenance}
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium ${tone} ${className}`}
    >
      <Icon className="h-3 w-3" aria-hidden />
      {label}
      {reason ? <span className="sr-only">: {reason}</span> : null}
    </span>
  );
};

/**
 * Visual treatment for a fabricated map. A caption alone is not enough: someone
 * screenshotting the panel must not be able to present synthesised output as
 * real, so the pixels themselves have to look wrong.
 */
export const provenanceOverlayStyle = (provenance?: Provenance | string | null) =>
  provenance === "fallback"
    ? { filter: "saturate(0.25) contrast(0.85)", opacity: 0.75 }
    : undefined;
