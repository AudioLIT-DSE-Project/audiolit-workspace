from enum import Enum


class Provenance(str, Enum):
    MEASURED = "measured"        # produced by the model on this input
    FALLBACK = "fallback"        # synthesised stand-in, NOT model output
    UNAVAILABLE = "unavailable"  # could not be produced at all


def provenance_fields(source: Provenance, reason: str | None = None) -> dict:
    """Returns {"provenance": ..., "provenance_reason": ...} for merging
    into any XAI response payload."""
    if source == Provenance.FALLBACK:
        if not reason or not reason.strip():
            raise ValueError("Provenance.FALLBACK requires a non-empty reason string explaining why fallback was used.")
        return {
            "provenance": source.value,
            "provenance_reason": reason.strip(),
        }
    return {
        "provenance": source.value,
        "provenance_reason": reason.strip() if reason else None,
    }
