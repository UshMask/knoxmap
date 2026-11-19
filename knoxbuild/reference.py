"""Compare generated counts with external data without filling in missing values."""
from __future__ import annotations

import re


def compare_count(metric, generated, *, source, field, raw, comparable=True):
    """Compare counts on the same area and scope. Delta is generated minus reference."""
    reference = None
    if raw is not None and re.fullmatch(r"[0-9]+", str(raw).strip()):
        try:
            reference = int(raw)
        except ValueError:
            pass

    delta = None
    if raw is None:
        status = "missing_reference"
    elif reference is None:
        status = "invalid_reference"
    elif not comparable:
        status = "not_comparable"
    else:
        delta = generated - reference
        status = "below_reference" if delta < 0 else "above_reference" if delta > 0 else "match"
    return {"metric": metric, "reference": {"source": source, "field": field,
                                            "raw": raw, "value": reference},
            "generated": generated, "delta": delta, "status": status}
