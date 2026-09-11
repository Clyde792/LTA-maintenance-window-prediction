"""Causal monitoring status, separate from mechanical health and RUL."""
import numpy as np
import pandas as pd


def evidence_status(row, *, as_of, last_supported=None, max_age_days=1):
    """Daily replay clock uses aggregate availability, never the observation date."""
    now = pd.Timestamp(as_of)
    available = pd.to_datetime(row.get("available_at"))
    missing = [c.removesuffix("_completeness") for c in row.index
               if c.endswith("_completeness") and
               (pd.isna(row[c]) or row[c] < .9)]
    if pd.isna(available):
        status = "missing"
    elif available > now:
        status = "not_available"
    elif now - available > pd.Timedelta(days=max_age_days):
        status = "stale"
    elif not pd.notna(row.get("context_supported")) or not bool(row.get("context_supported")):
        status = "outside_conditions"
    elif not pd.notna(row.get("data_quality_ok")) or not bool(row.get("data_quality_ok")) or missing:
        status = "incomplete"
    elif not np.isfinite(row.get("health_index_smooth", np.nan)):
        status = "warming_up"
    else:
        status = "supported"
        last_supported = available
    # Never disclose future support or channels from an unavailable aggregate.
    if last_supported is not None and pd.Timestamp(last_supported) > now:
        last_supported = None
    if status == "not_available":
        missing = []
    return {"status": status, "missingChannels": missing,
            "lastSupportedAt": pd.Timestamp(last_supported).isoformat() if last_supported is not None else None}
