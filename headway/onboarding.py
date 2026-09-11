"""Optional onboarding using only a new asset's historical reference telemetry.

This adapts location and scale, not the fleet context model, health-index weights
or failure model. A complete reference is not proof that the asset is healthy;
eligibility must ultimately be established from maintenance/inspection records.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .normalise import MAD_TO_SIGMA


@dataclass
class OnboardedPipeline:
    pipeline: object
    ready_at: dict
    rejected: dict
    provenance: dict = field(default_factory=dict)
    experimental: bool = False

    def transform(self, cycles):
        out = self.pipeline.transform(cycles)
        ready = pd.to_datetime(out.asset_id.map(self.ready_at))
        accepted = ready.notna()
        out["onboarding_ready_at"] = pd.to_datetime(ready)
        out["onboarding_status"] = np.where(accepted, "asset_reference", "not_onboarded")
        out["reference_verified"] = accepted & (not self.experimental)
        out["reference_evidence_id"] = out.asset_id.map({a:p["evidence_id"] for a,p in self.provenance.items()})
        out.loc[accepted, "baseline_source"] = "onboarded_asset_reference"
        pending = accepted & (out.available_at < ready)
        out.loc[pending, "reference_verified"] = False
        out.loc[pending, "reference_evidence_id"] = None
        out.loc[pending, "data_quality_ok"] = False
        out.loc[pending, "quality_ready"] = False
        out.loc[pending, "onboarding_status"] = "reference_not_yet_available"
        refused = out.asset_id.isin(self.rejected)
        out.loc[refused, "data_quality_ok"] = False
        out.loc[refused, "quality_ready"] = False
        out.loc[refused, "onboarding_status"] = "reference_rejected"
        return out


def onboard(pipeline, reference, *, as_of, reference_days=21, provenance=None, experimental=False):
    """Return an independent adapted pipeline; never inspect fault outcomes.

    The caller supplies the reference cut, not an entire test trajectory. Refuse
    future observations, existing training assets, incomplete calendar windows
    and degenerate scales. Rejected assets must abstain rather than silently use
    the fleet fallback as if onboarding had succeeded.
    """
    if pipeline.multivariate is None:
        raise RuntimeError("fit the fleet pipeline before onboarding")
    if reference_days < 5 or reference.empty:
        raise ValueError("at least five reference days and non-empty telemetry required")
    as_of = pd.Timestamp(as_of)
    if reference.ts.isna().any() or (reference.ts >= as_of).any():
        raise ValueError("reference contains observations unavailable at as_of")
    existing = set(next(iter(pipeline.baselines.values())).loc_.index)
    if set(reference.asset_id) & existing:
        raise ValueError("onboarding must not overwrite assets used for fleet training")
    # Remove labels before even aggregating the reference frame.
    ref = reference.drop(columns=[c for c in ("fault_confirmed", "fault_mode") if c in reference])
    daily = pipeline._daily(ref)
    result = OnboardedPipeline(deepcopy(pipeline), {}, {}, experimental=experimental)
    for asset, g in daily.groupby("asset_id"):
        first = g.day.min()
        ready = first + pd.Timedelta(days=reference_days)
        record = (provenance or {}).get(asset, {})
        if not experimental:
            try:
                valid = (record.get("eligible") is True
                    and all(isinstance(record.get(k), str) and record[k].strip() for k in ("reviewer", "evidence_id"))
                    and pd.Timestamp(record["reference_start"]) <= first
                    and pd.Timestamp(record["reference_end"]) >= ready
                    and ready <= pd.Timestamp(record["reviewed_at"]) <= as_of)
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                result.rejected[asset] = "missing or invalid reviewed reference provenance"
                continue
            result.provenance[asset] = deepcopy(record)
        g = g[(g.day < ready) & (g.available_at <= as_of)]
        if (ready > as_of or g.day.nunique() != reference_days
                or not g.data_quality_ok.all() or not g.context_supported.all()):
            result.rejected[asset] = "incomplete or unsupported reference window"
            continue
        params = {}
        for signal in pipeline.levels:
            values = g[f"res_{signal}"]
            location = float(values.median())
            scale = float(MAD_TO_SIGMA * (values - location).abs().median())
            if not np.isfinite(location) or not np.isfinite(scale) or scale <= 1e-9:
                break
            params[signal] = (location, scale)
        if len(params) != len(pipeline.levels):
            result.rejected[asset] = "degenerate reference signal"
            continue
        for signal, (location, scale) in params.items():
            baseline = result.pipeline.baselines[signal]
            baseline.loc_.loc[asset] = location
            baseline.scale_.loc[asset] = scale
        result.ready_at[asset] = ready if experimental else pd.Timestamp(record["reviewed_at"])
    return result
