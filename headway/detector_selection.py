"""Select a frozen detector using past validation only, then assess later data.

Policy thresholds are illustrative. No validation qualification means no winner.
Scores must already come from causally fitted preprocessing and detectors.
"""
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .evaluate.metrics import evaluate_replay


@dataclass(frozen=True)
class SelectionPolicy:
    threshold_quantile: float = .99
    daily_budget: int = 2
    cooldown_days: float = 3.
    minimum_fault_groups: int = 3
    minimum_recall: float = .8
    maximum_false_alerts_per_asset_month: float = 1.
    minimum_score_fraction: float = .9

    def __post_init__(self):
        if (not 0 < self.threshold_quantile < 1 or type(self.daily_budget) is not int
                or self.daily_budget < 1 or not np.isfinite(self.cooldown_days) or self.cooldown_days < 0
                or type(self.minimum_fault_groups) is not int or self.minimum_fault_groups < 1
                or not 0 <= self.minimum_recall <= 1 or not 0 <= self.minimum_score_fraction <= 1
                or not np.isfinite(self.maximum_false_alerts_per_asset_month)
                or self.maximum_false_alerts_per_asset_month < 0):
            raise ValueError("invalid selection policy")


def validate_scores(scores, columns):
    required = {"asset_id", "day", "available_at", *columns}
    if required - set(scores):
        raise ValueError(f"missing score columns: {sorted(required - set(scores))}")
    if not scores.index.is_unique or scores.duplicated(["asset_id", "available_at"]).any():
        raise ValueError("unique rows and asset prediction times required")
    if scores[["day", "available_at"]].isna().any().any():
        raise ValueError("missing prediction times")
    if (scores.available_at < scores.day + pd.Timedelta(days=1)).any():
        raise ValueError("daily scores available before observation day ends")


def measured(scores, name, episodes, threshold, policy):
    eligible = np.isfinite(pd.to_numeric(scores[name], errors="coerce"))
    result = evaluate_replay(scores, name, episodes, threshold,
        daily_budget=policy.daily_budget, cooldown_days=policy.cooldown_days, name=name).as_row()
    result["score_fraction"] = float(eligible.mean()) if len(scores) else 0.
    result["episode_recall"] = result["detected"] / len(episodes) if len(episodes) else None
    result["fault_groups"] = int(episodes.train_id.nunique())
    result["observation_rows"] = len(scores)
    return result


def select_detector(scores, episodes, columns, *, calibration_start, calibration_end,
                    validation_end, policy=SelectionPolicy()):
    """Freeze one threshold per model, rank models on validation only.

    Faults that straddle a split onset remain outside validation; their asset's
    validation rows are excluded from false-alert scoring. The caller must supply
    labels known by validation_end, with reliable coverage of the evaluated fleet.
    Later episodes/rows are ignored, so they cannot choose the model.
    """
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("distinct candidate score names required")
    validate_scores(scores, columns)
    c0, c1, v1 = map(pd.Timestamp, (calibration_start, calibration_end, validation_end))
    if not c0 < c1 < v1:
        raise ValueError("calibration and validation must be ordered non-overlapping periods")
    calibration = scores[(scores.available_at > c0) & (scores.available_at <= c1)]
    validation = scores[(scores.available_at > c1) & (scores.available_at <= v1)]
    known = episodes[episodes.fault_ts <= v1]
    complete = known[(known.onset_ts > c1) & (known.fault_ts > c1)]
    crossing_assets = set(known.loc[(known.onset_ts <= c1) & (known.fault_ts > c1), "asset_id"])
    validation = validation[~validation.asset_id.isin(crossing_assets)]
    complete = complete[~complete.asset_id.isin(crossing_assets)]
    candidates = []
    for name in columns:
        values = pd.to_numeric(calibration[name], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) < 5:
            candidates.append({"name": name, "qualified": False, "reasons": ["insufficient threshold calibration scores"]})
            continue
        threshold = float(values.quantile(policy.threshold_quantile))
        stats = measured(validation, name, complete, threshold, policy)
        reasons = []
        if stats["fault_groups"] < policy.minimum_fault_groups:
            reasons.append("too few independent validation fault groups")
        if stats["episode_recall"] is None or stats["episode_recall"] < policy.minimum_recall:
            reasons.append("validation episode recall below policy")
        if stats["score_fraction"] < policy.minimum_score_fraction:
            reasons.append("too many observations without a score")
        if stats["false_alerts_per_asset_month"] > policy.maximum_false_alerts_per_asset_month:
            reasons.append("false-alert workload above policy")
        candidates.append({**stats, "threshold": threshold, "qualified": not reasons, "reasons": reasons})
    qualified = [r for r in candidates if r["qualified"]]
    # Same budget and policy. Prioritise episode recall, then false-alert workload,
    # then worst warning time; alphabetic name is a deterministic tie-break.
    qualified.sort(key=lambda r: (-r["episode_recall"], r["false_alerts_per_asset_month"],
                                  -r["worst_lead_d"], r["name"]))
    winner = qualified[0] if qualified else None
    return {"policy": asdict(policy), "calibration_start": str(c0), "calibration_end": str(c1),
            "validation_end": str(v1), "selected_model": winner["name"] if winner else None,
            "selected_threshold": winner["threshold"] if winner else None,
            "reason": "qualified on validation only" if winner else "no candidate meets the validation evidence and workload gates",
            "excluded_crossing_assets": sorted(crossing_assets), "candidates": candidates,
            "limitations": ["Not a universal best model or operational approval.",
                "Unlabelled alerts are only false alerts if fault annotation coverage is complete.",
                "Validation results require representative independent fault groups."]}


def assess_selected(selection, scores, episodes):
    """Evaluate only the already-selected model; never revise the winner here."""
    name = selection["selected_model"]
    if name is None:
        return {"status": "not_evaluated", "reason": "no validation-qualified model"}
    validate_scores(scores, [name])
    cutoff = pd.Timestamp(selection["validation_end"])
    test = scores[scores.available_at > cutoff]
    if test.empty:
        return {"status": "not_evaluated", "reason": "no subsequent observations"}
    end = test.available_at.max()
    known = episodes[episodes.fault_ts <= end]
    crossing = set(known.loc[(known.onset_ts <= cutoff) & (known.fault_ts > cutoff), "asset_id"])
    test = test[~test.asset_id.isin(crossing)]
    eps = known[(known.onset_ts > cutoff) & ~known.asset_id.isin(crossing)]
    result = measured(test, name, eps, selection["selected_threshold"], SelectionPolicy(**selection["policy"]))
    return {"status": "evaluated", **result, "excluded_crossing_assets": sorted(crossing)}
