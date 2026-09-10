"""
Headway - normalisation.

Two stages, in this order, because they answer two different questions:

  ConditionNormaliser   "what SHOULD this reading be, given today's weather,
                         crowding and time of day?"  -> residual
  AssetBaseline         "is this residual unusual FOR THIS PARTICULAR DOOR?"
                         -> health_index, in robust sigma units

Measured by scripts/build_health.py at an equal budget of 150 asset-day alerts
over 9 episodes:

    raw fleet-wide threshold      7/9 found, 12.2 d median, 3.5 d worst, 56% precise
    + AssetBaseline               9/9 found, 12.2 d median, 8.5 d worst, 90% precise
    + ConditionNormaliser         9/9 found, 13.1 d median, 9.5 d worst, 91% precise

Two things worth holding on to:

Coverage comes from AssetBaseline. It is what takes 7/9 to 9/9 and 56% to 90%
precision, and it contributes most of the worst-case gain (+5.0 d, against
+1.0 d for the condition model). A fleet-wide threshold flags the stiffest
healthy door before it flags the worst degrading one, no matter how well you
model the weather. Condition normalisation adds on top of a per-asset baseline;
it does not substitute for one.

That attribution is implementation-sensitive, though - not a law. Run the same
ablation in scripts/check_premise.py, whose baseline is a plain offset with no
robust rescaling, and the split inverts: the baseline contributes +0.0 d of
worst-case and the condition model +6.0 d. The stages are substitutes at the
margin. Claim the combination, not a ranking.

`health_index` is deliberately unit-free and robust-scaled, so it is comparable
across doors, across fleets, and across operators - which is the standardised
condition-monitoring parameter the Rail Reliability Taskforce asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Context columns that are angles, not magnitudes. Expanded as harmonics so that
# 23:50 and 00:10 are near each other rather than 24 units apart.
CYCLIC: dict[str, float] = {"hour_of_day": 24.0}

MAD_TO_SIGMA = 1.4826   # scales median-absolute-deviation to a Gaussian sigma


def _design(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """Expand context columns into a design matrix.

    Magnitudes get a quadratic term (grease viscosity vs temperature is not
    linear), angles get two harmonics. Kept in one function so that fit and
    transform can never disagree about column order.
    """
    parts: list[np.ndarray] = []
    names: list[str] = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0)
        if c in CYCLIC:
            period = CYCLIC[c]
            for k in (1, 2):
                parts += [np.sin(2 * np.pi * k * x / period), np.cos(2 * np.pi * k * x / period)]
                names += [f"{c}_sin{k}", f"{c}_cos{k}"]
        else:
            parts += [x, x**2]
            names += [c, f"{c}^2"]
    return np.column_stack(parts), names


@dataclass
class ConditionNormaliser:
    """Model E[signal | operating conditions]; expose the residual.

    Fitted WITHOUT fault labels. At a base rate of 2e-05 the degradation present
    in the training data biases the fit by a negligible amount - and `robust`
    trims the worst residuals and refits, so even heavier contamination (a real
    fleet with several sick assets) does not drag the expected-value model up
    toward the faults it is supposed to reveal.
    """

    signal: str
    context: list[str]
    alpha: float = 1.0
    robust: bool = True
    trim: float = 0.01          # fraction of largest |residual| dropped before refit

    model_: Ridge | None = field(default=None, init=False)
    names_: list[str] = field(default_factory=list, init=False)
    mu_: np.ndarray | None = field(default=None, init=False)
    sd_: np.ndarray | None = field(default=None, init=False)

    def _standardise(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            self.mu_ = X.mean(axis=0)
            self.sd_ = np.where(X.std(axis=0) < 1e-12, 1.0, X.std(axis=0))
        return (X - self.mu_) / self.sd_

    def fit(self, df: pd.DataFrame) -> ConditionNormaliser:
        X, self.names_ = _design(df, self.context)
        Xs = self._standardise(X, fit=True)
        y = pd.to_numeric(df[self.signal], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)

        self.model_ = Ridge(alpha=self.alpha).fit(Xs[ok], y[ok])

        if self.robust and ok.sum() > 100:
            r = np.abs(y[ok] - self.model_.predict(Xs[ok]))
            keep = r <= np.quantile(r, 1.0 - self.trim)
            self.model_ = Ridge(alpha=self.alpha).fit(Xs[ok][keep], y[ok][keep])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("ConditionNormaliser.fit() must be called before transform()")
        X, names = _design(df, self.context)
        if names != self.names_:
            raise ValueError(
                f"context layout changed between fit and transform.\n"
                f"  fitted:  {self.names_}\n  given:   {names}"
            )
        expected = self.model_.predict(self._standardise(X, fit=False))
        y = pd.to_numeric(df[self.signal], errors="coerce").to_numpy(dtype=float)
        return df.assign(expected=expected, residual=y - expected)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def explain(self) -> pd.DataFrame:
        """Standardised coefficients - how much each condition moves the signal.

        Feeds the 'why' line on the Aspect Card, and is worth showing in the
        pitch: it is the model stating, in the signal's own units, that a hot
        crowded afternoon is worth more than three weeks of early wear.
        """
        if self.model_ is None:
            raise RuntimeError("fit() first")
        return (pd.DataFrame({"term": self.names_, "coef": self.model_.coef_})
                  .assign(abs_coef=lambda d: d.coef.abs())
                  .sort_values("abs_coef", ascending=False, ignore_index=True)
                  .drop(columns="abs_coef"))


@dataclass
class AssetBaseline:
    """Per-asset location and robust scale, learned over a reference period.

    Turns a residual into `health_index`: how many robust sigmas this door is
    above its own normal. Unit-free by construction, so a door on the North-South
    Line and a bogie on a different operator's fleet are directly comparable.

    Assets unseen at fit time fall back to the fleet median location and scale
    rather than raising - a new train entering service must not crash the
    dashboard.
    """

    value: str = "residual"
    by: str = "asset_id"
    reference_days: int = 21
    min_periods: int = 5

    loc_: pd.Series | None = field(default=None, init=False)
    scale_: pd.Series | None = field(default=None, init=False)
    fleet_loc_: float = field(default=0.0, init=False)
    fleet_scale_: float = field(default=1.0, init=False)

    def fit(self, df: pd.DataFrame, ts: str = "day") -> AssetBaseline:
        first = df.groupby(self.by)[ts].transform("min")
        ref = df[df[ts] < first + pd.Timedelta(days=self.reference_days)]

        g = ref.groupby(self.by)[self.value]
        loc = g.median()
        scale = g.apply(lambda s: MAD_TO_SIGMA * (s - s.median()).abs().median())
        counts = g.size()

        # Too few reference points, or a degenerate scale, means we do not trust
        # this asset's own baseline yet - fall back to the fleet.
        self.fleet_loc_ = float(loc.median())
        self.fleet_scale_ = float(scale[scale > 0].median()) if (scale > 0).any() else 1.0

        loc = loc.where(counts >= self.min_periods, self.fleet_loc_)
        scale = scale.where((scale > 1e-9) & (counts >= self.min_periods), self.fleet_scale_)

        self.loc_, self.scale_ = loc, scale
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.loc_ is None:
            raise RuntimeError("AssetBaseline.fit() must be called before transform()")
        loc = df[self.by].map(self.loc_).fillna(self.fleet_loc_).to_numpy(dtype=float)
        scale = df[self.by].map(self.scale_).fillna(self.fleet_scale_).to_numpy(dtype=float)
        return df.assign(health_index=(df[self.value].to_numpy(dtype=float) - loc) / scale)

    def fit_transform(self, df: pd.DataFrame, ts: str = "day") -> pd.DataFrame:
        return self.fit(df, ts=ts).transform(df)


@dataclass
class MultivariateHealthIndex:
    """Collapse several normalised residuals into ONE directional health index.

    Motivated by losing our own tournament. A univariate index built on the
    primary signal placed mid-field, beaten by six detectors that consumed every
    normalised feature at once - because door wear also shows up in travel
    distance and cycle duration, and a single-signal index throws that away.

    WHY NOT JUST MAHALANOBIS DISTANCE
    ---------------------------------
    Distance is unsigned. It treats an asset that has drifted BETTER than its
    baseline as exactly as alarming as one that has drifted worse, and it cannot
    be projected forward to a failure threshold because it never decreases in a
    meaningful direction. RUL needs a quantity that rises with wear and only
    with wear.

    WHAT THIS DOES INSTEAD
    ----------------------
    1. Orient every signal so that larger means worse, using the contract's
       declared orientation. A worn door draws more current and travels LESS
       far; summing those raw directions cancels real evidence.
    2. Whiten against the covariance of the reference period, so four correlated
       current measurements do not count as four independent votes.
    3. Project onto the direction in which all oriented signals rise together -
       which is what degradation does - and scale so the result is standard
       normal on healthy data.

    The output stays a single scalar in sigma units, so RUL, the aspect
    thresholds and the Aspect Card keep working unchanged. That property is the
    reason this is a drop-in rather than a rewrite.
    """

    signals: list[str]
    orientation: dict[str, int] = field(default_factory=dict)
    reference_days: int = 21
    ridge: float = 1e-6          # stabilises the whitening on near-singular covariance
    mode: str = "projection"     # "projection" (directional) | "mahalanobis" (unsigned)

    mean_: np.ndarray | None = field(default=None, init=False)
    whiten_: np.ndarray | None = field(default=None, init=False)
    scale_: float = field(default=1.0, init=False)

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for s in self.signals:
            v = pd.to_numeric(df[s], errors="coerce").to_numpy(dtype=float)
            cols.append(self.orientation.get(s, 1) * v)
        X = np.column_stack(cols)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, daily: pd.DataFrame, ts: str = "day") -> MultivariateHealthIndex:
        ref = daily[daily[ts] < daily[ts].min() + pd.Timedelta(days=self.reference_days)]
        X = self._matrix(ref if len(ref) > len(self.signals) * 5 else daily)

        self.mean_ = X.mean(axis=0)
        cov = np.cov(X - self.mean_, rowvar=False)
        cov = np.atleast_2d(cov) + self.ridge * np.eye(len(self.signals))

        # Symmetric inverse square root: the whitening transform.
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, self.ridge)
        self.whiten_ = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

        # After whitening, the "everything rises together" projection has
        # variance 1 by construction; rescale on the reference anyway so the
        # units survive a near-singular covariance.
        z = (X - self.mean_) @ self.whiten_
        proj = z.sum(axis=1) / np.sqrt(len(self.signals))
        sd = float(proj.std())
        self.scale_ = sd if sd > 1e-9 else 1.0
        return self

    def transform(self, daily: pd.DataFrame) -> pd.DataFrame:
        if self.whiten_ is None:
            raise RuntimeError("MultivariateHealthIndex.fit() must be called before transform()")
        z = (self._matrix(daily) - self.mean_) @ self.whiten_
        if self.mode == "mahalanobis":
            idx = np.sqrt((z**2).sum(axis=1))
        else:
            idx = z.sum(axis=1) / np.sqrt(len(self.signals)) / self.scale_
        return daily.assign(health_index=idx)

    def fit_transform(self, daily: pd.DataFrame, ts: str = "day") -> pd.DataFrame:
        return self.fit(daily, ts=ts).transform(daily)
