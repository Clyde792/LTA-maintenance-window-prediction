"""
Headway - remaining useful life, with honest bounds.

This module turns a health index into the thing an engineer can act on: a
MARGIN. How many days do I have?

THE PROJECTION MATCHES THE PHYSICS, AND IS STILL CONFORMALLY CORRECTED
----------------------------------------------------------------------
The point estimate extrapolates the health index to a failure threshold. A
straight line is the obvious choice and it is systematically wrong: wear is
convex, so today's slope understates tomorrow's, the line arrives late, and the
margin OVERSTATES the time available - by a median of +8.7 days, on 77% of
in-window days, measured across four independently seeded fleets. That is the
dangerous direction of error: it tells an engineer to defer when they should
not.

The default is therefore LOG-LINEAR: treat the index as growing by a constant
fraction per day (k = slope / hi), so time-to-threshold = ln(threshold/hi)/k.
Same holdout: median error -1.2 d, MAE 2.7 d - a 69% MAE reduction from one
change of functional form, with no parameters fitted to the 9 episodes.

A near-unbiased point estimate is not a licence to trust it. The conformal
layer stays, calibrated leave-one-EPISODE-out - it just no longer spends its
entire correction patching a known bias, so the bound it produces is tighter
for the same coverage.

THE RECOMMENDATION CONSUMES THE LOWER BOUND, NEVER THE POINT ESTIMATE
---------------------------------------------------------------------
Two assets with the same projected RUL but different uncertainty get different
aspects: the one we understand less is acted on sooner. Uncertainty is an input
to the decision, not a caveat printed underneath it.

THE CONFORMITY MODE IS AN EMPIRICAL QUESTION, AND THE ANSWER HAS CHANGED
------------------------------------------------------------------------
This story has two acts, and both are kept because each act killed a plausible
belief:

Act one, under the LINEAR projection. Its ~9-day optimistic bias was
stage-dependent - worst early in a degradation, mild late - so a flat additive
shift had to be huge to cover it and collapsed every bound to ~0 (safe,
useless, sharpness 0.00). Mondrian bands over health-index volatility won,
because volatility proxied trajectory stage: separate corrections per stage
(x0.19 / x0.32 / x0.37) patched the stage-dependent bias piecewise.

Act two, under the LOG-LINEAR projection. The stage-dependent bias is gone,
and the machinery that existed to patch it stopped earning its keep: the
mondrian bands converge to near-identical values (x0.54 / x0.61 / x0.52), and
additive - vacuous in act one - becomes the best mode on the criteria that
matter: best held-out coverage, tightest interval, 100% safety inside the
withdraw zone, and WITHDRAW reached on 9/9 episodes where the multiplicative
modes miss 2. A multiplicative bound scales DOWN with the estimate, so near
failure it can still sit above a truth of hours; a flat shift crushes small
margins to zero, which is exactly the conservatism the final days deserve.

Selection lives in scripts/build_rul.py and re-decides on evidence at every
run: hold coverage, hold withdraw-zone safety, then take the sharpest
survivor. Each criterion exists because a single-metric selection chose wrong
once - coverage alone picked the vacuous bound, sharpness alone picked the one
that fails near failure. The mondrian machinery stays implemented: the moment
real data shows stage-dependent bias again, it re-arms.

HONEST CAVEAT
-------------
Conformal coverage assumes exchangeability. Days within one episode are
strongly correlated and episodes differ from each other, so pooled per-day
residuals are not exchangeable in the strict sense. Leave-one-episode-out is
the right fold unit and keeps calibration honest about unseen assets, but
`coverage_` is EMPIRICAL, not guaranteed. With 9 episodes read it as a
well-calibrated heuristic, not a proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HI = "health_index_smooth"
SLOPE = "health_index_slope"


def _project(hi: np.ndarray, slope: np.ndarray, threshold: float,
             min_slope: float, horizon: float,
             form: str = "loglinear") -> np.ndarray:
    """Days until `hi` reaches `threshold`, extrapolating the current trend.

    form="linear"     straight line: (threshold - hi) / slope. Systematically
                      optimistic on convex wear - measured at a median +8.7 days
                      too long, on 77% of in-window days, across four fleets.
    form="loglinear"  constant fractional growth: the index is treated as
                      locally exponential, hi(t) ~ hi * exp(k t) with
                      k = slope / hi, so time-to-threshold = ln(threshold/hi)/k.
                      Median error -1.2 d, MAE 2.7 d on the same holdout - the
                      convexity the linear form ignores is mostly captured by
                      assuming growth compounds. Chosen as the default on that
                      evidence (scripts/build_rul.py section [1] reproduces it).

    The log form needs hi > 0 to be defined, and for a barely-elevated index it
    extrapolates from almost nothing - so any row with hi below 1 sigma falls
    back to the straight line rather than manufacturing a confident countdown
    out of noise.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        linear = (threshold - hi) / slope
        if form == "loglinear":
            k = slope / hi
            log_rul = np.log(threshold / hi) / k
            rul = np.where(hi >= 1.0, log_rul, linear)
        else:
            rul = linear
    rul = np.where(slope <= min_slope, np.inf, rul)   # not on a failure path
    rul = np.where(hi >= threshold, 0.0, rul)         # already there
    return np.where(np.isfinite(rul), np.clip(rul, 0.0, horizon), rul)


@dataclass
class ConformalRUL:
    """Projected remaining useful life with a calibrated lower bound.

    Args:
        alpha: miscoverage rate. 0.10 targets a ~90% lower bound.
        mode: "additive" (default) shifts the projection by a calibrated
            constant - under the log-linear form this is the safest mode near
            failure and the evidence-selected production choice. "ratio" uses a
            global multiple; "mondrian" a multiple per volatility band - both
            kept, and re-selected by build_rul.py whenever the evidence moves.
        n_bins: volatility bands for mondrian.
        min_slope: sigma/day below which we decline to project. Healthy assets
            sit at a 99th-percentile slope of ~0.18 on the multivariate index
            (~0.22 on the old univariate one), so 0.25 clears the healthy noise
            floor with margin either way.
    """

    alpha: float = 0.10
    # Point-estimate form. "loglinear" (default) captures the compounding of
    # convex wear; "linear" is kept for the before/after in build_rul.py.
    projection: str = "loglinear"
    # Default matches what the evidence currently selects under the log-linear
    # form (see the module docstring's two-act history). build_rul.py re-decides
    # on every run and is the authority.
    mode: str = "additive"
    n_bins: int = 3
    difficulty: str = "health_index_vol"
    # Calibration points a band needs before we trust its own quantile rather
    # than the pooled one. Below this the quantile is noise, and mondrian
    # correctly degrades to a single global ratio - which is what happens on
    # small fleets with few episodes.
    min_per_bin: int = 20
    min_slope: float = 0.25
    horizon: float = 90.0

    threshold_: float = field(default=np.nan, init=False)
    edges_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    lo_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    hi_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    coverage_: float = field(default=np.nan, init=False)
    n_calib_: int = field(default=0, init=False)

    # --------------------------------------------------------------- helpers
    @property
    def _n(self) -> int:
        return self.n_bins if self.mode == "mondrian" else 1

    def _bins(self, df: pd.DataFrame) -> np.ndarray:
        """Volatility band per row. Single band unless mode is mondrian."""
        if self.mode != "mondrian" or not len(self.edges_):
            return np.zeros(len(df), dtype=int)
        v = pd.to_numeric(df.get(self.difficulty), errors="coerce").to_numpy(float)
        v = np.nan_to_num(v, nan=float(np.median(self.edges_)))
        return np.clip(np.digitize(v, self.edges_), 0, self._n - 1)

    def _score(self, true: np.ndarray, hat: np.ndarray) -> np.ndarray:
        return (true - hat) if self.mode == "additive" else (true / hat)

    def _apply(self, hat: np.ndarray, q: np.ndarray, bins: np.ndarray) -> np.ndarray:
        qq = q[bins]
        return (hat + qq) if self.mode == "additive" else (hat * qq)

    @staticmethod
    def _threshold_from(daily: pd.DataFrame, episodes: pd.DataFrame) -> float:
        """Failure threshold = typical health index at a confirmed fault."""
        vals = []
        for _, e in episodes.iterrows():
            a = daily[(daily.asset_id == e.asset_id) & (daily.day <= e.fault_ts)]
            if len(a) and np.isfinite(a[HI].iloc[-1]):
                vals.append(float(a[HI].iloc[-1]))
        return float(np.median(vals)) if vals else np.nan

    def _episode_scores(self, daily: pd.DataFrame, e: pd.Series,
                        threshold: float) -> tuple[np.ndarray, np.ndarray]:
        """(scores, bins) over one episode's degradation window."""
        w = daily[(daily.asset_id == e.asset_id)
                  & (daily.day >= pd.Timestamp(e.onset_ts).normalize())
                  & (daily.day < pd.Timestamp(e.fault_ts).normalize())]
        if not len(w):
            return np.array([]), np.array([], dtype=int)

        hat = _project(w[HI].to_numpy(float), w[SLOPE].to_numpy(float),
                       threshold, self.min_slope, self.horizon,
                       form=self.projection)
        true = (pd.Timestamp(e.fault_ts) - w.day).dt.total_seconds().to_numpy() / 86400.0
        ok = np.isfinite(hat) & np.isfinite(true) & (hat > 0)
        if not ok.any():
            return np.array([]), np.array([], dtype=int)
        return self._score(true[ok], hat[ok]), self._bins(w)[ok]

    def _quantiles(self, scores: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-band conformal quantiles, with a pooled fallback for thin bands."""
        pooled_lo = float(np.quantile(scores, self.alpha))
        pooled_hi = float(np.quantile(scores, 1.0 - self.alpha))
        lo = np.full(self._n, pooled_lo)
        hi = np.full(self._n, pooled_hi)
        for b in range(self._n):
            s = scores[bins == b]
            if len(s) >= self.min_per_bin:
                lo[b] = float(np.quantile(s, self.alpha))
                hi[b] = float(np.quantile(s, 1.0 - self.alpha))
        return lo, hi

    # ------------------------------------------------------------------- fit
    def fit(self, daily: pd.DataFrame, episodes: pd.DataFrame) -> ConformalRUL:
        eps = episodes.reset_index(drop=True)

        if self.mode == "mondrian":
            # Edges must split the CALIBRATION population, not the whole fleet.
            # Volatility inside a degradation window is uniformly high, so
            # fleet-wide quantiles put every calibration row in the top band and
            # the model silently collapses back to a single global ratio.
            t_full = self._threshold_from(daily, eps)
            vals = []
            for _, e in eps.iterrows():
                w = daily[(daily.asset_id == e.asset_id)
                          & (daily.day >= pd.Timestamp(e.onset_ts).normalize())
                          & (daily.day < pd.Timestamp(e.fault_ts).normalize())]
                if not len(w) or not np.isfinite(t_full):
                    continue
                hat = _project(w[HI].to_numpy(float), w[SLOPE].to_numpy(float),
                               t_full, self.min_slope, self.horizon,
                               form=self.projection)
                v = pd.to_numeric(w.get(self.difficulty), errors="coerce").to_numpy(float)
                vals.append(v[np.isfinite(hat) & (hat > 0) & np.isfinite(v)])
            pool = np.concatenate(vals) if vals else np.array([])
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_ = np.quantile(pool, qs) if len(pool) else np.array([])

        per_ep: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for i, e in eps.iterrows():
            t_fold = self._threshold_from(daily, eps.drop(i))
            per_ep[i] = (self._episode_scores(daily, e, t_fold)
                         if np.isfinite(t_fold) else (np.array([]), np.array([], dtype=int)))

        # Honest coverage: bound each episode using ONLY the other episodes.
        # Measuring against a quantile of the pooled scores would return
        # 1 - alpha by construction - a tautology, not evidence.
        covered = total = 0
        for i, (own_s, own_b) in per_ep.items():
            cal = [per_ep[j] for j in per_ep if j != i and len(per_ep[j][0])]
            if not len(own_s) or not cal:
                continue
            cs = np.concatenate([c[0] for c in cal])
            cb = np.concatenate([c[1] for c in cal])
            if len(cs) < 10:
                continue
            lo_i, _ = self._quantiles(cs, cb)
            covered += int((own_s >= lo_i[own_b]).sum())
            total += len(own_s)
        self.coverage_ = covered / total if total else np.nan

        non_empty = [v for v in per_ep.values() if len(v[0])]
        if not non_empty:
            self.n_calib_ = 0
            self.lo_ = np.ones(self._n) if self.mode != "additive" else np.zeros(self._n)
            self.hi_ = self.lo_.copy()
            self.threshold_ = self._threshold_from(daily, eps)
            return self

        scores = np.concatenate([v[0] for v in non_empty])
        bins = np.concatenate([v[1] for v in non_empty])
        self.n_calib_ = int(len(scores))
        self.threshold_ = self._threshold_from(daily, eps)
        self.lo_, self.hi_ = self._quantiles(scores, bins)
        return self

    # --------------------------------------------------------------- predict
    def predict(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Add rul_point, rul_lower, rul_upper (days).

        rul_lower is the only one the recommendation is allowed to use.
        """
        if not np.isfinite(self.threshold_):
            raise RuntimeError("ConformalRUL.fit() must be called before predict()")

        point = _project(daily[HI].to_numpy(float), daily[SLOPE].to_numpy(float),
                         self.threshold_, self.min_slope, self.horizon,
                         form=self.projection)
        bins = self._bins(daily)
        lower = self._apply(point, self.lo_, bins)
        upper = self._apply(point, self.hi_, bins)

        finite = np.isfinite(point)
        lower = np.where(finite, np.clip(lower, 0.0, self.horizon), np.inf)
        upper = np.where(finite, np.clip(upper, 0.0, self.horizon), np.inf)
        return daily.assign(rul_point=point, rul_lower=lower, rul_upper=upper)

    def fit_predict(self, daily: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
        return self.fit(daily, episodes).predict(daily)

    def report(self) -> str:
        unit = "d" if self.mode == "additive" else "x"
        band = ", ".join(
            f"{'steady' if b == 0 else 'jumpy' if b == self._n - 1 else 'mid'}"
            f" {unit}{self.lo_[b]:.3f}" for b in range(self._n)
        )
        return (f"ConformalRUL(mode={self.mode}, alpha={self.alpha})\n"
                f"  failure threshold   {self.threshold_:.1f} sigma\n"
                f"  calibration points  {self.n_calib_:,} (leave-one-episode-out)\n"
                f"  lower correction    {band}\n"
                f"  empirical coverage  {self.coverage_:.1%}")
