"""RUL projection with empirical residual bounds.

Bounds describe historical group coverage, not individual failure probabilities.
Cross-fitting reduces label reuse; correlated trajectories and cross-fitted
residuals do not have the standard split-conformal finite-sample guarantee.
Use an independent chronological evaluation for performance claims.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd

HI = "health_index_smooth"
SLOPE = "health_index_slope"

def prediction_time(df):
    """Full-day aggregates are only available after their day closes."""
    return pd.to_datetime(df["available_at"] if "available_at" in df else df["day"])

def _project(hi, slope, threshold, min_slope, horizon, form="loglinear"):
    if form not in ("linear", "loglinear"):
        raise ValueError("projection must be linear or loglinear")
    hi, slope = np.asarray(hi, float), np.asarray(slope, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        linear = (threshold - hi) / slope
        value = np.where(hi >= 1, np.log(threshold / hi) / (slope / hi), linear) if form == "loglinear" else linear
    # A non-growing trend provides no countdown; it does not imply infinite life.
    value = np.where(np.isfinite(hi) & np.isfinite(slope) & (slope > min_slope), value, np.nan)
    value = np.where(np.isfinite(hi) & (hi >= threshold), 0., value)
    return np.where(np.isfinite(value), np.clip(value, 0., horizon), np.nan)

@dataclass
class ConformalRUL:
    alpha: float = .10
    projection: str = "loglinear"
    mode: str = "additive"
    n_bins: int = 3
    difficulty: str = "health_index_vol"
    min_per_bin: int = 20
    min_slope: float = .25
    horizon: float = 90.
    elevated_hi: float = 3.
    threshold_: float = field(default=np.nan, init=False)
    edges_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    lo_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    hi_: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    coverage_: float = field(default=np.nan, init=False)
    n_calib_: int = field(default=0, init=False)
    n_groups_: int = field(default=0, init=False)
    n_eval_groups_: int = field(default=0, init=False)
    fitted_: bool = field(default=False, init=False)

    @property
    def _n(self):
        return self.n_bins if self.mode == "mondrian" else 1

    def _bins(self, df):
        if self.mode != "mondrian" or not len(self.edges_):
            return np.zeros(len(df), dtype=int)
        v = pd.to_numeric(df[self.difficulty], errors="coerce").to_numpy(float)
        return np.clip(np.digitize(v, self.edges_), 0, self._n - 1)

    def _score(self, true, hat):
        return true - hat if self.mode == "additive" else true / hat

    def _apply(self, hat, q, bins):
        return hat + q[bins] if self.mode == "additive" else hat * q[bins]

    @staticmethod
    def _threshold_from(daily, episodes):
        vals = []
        when = prediction_time(daily)
        for e in episodes.itertuples():
            # Exclude aggregates containing observations after the fault.
            a = daily[(daily.asset_id == e.asset_id) & (when <= e.fault_ts)].sort_values("day")
            a = a[np.isfinite(a[HI])]
            if len(a):
                vals.append(float(a[HI].iloc[-1]))
        return float(np.median(vals)) if vals else np.nan

    @staticmethod
    def _groups(daily, episodes):
        if "train_id" in daily:
            mapping = daily.groupby("asset_id").train_id.first()
            return episodes.asset_id.map(mapping).fillna(episodes.asset_id).astype(str)
        return episodes.asset_id.astype(str)

    def _episode_scores(self, daily, e, threshold):
        when = prediction_time(daily)
        w = daily[(daily.asset_id == e.asset_id) & (when >= e.onset_ts) & (when < e.fault_ts)]
        if not len(w):
            return np.array([]), np.array([], dtype=int)
        hat = _project(w[HI].to_numpy(float), w[SLOPE].to_numpy(float),
                       threshold, self.min_slope, self.horizon, self.projection)
        true = (pd.Timestamp(e.fault_ts) - prediction_time(w)).dt.total_seconds().to_numpy() / 86400
        ok = np.isfinite(hat) & np.isfinite(true)
        if self.mode != "additive":
            ok &= hat > 0
        if self.mode == "mondrian":
            ok &= np.isfinite(w[self.difficulty].to_numpy(float))
        return self._score(true[ok], hat[ok]), self._bins(w)[ok]

    def _quantiles(self, scores, bins):
        # Rank correction, without claiming independence of episode-days.
        def ranks(s):
            s = np.sort(s[np.isfinite(s)])
            n = len(s)
            k = int(np.floor((n + 1) * self.alpha))
            j = int(np.ceil((n + 1) * (1 - self.alpha)))
            return (s[k - 1] if k >= 1 else -np.inf,
                    s[j - 1] if j <= n else np.inf)
        l, h = ranks(scores)
        lo, hi = np.full(self._n, l), np.full(self._n, h)
        for b in range(self._n):
            s = scores[bins == b]
            if len(s) >= self.min_per_bin:
                lo[b], hi[b] = ranks(s)
        return lo, hi

    def _fit_core(self, daily, episodes):
        if not 0 < self.alpha < .5 + 1e-12 or self.mode not in ("additive", "ratio", "mondrian"):
            raise ValueError("invalid alpha or conformity mode")
        self.fitted_ = True
        eps = episodes.reset_index(drop=True)
        groups = self._groups(daily, eps)
        self.n_groups_ = groups.nunique()
        self.threshold_ = self._threshold_from(daily, eps)
        self.edges_ = np.array([])
        if self.mode == "mondrian" and self.difficulty in daily:
            # All edge fitting happens inside this training fold.
            v = daily[self.difficulty].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                self.edges_ = np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1])
        pieces = []
        for group in groups.unique():
            train_eps = eps[groups != group]
            held_eps = eps[groups == group]
            threshold = self._threshold_from(daily, train_eps)
            if not np.isfinite(threshold):
                continue
            for _, e in held_eps.iterrows():
                s, b = self._episode_scores(daily, e, threshold)
                if len(s):
                    pieces.append((s, b))
        self.n_calib_ = sum(len(s) for s, _ in pieces)
        self.lo_, self.hi_ = np.full(self._n, np.nan), np.full(self._n, np.nan)
        if self.n_calib_:
            self.lo_, self.hi_ = self._quantiles(
                np.concatenate([s for s, _ in pieces]), np.concatenate([b for _, b in pieces]))
        return self

    def out_of_fold_predict(self, daily, episodes):
        """Outer train/asset holdout. Preprocessing must also be fitted separately
        by the caller; these predictions only validate the RUL stage."""
        eps = episodes.reset_index(drop=True)
        groups = self._groups(daily, eps)
        outputs = []
        for group in groups.unique():
            assets = eps.loc[groups == group, "asset_id"]
            held = daily.train_id.astype(str).eq(group) if "train_id" in daily else daily.asset_id.isin(assets)
            model = replace(self)._fit_core(daily[~held], eps[groups != group])
            pred = model.predict(daily[held])
            pred["evaluation_group"] = group
            outputs.append(pred)
        return pd.concat(outputs).sort_index() if outputs else pd.DataFrame()

    def fit(self, daily, episodes, *, evaluate=True):
        self._fit_core(daily, episodes)
        self.coverage_, self.n_eval_groups_ = np.nan, 0
        if evaluate:
            oof = self.out_of_fold_predict(daily, episodes)
            by_group = {}
            for e in episodes.itertuples():
                if oof.empty:
                    break
                t = prediction_time(oof)
                w = oof[(oof.asset_id == e.asset_id) & (t >= e.onset_ts) & (t < e.fault_ts)]
                truth = (pd.Timestamp(e.fault_ts) - prediction_time(w)).dt.total_seconds() / 86400
                ok = np.isfinite(w.rul_lower)
                if ok.any():
                    key = str(w.evaluation_group.iloc[0])
                    by_group.setdefault(key, []).extend((w.rul_lower[ok] <= truth[ok]).tolist())
            self.n_eval_groups_ = len(by_group)
            if by_group:
                self.coverage_ = float(np.mean([np.mean(v) for v in by_group.values()]))
        return self

    def predict(self, daily, *, as_of=None, stale_after_hours=24.):
        if not self.fitted_:
            raise RuntimeError("ConformalRUL.fit() must be called before predict()")
        hi, slope = daily[HI].to_numpy(float), daily[SLOPE].to_numpy(float)
        state = np.full(len(daily), "insufficient_history", dtype=object)
        known = np.isfinite(hi) & np.isfinite(slope)
        state[known] = "valid"
        quiet = known & (slope <= self.min_slope)
        state[quiet & (hi < self.elevated_hi)] = "no_worsening_trend"
        state[quiet & (hi >= self.elevated_hi)] = "elevated_no_trend"
        point = _project(hi, slope, self.threshold_, self.min_slope, self.horizon, self.projection)
        if not np.isfinite(self.threshold_) or self.n_calib_ == 0:
            state[known] = "uncalibrated"
        crossed = np.isfinite(hi) & np.isfinite(self.threshold_) & (hi >= self.threshold_)
        state[crossed] = "threshold_exceeded"
        # The projection is capped; a finite cap is not evidence of a long life.
        state[(state == "valid") & (point >= self.horizon)] = "outside_horizon"
        if self.mode == "mondrian":
            state[~np.isfinite(daily[self.difficulty].to_numpy(float))] = "insufficient_history"
        for col, label in (("data_quality_ok", "insufficient_data"), ("context_supported", "outside_training_conditions")):
            if col in daily:
                state[~daily[col].fillna(False).to_numpy(bool)] = label
        if "stale_data" in daily:
            state[daily.stale_data.fillna(True).to_numpy(bool)] = "stale_data"
        if as_of is not None:
            if stale_after_hours <= 0:
                raise ValueError("stale_after_hours must be positive")
            now = pd.Timestamp(as_of)
            observed = pd.to_datetime(daily["last_observed_at"]) if "last_observed_at" in daily else prediction_time(daily)
            stale = ((now-observed).dt.total_seconds()/3600 > stale_after_hours) | observed.isna()
            state[stale.to_numpy()] = "stale_data"
            state[(prediction_time(daily)>now).to_numpy()] = "not_yet_available"
        lower, upper = np.full(len(daily), np.nan), np.full(len(daily), np.nan)
        valid = state == "valid"
        if self.n_calib_:
            bins = self._bins(daily)
            lower[valid] = np.maximum(0, self._apply(point, self.lo_, bins)[valid])
            upper[valid] = np.maximum(0, self._apply(point, self.hi_, bins)[valid])
        lower[state == "threshold_exceeded"] = 0.
        # Exceeding the learned threshold does not establish the actual failure time.
        point[~np.isin(state, ["valid", "threshold_exceeded", "outside_horizon"])] = np.nan
        return daily.assign(rul_point=point, rul_lower=lower, rul_upper=upper, prediction_state=state)

    def fit_predict(self, daily, episodes):
        return self.fit(daily, episodes).predict(daily)

    def report(self):
        return (f"RUL {self.projection}/{self.mode}: threshold={self.threshold_:.2f}\n"
                f"  {self.n_calib_} correlated calibration rows; {self.n_groups_} train/asset groups\n"
                f"  outer-group empirical coverage={self.coverage_:.1%} ({self.n_eval_groups_} evaluated groups)\n"
                "  RUL-stage diagnostic only; no individual failure probability or coverage guarantee.")
