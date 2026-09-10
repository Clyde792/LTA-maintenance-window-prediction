"""Training-frozen operating-condition normalisation and health indices.

A standardised index measures departure from a reference population. Equal
indices across fleets or subsystems do not establish equal failure risk.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

CYCLIC = {"hour_of_day": 24.}
MAD_TO_SIGMA = 1.4826

def _design(df, cols, fills):
    parts, names = [], []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        x = np.where(np.isfinite(x), x, fills[c])
        if c in CYCLIC:
            for k in (1, 2):
                parts.extend([np.sin(2*np.pi*k*x/CYCLIC[c]), np.cos(2*np.pi*k*x/CYCLIC[c])])
                names.extend([f"{c}_sin{k}", f"{c}_cos{k}"])
        else:
            parts.extend([x, x*x])
            names.extend([c, f"{c}^2"])
    return np.column_stack(parts), names

@dataclass
class ConditionNormaliser:
    signal: str
    context: list[str]
    alpha: float = 1.
    robust: bool = True
    trim: float = .01
    model_: Ridge | None = field(default=None, init=False)
    names_: list = field(default_factory=list, init=False)
    mu_: np.ndarray | None = field(default=None, init=False)
    sd_: np.ndarray | None = field(default=None, init=False)
    fills_: dict = field(default_factory=dict, init=False)
    ranges_: dict = field(default_factory=dict, init=False)

    def fit(self, df):
        if df.empty or not self.context:
            raise ValueError("non-empty training data and context are required")
        self.fills_, self.ranges_ = {}, {}
        for c in self.context:
            x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if not x.notna().any():
                raise ValueError(f"no training observations for context {c}")
            self.fills_[c] = float(x.median())
            self.ranges_[c] = (float(x.min()), float(x.max()))
        X, self.names_ = _design(df, self.context, self.fills_)
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_ = np.where(self.sd_ < 1e-12, 1., self.sd_)
        X = (X-self.mu_)/self.sd_
        y = pd.to_numeric(df[self.signal], errors="coerce").to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() < 5:
            raise ValueError("insufficient signal observations for normalisation")
        self.model_ = Ridge(alpha=self.alpha).fit(X[ok], y[ok])
        if self.robust and ok.sum() > 100:
            residual = np.abs(y[ok]-self.model_.predict(X[ok]))
            keep = residual <= np.quantile(residual, 1-self.trim)
            self.model_ = Ridge(alpha=self.alpha).fit(X[ok][keep], y[ok][keep])
        return self

    def transform(self, df):
        if self.model_ is None:
            raise RuntimeError("ConditionNormaliser.fit() must be called first")
        X, names = _design(df, self.context, self.fills_)
        if names != self.names_:
            raise ValueError("context layout changed between fit and transform")
        expected = self.model_.predict((X-self.mu_)/self.sd_)
        y = pd.to_numeric(df[self.signal], errors="coerce").to_numpy(float)
        missing, supported = np.zeros(len(df), bool), np.ones(len(df), bool)
        for c in self.context:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            absent = ~np.isfinite(x)
            missing |= absent
            lo, hi = self.ranges_[c]
            # Modest range padding is a diagnostic policy, not a drift guarantee.
            pad = max((hi-lo)*.1, 1e-9)
            supported &= ~absent & (x >= lo-pad) & (x <= hi+pad)
        return df.assign(expected=expected, residual=np.where(np.isfinite(y), y-expected, np.nan),
            context_missing=missing, context_supported=supported)

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def explain(self):
        if self.model_ is None:
            raise RuntimeError("fit() first")
        return pd.DataFrame({"term":self.names_, "coef":self.model_.coef_}).assign(
            abs_coef=lambda d:d.coef.abs()).sort_values("abs_coef",ascending=False).drop(columns="abs_coef")

@dataclass
class AssetBaseline:
    value: str = "residual"
    by: str = "asset_id"
    reference_days: int = 21
    min_periods: int = 5
    loc_: pd.Series | None = field(default=None, init=False)
    scale_: pd.Series | None = field(default=None, init=False)
    fleet_loc_: float = field(default=0., init=False)
    fleet_scale_: float = field(default=1., init=False)

    def fit(self, df, ts="day"):
        first = df.groupby(self.by)[ts].transform("min")
        ref = df[df[ts] < first+pd.Timedelta(days=self.reference_days)].copy()
        ref[self.value] = pd.to_numeric(ref[self.value],errors="coerce").replace([np.inf,-np.inf],np.nan)
        if ref[self.value].notna().sum() < self.min_periods:
            raise ValueError("insufficient reference observations for baseline")
        g = ref.groupby(self.by)[self.value]
        loc = g.median()
        scale = g.apply(lambda s:MAD_TO_SIGMA*(s-s.median()).abs().median())
        counts = g.count()
        self.fleet_loc_ = float(loc.median())
        self.fleet_scale_ = float(scale[scale>1e-9].median()) if (scale>1e-9).any() else 1.
        self.loc_ = loc.where(counts>=self.min_periods,self.fleet_loc_)
        self.scale_ = scale.where((scale>1e-9)&(counts>=self.min_periods),self.fleet_scale_)
        return self

    def transform(self, df):
        if self.loc_ is None:
            raise RuntimeError("AssetBaseline.fit() must be called first")
        known = df[self.by].isin(self.loc_.index)
        loc = df[self.by].map(self.loc_).fillna(self.fleet_loc_).to_numpy(float)
        scale = df[self.by].map(self.scale_).fillna(self.fleet_scale_).to_numpy(float)
        y = pd.to_numeric(df[self.value],errors="coerce").to_numpy(float)
        return df.assign(health_index=np.where(np.isfinite(y),(y-loc)/scale,np.nan),
                         baseline_source=np.where(known,"asset_reference","fleet_fallback"))

    def fit_transform(self, df, ts="day"):
        return self.fit(df,ts).transform(df)

@dataclass
class MultivariateHealthIndex:
    signals: list[str]
    orientation: dict = field(default_factory=dict)
    reference_days: int = 21
    ridge: float = 1e-6
    mode: str = "projection"
    mean_: np.ndarray | None = field(default=None, init=False)
    whiten_: np.ndarray | None = field(default=None, init=False)
    weights_: np.ndarray | None = field(default=None, init=False)
    scale_: float = field(default=1., init=False)

    def _matrix(self, df):
        X = np.column_stack([self.orientation.get(s,1)*pd.to_numeric(df[s],errors="coerce").to_numpy(float)
                             for s in self.signals])
        return np.where(np.isfinite(X),X,np.nan)

    def fit(self, daily, ts="day"):
        ref = daily[daily[ts] < daily[ts].min()+pd.Timedelta(days=self.reference_days)]
        X = self._matrix(ref)
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) <= max(5,len(self.signals)*5):
            raise ValueError("insufficient complete reference rows; future data cannot replace the reference")
        self.mean_ = X.mean(0)
        cov = np.atleast_2d(np.cov(X-self.mean_,rowvar=False))+self.ridge*np.eye(len(self.signals))
        vals, vecs = np.linalg.eigh(cov)
        self.whiten_ = vecs @ np.diag(1/np.sqrt(np.maximum(vals,self.ridge))) @ vecs.T
        # Clipping the effective weights prevents one worsening channel from
        # decreasing the directional score. Distance still covers other directions.
        self.weights_ = np.maximum(self.whiten_.sum(axis=1),0)
        if not self.weights_.any():
            self.weights_ = np.ones(len(self.signals))
        self.scale_ = max(float(((X-self.mean_)@self.weights_).std()),1e-9)
        return self

    def transform(self, daily):
        if self.whiten_ is None:
            raise RuntimeError("MultivariateHealthIndex.fit() must be called first")
        X = self._matrix(daily)-self.mean_
        z = X@self.whiten_
        distance = np.sqrt((z*z).sum(axis=1))
        contribution = X*self.weights_/self.scale_
        idx = distance if self.mode == "mahalanobis" else contribution.sum(axis=1)
        out = daily.assign(health_index=idx, anomaly_distance=distance,
                           health_inputs_complete=np.isfinite(X).all(axis=1))
        for j,s in enumerate(self.signals):
            out[f"contribution_{s}"] = contribution[:,j]
        return out

    def fit_transform(self, daily, ts="day"):
        return self.fit(daily,ts).transform(daily)
