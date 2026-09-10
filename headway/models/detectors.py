"""
Headway - the detector zoo for the Proving Ground.

PS3 asks two things: build a tool, and "identify the best models to be used to
detect future anomalies". This module is the second half. Every detector here
implements the same tiny interface and is judged on the same operational
metrics, at the same alert budget, under the same protocol.

THE PROTOCOL, AND WHY IT IS STRICTER THAN IT LOOKS
--------------------------------------------------
Every detector is fitted ONLY on a reference window at the start of the record
and then scores the entire history forward. No detector sees a label, and none
sees data from beyond its fitting window at fit time. This matters because the
headline metric is warning lead time: a model fitted on the whole record has
already seen the failure it is being congratulated for predicting, and every
lead-time number it produces is fiction.

Scores are also smoothed identically (a common trailing rolling median) so the
comparison measures the DETECTOR, not who happened to smooth harder.

Convention: higher score = worse health, always.

WHAT THE ZOO IS FOR
-------------------
The interesting question is not "which library wins". It is whether the model
family matters more than what you feed it. So the same families appear twice,
once on raw per-cycle aggregates and once on condition-normalised, per-asset
baselined features - which isolates the contribution of the preprocessing from
the contribution of the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class Detector:
    """Fit on a reference window; score every row. Higher means worse.

    Deliberately not a dataclass, and it declares no `needs` default: a mutable
    class attribute here is inherited as a dataclass field default by every
    subclass, which Python rejects outright. Subclasses that consume a feature
    matrix declare `needs` themselves.
    """

    name: str = "detector"

    def fit(self, reference: pd.DataFrame) -> "Detector":
        raise NotImplementedError

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # -- shared plumbing ----------------------------------------------------
    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.needs].to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError("missing detector inputs require abstention; use run() to handle incomplete rows")
        return X


# --------------------------------------------------------------------------
# Baselines an engineer would recognise
# --------------------------------------------------------------------------

@dataclass
class RawThreshold(Detector):
    """The naive detector: rank the fleet on the raw signal, as measured.

    This is what most teams will build, and it is the thing to beat. It has no
    idea that doors differ from each other or that afternoons are hot.
    """

    column: str
    name: str = "raw signal (fleet-wide)"

    def fit(self, reference: pd.DataFrame) -> "RawThreshold":
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        return daily[self.column].to_numpy(dtype=float)


@dataclass
class EWMAChart(Detector):
    """Exponentially weighted moving average control chart, per asset.

    The genuine industrial incumbent - decades of SPC practice, no ML at all.
    Including it keeps us honest: if a control chart matches the models, the
    models are not earning their complexity.
    """

    column: str
    lam: float = 0.15
    name: str = "EWMA control chart"
    centre_: pd.Series | None = field(default=None, init=False)
    sigma_: pd.Series | None = field(default=None, init=False)

    def fit(self, reference: pd.DataFrame) -> "EWMAChart":
        g = reference.groupby("asset_id")[self.column]
        self.centre_ = g.median()
        s = g.std()
        self.sigma_ = s.where(s > 1e-9, float(s.median() or 1.0))
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        # An asset absent from the reference window - a train that entered
        # service after fitting - must fall back to the fleet, not to NaN. A
        # NaN score drops silently out of the ranking, so the new train would
        # never be alerted on at all.
        fleet_centre = float(np.nanmedian(self.centre_)) if len(self.centre_) else 0.0
        fleet_sigma = float(np.nanmedian(self.sigma_)) if len(self.sigma_) else 1.0
        fleet_sigma = fleet_sigma if fleet_sigma > 1e-9 else 1.0

        centre = daily.asset_id.map(self.centre_).astype(float).fillna(fleet_centre)
        sigma = daily.asset_id.map(self.sigma_).astype(float).fillna(fleet_sigma)
        sigma = sigma.where(sigma > 1e-9, fleet_sigma)
        z = (daily[self.column].astype(float) - centre) / sigma
        ewma = (z.groupby(daily.asset_id, sort=False)
                 .transform(lambda s: s.ewm(alpha=self.lam, adjust=False).mean()))
        return ewma.to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Unsupervised anomaly detectors
# --------------------------------------------------------------------------

@dataclass
class IsolationForestDetector(Detector):
    """Isolation Forest - the default reach for tabular anomaly detection."""

    needs: list[str]
    name: str = "isolation forest"
    n_estimators: int = 300
    seed: int = 0
    scaler_: StandardScaler | None = field(default=None, init=False)
    model_: IsolationForest | None = field(default=None, init=False)

    def fit(self, reference: pd.DataFrame) -> "IsolationForestDetector":
        self.scaler_ = StandardScaler().fit(self._matrix(reference))
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, contamination="auto"
        ).fit(self.scaler_.transform(self._matrix(reference)))
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        X = self.scaler_.transform(self._matrix(daily))
        return -self.model_.score_samples(X)      # higher = more anomalous


@dataclass
class PCAReconstruction(Detector):
    """Reconstruction error against a subspace fitted on healthy operation.

    The linear stand-in for an autoencoder. On a few thousand rows it is the
    honest choice - a deep autoencoder here would be fitting noise with more
    parameters and a longer training loop, not learning more.
    """

    needs: list[str]
    n_components: int = 3
    name: str = "PCA reconstruction error"
    scaler_: StandardScaler | None = field(default=None, init=False)
    model_: PCA | None = field(default=None, init=False)

    def fit(self, reference: pd.DataFrame) -> "PCAReconstruction":
        X = self._matrix(reference)
        self.scaler_ = StandardScaler().fit(X)
        k = min(self.n_components, X.shape[1])
        self.model_ = PCA(n_components=k, random_state=0).fit(self.scaler_.transform(X))
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        Z = self.scaler_.transform(self._matrix(daily))
        return np.sqrt(((Z - self.model_.inverse_transform(self.model_.transform(Z))) ** 2).sum(axis=1))


@dataclass
class MahalanobisDetector(Detector):
    """Distance from the healthy operating envelope, accounting for covariance.

    Cheap, deterministic, and it takes correlations between signals seriously -
    which the raw threshold does not.
    """

    needs: list[str]
    name: str = "Mahalanobis distance"
    scaler_: StandardScaler | None = field(default=None, init=False)
    model_: EmpiricalCovariance | None = field(default=None, init=False)

    def fit(self, reference: pd.DataFrame) -> "MahalanobisDetector":
        X = self._matrix(reference)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = EmpiricalCovariance().fit(self.scaler_.transform(X))
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        return self.model_.mahalanobis(self.scaler_.transform(self._matrix(daily)))


@dataclass
class LOFDetector(Detector):
    """Local Outlier Factor in novelty mode - density rather than distance."""

    needs: list[str]
    n_neighbors: int = 35
    name: str = "local outlier factor"
    scaler_: StandardScaler | None = field(default=None, init=False)
    model_: LocalOutlierFactor | None = field(default=None, init=False)

    def fit(self, reference: pd.DataFrame) -> "LOFDetector":
        X = self._matrix(reference)
        self.scaler_ = StandardScaler().fit(X)
        self.model_ = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, max(len(X) - 1, 2)), novelty=True
        ).fit(self.scaler_.transform(X))
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        return -self.model_.score_samples(self.scaler_.transform(self._matrix(daily)))


@dataclass
class Passthrough(Detector):
    """Score a column that has already been computed - the Headway pipeline."""

    column: str
    name: str = "headway"

    def fit(self, reference: pd.DataFrame) -> "Passthrough":
        return self

    def score(self, daily: pd.DataFrame) -> np.ndarray:
        return daily[self.column].to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(
    detectors: dict[str, Detector],
    daily: pd.DataFrame,
    reference_days: int = 30,
    smooth_days: int = 5,
) -> pd.DataFrame:
    """Fit every detector on the reference window and score the full history.

    Returns `daily` with one score column per detector, all smoothed the same
    way so the comparison is about the detector rather than the smoothing.
    """
    out = daily.sort_values(["asset_id", "day"]).copy()
    start = out.day.min()
    reference = out[out.day < start + pd.Timedelta(days=reference_days)]

    for key, det in detectors.items():
        needs = det.needs if hasattr(det, "needs") else [det.column]
        complete = np.isfinite(out[needs].to_numpy(float)).all(axis=1)
        ref_complete = np.isfinite(reference[needs].to_numpy(float)).all(axis=1)
        if ref_complete.sum() < 5:
            out[key] = np.nan
            continue
        det.fit(reference[ref_complete])
        raw = pd.Series(np.nan,index=out.index,dtype=float)
        if complete.any():
            raw.loc[out.index[complete]] = det.score(out[complete])
        out[key] = np.nan
        for _,g in out.groupby("asset_id",sort=False):
            signal = pd.Series(raw.loc[g.index].to_numpy(float),index=pd.DatetimeIndex(g.day))
            smooth = signal.rolling(f"{smooth_days}D",min_periods=min(3,smooth_days)).median().where(signal.notna())
            out.loc[g.index,key] = smooth.to_numpy()
        if "data_quality_ok" in out:
            out.loc[~out.data_quality_ok.fillna(False),key] = np.nan
        if "context_supported" in out:
            out.loc[~out.context_supported.fillna(False),key] = np.nan
    return out.reindex(daily.index)
