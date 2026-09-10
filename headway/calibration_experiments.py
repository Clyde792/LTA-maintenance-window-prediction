"""Experimental equal-group residual calibration, not a conformal guarantee.

Every training train contributes equal total weight to the empirical residual
distribution. Long observed degradation trajectories cannot dominate merely by
providing more daily rows. Kept separate from the production default.
"""
import numpy as np
from .rul import ConformalRUL


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if len(values) == 0 or values.shape != weights.shape or not 0 <= q <= 1:
        raise ValueError("invalid weighted quantile inputs")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("finite values and positive weights required")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / weights.sum()
    i = min(np.searchsorted(cumulative, q, side="left"), len(values) - 1)
    return float(values[order[i]])


class GroupBalancedRUL(ConformalRUL):
    def _fit_core(self, daily, episodes):
        if self.mode != "additive":
            raise ValueError("group-balanced experiment supports additive residuals only")
        super()._fit_core(daily, episodes)
        eps = episodes.reset_index(drop=True)
        groups = self._groups(daily, eps)
        values, weights = [], []
        for group in groups.unique():
            threshold = self._threshold_from(daily, eps[groups != group])
            if not np.isfinite(threshold):
                continue
            pieces = [self._episode_scores(daily, e, threshold)[0]
                      for _, e in eps[groups == group].iterrows()]
            s = np.concatenate(pieces) if pieces else np.array([])
            s = s[np.isfinite(s)]
            if len(s):
                values.extend(s)
                weights.extend(np.full(len(s), 1 / len(s)))
        if values:
            self.lo_ = np.array([weighted_quantile(values, weights, self.alpha)])
            self.hi_ = np.array([weighted_quantile(values, weights, 1 - self.alpha)])
        return self
