"""Preprocessing fitted only on an explicit historical reference frame."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from . import contract, features
from .normalise import ConditionNormaliser, AssetBaseline, MultivariateHealthIndex
from . import duty as duty_mod

# Usage since service can carry degradation itself. Preserve it as a feature;
# do not regress it out of wear. A load proxy requires validation on real data.
NORMALISATION_CONTEXT = ["ambient_temp_c","load_proxy","hour_of_day"]

@dataclass
class HealthPipeline:
    subsystem: str = "door"
    normalisers: dict = field(default_factory=dict,init=False)
    baselines: dict = field(default_factory=dict,init=False)
    multivariate: object = field(default=None,init=False)
    fitted_at: object = field(default=None,init=False)
    # Peak service bands and their reference loads, frozen at fit for duty assessment.
    peak_bands: list = field(default_factory=list,init=False)
    load_peak: float = field(default=float("nan"),init=False)
    load_offpeak: float = field(default=float("nan"),init=False)

    @property
    def levels(self):
        return [s for s in contract.get(self.subsystem).signals if not s.endswith(("_flag","_count"))]

    def _residualise(self, cycles):
        c=cycles.copy()
        supported=c.context_supported.fillna(False).to_numpy(bool).copy() if "context_supported" in c else np.ones(len(c),bool)
        for s,m in self.normalisers.items():
            v=m.transform(c)
            c[f"res_{s}"]=v.residual
            supported &= v.context_supported.to_numpy(bool)
        c["context_supported"]=supported
        return c

    def _daily(self, cycles):
        return features.to_daily(self._residualise(cycles),self.subsystem,extra=tuple(f"res_{s}" for s in self.levels))

    def _cycle_index(self, c):
        """The card's index applied per cycle: same baselines, same weights.
        Only used for within-day load contrast; daily decisions still come from
        the daily aggregate."""
        for s,b in self.baselines.items():
            c[f"{s}_hx"]=b.transform(c).health_index
        return self.multivariate.transform(c).health_index

    def fit(self, reference):
        if reference.empty:
            raise ValueError("historical reference data required")
        self.fitted_at=reference.ts.max().floor("D")+pd.Timedelta(days=1)
        self.normalisers={s:ConditionNormaliser(s,NORMALISATION_CONTEXT).fit(reference) for s in self.levels}
        self.peak_bands,self.load_peak,self.load_offpeak=duty_mod.peak_bands(reference)
        d=self._daily(reference)
        self.baselines={}
        for s in self.levels:
            b=AssetBaseline(value=f"res_{s}").fit(d)
            self.baselines[s]=b
            d[f"{s}_hx"]=b.transform(d).health_index
        self.multivariate=MultivariateHealthIndex(
            signals=[f"{s}_hx" for s in self.levels],
            orientation={f"{s}_hx":contract.get(self.subsystem).sign(s) for s in self.levels}).fit(d)
        return self

    def transform(self, cycles):
        if self.multivariate is None:
            raise RuntimeError("HealthPipeline.fit() required")
        c=self._residualise(cycles)
        d=features.to_daily(c,self.subsystem,extra=tuple(f"res_{s}" for s in self.levels))
        c["cycle_index"]=self._cycle_index(c)
        d=d.merge(duty_mod.cycle_stats(c),on=["asset_id","day"],how="left",validate="one_to_one")
        d[list(duty_mod.STATS)]=d[list(duty_mod.STATS)].fillna(0.)
        for s,b in self.baselines.items():
            transformed=b.transform(d)
            d[f"{s}_hx"]=transformed.health_index
            d["baseline_source"]=transformed.baseline_source
        d=self.multivariate.transform(d)
        d["uni_index"]=d[f"{contract.get(self.subsystem).primary}_hx"]
        d=features.add_trend(d)
        d=features.add_trend(d,col="uni_index")
        d["data_quality_ok"] &= d.health_inputs_complete & (d.available_at>=self.fitted_at)
        d["preprocessing_fitted_at"]=self.fitted_at
        return d
