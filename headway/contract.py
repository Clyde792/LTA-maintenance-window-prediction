"""
Headway - the Feature Contract.

This is the keystone of the system. Everything downstream of ingest - condition
normalisation, the model tournament, RUL, the Aspect Card, the Possession Planner
- speaks ONLY this schema. Nothing downstream ever sees a raw vendor column name.

Consequence, and the entire reason this file exists first: when NEBULA X hands us
the real door/bogie data at 20:00 on Friday 18 Sep, we write exactly one new file
(headway/adapters/nebulax.py) that maps their columns onto these names. If their
schema is not what we guessed, we change that adapter and nothing else. Friday
night is data-binding, not infrastructure.

THE CANONICAL UNIT IS A CYCLE.
  door  : one complete open or close movement
  bogie : one measurement window (e.g. one inter-station run)

Each row is one cycle of one asset. Columns fall into four groups:

  IDENTITY  who and when. Always required.
  SIGNAL    the health-bearing measurements. Subsystem-specific.
  CONTEXT   operating conditions we normalise AGAINST, so that heat, crowding
            and time-of-day cannot masquerade as degradation. Always required.
  LABEL     sparse, delayed, human-confirmed fault records. Mostly null.

The SIGNAL/CONTEXT split is not bookkeeping - it is the modelling thesis.
We never score a raw signal. We model E[signal | context] and score the residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------

IDENTITY: list[str] = [
    "ts",          # datetime64[ns] - cycle start, tz-naive local (SGT)
    "asset_id",    # str - the maintainable unit, e.g. "TRN047-DOOR-3"
    "train_id",    # str - e.g. "TRN047"
    "subsystem",   # str - "door" | "bogie"
]

# Operating conditions. These are the confounds. A model that scores a signal
# without conditioning on these will flag a hot crowded Tuesday as a fault.
CONTEXT: list[str] = [
    "ambient_temp_c",        # saloon/ambient temperature
    "load_proxy",            # crowding proxy in [0,1] (dwell, load weigh, headcount)
    "hour_of_day",           # 0-23, carries diurnal duty patterns
    "cycles_since_service",  # usage since last maintenance touch
]

# Sparse ground truth, from the "manually verified fault data" in the brief.
# Expect tens of positives against hundreds of thousands of cycles.
LABEL: list[str] = [
    "fault_confirmed",  # bool - True only on the cycle where a fault was verified
    "fault_mode",       # str - e.g. "roller_wear"; null when no fault
]


@dataclass(frozen=True)
class Subsystem:
    """Declares the health signals carried by one kind of asset."""

    name: str
    signals: list[str]
    # The signal we treat as the primary degradation axis for RUL. Must be one
    # of `signals`, and must be oriented so that LARGER = WORSE.
    primary: str
    description: str
    # Which way each signal moves as the asset degrades: +1 larger is worse,
    # -1 smaller is worse. Only the exceptions need listing; +1 is assumed.
    #
    # This is load-bearing for the multivariate health index. A worn door draws
    # MORE current and travels LESS far, so summing the two raw directions
    # cancels part of the very degradation we are trying to see. Getting an
    # orientation wrong does not raise - it quietly subtracts evidence.
    orientation: dict[str, int] = field(default_factory=dict)

    def sign(self, col: str) -> int:
        return self.orientation.get(col, 1)

    @property
    def columns(self) -> list[str]:
        return IDENTITY + self.signals + CONTEXT + LABEL


# --------------------------------------------------------------------------
# Subsystem registry
# --------------------------------------------------------------------------
# Signal names follow the door-cycle / rolling-stock condition monitoring
# literature, so the odds of a near-direct column mapping on the night are good.

DOOR = Subsystem(
    name="door",
    signals=[
        "cycle_duration_s",     # time to complete the movement
        "peak_current_a",       # peak motor current
        "mean_current_a",       # mean motor current over the cycle
        "current_integral_as",  # integral of i dt: charge in ampereseconds, not energy.
        "travel_mm",            # leaf travel distance achieved
        "obstruction_flag",     # 0/1 - obstruction detected this cycle
        "retry_count",          # re-open / re-close attempts
    ],
    primary="current_integral_as",
    description="Door Control Unit telemetry, one row per open or close movement.",
    # A worn mechanism binds and under-travels: less distance is worse.
    orientation={"travel_mm": -1},
)

BOGIE = Subsystem(
    name="bogie",
    signals=[
        "bearing_temp_rise_c",  # bearing temp ABOVE ambient - already partly normalised
        "vibration_rms_g",      # broadband vibration energy
        "vibration_kurtosis",   # impulsiveness - classic early bearing defect indicator
        "suspension_defl_mm",   # secondary suspension deflection
        "wheel_impact_g",       # peak wheel/rail impact
    ],
    primary="vibration_rms_g",
    description="Bogie condition monitoring, one row per inter-station run.",
    # All bogie signals rise with deterioration; suspension deflection is the
    # one to revisit against real data, since it can move either way.
    orientation={},
)

SUBSYSTEMS: dict[str, Subsystem] = {s.name: s for s in (DOOR, BOGIE)}


def get(name: str) -> Subsystem:
    """Look up a subsystem, with an error that tells you what actually exists."""
    try:
        return SUBSYSTEMS[name]
    except KeyError:
        raise KeyError(
            f"Unknown subsystem {name!r}. Registered: {sorted(SUBSYSTEMS)}. "
            f"Add a Subsystem() to headway/contract.py to support a new asset type."
        ) from None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass
class ValidationReport:
    subsystem: str
    n_rows: int
    ok: bool
    errors: list[str]
    warnings: list[str]

    def __str__(self) -> str:
        head = "PASS" if self.ok else "FAIL"
        lines = [f"[{head}] contract check - subsystem={self.subsystem} rows={self.n_rows:,}"]
        lines += [f"  ERROR   {e}" for e in self.errors]
        lines += [f"  note    {w}" for w in self.warnings]
        if self.ok and not self.warnings:
            lines.append("  all required columns present, typed and in range")
        return "\n".join(lines)

    def raise_if_bad(self) -> ValidationReport:
        if not self.ok:
            raise ValueError(str(self))
        return self


def validate(df: pd.DataFrame, subsystem: str) -> ValidationReport:
    """Check a frame against the contract.

    Written to be read at 02:00 by someone who is tired: every failure names the
    exact column and what was expected, so a bad adapter mapping is a 30-second
    fix rather than a debugging session.
    """
    sub = get(subsystem)
    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in sub.columns if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")

    extra = [c for c in df.columns if c not in sub.columns]
    if extra:
        warnings.append(f"columns not in contract (ignored downstream): {extra}")

    if "ts" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        errors.append("'ts' must be datetime64 - use pd.to_datetime() in the adapter")

    for col in sub.signals + CONTEXT:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"'{col}' must be numeric, got {df[col].dtype}")

    if sub.primary not in df.columns:
        errors.append(f"primary degradation signal '{sub.primary}' is absent")

    # Range and sanity checks are warnings, not errors: real data is messy and we
    # would rather run on it than refuse it.
    if "load_proxy" in df.columns:
        s = df["load_proxy"].dropna()
        if len(s) and (s.min() < 0 or s.max() > 1):
            warnings.append(
                f"'load_proxy' outside [0,1] (min={s.min():.2f} max={s.max():.2f}) - "
                f"rescale in the adapter so normalisation is comparable across fleets"
            )
    if "hour_of_day" in df.columns:
        # Fractional hours are expected (a cycle at 23:24 is 23.4), so the valid
        # range is the half-open interval [0, 24) - not [0, 23].
        s = df["hour_of_day"].dropna()
        if len(s) and (s.min() < 0 or s.max() >= 24):
            warnings.append(
                f"'hour_of_day' outside [0,24) (min={s.min():.2f} max={s.max():.2f})"
            )

    if len(df) == 0:
        warnings.append("frame is empty - nothing to check beyond column layout")

    for col in sub.signals + CONTEXT:
        if col in df.columns and len(df):
            frac = float(df[col].isna().mean())
            if frac > 0.5:
                warnings.append(f"'{col}' is {frac:.0%} null")

    if "fault_confirmed" in df.columns and len(df):
        n_pos = int(df["fault_confirmed"].fillna(False).astype(bool).sum())
        if n_pos == 0:
            warnings.append("no confirmed faults - evaluation will be unsupervised only")
        else:
            warnings.append(
                f"confirmed faults: {n_pos} of {len(df):,} cycles "
                f"(base rate {n_pos / len(df):.2e}) - report precision at THIS rate, never accuracy"
            )

    return ValidationReport(
        subsystem=subsystem,
        n_rows=len(df),
        ok=not errors,
        errors=errors,
        warnings=warnings,
    )
