# Headway

Headway turns train telemetry into reviewable maintenance decisions. This repository
currently contains a **synthetic door-telemetry prototype**, not a validated rail
safety system. Bogie columns are defined in the contract but bogie performance has
not been demonstrated.

The pipeline normalises operating context, compares signals against a historical
reference, builds a directional health index and a separate anomaly distance,
projects remaining useful life, and compares proposed maintenance windows against
an empirical lower RUL bound.

## Current behaviour

- Missing, stale, incomplete and unprojectable observations remain explicit states.
  They do not become infinite remaining life or zero failure risk.
- No worsening trend is a monitoring observation, not a claim that failure is impossible.
- A flat but elevated signal requires review.
- Daily aggregates become available at midnight after the telemetry day.
  Slopes and cooldowns use elapsed calendar time.
- Preprocessing fits an explicit historical reference. Imputation values are frozen.
  Missing health channels produce no combined health estimate.
- The ledger uses the exact bound on the aspect card. It displays window compatibility,
  not failure probabilities, risk budgets or guaranteed safe dates.
- Decision stability means point/bound action agreement. It is not confidence of correctness.
- The four maintenance thresholds are illustrative and must be validated with an operator.
  Missing evidence never clears an existing escalation.
- The adapter requires reviewed source units and identity joins. An amperesecond
  integral is charge, not energy. A successful schema check does not validate model applicability.

## Build and verify (PowerShell)

From this directory, using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe -B scripts/build_health.py
.\.venv\Scripts\python.exe -B scripts/build_rul.py
.\.venv\Scripts\python.exe -B scripts/build_aspects.py
.\.venv\Scripts\python.exe -B scripts/build_deferral.py
.\.venv\Scripts\python.exe -B scripts/build_ui.py
.\.venv\Scripts\python.exe -B scripts/validate_pipeline.py
.\.venv\Scripts\python.exe -B scripts/run_tournament.py
```

Open `ui/headway.html`. It is labelled a retrospective synthetic demonstration:
the demo RUL model is fitted on all available demo fault labels. Scrubbing earlier
dates does not turn those estimates into historical forecasts.

## Evaluation

`data/validation_report.json` is the independent evaluation output:

1. Chronological evaluation freezes training before the final 30% of the timeline.
   Only faults confirmed by the cutoff may train the RUL model.
2. End-to-end train holdouts exclude the entire held-out train from preprocessing,
   calibration and inner model selection. These are a separate generalisation check.
3. Report episode/group counts, abstention, lower-bound coverage and point error
   together. Coverage is only assessed on labelled degradation windows; it is not
   an unconditional guarantee for the whole fleet.

The detector tournament reports chronological alert replay with a fixed training
threshold, daily capacity and cooldown. Retrospective top-k ranking remains an
explicitly labelled diagnostic. Capture uses onset-to-fault time. The simulator's
signal-to-noise benchmark is not a physical detection limit.

Conformal-style rank corrections do not remove correlation between episode-days.
The residual bounds are empirical, and their nominal level does not imply an
individual door's conditional failure probability.

## Remaining evidence needed

Real telemetry with verified units, maintenance history, failure/inspection semantics,
censored follow-up, representative operating conditions and enough independent
fault episodes. Survival models and operator-specific scheduling should be evaluated
only after those inputs are available. Synthetic seed sweeps are not external validation.

Original files from the September audit are preserved in `.audit-backup/2026-09-10`.
Legacy exploratory scripts (`diagnose.py`, `holdout_check.py`, `check_premise.py`)
are not the authority for performance or submission claims.
