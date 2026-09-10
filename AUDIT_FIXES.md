# Headway audit fixes — 10 September 2026

The implementation defects identified in the audit have been corrected, and the
dashboard and evaluation artifacts have been rebuilt. These changes improve the
honesty of the decisions and measurements; they do not establish deployment readiness.

## Changes

- RUL carries explicit states for missing history, incomplete inputs, elevated but
  flat signals, stale observations, future observations, unsupported conditions,
  insufficient calibration and projections outside the supported horizon.
- Missing predictions no longer become infinite life, green assets or zero risk.
  Missing evidence cannot clear an earlier escalation.
- The ledger compares proposed windows with the exact lower margin used by the
  card. Individual failure percentages and “safe until” dates have been removed.
  Zero lower bounds mean no positive margin is established, not certain failure.
  A booked window can be specified as a timestamp and compared in elapsed hours.
- RUL outer folds exclude the entire held-out train/asset and refit threshold,
  calibration and volatility bands. A separate end-to-end evaluation also refits
  preprocessing without the held-out train. Correlated calibration rows remain
  an empirical limitation; rank correction is not a coverage guarantee.
- Full-day aggregates are available the following midnight. Fault-day data is not
  credited before it becomes available. Trend slopes, smoothing, cooldowns and
  hysteresis gaps use calendar time.
- Context imputation is frozen at training. Missing sensor channels remain missing.
  An insufficient reference cannot fall back to future data. Usage since service
  is preserved as a feature instead of being regressed out of wear.
- The multivariate index has non-negative effective weights in the declared wear
  directions, channel contributions and a separate unsigned anomaly distance.
- Adapter source-unit scaling occurs before derived features. Text “false” is not
  interpreted as true. Filled context is flagged for downstream abstention.
- The detector tournament now uses chronological alert replay with training-frozen
  thresholds, daily capacity and cooldown. Retrospective ranking is labelled as
  such. Capture no longer uses an asserted physical detection ceiling.
- Model selection cannot qualify a candidate whose median lower bound is zero.
  If no candidate qualifies, the fixed baseline is explicitly labelled exploratory.
- The light dashboard distinguishes unknown data from monitoring, shows decision
  stability instead of confidence, preserves chart gaps and labels the replay as
  a retrospective synthetic demonstration.

## Verification

- **174 tests passed**, including regression cases for fold isolation, missing
  telemetry, timestamps, calendar gaps, censoring, unit conversion, booleans,
  hysteresis, model selection and dashboard payloads.
- Full artifact rebuild and independent evaluation completed successfully.
- Artifact checks passed for **9,600 asset-days / 80 doors**, verifying ledger/card
  agreement, absence of deprecated probability fields, unknown-state behaviour,
  JSON validity and generated JavaScript syntax.
- Browser visual verification was blocked by the browser tool's local-file URL
  policy. No visual verification is claimed.

## Measured results after correction

Source: `data/validation_report.json`. All results below are synthetic.

| Measure | Future-period evaluation | Entire train held out |
|---|---:|---:|
| Fault groups evaluated | 6 | 9 |
| Labelled degradation rows | 107 | 301 |
| Rows with a finite lower margin | 96 | 182 |
| Abstention on labelled degradation rows | 10.3% | 39.5% |
| Mean absolute point error on scored rows | 3.62 days | 8.87 days |
| Mean of group lower-bound coverage rates | 97.2% | 72.9% |
| Median lower margin on scored rows | 0.00 days | 2.43 days |
| Scored rows with a positive lower margin | 31.3% | 62.6% |

The chronological split trains on faults confirmed by **24 June 2026**, then tests
on later observations. The train-holdout evaluation is a separate protocol, with
inner model selection inside each training fold. It is not simultaneously a
chronological holdout.

The directional detector found all six future-period episodes, with 6.05 days
minimum warning and 87.9% alert precision in that replay. See
`data/tournament_replay.csv` for the other models. This test table does not select
a deployment winner.

**Coverage must be read alongside abstention and positive-margin frequency.** A
zero lower bound covers every positive remaining life and provides no useful
deferral time. The unseen-train result falls substantially short of the nominal
90% lower-bound target; nine simulated fault episodes do not establish dependable
generalisation. No real-world safety rate is claimed.

## Addendum, later on 10 September: fit-for-duty and regenerated data

After the audit the generator gained a **wear-load interaction** (`load_wear_coeff`
in `headway/synth/doors.py`): a worn door is disproportionately worse under crowding.
This is a simulator assumption drawn from operator experience, not a measurement.
It is what `headway/duty.py` detects. The synthetic data was regenerated with the
same seed and every artifact rebuilt, so the table above no longer matches
`data/validation_report.json` exactly. Current figures: chronological 6/6 found,
6.05 d worst warning, 93.5% precision, 3.9 d MAE, 94.0% coverage (median bound
still 0 d); entire-train holdout 74.2% coverage, 9.0 d MAE, 37.5% abstention.
Every headline survived; `SUBMISSION.md` and `TECHNICAL_OVERVIEW.md` carry the
current numbers. Two design decisions were forced by evidence during the build:
the sensitivity window was cut from 7 to 3 days after it left restrictions on
freshly repaired doors, and a margin-based restriction rule was removed after it
fired on sub-day differences between two bounds and flickered. Test count is
now 197.

## What remains

Representative real telemetry, verified failure/inspection semantics, maintenance
and censoring records, and operator-specific action policy are still required.
Unseen-asset onboarding and calibration need further experiments. A conditional
survival model and the persistent operator deferral-decision record are future
extensions, not implemented claims.

Original files and generated artifacts are retained under
`.audit-backup/2026-09-10`. The current dashboard is `ui/headway.html`.
