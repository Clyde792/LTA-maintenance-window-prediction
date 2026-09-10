# New-asset onboarding experiment

Completed 10 September 2026. This is simulator evidence, not validation on an
operating railway. No experimental variant has been promoted into the dashboard.

## What changed

`headway/onboarding.py` provides optional per-asset reference adaptation. It uses
the first 21 complete calendar days to estimate each residual signal's median and
robust scale. Fleet context models, health-index weights and the failure model
remain frozen. The original pipeline is not mutated.

Future reference observations and overwriting fleet-training assets are rejected.
Incomplete, unsupported or degenerate references produce explicit abstention.
Decisions before the reference becomes available are masked. Fault labels are
removed before reference aggregation. These checks establish data eligibility;
they do not establish that the reference asset is healthy.

`headway/calibration_experiments.py` adds empirical residual calibration that gives
each training group equal total weight. The experiment compares four fixed arms:
fleet baseline, onboarding with pooled additive residuals, onboarding with
group-balanced additive residuals, and onboarding with ratio residuals.

## Held-out train results

Each target train is excluded from fleet preprocessing and failure-model fitting.
Only its initial reference telemetry may adapt its baseline. This protocol allows
other trains' later outcomes in training; it measures retrospective generalisation.

| Fleet | Variant | Point MAE, days | Lower-bound coverage | Abstention |
|---|---|---:|---:|---:|
| Existing development fleet | Fleet baseline | 8.87 | 72.9% | 39.5% |
| Existing development fleet | Onboarding, pooled | 3.95 | 85.3% | 37.2% |
| Existing development fleet | Onboarding, balanced | 3.95 | 89.8% | 37.2% |
| Existing development fleet | Onboarding, ratio | 3.95 | 86.1% | 37.2% |
| Fresh seed 20260910 | Fleet baseline | 8.03 | 79.6% | 39.3% |
| Fresh seed 20260910 | Onboarding, pooled | 4.16 | 88.9% | 37.0% |
| Fresh seed 20260910 | Onboarding, balanced | 4.16 | 88.9% | 37.0% |
| Fresh seed 20260910 | Onboarding, ratio | 4.16 | 86.5% | 37.0% |

Both fleets have nine evaluated fault-bearing train groups. The development fleet
has 301 eligible episode-days, of which 182 baseline and 189 onboarded rows are
scored. The replication has 300 eligible days, with the same scored counts.
MAE is calculated only where a prediction exists; baseline and onboarding therefore
do not have identical scored populations. Daily observations are correlated.
Coverage is averaged across evaluated groups and checks whether the lower bound
is at or below actual remaining life; it is not confidence in an individual decision.

## Unseen train and future outcomes

The stricter protocol additionally freezes training at 24 June 2026, 70% through
the timeline, and uses only other-train faults confirmed by that cutoff.

| Fleet | Variant | Point MAE, days | Lower-bound coverage | Abstention |
|---|---|---:|---:|---:|
| Development | Fleet baseline | 7.50 | 86.0% | 10.3% |
| Development | Onboarding, pooled | 3.46 | 95.8% | 10.3% |
| Development | Onboarding, balanced | 3.46 | 95.8% | 10.3% |
| Development | Onboarding, ratio | 3.46 | 94.4% | 10.3% |
| Replication | Fleet baseline | 2.89 | 100.0% | 0.0% |
| Replication | Onboarding, pooled | 1.62 | 100.0% | 0.0% |
| Replication | Onboarding, balanced | 1.62 | 100.0% | 0.0% |
| Replication | Onboarding, ratio | 1.62 | 100.0% | 0.0% |

Development evaluates six groups and 107 episode-days (96 scored). Replication
evaluates only three groups and 48 scored episode-days. Its 100% coverage is a
small-sample result. Pooled and balanced onboarding have a median lower bound of
zero in both chronological tests: high coverage alone does not mean useful
maintenance deferral recommendations.

## Decision

**Continue developing per-asset onboarding.** The reduction in point error repeats
on a fresh simulator seed and in the stricter chronological check. Before real
use, inspection or maintenance evidence must establish a suitable reference;
otherwise adaptation could absorb existing degradation into the baseline.

**Keep group-balanced calibration experimental.** Its development coverage gain
does not repeat on the fresh seed. Neither additive variant consistently reaches
the nominal 90% coverage target across the held-out train tests.

**Do not select ratio calibration merely for longer usable margins.** It increases
the fraction of eligible rows with a positive, correctly covered lower bound from
33.2% to 52.5% in development and 38.0% to 53.3% in replication, relative to pooled
onboarding. But near-fault coverage drops from 100% to 82.5% and 73.0%, respectively,
in the train-holdout tests. This tradeoff matters directly to deferral decisions.

The next evaluation should fix its protocol before examining new outcomes, include
matched-row point-error comparisons, and test contaminated onboarding references,
missing observations and operating-condition shifts. These stress tests can use
simulation. A defensible operational performance claim needs real telemetry,
verified units, inspection and maintenance records, fault timing and censored
follow-up across enough independent assets and failure episodes.

## Reproduce and inspect

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -B scripts/experiment_onboarding.py
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
```

The completed test suite passed **181 tests**, including reference isolation,
pre-readiness abstention, future-telemetry invariance and held-out-label isolation.

- [Fixed protocol and script hash](data/onboarding_experiment/protocol.json)
- [Full metrics and per-group results](data/onboarding_experiment/results.json)
- [Comparison table](data/onboarding_experiment/comparison.csv)

Prediction Parquet files in the same directory retain each fleet/protocol/arm's
outputs. The existing fleet was previously inspected; the replication uses the
same simulator family. Neither is an external validation dataset. No model was
automatically selected from these results.
