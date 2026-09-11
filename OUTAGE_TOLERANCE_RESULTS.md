# Outage-tolerant warning score — experiment, 11 September 2026

Experimental. Synthetic only. **No model is promoted**, no operational threshold
is changed, and the dashboard's detector and smoother are untouched. Nothing here
is evidence of performance on an operating railway; the organiser dataset has not
been received.

Current results: `data/outage_experiment/runs/dev-20260911T040628-2aabf349/`.
The flat files in `data/outage_experiment/` are the original run, preserved as a
historical artifact and marked `SUPERSEDED.md`.

## What this addresses

`ROBUSTNESS_RESULTS.md` recorded that under an every-third-day outage the
monitoring pipeline produced **no scores at all**. The cause is the admission
rule, not the underlying detector: `robustness.score_frozen` smooths over a
trailing three-*calendar-day* window and requires three observations inside it,
so three consecutive daily readings never accumulate.

The challenger (`headway/outage_tolerant.py`) keeps the same per-day scores and
the same abstention discipline, and changes only which observations may support
an assessment:

| Parameter | Meaning |
|---|---|
| `min_observations` | how much evidence is required before any score |
| `max_age_days` | how old a supporting observation may be |
| `max_gap_days` | largest elapsed gap permitted between consecutive supporting observations |

Observations may be irregularly spaced; admission is decided on **elapsed time**,
not row adjacency. The realised observation count, span and largest gap are
emitted alongside every score.

### What it deliberately does not do

* No interpolation, imputation or resampling — a missing reading stays missing.
* No carrying a score forward. Every assessment is anchored on a valid
  observation **for that day**; without one the output is an explicit status and
  a NaN score, never yesterday's number relabelled as today's.
* No unsupported operating conditions, in the current row *or in history*.
* No publishing a score the model could not produce: finite inputs do not
  guarantee finite output, so a non-finite model result is reported as
  `model_score_unavailable` rather than folded into the median.
* No future data.

Availability is reported separately from health: a `*_status` column
(`supported`, `unsupported_or_incomplete`, `incomplete_channels`,
`model_score_unavailable`, `insufficient_observations`, `gap_too_large`) says why
an assessment is missing, so a quiet detector during an outage is never readable
as a healthy asset.

## Protocol and freeze integrity

Each development run gets its own directory under `runs/<run_id>/`, holding the
protocol, the development results and — only if a candidate qualified — the
frozen policy. The freeze binds the policy to the SHA-256 of the protocol (both
as written and canonicalised), of the development results, and of the four code
files whose change could alter a result. `--stage eval` requires an explicit
`--run` and re-checks every one of those hashes; any drift, a missing artifact, a
freeze lifted into another run, or a development run that qualified nothing is
refused rather than silently accepted.

No hash covers the freeze file itself, so the evaluation stage additionally
**re-derives the winning policy from the hash-verified development results** and
refuses any freeze whose parameters, label or qualifying status do not match what
the sweep selected — including the runner-up, an unevaluated policy, or the same
label with a loosened parameter. Runs are never overwritten: a development stage
that targets an existing run directory is refused, and an evaluation whose
artifacts already exist must be re-run as `--reproduction <id>`, written beside
the original rather than over it.

The runner makes **no blindness claim**. It cannot know what has already been
scored on the machine it runs on, so `frozen_policy.json` records
`blinding: unasserted` and leaves that assertion to the operator.

**This run is stale under the current code.** The freeze-integrity changes above
altered `experiment_outage_tolerance.py`, and the freeze is bound to its hash, so
`--stage eval --run dev-20260911T040628-2aabf349` now refuses. That is the guard
working. The sweep was not re-run for a bookkeeping change; the numbers below are
the numbers that run produced, under the code it hashed. See
`runs/dev-20260911T040628-2aabf349/STALE_UNDER_CURRENT_CODE.md`.

* Fleets: 12 trains × 2 doors × 120 days, 6 episodes, 35 cycles/day.
* Splits: fit on days 0–30 of the training trains; calibrate thresholds on days
  30–84; replay after day 84. First train identity held out of all fitting.
* Each method is calibrated to **its own** 99th-percentile threshold on the same
  pre-replay span — the two produce different score distributions, so sharing one
  threshold would rig the comparison. (The thresholds do differ materially; see
  the alert-burden section.)
* Alerting: 2 alerts/day, 3-day cooldown, per population.
* Seeds: development **20260913, 20260914**; evaluation **20260915, 20260916**.
  Both pairs differ from ROBUSTNESS_RESULTS.md's (20260911, 20260912).
* Scenarios: clean, 30% primary-channel sample loss, every-third-day outage, a
  10-day contiguous blackout, +0.6 A.s sensor offset, +20 °C context corruption.
* Expected asset-days stay in every availability denominator, so an outage cannot
  flatter availability by removing its own rows.

### How the policy was chosen, and two caveats

Acceptance criteria and the 18-policy candidate grid were fixed before the first
development sweep ran. The sweep rejected wider windows: every `max_age_days` of
7 or 10 failed the clean-detection criterion, because a longer median window lags
a rising wear signal. Six policies qualified, all with `max_age_days = 5`.

Selected and frozen: **`min_observations=3, max_age_days=5, max_gap_days=3`**.

**Caveat 1.** The criteria were pre-registered; the *tie-break* order was added
after the sweep showed six qualifiers. It was applied without reference to any
evaluation-seed number — the evaluation seeds had not been scored when the first
`frozen_policy.json` was written. The tie-break prefers, in order: highest outage
availability, then more required observations, then fresher evidence, then
smaller tolerated gaps.

**Caveat 2.** The evaluation seeds are now **exposed**. The figures below come
from a re-run after three corrections (below), so this is a *reproduction on
exposed seeds*, not a blind test. The policy was not re-selected using evaluation
results: it was re-frozen from the development seeds under the corrected code,
and independently reproduced the same choice.

An earlier version of `frozen_policy.json` in this run carried the sentence "The
evaluation seeds had not been scored when this was written." That was inherited
from the first run and is **untrue of the regeneration**, which was written after
those seeds had already been scored under the previous code. The claim is
withdrawn; the file now records the run as a reproduction on exposed seeds, and
the runner no longer emits such a claim automatically.

## Results — evaluation seeds

Means over 3 detectors × 2 seeds. 216 expected asset-days per seed for the
affected assets; 3 post-cutoff episodes per seed.

### Availability — supported score as a fraction of expected asset-days

| Scenario | Affected assets: baseline | challenger | Whole fleet: baseline | challenger |
|---|---:|---:|---:|---:|
| Clean | 0.993 | 0.998 | 0.986 | 0.995 |
| **Every-third-day outage** | **0.000** | **0.664** | 0.737 | 0.911 |
| 10-day blackout | 0.660 | 0.664 | 0.902 | 0.911 |
| 30% channel sample loss | 0.000 | 0.000 | 0.737 | 0.745 |
| Unsupported context | 0.000 | 0.000 | 0.737 | 0.745 |
| Sensor offset | 0.993 | 0.998 | 0.986 | 0.995 |

The headline is precise, not rounded: under the every-third-day outage, of 216
expected affected asset-days 144 actually carried telemetry. The baseline scored
**0** of them. The challenger scored **143** — 99.3% of the days that had data.
The remaining shortfall is the 72 days that were genuinely absent, and they stay
in the denominator.

**Two scenarios are unimproved, by design.** With 30% sample loss the daily
aggregate fails its completeness gate, so there are no valid observations to
window over; with corrupted context every row fails `context_supported`. In both
cases 216/216 affected rows report `unsupported_or_incomplete`. Those failures
are upstream of the smoother and this change cannot — and must not — rescue them.

### Fault detection — CLEAN scenario, WHOLE-FLEET population

Six unique post-cutoff episodes exist across the two evaluation seeds (three per
seed). Each is assessed by three detectors, giving 18 detector–episode
evaluations. These are different denominators and are reported separately.

| | Baseline | Challenger |
|---|---:|---:|
| **Unique episodes** detected by at least one detector | **3 of 6** | **3 of 6** |
| Detector–episode evaluations detected | 7 of 18 | 6 of 18 |
| Median lead where detected | 1.9–2.9 d | 0.9–2.4 d |
| Worst lead | 0.0 d | 0.0 d |

At the level of unique faults the two methods are **identical**: both find the
same three episodes, all in seed 20260916. Seed 20260915 detected nothing with
either method on any detector. The 7-versus-6 difference is one detector–episode
evaluation — `mahalanobis` on seed 20260916, 3/3 → 2/3 — and does not change
which faults were found.

With six unique episodes, one seed detecting none, and correlated daily rows,
this comparison does not support a claim in either direction.

Note this contradicts development, where clean-detection regression was exactly
0.000 and the criterion passed. The pre-registered criterion did not generalise
to fresh seeds. That is a finding about the criterion, and a reason to distrust
detection conclusions at this sample size — not a reason to re-tune.

### Recovery after telemetry resumes

Measured against `available_at`: a day's aggregate lands the following midnight,
so this is when an engineer could actually receive the score, not when the
observation was taken. Every affected asset stays in the denominator, and an
asset that never recovers censors the median rather than dropping out of it.

Identical: **median 3.0 days, 6 of 6 assets recovered, uncensored — both methods,
both seeds.** After a clean daily resume the challenger needs three observations
within five days and the baseline needs three consecutive days: the same three
readings, available at the same midnight. The challenger offers **no recovery
advantage** when telemetry returns cleanly; its advantage is confined to
telemetry that returns *intermittently*.

### Alert burden, and sensor versus mechanism

Unmatched alerts per asset-month:

| Scenario (whole fleet) | Baseline | Challenger |
|---|---:|---:|
| Clean | 0.075 | 0.069 |
| Sensor offset | 0.139 | **0.318** |

On the affected assets in seed 20260915 the sensor offset drove the `raw`
detector from 11 alerts (1.53/asset-month) to **36 alerts (5.00/asset-month)**;
`mahalanobis` from 0.28 to 1.11. Seed 20260916 produced zero for both.

**Zero of these alerts matched a mechanical episode.** They are attributable to
the injected sensor perturbation, and — as ROBUSTNESS_RESULTS.md noted — such
alerts could still justify a sensor inspection rather than being simply wrong.

**Causation is unresolved.** An earlier draft of this report claimed the increase
followed from restored availability. That claim is withdrawn: it does not survive
the numbers. Availability moved 0.986 → 0.995, a 0.9-percentage-point change,
while `raw` alerts rose 11 → 36, a factor of 3.3. At least two other things
differ between the arms and either could dominate:

* **Thresholds differ.** Each method is calibrated separately, and the challenger's
  are lower — on seed 20260915, `mahalanobis` 1715.1 versus 1406.6, about 18%
  lower; `directional` 35.0 versus 33.5. A lower threshold alone produces more
  alerts under a biased sensor.
* **The aggregation differs.** A median over an elapsed-time window responds
  differently to a step offset than a median over three consecutive days.

Separating these would need an experiment holding threshold and aggregation
fixed while varying only availability. That has not been run. What can be stated
is the observation: under a sensor offset the challenger raised substantially
more unmatched alerts on one of two seeds, and none of them were mechanical.

## Corrections applied in this revision, and what they changed

| Correction | Effect on reported numbers |
|---|---|
| Require a finite **model output**, not just finite inputs, before admitting today's observation | **None here.** No detector returned a non-finite score on finite inputs in these runs; availability and detection are byte-identical before and after. It closes a latent hole in which `status="supported"` could accompany `score=NaN`. |
| Measure recovery against `available_at`, keep fractional days, and return every requested asset | **Recovery 2.0 → 3.0 days.** The previous figure measured observation dates and understated how long monitoring is dark by one aggregation delay. Coverage (6/6) and the baseline-versus-challenger tie are unchanged. |
| Censored recovery median; coverage checked before speed in selection | No change to the selected policy — `m3_a5_g3` was re-selected independently under the corrected criteria. |
| Versioned runs; freeze bound to protocol, development results and code hashes | No numerical effect. |
| Freeze re-derived from the hash-verified development results; runs and published artifacts never overwritten; blindness no longer asserted by the runner | No numerical effect. Closes a bypass in which the frozen `min_observations` could be edited from 3 to 2 in place and still be accepted, because no hash covered the freeze file. Makes this run's freeze stale under the current code, by design. |
| Report: withdraw the availability-causes-alerts claim; label the detection table; separate unique episodes from detector–episode evaluations | Detection is now reported as **3 of 6 unique episodes for both methods**, with 7-vs-6 shown as the detector–episode count it always was. |

## Limitations

* Small fixed synthetic fleets. Not external validation, and not evidence about
  rail hardware.
* Six unique evaluation episodes, one seed detecting none. Daily rows are
  correlated; these are not independent failures.
* The evaluation seeds are exposed; this revision is confirmatory, not blind.
* Whole-fleet and affected-asset replays hold separate alert budgets; their
  counts are not additive.
* Fault labels are unchanged by the sensor and context perturbations, so an
  unmatched alert is not necessarily a wasted inspection.
* The tie-break order was fixed after the first development sweep, not before it.
* Why the sensor-offset alert load rose is unresolved.
* The challenger addresses one failure mode. Aggregate incompleteness and
  unsupported context remain unaddressed and dominate two of the five stress
  scenarios.

## Promotion recommendation

**Do not promote. Keep as an experimental alternative.**

The change does what it was built to do — it converts a total monitoring outage
into 99.3% coverage of the days that had telemetry — and it does so without
inventing evidence, carrying scores forward, admitting unsupported readings, or
publishing a score the model could not produce.

It is not enough to justify replacing the deployed smoother:

1. It fixes one of three availability failures; the other two fail upstream.
2. It gives no recovery advantage after a clean resume.
3. It finds exactly the same unique faults as the baseline, on a sample far too
   small to interpret — and the development criterion that was supposed to
   protect detection did not generalise.
4. It coincided with a large increase in sensor-driven alerts on one seed, for
   reasons this experiment cannot attribute.

Before reconsidering: re-run on organiser data with real outage patterns; run the
threshold-and-aggregation-controlled experiment needed to explain the alert
increase; pair it with the sensor-versus-mechanism evidence proposed in
ROBUSTNESS_RESULTS.md; and treat the detection comparison as unresolved until
enough independent episodes exist to resolve it.

## Reproduce

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -B scripts/experiment_outage_tolerance.py --stage dev
# the development run prints the exact command, including its run id:
.\.venv\Scripts\python.exe -B scripts/experiment_outage_tolerance.py --stage eval --run <run_id>
.\.venv\Scripts\python.exe -B -m pytest tests/test_outage_tolerant.py tests/test_outage_experiment_freeze.py -q -p no:cacheprovider
```

`--stage eval` requires `--run` and refuses any policy whose protocol,
development results or code hashes no longer match, so it can neither choose its
own parameters nor silently reuse a stale freeze.
