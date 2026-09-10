# Headway — Time to act

*NEBULA X · PS3 · The Living Railway. Draft; all figures below are from synthetic
door telemetry and are reproducible from a clean checkout.*

Singapore's rail network is already reliable — MKBF runs near 1.7 million
train-km. So the problem in PS3 is not *finding* failures; a plain threshold does
that. The problem is finding them **early enough to choose** — because the
engineering window is barely two usable hours a night, and every night someone
decides which few jobs fit.

An anomaly score does not help with that decision. It says *something changed*.
Headway turns that into a **maintenance decision an engineer can review**: an
action, a lower bound on how long the door has, the data state behind it, and a
direct comparison against the window they are actually considering.

**The most useful artefact we built is not the model. It is the evaluation
apparatus** — built first, pointed at our own work, and it ruled against us
repeatedly until what remained was small enough to trust. This document is
organised around that, because it is also what makes the numbers believable.

---

## What it does

The dashboard opens on **Status** - the doors needing action tonight - with
**Review** (recently flagged, now recovered or awaiting a slot), a **Planner**
that holds tonight's engineering window, a **Fleet** grid, and a **Door detail**
view whose wear chart doubles as the date scrubber.

The Planner is where a decision gets recorded rather than just displayed. Jobs go
in from Status, get ordered, and close with **Done**. Anything not closed
**carries to the next night by itself**, tagged with how late it now is — and
priced against the same lower bound the card shows:

```
TRN025-DOOR-2   WITHDRAW · at least 41.7 d
   carried from 2026-07-26 · 3 nights late
   Another night is still inside the estimated lower margin.
```

That is the honest version of the survival curve the audit removed: no
probability, just *this delay is, or is not, inside the estimated margin*. We do
not estimate repair durations — no repair-time data exists in the contract, and
inventing one would be the same mistake.

Every door gets a **signal aspect** in the four-aspect language railways already
use — GREEN `MONITOR` / DOUBLE AMBER `PLAN` / AMBER `TONIGHT` / RED `WITHDRAW` —
plus an **Aspect Card**: the estimated margin in days, the evidence in plain
English, and the **window comparison**:

```
TRN023-DOOR-2 · DOUBLE AMBER · PLAN
    proposed window: in 3 days    Within estimated margin
                     in 7 days    Exceeds estimated margin
    Estimated margin only; operator review required.
```

Underneath the card sits a second line the aspect cannot express — **fit for
duty**:

```
    Fit for duty: Off-peak service only      avoid 07:00–10:00, 16:00–20:00
    index at peak load 66.2 / day average 49.7 / off-peak 43.4; failure level 54.2
```

Railway engineers told us the same thing in different words: a worn door does
not fail on average, it fails at 08:15. The fleet normaliser already removes the
crowding effect a *healthy* door shows; what is left on a worn door is an
**excess** load sensitivity, measured from that door's own crowded-versus-quiet
cycles over the same three days as its index. When it is present and
significant, Headway re-runs the card's own model at the reference peak load.
If the peak-conditioned index has reached the level at which doors failed while
the day average has not, the door is recommended **off-peak only** until
maintained. Duty is never a relaxation of the aspect — it only ever adds a
restriction — and a door whose day average has itself reached that level is
simply *withdraw*.

Three properties make that margin worth acting on:

- **Normalised for operating conditions.** On this data a hot or crowded run
  moves the primary signal ~4× more than three-weeks-out wear. Headway models the
  expected reading given conditions and scores the residual, per asset — a stiff
  healthy door never outranks a degrading one.
- **It is a lower bound, and it abstains.** When history is too short, the signal
  is flat-but-elevated, the observation is stale, or conditions fall outside
  training, the card says *"cannot assess"* — it does not fabricate a countdown.
- **It never claims a probability.** An empirical lower bound on a handful of
  episodes is not a survival curve. The ledger compares windows; it does not
  print "14% risk" or a guaranteed safe date, because we cannot calibrate one.

---

## What the evaluation found

Two protocols, both on synthetic data, both in `data/validation_report.json`.

### Chronological — train on the past, test on the future

Freeze training on faults confirmed by **24 June 2026** (70% through the
timeline); test on everything after.

| | result |
|---|---:|
| Future episodes detected | **6 / 6** |
| Worst-case warning | **6.05 days** |
| Alert precision (directional score) | 78.1% |
| False alerts | 0.07 per asset-month |
| RUL point error (MAE) | **3.5 days** |
| Lower-bound coverage / near-fault | 97.4% / 100% |
| Abstention on degradation rows | 13.8% |
| Doors restricted to off-peak before the fault | **6 / 6**, median 2 days ahead |
| Healthy asset-days wrongly restricted | **0** of 2,647 assessed |

Read alongside the number we are least happy with: the **median lower margin is
0 days**, and only 34% of scored rows carry a *positive* lower bound. Coverage of
97% is easy when half your bounds are "at least zero". The bound is honest, but
on this split it is often too conservative to schedule against.

The duty rows deserve their own caveat. The restriction is a *level* comparison,
not a forecast; it fires in the last days before failure, after the aspect has
already escalated. Its value is not earlier warning — it is telling the roster
what to do with the door *between* the escalation and the window. And the
wear-load interaction it detects is a **simulator assumption** (`load_wear_coeff`
in the generator), chosen from operator experience, not measured. It is the
first thing we want to test on real telemetry: do worn doors show a steeper
current-against-crowding slope than healthy ones?

### Whole train held out — the stricter test

Refit *everything* — preprocessing, calibration, model selection — with an entire
train removed, for each of the nine fault-bearing trains.

| | result |
|---|---:|
| Lower-bound coverage | **75.8%** (target 90%) |
| RUL point error (MAE) | 8.7 days |
| Abstention | 37.3% |

**This misses the target, and we say so on the dashboard.** Nine synthetic
episodes do not establish generalisation. The failure mode is specific and
fixable — see below.

### The tournament does not crown a winner

Under the same chronological replay (threshold frozen before the cutoff, daily
alert capacity, cooldown): the raw fleet threshold **misses 1 of 6** and
its worst warning is 0 days. Every model fed the *normalised* features finds all
six with ~6 days minimum warning and clusters within a few points of each other
— PCA, Mahalanobis, LOF and our directional score are all at 0.021
false-alerts/asset-month; isolation forest at 0.042.

That clustering is the finding: **the preprocessing does the work, not the
algorithm.** `data/tournament_replay.csv` has every row. We do not select a
deployment model from a six-episode replay.

---

## The generalisation gap, and the fix

The 76% held-out coverage has one cause: when a train Headway has never seen
enters the fleet, its per-asset baseline is estimated from contaminated data.

`headway/onboarding.py` addresses it with a 21-day per-asset reference adapter
(fleet models stay frozen). Measured on the same held-out-train protocol:

| | point MAE | coverage |
|---|---:|---:|
| Fleet baseline | 8.7 d | 75.8% |
| + onboarding adapter | **4.6 d** | **87–90%** |
| + onboarding, fresh simulator seed | 4.4 d | ~90% |

The point-error halving **replicates on a fresh seed** and holds in the stricter
chronological check. `ONBOARDING_RESULTS.md` has the full table.

**It is not in the demo.** Promoting it needs inspection evidence that a door's
first 21 days were actually healthy — otherwise the adapter absorbs existing
degradation into the baseline. That is exactly the kind of question the SMRT
depot and LTOC site visits can answer, and the kind of decision that belongs to
an operator, not to us.

---

## The corrections behind the numbers

Four times during the build the evidence overturned something we believed. Each
correction, and the measurement that forced it, is in the repo:

1. **Alert budget was counted per cycle.** One sick door consumed the whole
   budget and a spiky baseline won by accident — the result was *inverted*. An
   engineer gets one alert per door per day; the unit is asset-days.
2. **The tournament said we were losing.** Our own score was double-smoothed by
   the harness. An evaluation that returns a surprising verdict about your own
   model is usually measuring your harness.
3. **The inverse flaw, in our favour.** Condition models were fitted on all 120
   days, giving the normalised detectors a look-ahead the raw ones lacked.
   Refitted on the first 30 days only; every headline number survived.
4. **The RUL projection was optimistic by design.** A straight line to the
   failure threshold overstates remaining life by a median of +8.7 days (wear is
   convex). A log-linear projection cut point error by 69%, with nothing fitted
   to the episodes — and downstream, the volatility-band machinery we had
   carefully built turned out to only exist to patch that bias.

The full audit — explicit RUL data states, calendar-time availability, the
window-comparison rewrite — is in `AUDIT_FIXES.md`. Pre-audit files are retained
under `.audit-backup/`.

---

## What is genuinely unproven

Real telemetry with verified units and identity joins; confirmed failure and
inspection semantics; maintenance and censoring records; operator action policy;
and enough independent episodes to calibrate anything. The wear-load
interaction behind fit-for-duty is assumed, not measured, and the generator's
faults are not load-triggered — so nothing here shows that off-peak running
*extends* a door's life, only that its condition at peak load has reached the
failure level. Bogies are defined in the contract but not demonstrated. A conditional survival model and a persistent
operator decision-record are future extensions, not claims.

---

## Try it

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q          # 206 tests
.\.venv\Scripts\python.exe -B scripts\rebuild.py --validate
```

Open `ui/headway.html` — labelled a retrospective synthetic demonstration.
`data/validation_report.json` is the evaluation output of record.

- Repository: `[[repo URL]]`
- Demo video: `[[video URL]]`
- Team: `[[names and roles]]`
