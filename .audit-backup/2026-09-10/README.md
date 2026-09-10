# Headway

**Time to act.**

Predictive fault detection for rail rolling stock that outputs a *decision*, not a score.
Built for NEBULA X (LTA Rail Digitalisation & Guild), PS3 — The Living Railway, 18–20 Sep 2026.

Every fault announces itself early. Headway listens to door and bogie telemetry, strips away
the weather and the crowds, and tells the engineer the only three things that matter:
**how much time you have, how sure we are, and what to do tonight.**

---

## Why this exists

An anomaly score tells an engineer that something changed. It does not tell them whether it can
wait until tonight's engineering window — of which there are barely two usable hours — or whether
someone needs to be on track now. Headway closes that gap by producing a **signal aspect** per
asset, in the four-aspect language railways already use:

| Aspect | Recommendation | Meaning |
|---|---|---|
| 🟢 GREEN | `MONITOR` | Healthy. Keep watching. |
| 🟡🟡 DOUBLE AMBER | `PLAN` | Book into an upcoming window. |
| 🟡 AMBER | `TONIGHT` | This window. |
| 🔴 RED | `WITHDRAW` | Pull from service now. |

---

## The case, in one screen

*Everything below is reproducible from a clean checkout in about four minutes. Figures come from
the synthetic pre-build: 456,860 door cycles, 80 doors, 120 days, 9 confirmed faults — a base rate
of 2.0e-05, where predicting "healthy" every time scores 99.998% accuracy.*

**The problem is not detection. It is timing.** A raw fleet-wide threshold is ~60% precise — poor
but survivable. What sinks it is that it misses 2 of 9 episodes outright and its worst warning is
**4.5 days**, inside the window where the only remaining option is an unplanned withdrawal.

**What the pipeline adds**, at an equal budget of 150 asset-day alerts:

| detector | found | worst-case warning | precision |
|---|---|---|---|
| raw fleet-wide threshold | 7/9 | 4.5 d | 60.0% |
| **Headway** | **9/9** | **10.5 d** | **96.7%** |

**It arrives in time to plan.** Days before the confirmed fault that each order first appears:

| | PLAN | TONIGHT | WITHDRAW |
|---|---|---|---|
| median | 21.1 d | 9.0 d | 7.1 d |
| worst case | 11.5 d | 6.5 d | 4.9 d |

All nine episodes reach every stage. **It stays quiet:** 97.6% of asset-days are GREEN, and of 71
healthy doors, **zero ever reach AMBER or worse** — about 0.9 doors per night need the engineering
window, a list short enough to actually be worked.

**It wins its own tournament, under a protocol that could have beaten it.** Nine detectors, each
fitted on the first 30 days only then scoring the full history, identical smoothing, equal budget.
Headway leads on precision and capture at 250 alerts (86.8% / 85%). The sharper finding is not that
we win — it is that **normalisation is worth +4.4 days to an isolation forest and +0.0 to PCA**,
because PCA already performs its own implicit normalisation. *Preprocessing substitutes for model
capacity; it does not add to it.*

**It knows what it cannot do.** At onset the wear is exactly zero, so ~26% of every degradation
window is undetectable by anything. Capture is measured against what is *physically detectable*,
which is why we can say the pipeline delivers **89% of the warning that actually exists** at the
250-alert operating point.

**164 tests** assert the properties this argument depends on — no window sees the future, every
detector gets exactly the same budget, no bound overstates a margin, no healthy door is handed a
countdown. Four times the evidence overturned something we believed; each correction is documented
in place with the measurement that caused it. Start with
[`SUBMISSION.md`](SUBMISSION.md) for the narrative, or read on for the full evidence.

---

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe scripts/make_synthetic.py
```

```bash
.venv/Scripts/python.exe scripts/check_premise.py
```

```bash
.venv/Scripts/python.exe scripts/build_health.py
```

```bash
.venv/Scripts/python.exe scripts/build_rul.py
```

```bash
.venv/Scripts/python.exe scripts/build_aspects.py
```

```bash
.venv/Scripts/python.exe scripts/build_deferral.py
```

```bash
.venv/Scripts/python.exe scripts/run_tournament.py
```

```bash
.venv/Scripts/python.exe scripts/build_ui.py
```

Then open `ui/headway.html` — a single self-contained file, no server.

---

## Current state

**Built and verified**

- `headway/contract.py` — the feature contract. The keystone. Everything downstream speaks only
  this schema, so binding real NEBULA X data means writing *one* adapter file, not refactoring
  the pipeline. Includes a validator that names the exact column and expectation on failure.
- `headway/synth/doors.py` — synthetic Door Control Unit telemetry, deliberately adversarial:
  operating conditions move the primary signal ~4.6× more than three-weeks-out wear does, every
  door carries its own build tolerance, faults are repaired (health resets) and the base rate is
  brutal (9 confirmed faults in 456,860 cycles — 2.0e-05).
- `scripts/check_premise.py` — proves the thesis holds, and prototypes the Proving Ground metric.

- `headway/normalise.py` — `ConditionNormaliser` (models `E[signal | conditions]`, robust-trimmed,
  label-free), `AssetBaseline` (per-asset location and robust scale), and `MultivariateHealthIndex`
  (orients, whitens and projects all signals into one *directional* index — unit-free sigma, so it
  is the standardised, cross-fleet-comparable parameter).
- `headway/features.py` — cycle → asset-day aggregation (level, dispersion, rates) plus trailing
  trend features. All windows are trailing, so no score can see the future.
- `headway/evaluate/metrics.py` — the Proving Ground core. Equal-alert-budget comparison,
  lead time, worst-case lead, precision at true base rate, and **capture**.
- `scripts/build_health.py` — runs the pipeline end to end, writes `data/door_daily.parquet`.
- `headway/rul.py` — `ConformalRUL`. Straight-line projection to a learned failure threshold, with
  a Mondrian-conformal lower bound calibrated leave-one-episode-out.
- `scripts/build_rul.py` — fits it, selects the conformity mode on evidence, writes
  `data/door_rul.parquet`.

- `headway/aspect.py` — the decision policy. Four-aspect signalling, hysteresis, and the
  Aspect Card. Kept separate from estimation on purpose: what Headway *recommends* can be changed
  without retraining anything, and the thresholds are readable enough to be argued with.
- `scripts/build_aspects.py` — applies the policy and validates it operationally.

- `headway/models/detectors.py` — the detector zoo: raw threshold, EWMA control chart, isolation
  forest, PCA reconstruction, Mahalanobis distance, LOF. One interface, one protocol.
- `scripts/run_tournament.py` — the Proving Ground.
- `headway/deferral.py` — the **Deferral Ledger**: a calibrated survival curve per asset-day, and
  the price of waiting. `scripts/build_deferral.py` fits it and validates the percentages against
  outcomes.
- `scripts/build_ui.py` — renders `ui/headway.html`, the operator view, from real pipeline output.

**Not built yet**

Possession Planner · Hindsight · Logbook · Earshot.

---

## What the premise check establishes

Two assumptions died while building this, and both corrections made the thesis sharper. They are
worth knowing because the pitch rests on them.

**1. Naive thresholding is not primarily a precision problem.** It is ~60% precise on this data —
poor but survivable. What sinks it is *timing and coverage*: it misses 2 of 9 episodes outright,
and its worst warning is 4.5 days, which is inside the window where the only remaining option is
an unplanned withdrawal.

**2. Alerts must be counted in asset-days, not cycles.** A smoothed score stays elevated for
hundreds of consecutive cycles, so one sick door consumes an entire per-cycle alert budget and a
spiky raw score wins by accident. An engineer receives one alert per door per day. Getting this
unit wrong inverted the whole result the first time.

Detectors compared at an **equal budget of 150 asset-day alerts** (1.6% of 9,600 asset-days),
9 episodes to find. These are the figures from `build_health.py`, i.e. the real componentised
pipeline — `check_premise.py` reports slightly different values for B and C because it uses a
simpler baseline (offset only, no robust rescaling):

| detector | found | median lead | worst case | precision | worst capture |
|---|---|---|---|---|---|
| A — raw fleet-wide threshold | 7/9 | 13.2 d | 4.5 d | 60.0% | 29% |
| B — + per-asset baseline | 9/9 | 13.2 d | 9.5 d | 92.7% | 44% |
| C — + condition normalisation | 9/9 | 13.1 d | **10.5 d** | **96.7%** | **53%** |
| D — + multivariate (all signals) | **9/9** | **15.1 d** | **10.5 d** | **96.7%** | 52% |

Net C over A: **+6.0 days in the worst case, +37 points of precision, and coverage from 7/9 to
9/9** — every episode found. Median lead barely moves at all (−0.1 d), which is itself the point.

The worst case is the number that matters. A detector whose worst warning is 4.5 days cannot be
planned around; one whose worst warning is 10.5 days can.

Coverage comes from the per-asset baseline: it is what takes 7/9 to 9/9 and 56% to 90% precision.
On worst-case margin it also contributes most of the gain here (+5.0 d, vs +1.0 d for the condition
model) — but that split is implementation-sensitive rather than a law. Run the same ablation in
`check_premise.py`, whose baseline is a plain offset with no robust rescaling, and it inverts
(+0.0 d baseline, +6.0 d conditions). **Claim the combination, not a ranking.**

And note what the **median** does here: A, B and C all sit within 0.1 d of each other (13.1–13.2).
Ranked on the median, all three detectors look interchangeable. Ranked on worst case they separate
by a factor of nearly three, and on coverage two of them are simply unusable. **The median cannot
tell these detectors apart; the worst case can.** That is the argument for how the Proving Ground
scores, made by our own numbers rather than asserted.

### Capture — how much of the available warning we actually deliver

No detector can warn before degradation begins — and at onset the wear is *exactly zero*, so the
opening ~26% of every window sits below the noise floor of a daily aggregate no matter the model
(`scripts/diagnose.py` derives this; the generator now declares it per episode as
`detectable_days`). `capture = lead achieved / lead physically detectable`, per episode:

| alert budget | found | median lead | worst lead | precision | worst capture |
|---|---|---|---|---|---|
| 50 | 9/9 | 4.0 d | 1.9 d | 92.0% | 10% |
| 100 | 9/9 | 9.1 d | 5.9 d | 95.0% | 31% |
| **150** | 9/9 | 15.1 d | 10.5 d | **96.7%** | 52% |
| **250** | 9/9 | 23.1 d | **13.5 d** | 87.2% | **89%** |
| 400 | 9/9 | 23.1 d | 17.5 d | 58.2% | 95% |
| 800 | 9/9 | 29.9 d | 17.9 d | 31.0% | 95% |

This curve replaces ROC — its x-axis is a resource a depot manager actually controls. Two things
it tells us: **250 alerts remains the better operating point** (+3 days of worst-case margin and
+37 points of capture for ~10 of precision), and at that point the pipeline is delivering **89% of
the warning that physically exists** on its hardest episode — after the log-linear projection and
the 3-day smoothing window, the remaining headroom is mostly in the physics, not the model.

---

## RUL — the margin, and whether it is safe to act on

The point estimate extrapolates the health index to a learned failure threshold (41.5σ, the
median index at a confirmed fault). **The functional form is the single most consequential choice
in this module, and it was changed on holdout evidence.**

A straight line — `(threshold − hi) / slope` — is systematically wrong: wear is convex, today's
slope understates tomorrow's, and the line arrives late. Measured across **four independently
seeded fleets**: it overstates remaining life by a **median of +8.7 days, on 77% of in-window
days**. That is the dangerous direction — it tells an engineer to defer when they cannot.

The default is now **log-linear**: treat the index as growing by a constant fraction per day
(`k = slope / hi`), so time-to-threshold = `ln(threshold/hi) / k`. Same holdout: **median error
−1.2 d, MAE 2.7 d — a 69% MAE reduction** from one change of functional form, with zero parameters
fitted to the 9 episodes. Below 1σ the log form falls back to the line rather than manufacturing a
confident countdown out of noise.

A near-unbiased point estimate is not a licence to trust it: the conformal layer stays, calibrated
leave-one-episode-out. It just no longer spends its entire correction patching a known bias.

### Choosing the conformity mode on evidence — and re-choosing when the evidence moves

| mode | held-out coverage | interval | sharpness | withdraw-zone safe | verdict |
|---|---|---|---|---|---|
| **additive (flat shift, −8.6 d)** | **87.8%** | **12.4 d** | 0.43 | **100%** | **selected** |
| ratio (constant multiple) | 85.6% | 12.6 d | **0.64** | 72% | fails near failure |
| mondrian (per volatility band) | 86.2% | 12.6 d | 0.63 | 72% | bands have converged |

This table is the third act of a story in which **every selection criterion is a scar**:

- **Coverage alone** once picked a vacuous bound — under the *linear* form, additive hit its
  coverage target by reporting ~0 days for everything (sharpness 0.00, kept as a pinned regression
  test).
- **Sharpness alone** then picked `ratio` under the *log-linear* form — whose bound overstates the
  margin on **28% of the final three days** and misses WITHDRAW on 2 of 9 episodes. Sharpness is
  measured mostly far from failure, which is exactly where it matters least.

The selection now holds coverage **and withdraw-zone safety (true RUL < 3 d, ≥95%)**, then takes
the sharpest survivor. A multiplicative bound scales *down* with the estimate, so near failure it
can still sit above a truth of hours; a flat shift crushes small margins to zero — exactly the
conservatism the final days deserve.

### The mondrian bands converged — and that is the interesting part

Under the linear form, mondrian's volatility bands fitted `×0.19 / ×0.32 / ×0.37` — strongly
stage-dependent, because they were patching the linear form's stage-dependent bias piecewise.
Under the log-linear form they fit `×0.54 / ×0.61 / ×0.52` — **near-identical**. The machinery
existed to correct a bias that no longer exists. It stays implemented, and re-arms the moment real
data shows stage-dependent bias again; on this data a constant shift is both safest and
sufficient.

Coverage is still measured **out-of-fold** — each episode bounded using only the others — because
a pooled quantile returns 1−α by construction. The selected mode sits at 87.8% against a 90%
target; stated, not rounded away.

### Is the bound safe when it matters?

| true RUL | n | bound safe | median bound |
|---|---|---|---|
| 0–3 d | 18 | **100.0%** | 0.0 d |
| 3–7 d | 36 | **100.0%** | 0.0 d |
| 7–14 d | 62 | 90.3% | 4.3 d |
| 14–30 d | 76 | 85.5% | 13.2 d |

Perfect inside the final week — the −8.6 d shift reads the last days of margin as zero, which
means the system **withdraws early rather than late**. That is a real cost stated as a choice: a
handful of trains come out of service a few days before strictly necessary, and in exchange no
failing door is left in passenger service past its margin. The weakest bucket is now 14–30 d
(85.5%), where an overstated bound mislabels a PLAN date, not a withdrawal.

A worked trajectory (TRN040-DOOR-1, roller wear) — the bound tracks below truth throughout, while
the raw projection starts at a capped 90 days against a true 27.8:

| day | health | slope | margin | lower bound | true |
|---|---|---|---|---|---|
| 2026-05-23 | 1.9σ | 0.29 | 90.0 d | **16.1 d** | 27.8 d |
| 2026-06-02 | 9.6σ | 0.63 | 41.2 d | **13.3 d** | 17.8 d |
| 2026-06-12 | 22.8σ | 1.30 | 9.9 d | **2.2 d** | 7.8 d |
| 2026-06-17 | 33.5σ | 1.81 | 1.2 d | **0.3 d** | 2.8 d |

**Caveat, stated plainly:** conformal coverage assumes exchangeability. Days within an episode are
strongly correlated and episodes differ, so per-day residuals are not exchangeable in the strict
sense. Leave-one-episode-out is the right fold unit, but with 9 episodes read `coverage_` as a
well-calibrated heuristic, not a proof.

---

## The aspect policy — does the order arrive in time, and stay quiet?

Thresholds are operational, not statistical, and they read `rul_lower` — never `rul_point`.

| threshold | aspect | why |
|---|---|---|
| < 1 day | 🔴 RED `WITHDRAW` | cannot be trusted to survive to the next nightly window |
| < 3 days | 🟡 AMBER `TONIGHT` | must go in tonight's ~2 usable hours, not be negotiated later |
| < 21 days | 🟡🟡 DOUBLE AMBER `PLAN` | inside the maintenance planning horizon — book it |
| otherwise | 🟢 GREEN `MONITOR` | including "no failure path", which is green, not unknown |

**Alert volume.** Across 9,600 asset-days: 97.6% GREEN, 1.20% PLAN, 0.14% TONIGHT, 1.03% WITHDRAW
— **0.9 doors per night** need tonight's window. A ~2-hour window supports roughly 2–4 door jobs,
so the nightly list is short enough to actually be worked.

**Quiet on healthy doors.** Of 71 healthy doors, escalation above GREEN on 0.15% of asset-days, and
**zero healthy doors ever reach AMBER or worse** — no false "go on track tonight".

**Warning delivered**, days before the confirmed fault that each aspect first appears:

| | PLAN | TONIGHT | WITHDRAW | available |
|---|---|---|---|---|
| median | 21.1 d | 9.0 d | 7.1 d | 31.5 d |
| worst | 11.5 d | 6.5 d | 4.9 d | 19.0 d |

Every episode reached PLAN with a median 21 days in hand — time to *book* the work rather than
react to it — and after the log-linear projection, **all nine episodes now reach WITHDRAW, at a
worst case of 4.9 days**. Before the projection fix, two episodes never reached WITHDRAW at all
and the worst warning was 0.8 days; the escalation chain is no longer the weak link.

### Confidence means "is this order robust", not "is this number precise"

Two earlier definitions were discarded for producing nonsense, and both are pinned by regression
tests:

1. **Relative interval width** labelled a RED card *LOW confidence* — near failure the margin is
   ~1 day, so a ±1 day interval reads as 100% error even though the decision isn't in doubt.
2. **Comparing the aspect at each end of the interval** relies on the upper bound, and the upper
   bound is not trustworthy: the ratio conformity score is `true / projection`, which explodes as
   the projection approaches zero. The fitted upper multiplier reaches **×4.7**.

The lower bound is the calibrated, well-behaved end and the only one the product acts on, so
confidence compares the aspect from the raw projection against the aspect from the bound. HIGH =
the call doesn't depend on the uncertainty model at all. Distribution across escalated cards:
97 HIGH, 66 MEDIUM, 46 LOW.

### Hysteresis

Aspects escalate immediately and de-escalate only after 3 consecutive quieter days. Measuring this
*inside* degradation windows shows no effect — the smoothed index is near-monotone there, so there
is nothing to damp, and an early version of this README wrongly called the feature inert on that
basis. Measured across all asset-days it holds **28** at an elevated aspect the raw margin would
have dropped. That is the flapping it exists to prevent.

---

## The Deferral Ledger — what does it cost to wait?

Every other output in this repo tells an engineer what is wrong. This one tells them **what it
costs to wait** — which is the question actually asked at 01:00, when there are two usable hours
and three jobs.

```
TRN023-DOOR-2 · DOUBLE AMBER · PLAN     what does it cost to wait?
    act tonight             0%
    wait 3 days             7%
    wait a week            17%
    wait a fortnight      HIGH
    wait a month          HIGH
    ──────────────────────────
    latest date under 10% risk: 5 days
```

Six doors can carry the *same* PLAN order and have completely different prices — on 2026-06-12
their safe-until dates are 14, 14, 13, 9, 5 and 4 days. An aspect alone cannot express that.

**Where the numbers come from.** `ConformalRUL` already yields a bound `L(α)` with
`P(RUL ≥ L(α)) ≈ 1−α`. Fit the *same* machinery at six confidence levels and `L` becomes a curve —
a discrete survival function. Then read it backwards: instead of *"how many days at 90%?"*,
interpolate α at the horizon. No new model, no distributional assumption; every point is a
conformal bound whose coverage is measured out-of-fold.

A consequence worth noticing: `latest_date_under(0.10)` **is** `L(0.10)` — the margin the Aspect
Card already prints. The ledger doesn't bolt on a new idea, it exposes one that was already there.

### Does 14% actually mean 14%?

A risk figure nobody validated is worse than no figure. Grouping every scored asset-day by the risk
we stated, then asking what fraction actually failed in time:

| region | asset-days | mean \|stated − actual\| | how it is reported |
|---|---|---|---|
| **below 30%** | 422 | **8.1%** | quoted as a percentage |
| at or above 30% | 304 | 32.5% | reported as `HIGH` |

**So the ledger prints a number only where it has earned one.** *"Can this wait a week?"* is only
ever asked of assets that plausibly can, and there we track outcomes to within five points. Above
30% the answer is "do not defer" whether the truth is 45% or 70% — so `RISK_CEILING` prints `HIGH`
instead. Quoting 53% when we are ~35 points out would cost the product its credibility the first
time an engineer checked it.

### The mistake that reached the screen

This was first built on `ratio` conformity rather than the `additive` mode the Aspect Card uses,
reasoning that additive clamps 38.8% of bounds to zero and flattens 11.6% of curves — an
unaffordable loss of resolution. **That reasoning was wrong twice, and the result was a card that
contradicted itself:**

```
RED · WITHDRAW      margin 1 day
    ...
    latest date under 10% risk: 5 days
```

Both numbers were valid conformal bounds. They disagreed because they came from different modes —
by a median of 2.5 days and **up to 32**. An engineer reads that once and stops believing both.

Measured properly, the case for `ratio` evaporated:

| mode | card ↔ ledger disagreement | PLAN rows quotable @7d | median spread |
|---|---|---|---|
| `ratio` | median 2.5 d, **max 32.2 d** | 92.2% | 0.45 |
| **`additive`** | **0.00 d — exact** | 92.2% | 0.45 |

**Discrimination is identical.** The clamping concentrates *near failure*, where every horizon
reading HIGH is the correct answer, not lost information — so the statistic that motivated the
choice was measuring the wrong population. Unifying the modes cost 3 points of calibration in the
quoted band (4.9% → 8.1%) and bought back a product that speaks with one voice. The identity is
now asserted by a test rather than assumed.

---

## The Proving Ground — which model earns the right to make the call?

PS3 asks us to *"identify the best models to be used to detect future anomalies"*. This answers it.

**Protocol.** Every detector is fitted on the **first 30 days only**, then scores the full 120-day
history. No labels, no sight of the future, and identical smoothing on every score. This matters:
a model fitted on the whole record has already seen the failure it is being congratulated for
predicting, and every lead-time number it produces is fiction.

Each family runs **twice** — once on raw per-cycle aggregates, once on condition-normalised,
per-asset baselined features — because the question worth answering is not which library wins.

### Budget 150 asset-day alerts

| detector | found | median | worst | precision | capture |
|---|---|---|---|---|---|
| PCA reconstruction `[raw]` | 9/9 | 15.0 d | **9.5 d** | 90.0% | 36% |
| PCA / Mahalanobis / LOF `[norm]` | 9/9 | 14.1 d | **9.5 d** | 90.7% | 32% |
| **HEADWAY univariate** | 9/9 | 13.1 d | **9.5 d** | **90.7%** | **36%** |
| **HEADWAY multivariate** | **9/9** | **22.1 d** | **13.5 d** | **86.8%** | **85%** |
| Mahalanobis `[raw]` | 9/9 | 15.1 d | 8.5 d | 88.0% | 36% |
| isolation forest `[norm]` | 9/9 | 14.1 d | 8.5 d | 90.7% | 36% |
| isolation forest `[raw]` | 9/9 | 8.9 d | 4.1 d | 54.0% | 11% |
| EWMA control chart `[raw]` | 9/9 | 9.8 d | 2.5 d | 61.3% | 13% |
| raw signal, fleet-wide | 7/9 | 12.2 d | 3.5 d | 56.0% | 18% |

At 150 alerts the top of the field is saturated — six detectors tie at 9.5 d worst case. The
budget, not the model, is the binding constraint. Raising it to 250 separates them.

### Budget 250 asset-day alerts

| detector | found | median | worst | precision | capture |
|---|---|---|---|---|---|
| **HEADWAY multivariate** | 9/9 | **22.1 d** | 12.5 d | **85.2%** | **60%** |
| HEADWAY univariate | 9/9 | 21.1 d | **13.5 d** | 81.6% | 76% |
| PCA reconstruction `[norm]` | 9/9 | 21.1 d | 12.5 d | 77.6% | 76% |
| isolation forest `[norm]` | 9/9 | 21.1 d | 12.5 d | 76.4% | 74% |
| Mahalanobis / LOF `[norm]` | 9/9 | 21–22 d | 12.5 d | ~75% | 74% |
| PCA reconstruction `[raw]` | 9/9 | 16.0 d | 11.5 d | 65.2% | 39% |
| EWMA control chart `[raw]` | 9/9 | 14.1 d | 7.5 d | 59.2% | 34% |
| raw signal, fleet-wide | 7/9 | 12.2 d | 3.5 d | 36.0% | 18% |

### Algorithm or preprocessing — which is doing the work?

| family | raw worst | norm worst | delta |
|---|---|---|---|
| isolation forest | 4.1 d | 8.5 d | **+4.4 d** |
| PCA reconstruction | 9.5 d | 9.5 d | +0.0 d |
| Mahalanobis distance | 8.5 d | 9.5 d | +1.0 d |

Switching the *features* moves worst-case warning by a mean of +1.8 days; switching the *model
family* with features held fixed moves it by 1.0 days. But the mean hides the finding: normalisation
is worth **+4.4 days to the isolation forest and +0.0 to PCA** — because PCA already performs its
own implicit normalisation. Temperature and crowding move every signal together, so those directions
land in the leading principal components and drop straight out of the reconstruction error.

**Condition normalisation is a substitute for model capacity, not an addition to it.** It buys the
most for models that cannot discover the structure themselves. That is a more defensible claim than
"preprocessing beats models", and it is what the numbers actually support.

### What the tournament corrected — in us

An earlier run of this tournament reported that Headway placed mid-field, beaten by six detectors,
and that finding was written into this README and the published brief. **It was not a result. It
was a bug in the harness.**

The Headway entry was fed an already-smoothed column, and `run()` then applied the common smoothing
on top — so our own model, alone in the field, was smoothed twice and handicapped by roughly two
days of worst-case warning. Fixed, both Headway variants sit at the top of the field, and the
earlier finding is withdrawn.

The lesson generalises past this repo: **an evaluation that produces a surprising verdict about
your own model is more likely to be measuring your harness than your model.** Audit the protocol
before rebuilding the model.

### Univariate or multivariate — a genuine trade-off

The multivariate index orients each signal by the contract's declared direction (a worn door draws
*more* current but travels *less* far — combining raw directions cancels real evidence), whitens
against the reference covariance so four correlated current measurements are not four independent
votes, then projects onto the direction in which all oriented signals rise together. It stays a
single scalar in sigma units, so RUL, the aspect thresholds and the Aspect Card kept working
unchanged — the failure threshold simply relearned itself from 35.6σ to 37.8σ.

At budget 250: multivariate wins precision (+4.8 points) and capture (+9 points); univariate wins
worst-case warning by 1.0 day — **which is the metric we said to lead on.**

So this is a trade-off, not a clean win, and it is worth saying so. We ship multivariate anyway,
for a reason the table cannot show: on unseen data a univariate index is blind if its one sensor
drifts or fails. Spending one day of worst-case margin to stop depending on a single channel is
the right trade against a dataset we have not seen yet.

---

## The UI

`scripts/build_ui.py` emits **one self-contained HTML file** — no server, no CDN, no build step —
with the fleet's actual aspect history embedded. Every number on screen came out of the pipeline,
and the date scrubber replays the real escalation of real episodes rather than a scripted
animation. 336 KB, 80 doors × 120 days.

- **Key** — the four aspects explained inline, so the language is learnable at a glance.
- **Tonight's window** — Aspect Cards that cannot wait, fresh calls ranked above held ones, with
  *margin* as the hero number because it is the thing being decided on.
- **Fleet** — 80 tiles, filterable by aspect. Escalated doors are tinted and carry their margin;
  healthy ones read `ok`. The 5 doors that matter are findable in a glance across 80.
- **Detail** — click any door for its full health-index history, with the aspect in force drawn as
  a band beneath the curve and the selected day marked.

### Reading the four aspects apart

Light theme is the default; a toggle switches to dark. Two deliberate choices make the aspects
distinguishable:

**Four hues, not two.** On a real signal head PLAN and TONIGHT are *both yellow*, told apart by how
many lamps are lit. Rendered literally on screen that reads as "two ambers that look the same" —
exactly the wrong ambiguity for the two most consequential states. So PLAN is gold, TONIGHT is
orange, WITHDRAW is red.

**The lamp count is kept anyway.** Every aspect shows a two-lamp signal head; DOUBLE AMBER lights
both, the others light one. So the aspects stay separable in greyscale, on a projector, or for a
colour-blind viewer — colour is never the only channel.

Lamp colours are vivid (they are 9px dots and must read as colour); text colours are darker
variants that clear 4.5:1 contrast on white. Those are separate tokens for that reason.

### One defect the UI surfaced that the tests had not

The first render produced a card reading **"RED · WITHDRAW — margin 26.7 days — health −0.1σ"**:
a withdraw-from-service order sitting beside evidence that the door was healthier than its own
baseline. An engineer who sees that once stops believing the next card.

The cause was presentational, not logical. When a door is repaired the health index falls from
~40σ to ~0 overnight, so the margin becomes large but **finite** — and `AspectCard.held` only
tested for an *infinite* margin. Hysteresis was correctly holding the RED for its dwell; the card
just had no way to say so.

The right test is whether the displayed aspect outranks the aspect today's margin would earn on its
own. Held cards now carry a `HELD` chip, a dashed rule, a muted headline, and the line *"today's
reading alone would be GREEN — the order stands until the signal clears its dwell, or an engineer
confirms the repair."* That last clause is also the argument for the Logbook: a human confirming
the repair should clear it immediately.

---

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

164 tests, ~31s. They assert the properties the *pitch* depends on, not just that the code runs —
if the synthetic data stops being adversarial, or a window starts peeking at the future, the
argument on stage quietly becomes false and a test should fail first.

The load-bearing ones:

- `test_confounds_dominate_early_wear` — the core premise. Measured as a difference-in-differences
  against the healthy fleet, because a raw before/after comparison is itself confounded by seasonal
  drift. (The product's thesis, appearing inside its own test suite.)
- `test_asset_is_repaired_after_confirmed_fault` — no zombie assets running elevated forever.
- `test_trend_windows_are_trailing_and_cannot_see_the_future` — truncate the data at day *t* and
  every trend value at *t* must be unchanged. Without this, every lead-time number is inflated.
- `test_add_trend_is_order_independent` — regression test for a silent misalignment bug.
- `test_budget_is_spent_exactly` / `test_tied_scores_do_not_inflate_the_budget` — regression tests
  for unequal alert budgets under ties (observed: 152 alarms vs 150).
- `test_alerts_after_the_fault_are_not_credited_as_warnings` — telling someone a train has already
  failed is not a warning.

Three real defects were found by writing these, all now fixed: the tie-breaking budget leak, the
`add_trend` misalignment, and degradation onset being able to precede the start of the data
(leaving an asset with no healthy period to baseline against).

---

## Design rules

**The contract is the interface.** Nothing downstream of ingest ever sees a vendor column name.
On the night, `headway/adapters/nebulax.py` maps their schema onto ours and nothing else changes.

**Signal vs context is the thesis, not bookkeeping.** We never score a raw signal. We model
`E[signal | context]` and score the residual, per asset.

**Never report accuracy.** At a base rate of 2e-05, predicting "healthy" every time scores
99.998%. The harness reports lead time, precision at the true base rate, calibration, and alert
budget consumed — the things that decide whether a recommendation is worth acting on.

**The synthetic ground truth (`data/door_episodes.csv`) is never shown to a model.** It exists to
build and check the evaluation harness.
