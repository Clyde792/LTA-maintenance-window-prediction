# Headway — technical overview

A map for reading the codebase. Written 10 Sep 2026, after the September audit.
Everything here concerns the **synthetic prototype**; no claim is made about a
real railway.

---

## 1. The problem, and the shape of the answer

PS3 asks for anomaly detection on train telemetry. But Singapore's network is
already reliable, and the binding constraint is the **nightly engineering
window** — roughly two usable hours. So detection is the easy half. The hard
half is turning a changing signal into *"can this wait until Thursday, or does
someone go now?"* — a decision an engineer can review and defer.

Headway is a pipeline that produces, per door per day: an **action**, a **lower
bound on remaining days**, an explicit **data state**, and a **window
comparison** (does a proposed maintenance date fall inside or outside that
bound). It never emits a failure probability, because a handful of episodes
cannot calibrate one.

The centrepiece is not the model. It is the **evaluation apparatus** — built
first, and it caught four wrong beliefs during the build and a 74% generalisation
gap during the audit. That is what makes the numbers trustworthy, and it is the
thing to understand first.

---

## 2. The pipeline, stage by stage

Data flows: **cycles → contract → normalise → per-asset baseline → multivariate
index → trend → RUL → aspect policy → fit-for-duty → window comparison →
dashboard**. Each stage is a component with its own tests.

### 2.1 The Feature Contract — `headway/contract.py`

One canonical schema (identity / signal / context / label) that every downstream
module speaks. Real vendor data binds through **one adapter file**
(`headway/adapters/nebulax.py`); nothing else changes.

- **Design choice:** the signal/context split *is* the modelling thesis. Signals
  degrade; context (temperature, crowding, hour) is what we normalise *against*.
- **Why it matters:** on Friday 18 Sep, binding real data is a 30-minute mapping
  exercise, not a refactor. `scripts/bind_data.py` even guesses the mapping.
- **Weakness:** the guessed column names and the bogie signal list are
  literature-based assumptions. If the real schema is very different, the adapter
  gets bigger.

### 2.2 Condition normalisation — `headway/normalise.py` (`ConditionNormaliser`)

Models the *expected* reading given operating conditions, and scores the
**residual**. On this data a hot or crowded run moves the primary signal ~3.8×
more than three-weeks-out wear — so thresholding the raw signal ranks the
weather, not the fault.

- **Design choice:** ridge regression, fitted on a frozen historical reference
  window only. Imputation values are frozen at training.
- **Weakness:** it is linear-plus-interactions. A real confound that is
  non-linear in an un-modelled way would leak into the residual. `load_proxy`
  (crowding) is itself a proxy that needs validation on real data.

### 2.3 Per-asset baseline — `headway/normalise.py` (`AssetBaseline`)

Each door has its own build tolerance. This subtracts each asset's own reference
median and divides by its robust scale, giving a **health index in sigma units**
— comparable across doors, fleets and subsystems.

- **This is the single biggest contributor** to detection quality: it is what
  takes the tournament from "misses 2 of 6" to "finds all 6".
- **Weakness — and it is the important one:** the reference window is assumed
  healthy. For a train Headway has never seen, that assumption is contaminated,
  and coverage drops to 74% (see §3). The onboarding adapter (§4) is the fix.

### 2.4 Multivariate health index — `headway/normalise.py` (`MultivariateHealthIndex`)

Combines the per-signal health indices into **one directional scalar** — oriented
so every signal points the same way as wear, whitened so correlated current
measurements don't count as four independent votes, projected onto the
"everything worsens together" direction. Also emits a separate *unsigned* anomaly
distance and per-channel contributions.

- **Design choice:** directional, not a plain Mahalanobis distance, because RUL
  needs a quantity that rises *with wear and only with wear*.
- **Finding:** once any decent model gets these features, the models cluster
  within a few points of each other. **The preprocessing does the work, not the
  algorithm.** (`scripts/run_tournament.py`, `data/tournament_replay.csv`.)

### 2.5 Trend features — `headway/features.py` (`add_trend`)

Trailing rolling median (smoothing), least-squares slope in sigma/day, and
volatility — all on **elapsed calendar time**, so a data gap compresses rather
than distorts the fit. Daily aggregates become available at midnight *after* the
telemetry day; nothing is credited before then.

- **Weakness:** the smoothing window (3 days) was tuned on a holdout of 4
  simulator seeds — better than tuning on one, still not real data. Dispersion
  (IQR) features carry no signal *in this synthetic data* because the generator
  scales the mean of each signal with wear and leaves the variance alone; real
  worn mechanisms do show a dispersion signal, so this is a blind spot, not a
  dead end.

### 2.6 Remaining useful life — `headway/rul.py` (`ConformalRUL`)

Projects the health index forward to a learned failure threshold, then wraps it
in an **empirical lower bound** calibrated leave-one-episode-out.

- **The projection is log-linear**, not a straight line. A straight line
  overstates remaining life by a median of +8.7 days (wear is convex); the log
  form cut point error by 69% with nothing fitted to the episodes.
- **Explicit states**, not silent failure: `insufficient_history`,
  `no_worsening_trend`, `elevated_no_trend`, `threshold_exceeded`,
  `outside_horizon`, `stale_data`, `outside_training_conditions`. A missing
  prediction is *never* "infinite life / green / zero risk".
- **Weakness — stated plainly in the module docstring:** correlated daily rows
  and cross-fitted residuals do **not** have the standard split-conformal
  finite-sample guarantee. The bound is an *empirical* group-coverage statistic.
  On the chronological split its median value is **0 days** — honest, but often
  too conservative to schedule against.

### 2.7 Aspect policy — `headway/aspect.py` (`AspectPolicy`)

Maps the lower bound to GREEN / DOUBLE AMBER / AMBER / RED via four day
thresholds (1 / 3 / 21), with hysteresis (escalate fast, de-escalate slow).
**Kept in its own file on purpose** — the recommendation can change without
retraining anything, and an engineer can read and argue with the thresholds.

- **Weakness:** the thresholds are illustrative demo policy. Real ones are an
  operator decision informed by what a false alarm actually costs in the window
  — a site-visit question.

### 2.8 Window comparison — `headway/deferral.py` (`DeferralLedger`)

Given a proposed maintenance date, returns a label: *Within estimated margin /
Exceeds estimated margin / No positive margin established / Threshold exceeded /
Cannot assess*. It consumes the **exact** bound the aspect card shows, so the two
can never contradict each other.

- **History worth knowing:** this was originally a "survival curve" that priced
  any wait as a calibrated probability. The audit removed it — a lower bound on
  9 episodes is not a distribution, and printing "14% risk" invited a
  quantitative decision on a number we could not defend. What remains is honest
  and still useful: *this proposed window is / is not inside the estimated
  margin*.
- **Weakness:** it is now a labelling function, not a headline feature. If a
  "wow" differentiator is needed, this is where a *defensible* one would be
  rebuilt.

### 2.9 Fit-for-duty — `headway/duty.py` (`DutyAssessor`)

The aspect answers *when* to act. Duty answers *how to run the door meanwhile*:
all service, off-peak only, or withdraw. It exists because of an operational
claim from rail practitioners that the modelling had no way to express: a worn
mechanism is not just worse on average, it is disproportionately worse **under
load** — it holds together at 14:00 and fails at 08:15.

- **How it is measured.** The fleet normaliser already removes the load effect a
  healthy door shows. What remains for a worn door is an *excess* sensitivity:
  the cycle-level index rises with crowding more than the fleet model expects.
  The pipeline keeps six sufficient statistics of (load, index) per asset-day;
  their trailing 3-day sums give the pooled within-asset regression exactly. The
  slope must be positive and significant (t ≥ 2.5) before it is used at all.
- **What is done with it.** The index is shifted to the reference **peak load**
  (the 90th-percentile crowding inside the peak bands, which are themselves
  derived from the reference load profile, not hard-coded), and the card's own
  RUL model and aspect policy are re-run on the shifted index. The label is a
  **level rule**: `off_peak_only` when the peak-conditioned index is at or beyond
  the failure level while the day average is not; `withdraw` when the day average
  itself is. Duty is never a relaxation of the aspect.
- **Why the window is 3 days.** A first version used 7 and produced restrictions
  on freshly repaired doors — the sensitivity window straddled the depot visit
  while the level had already dropped. Aligning it with the index smoothing
  window removed that; `tests/test_duty.py` pins it.
- **Why a level rule, not a margin rule.** Re-running the projection at peak
  load changes the lower bound by less than a day near failure, and the bound is
  already 0 there; as an aspect escalation the feature was vacuous. The level
  comparison is where the peak signal actually is. On the chronological holdout
  (trained before 24 June, six future episodes) all six doors are restricted
  before the fault, median 5.6 days ahead, with 0 of 2,654 healthy asset-days
  restricted; in the retrospective demo 6 of 9, median 3.5 days. The
  restriction is contiguous and runs straight into *withdraw* — a first version
  flickered on and off, which is what killed the margin rule.
- **Weakness, stated plainly:** the wear-load interaction is a *simulator
  assumption* (`load_wear_coeff` in `headway/synth/doors.py`), chosen from
  operator experience, not measured. The peak projection reuses the fleet
  calibration on a shifted index and is not separately calibrated. And the
  generator's faults are not load-triggered, so nothing here shows that running
  a door off-peak *extends* its life — only that its condition at peak load has
  reached the level at which doors failed. On real data this is the first
  claim to test at the SMRT depot: do worn doors show a steeper current-vs-load
  slope than healthy ones?

### 2.10 The dashboard — `scripts/build_ui.py` → `ui/headway.html`

One self-contained HTML file, no server. Every figure comes from the pipeline.
Light theme, explicit unknown states, "decision stability" (do the point
estimate and the bound agree on an action) rather than "confidence", chart gaps
preserved, labelled a retrospective synthetic demonstration.

---

## 3. The evaluation apparatus — why it is the centrepiece

`scripts/validate_pipeline.py` → `data/validation_report.json`. Two protocols:

**Chronological (train on the past, test on the future).** Freeze training on
faults confirmed by 24 June; test on later observations.

| | result | read alongside |
|---|---:|---|
| Future episodes found | 6 / 6 | — |
| Worst-case warning | 6.05 d | — |
| Alert precision | 93.5% | 0.02 false alerts per asset-month |
| RUL point error | 3.9 d MAE | — |
| Lower-bound coverage | 94.0% | median bound is **0 days**; only 33% positive |
| Abstention on degradation rows | 8.4% | — |
| Restricted to off-peak before the fault | 6 / 6 | median 5.6 d ahead; 0 of 2,654 healthy asset-days restricted |

**Whole train held out (stricter).** Refit *everything* with an entire train
removed, for each of 9 fault trains.

| | result |
|---|---:|
| Lower-bound coverage | **74.2%** (target 90%) |
| RUL point error | 9.0 d MAE |
| Abstention | 37.5% |

The stricter test **misses the target**, and the dashboard says so.

**The tournament** (`scripts/run_tournament.py`) does not crown a winner: under
chronological replay with a frozen threshold, the raw fleet threshold misses 2
of 6 and every normalised model finds all 6 and clusters together.

**Four corrections during the build** (each with its measurement, in
`AUDIT_FIXES.md` and the git history):

1. Alert budget counted per cycle → *inverted* the result. Unit is asset-days.
2. Tournament harness double-smoothed our own model → it "lost". Fixed → it leads.
3. Condition models fitted on all 120 days → look-ahead in our favour. Refitted
   on 30 → every headline survived.
4. Straight-line RUL projection optimistic by +8.7 d → log-linear, error −69%.

---

## 4. The generalisation gap and the fix — `headway/onboarding.py`

The 74% held-out coverage has one cause: an unseen train's per-asset baseline is
estimated from contaminated data. A 21-day per-asset reference adapter (fleet
models stay frozen) halves point error (9.0 → 4.6 d) and lifts coverage to
87–90%. **It replicates on a fresh simulator seed** (`ONBOARDING_RESULTS.md`).

It is **not in the demo** — promoting it needs inspection evidence that a door's
first 21 days were healthy, or the adapter absorbs existing degradation. That is
an operator question and a site-visit question.

---

## 5. Honest ledger — pros and cons

### Genuine strengths

| | why it holds up |
|---|---|
| **The decision framing** | Output is an action + a reviewable margin + a window comparison, not a score. Directly answers the constraint (the 2-hour window) that PS3's own framing names. |
| **The evaluation apparatus** | Two real holdout protocols, chronological + train-level, with abstention and positive-margin frequency reported *beside* coverage. Most teams will report accuracy on a base rate where "healthy always" scores 99.998%. |
| **Condition normalisation** | Measured 3.8× confound-to-signal ratio; the residual approach is what makes early warning possible at all. |
| **Explicit unknown states** | A missing or unprojectable observation is a named state, never a silent green. This is the property a practising reliability engineer checks for first. |
| **The audit trail** | Four documented corrections where the evidence beat our intuition. Credibility, not weakness, in front of engineers who have been burned by confident CM systems. |
| **Binding cost** | One adapter file. Friday night is a mapping exercise. |

### Real weaknesses, ranked by how much they matter

| | severity | status |
|---|---|---|
| **74% held-out coverage** vs 90% target | high | fix exists (onboarding), not shipped; needs operator input |
| **Median lower bound = 0 days** on the chronological split | high | honest but often unschedulable; the bound is conservative by construction |
| **9 synthetic episodes** | high | nothing about generalisation is established; the effective sample size is *groups*, not daily rows |
| **No calibrated probability** | medium | deliberate — but it means the "wow" deferral feature is now a label |
| **Conformal guarantee does not hold** | medium | correlated rows + cross-fitting; stated in the docstring; the empirical evaluation is the fallback |
| **Thresholds are demo policy** | medium | by design; needs an operator |
| **Bogies undemonstrated** | medium | contract defined, no data, no adapter |
| **Wear-load interaction is assumed** | medium | fit-for-duty rests on a simulator coefficient; first thing to test on real data |
| **Dispersion features are a blind spot** | low | synthetic-data limitation, not a doors fact |
| **Tonal inconsistency in docstrings** | cosmetic | some modules still carry pre-audit verbose docstrings |

---

## 6. Reading path

**30 minutes — the shape.** `SUBMISSION.md` (the narrative), then this file,
then open `ui/headway.html` and scrub the date.

**1 hour — the technical core.** In dependency order:
`headway/contract.py` (skim the docstring) → `headway/normalise.py` (all three
classes) → `headway/rul.py` (the docstring and `_project` / the state machine) →
`headway/aspect.py` → `headway/duty.py` → `headway/deferral.py` →
`headway/pipeline.py` (how they compose).

**1 hour — the evaluation.** `headway/evaluate/metrics.py` docstring →
`scripts/validate_pipeline.py` → `data/validation_report.json` →
`AUDIT_FIXES.md` → `ONBOARDING_RESULTS.md`.

**Reference, not front-to-back:** `headway/models/detectors.py`,
`headway/synth/doors.py` (how the test data is built and why it is adversarial),
`tests/test_audit_fixes.py` (the regression cases pin exactly what was fixed).

**Do not treat as authoritative:** `scripts/diagnose.py`,
`scripts/holdout_check.py`, `scripts/check_premise.py` — labelled legacy
exploration; superseded by `validate_pipeline.py`.
