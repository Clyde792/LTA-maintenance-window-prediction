# Devpost submission copy — Headway

**Status: draft, prepared 31 Aug 2026.** Everything here is written against the synthetic
pre-build. Placeholders marked `[[…]]` are the things that only exist after Friday 18 Sep —
fill them in before the 16:00 Saturday deadline and delete this banner.

Gate one is judged on Saturday night **without us in the room**. This page carries the whole
argument on its own. Written to be read top-to-bottom in four minutes by someone who has already
read twenty other submissions.

---

## Project name

**Headway**

## Tagline

Time to act.

## Elevator pitch (Devpost caps this at ~200 characters)

> Trains announce their failures weeks early. Headway turns that whisper into an order: how many
> days you have, how confident we are, and whether it can wait for tonight's engineering window.

---

## Inspiration

Singapore's rail network is already very reliable — MKBF runs around 1.7 million train-km against
a 1 million target. So the problem is not "we cannot find failures". Naive detectors find them.
The problem is that they find them **too late to choose**.

The constraint everything in LTA's world bends around is the engineering window: the network
closes roughly 00:00–05:30, but once equipment is deployed and the track is prepared, crews get
**barely two usable working hours**. Every night, someone decides which handful of jobs fit.

An anomaly score does not help with that decision. It says *something changed*. It does not say
whether this can wait until tonight, until next week, or whether a train needs to come out of
service now. That gap — between a score and an order — is the entire product.

We also took the second half of PS3 literally. It asks us to **"identify the best models to be
used to detect future anomalies."** That is an evaluation question, and we treated it as the
harder and more interesting one.

## What it does

Headway takes per-cycle door and bogie telemetry and gives every asset a **signal aspect** — the
four-aspect language railways already use, so no one has to learn a new vocabulary:

| Aspect | Order | Meaning |
|---|---|---|
| 🟢 GREEN | `MONITOR` | Healthy. Keep watching. |
| 🟡🟡 DOUBLE AMBER | `PLAN` | Book into an upcoming engineering window. |
| 🟡 AMBER | `TONIGHT` | This window. It cannot wait for the next one. |
| 🔴 RED | `WITHDRAW` | Take it out of service now. |

Each aspect comes with an **Aspect Card**: a margin in days, a confidence, and the reasoning in
plain English — *"26.6σ above this door's own baseline, rising 1.42σ/day, already normalised for
30 °C and 33% loading."*

Three things make that margin trustworthy rather than decorative:

**It is normalised for operating conditions.** On our data, a hot afternoon or a crowded peak-hour
run moves the primary signal **4.6× more than three-weeks-out wear does**. Threshold the raw
signal and you are ranking the weather. Headway models `E[signal | conditions]` and scores the
residual, per asset — so a stiff healthy door never outranks a genuinely degrading one.

**It is a calibrated lower bound, not a point estimate.** The recommendation reads the bound, never
the projection. Two doors with the same projected margin get different orders if one is understood
less well.

**It knows when it is late.** Every alert is measured against the warning that was *physically
available* on that episode, not the warning we would like to claim.

## How we built it

A five-stage pipeline, each stage a component with its own tests:

```
contract → condition normalisation → per-asset baseline → multivariate health index
        → conformal RUL → aspect policy → operator UI
```

- **The Feature Contract** (`headway/contract.py`) is the keystone. Everything downstream speaks
  one canonical schema — identity, signals, operating context, sparse labels — so real data binds
  through a *single adapter file*, not a refactor. The signal/context split is the modelling
  thesis, not bookkeeping.
- **Condition normalisation + per-asset baselining** produce a unit-free `health_index` in sigma,
  comparable across doors, fleets and operators — the standardised condition-monitoring parameter
  the Feb 2026 Rail Reliability Taskforce asked for.
- **Conformal RUL** projects to a learned failure threshold and wraps it in a lower bound
  calibrated leave-one-episode-out.
- **The aspect policy** lives in its own file, deliberately: what Headway *recommends* can be
  changed without retraining anything, and the thresholds are readable enough for an engineer to
  argue with.
- **The Proving Ground** runs nine detectors under one protocol — each fitted on the first 30 days
  only, then scoring the full history, with identical smoothing.
- **The UI** is a single self-contained HTML file. No server, no CDN. Every figure on screen came
  out of the pipeline, and the date scrubber replays real episodes rather than an animation.

## Challenges we ran into

Honestly, the hardest problem was **not trusting our own results**. Four times the evidence
overturned something we believed, and each correction is in the repo with the measurement that
caused it:

**Our metric was measuring the wrong unit.** We counted alert budget per *cycle*. A smoothed score
stays elevated for hundreds of consecutive cycles, so one sick door consumed an entire budget and
a spiky raw baseline won by accident — the result was inverted. An engineer gets one alert per door
per day; **asset-days** is the honest unit.

**Our tournament said we were losing.** Headway placed mid-field, beaten by six detectors. It was
a bug in our harness — our entry was fed an already-smoothed column and the runner smoothed it
again, so our model alone was smoothed twice. Fixed, it leads the field. The lesson generalises:
*an evaluation returning a surprising verdict about your own model is more likely measuring your
harness than your model.*

**Then we found the inverse flaw — one in our favour.** The condition models feeding the
normalised detectors were fitted on all 120 days, giving them a look-ahead the raw detectors never
had. Refitted on the first 30 days only, every headline number survived unchanged. The claim now
stands on a clean protocol rather than a hopeful one.

**Our uncertainty model was patching the wrong problem.** A straight-line projection overstates
remaining life by a median of **+8.7 days on 77% of days** — because wear is convex and today's
slope understates tomorrow's. We had been spending the entire conformal correction covering that
bias. Switching to a log-linear form cut the point estimate's error by **69%**, and downstream the
volatility-band machinery we had carefully built converged to near-identical values: it had only
ever existed to patch the bias the form change removed.

## Accomplishments that we're proud of

**We answered PS3's second clause with a number, not an opinion.** Nine detectors, one protocol,
equal alert budget. The finding is sharper than "our model wins": condition normalisation is worth
**+4.4 days to an isolation forest and +0.0 to PCA** — because PCA already performs its own
implicit normalisation. *Normalisation is a substitute for model capacity, not an addition to it.*

**We measured the ceiling, not just the score.** At onset the wear is exactly zero, so roughly
**26% of every degradation window is undetectable** no matter the model. Measuring against the
full window flattered our headroom by a third. Our capture metric now divides by what is
physically detectable — which is how we can say the pipeline delivers **89% of the warning that
actually exists**, and mean it.

**The system is quiet.** 97.6% of asset-days are GREEN. Of 71 healthy doors, **zero ever reach
AMBER or worse**. Roughly **0.9 doors per night** need the engineering window — a list short enough
to actually be worked.

**It warns in time to plan.** Every episode reaches PLAN with a **median 21 days** in hand, and all
nine reach WITHDRAW with a worst case of **4.9 days**.

**147 tests**, asserting the properties the argument depends on — that no window can see the
future, that every detector gets exactly the same budget, that a bound never overstates the margin,
that a healthy door is never handed a countdown.

## What we learned

That the interesting question in predictive maintenance is rarely the model.

We can now show that swapping the **model family** moves worst-case warning by about a day, while
swapping the **features** moves it by four — and that the single largest improvement we made all
weekend was changing a straight line to an exponential, which is physics, not machine learning.

We also learned to distrust single-metric selection, the hard way, twice. Coverage alone chose a
bound that was perfectly safe and completely useless. Sharpness alone then chose a bound that
fails on 28% of the final three days before a failure. Every criterion in our selection rule is a
scar from a measured mistake.

## What's next

- **Possession Planner** — pack the aspects into tonight's two usable hours, and show what a false
  alarm costs in window-minutes.
- **Logbook** — one-tap confirm/overrule, so engineer judgement becomes training data. It also
  clears a held order instantly, which is the right answer to a repaired door.
- **Earshot** — sonified door cycles. Veterans diagnose by ear; the system should too.
- **Bogies.** The contract already declares them; only an adapter is missing.

## Built with

`python` `pandas` `numpy` `scikit-learn` `conformal-prediction` `pyarrow` `pytest` `html` `css`
`javascript` `svg`

## Try it out

- Repository: `[[repo URL]]`
- Operator UI: open `ui/headway.html` — one self-contained file, no server
- Reproduce every figure: `scripts/make_synthetic.py` → `build_health.py` → `build_rul.py` →
  `build_aspects.py` → `run_tournament.py`
- Demo video: `[[video URL]]`

---

## Pre-submission checklist

- [ ] Rerun the full chain on the **real NEBULA X data** and replace every figure in this file
- [ ] `scripts/bind_data.py <their file>` → paste mapping into `adapters/nebulax.py`
- [ ] Record the demo video (Hindsight replay is the strongest 30 seconds)
- [ ] Screenshots: Tonight's window, Fleet, Door detail
- [ ] Team names and roles
- [ ] Confirm the repo is public before 16:00 Saturday
