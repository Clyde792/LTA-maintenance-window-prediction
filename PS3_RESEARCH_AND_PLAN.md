# Headway: PS3 alignment and next implementation priorities

Verified against the public competition website on 11 September 2026. The team
has not received the organiser dataset. Existing performance evidence is synthetic.

## What PS3 actually asks

PS3 calls for AI/ML anomaly detection across train-system telemetry and identifying
the best models for future anomalies. Door data, bogie data and manually verified
fault data are listed as **possible** datasets. The page does not specify schemas,
sampling frequencies, label availability or a dataset release time. Judging names
technical execution, problem fit, ease of use and real-world impact.
[Official Nebula X page](https://nebulax.com.sg/)

Our interpretation: anomaly detection and credible model comparison are the core
deliverables. Scheduling and repair verification demonstrate the intervention
workflow, but should not displace the detection experiment. Scheduling is also the
focus of PS1, so a planner-heavy pitch could obscure our PS3 contribution.

## What exists and what remains

| Capability | Actual status |
|---|---|
| Door feature extraction and context normalisation | Implemented; demonstrated only with synthetic data |
| Directional health index and anomaly distance | Implemented |
| Multiple detector comparison | Implemented; existing tournament is a retrospective final-period comparison |
| Validation-based detector selection | Added in this continuation; no automatic deployment promotion |
| Mapped-data readiness report | Added in this continuation; schema/data checks, not model validation |
| RUL and empirical lower margin | Implemented; unseen-train calibration remains unresolved |
| Maintenance planner and repair-verification display | Implemented prototype workflow |
| Bogie fault detection | Contract exists; real features and performance not demonstrated |
| Raw waveform door diagnostics | Candidate extension; raw waveform availability unknown |
| Confirmed performance on organiser data | Not possible until the dataset arrives |

## Research-backed solution directions

### 1. Select models before final testing — implemented now

Fit preprocessing and detectors on a historical reference, set score thresholds
on a subsequent calibration period, select using a separate validation period,
and assess only the frozen selection on the later test period. This addresses
PS3's model-selection requirement directly. Scikit-learn warns against test-set
leakage and describes time-ordered validation for temporally ordered data.
[Leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html),
[temporal splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)

Use episode recall, warning lead, false-alert workload and score availability.
NAB demonstrates why early-warning timing matters in anomaly detection; our
rail episode metrics are custom and are not the NAB scoring function.
[NAB primary repository](https://github.com/numenta/NAB)

New files: `headway/detector_selection.py`, `scripts/select_ps3_detector.py`.
Fixed experiment: first 30 days fit, through 50% calibrate, through 70% validate,
last 30% test. Minimum three validation fault-bearing train groups, 80% episode
recall, at most one false alert per observed asset-month and at least 90% scored
rows. These are illustrative gates, not operator requirements. Qualification is
not an individual confidence estimate. If no model qualifies, report no winner.

The shared detector runner freezes reference fitting and uses trailing smoothing.
The selector consumes scores and cannot itself prove their upstream causal origin.
Episodes crossing evaluation boundaries are excluded with their asset's rows in
that evaluation. Unconfirmed alerts can only be called false alarms if annotation
coverage is reliable. Correlated days are not independent fault episodes.

The existing fleet was previously examined, so this is development evidence even
though the code separates validation and test. It must be repeated on untouched
organiser data or an independently reserved fleet. New outputs live in
`data/ps3_selection/`; the dashboard model and old tournament are not overwritten.

### 2. Dataset readiness and label audit — implemented now

`headway/data_readiness.py` and `scripts/ps3_readiness.py` check mapped column
availability, finite signals, timestamps, duplicate asset timestamps, identity
joins, context support, history length and usable fault groups. Unit, identity
and fault-label verification require explicit caller attestations.

The report separates readiness for a reviewed pipeline trial from readiness for
labelled evaluation. Fault-free or unlabelled data may support exploratory anomaly
detection, but cannot establish verified warning-time or RUL performance. This is
consistent with the distinction between outlier and novelty detection in the
[scikit-learn documentation](https://scikit-learn.org/stable/modules/outlier_detection.html).

Duplicate timestamps are flagged for review, not silently deleted: they might
represent duplicate records, distinct cycles with coarse timestamps, or a schema
that is not cycle-level at all. The synthetic test fixture also contains repeated
asset timestamps. Passing readiness does not establish a healthy reference or
safe operational use. Bogie compatibility is assessed against the existing
provisional contract and must be revised once the actual schema is known.

### 3. Door waveform signatures — conditional on raw data

Published train-door work investigates motor-current signals for fault diagnosis,
including conventional and deep-learning methods. A newer study uses wavelet
features for electrical and mechanical disturbances. Those findings motivate a
feature experiment; they do not establish accuracy on our fleet.
[Train-door comparative study](https://www.mdpi.com/1424-8220/19/23/5160),
[wavelet motor-current study](https://www.mdpi.com/1424-8220/26/9/2898)

If raw per-cycle traces exist, evaluate duration-normalised shape, starting peak,
movement-phase charge, transient spikes and multiscale energy against our current
aggregate features. Split by whole train/cycle, not neighbouring waveform samples.
Keep opening and closing movements separate until their semantics are understood.
Do not reconstruct imaginary waveforms from daily means or copy a paper's wavelet
levels without checking sample rate and mechanism. No waveform accuracy is claimed.

### 4. Bogie temperature residuals — conditional on relevant channels

Research on axle-box bearings uses temperature along with operating variables
such as speed and ambient temperature for early warning. This supports modelling
expected temperature before detecting unusual residual behaviour, rather than
calling every hot reading wear.
[Axle-box early warning study](https://www.mdpi.com/1424-8220/20/3/823)

If supplied, test temperature above ambient, context-adjusted temperature rise,
rate of change and comparisons with mechanically comparable bearings. Add
vibration features only if suitable sampled vibration exists. Bogie models should
be evaluated separately from doors; do not transfer the door failure threshold.

### 5. Robustness and abstention — next experiment

Before adding a more complex model, test missing channels, sensor offsets,
context shifts, contaminated reference periods and unseen trains. Freeze the
stress protocol first and report both missed episodes and abstention. Compare
point errors on the same scored rows when assessing onboarding. These stress
tests can start with simulation; operational rates need verified real outcomes.

## Commands

### Completed verification in this continuation

All 15 new selector/readiness tests passed. The full synthetic model-selection
experiment completed. It selected **no winner**: zero complete fault episodes
fall wholly inside the fixed validation period after excluding assets with
boundary-crossing episodes. Three fault-bearing assets cross the calibration /
validation boundary. This is a sample/split limitation, not evidence of bad
detectors. The final selected-model test was consequently not run; the existing
independent tournament remains separate evidence. Do not retune these boundaries
after inspecting outcomes and call the new result an untouched test.

The readiness report examined 456,840 cycle rows over 120 observed days and found
20 repeated asset timestamps. It also correctly leaves unit, identity and label
attestations outstanding because the CLI was run without those review flags.
No records were automatically deduplicated or changed.

See `data/ps3_selection/selection.json`, `test.json`, `protocol.json` and
`readiness.json` for the exact output. These artifacts concern the current
synthetic dataset only.

```powershell
.\.venv\Scripts\python.exe -B scripts/ps3_readiness.py data/door_cycles.parquet --episodes data/door_episodes.csv
.\.venv\Scripts\python.exe -B scripts/select_ps3_detector.py
.\.venv\Scripts\python.exe -B -m pytest tests/test_detector_selection.py tests/test_data_readiness.py -q -p no:cacheprovider
```

The first command deliberately does not attest that units or labels have been
reviewed. Add the corresponding CLI flags only after completing those reviews.
Use `scripts/bind_data.py` for initial mapping before running readiness on new
mapped cycle Parquet. Do not point the synthetic selection runner at organiser
files until reference eligibility, split dates and label semantics are reviewed.

## Questions for the organiser or depot mentors

Robustness follow-up: [ROBUSTNESS_RESULTS.md](ROBUSTNESS_RESULTS.md) records the
implemented fixed-seed stress experiments, measured monitoring gaps, reference
contamination and priorities for the next iteration. These are synthetic
results; the experiments do not promote a deployment model.

- Are rows waveform samples, cycles, events, periodic snapshots or daily summaries?
- What are the timestamps, timezone, signal units and sampling rates?
- How do sensor/door/bogie identifiers join to train and maintenance records?
- Does a verified fault label mean discovery, functional failure, inspection or repair?
- Are fault-free periods actively verified, and are incomplete follow-ups identified?
- Which operating conditions and door directions are recorded?
- Is there an independently held-out fleet or period for final evaluation?
- How many alerts can engineers investigate, and what warning time is useful?

Do not assume the dataset handover is at a particular hour: the public page does
not establish that. Do not claim the adapter is merely a fixed-duration mapping
exercise; signal and label semantics determine which models are applicable.
