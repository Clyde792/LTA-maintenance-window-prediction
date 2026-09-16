# Data handover plan — official Nebula X PS3 dataset

A read-only audit of what Headway can do with the organiser's data the moment it
arrives, checked against the code rather than against intentions. No schema is
assumed: PS3's public page lists door data, bogie data and manually verified
fault data as *possible* datasets and specifies no format.

Two things this document deliberately does not do. It **does not estimate model
performance** on data nobody has seen. It **does not promise how long onboarding
will take** — the adapter is a mapping exercise only if the signal and label
semantics turn out to be compatible, and that is not knowable in advance.

Audited 11 September 2026 against `headway/adapters/`, `headway/data_readiness.py`,
`headway/detector_selection.py`, `headway/evaluate/metrics.py`, `headway/rul.py`,
`headway/pipeline.py` and `PS3_RESEARCH_AND_PLAN.md`.

---

## 1 · What we need to be told

These are not wishes. Each one is a value some existing check reads, refuses
without, or silently mis-handles if guessed.

### Schema and row semantics

The contract's unit is **one cycle of one asset** — for a door, one complete open
or close movement. Everything downstream aggregates from there.

* Is a row a waveform sample, a cycle, an event, a periodic snapshot, or a daily
  summary? These are four different ingestion paths (§3), and one of them is
  unsupported today.
* If a row is a half-cycle, is there a direction column? `headway/adapters/nebulax.py`
  says in its header comment to filter to close movements rather than mixing
  them; nothing enforces that automatically.
* Column names in full. `suggest_mapping` matches an alias table on a squashed
  form (lowercase, alphanumerics only) with a 0.72 fuzzy cutoff, so
  `Peak Current (A)` resolves, but a site-specific code will not.

### Units

Units are **not** inferred. `Adapter.unit_scales` applies a declared multiplier
per column before any derivation, and it is validated (`finite`, `> 0`, and the
column must be a contract signal or context). It is not populated by default.

We need the unit of every mapped column, because two derivations multiply and
divide across columns — `current_integral_as = mean_current_a × cycle_duration_s`
and its inverse — so a millisecond duration or a milliamp current corrupts a
derived channel silently. `contract.validate` warns if `load_proxy` falls outside
`[0, 1]` or `hour_of_day` outside `[0, 24)`, but those are the only range checks.

### Timestamps and timezone

* Timestamp column, its format, and **its timezone**. The adapter coerces with
  `pd.to_datetime(..., errors="coerce")`; unparseable values become `NaT` and
  `assess_readiness` blocks on them.
* Does the timestamp mark cycle start, cycle end, or logging time? The daily
  aggregate floors to a calendar day and treats a day's aggregate as available at
  the **following midnight**; the whole replay clock rests on that.
* The pipeline treats naive timestamps as Asia/Singapore. A UTC export needs
  explicit conversion at the adapter, not downstream.

### Timestamps: what each value means

Output is always **naive Asia/Singapore**, because that is what the pipeline
assumes. Four input cases, and the third is the one that used to go wrong:

| Input column | Behaviour |
|---|---|
| All offset-aware | Each instant preserved, converted to local, zone dropped |
| All naive, `source_timezone` declared | Localised to that zone, converted, zone dropped |
| All naive, nothing declared | Passed through unchanged; the provenance records that no zone was declared |
| **Mixed** aware and naive | Each value keeps its own meaning: an offset value keeps its instant, a naive value is localised to the declared zone. **Refused** if no `source_timezone` is declared, because the naive values cannot be interpreted |

Mixedness is decided from the source text, not from the parser. Given a mixed
column, pandas 3 returns a tz-aware series with every naive value coerced to
`NaT` — silently deleting exactly the rows whose meaning is in question — so
relying on it would have been circular.

A DST-**ambiguous** local time (01:30 on the autumn fall-back) or a
**non-existent** one (02:30 on the spring forward) becomes `NaT` rather than a
guess. `assess_readiness` then blocks on invalid timestamps, which is the
correct outcome: the source must say which instant it meant.

### Publication: recoverable, not atomic

An export writes two files, and two renames are not an atomic pair. The sequence
is: write both to `.partial`; move any existing pair aside to `.backup`; rename
the new pair into place; on **any** failure, remove what was published and move
the backups back; drop the backups only once both files are live. A failed
re-export therefore leaves the previous export byte-identical, which is checked
by failing at each publication step in turn.

**The honest limitation:** this is recoverable, not transactional. A process
killed between the two final renames can leave the new Parquet beside the old
sidecar, with both `.backup` files still on disk. A later run **refuses to
start** while those leftovers exist, because their presence means an earlier run
did not finish and a person should decide which pair is the good one.

### Nothing writes over a source

Before anything is read, the input file, the mapping, `--out`, the derived
sidecar path and `--suggest-out` are resolved and compared. Any two that are the
same file on disk is a refusal — including via a different spelling of the same
path. This check is **not** conditional on `--overwrite`: that flag authorises
replacing an export, and can never authorise writing over source telemetry or a
reviewed mapping. An existing draft is also not overwritten without `--overwrite`.

### Asset joins

* The key telemetry is recorded against, and the key the verified fault records
  are recorded against. If they differ — a DCU serial against a car number, say —
  a lookup is required. `nebulax.py` says to build it there, not in the pipeline.
* `train_id` is derived, if absent, as the leading token of `asset_id` split on
  `[-_ ]`. That is a guess about identifier structure and must be confirmed.
* `assess_readiness` **blocks** if any asset maps to more than one train, if any
  `asset_id`/`train_id` is null or blank, or if fault-record assets cannot all be
  joined to telemetry.

### Sampling or cycle boundaries

* Sampling rate, if rows are samples; cycle delimitation rule, if rows are cycles.
* `assess_readiness` **blocks on duplicate `(asset_id, ts)` pairs** — it will not
  deduplicate them for you, because a repeated timestamp means the cycle or event
  identity is unresolved. On our own synthetic data this found 20 such rows.
* At least **30 observed calendar days** are required by the current reference
  configuration, and readiness blocks below that.

### Operating conditions

The SIGNAL/CONTEXT split is the modelling thesis: we score residuals after
modelling `E[signal | context]`. Context is `ambient_temp_c`, `load_proxy`,
`hour_of_day`, `cycles_since_service`.

If a context column is absent the adapter fills a constant (ambient 28.0,
load 0.5, cycles-since-service 0.0, hour 12.0), reports it loudly, **and sets
`context_supported = False`**. That flag then blocks readiness for the current
pipeline. So a missing confound is not a soft degradation — it is an explicit
decision point about whether to proceed with that correction inert.

### Fault-label meaning

The single largest unknown, and the one that decides which metrics are even
computable.

* Does a verified fault mean discovery, functional failure, inspection finding,
  or completed repair?
* **Is a degradation onset time available?** This decides which evaluator can
  be used at all, and the distinction matters more than it first looks.

  *What the current evaluator requires.* `headway/evaluate/metrics.py` credits an
  alert to an episode only when the prediction time lies in
  `[onset_ts, fault_ts)`. An alert *before* onset is not credited and counts
  toward the unmatched burden. Our episode schema is
  `asset_id, train_id, onset_ts, fault_ts`, and `assess_readiness` blocks without
  those four. With confirmation-only labels this evaluator **cannot be run** —
  not for lead time, and not for fault-event detection either, because "detected"
  is defined by that same onset-bounded window.

  *What an alternative would require.* Confirmation-only labels can still support
  detection and lead-time reporting under a **different, explicitly agreed
  event-matching protocol** — for example, crediting the first alert inside a
  fixed window before confirmation, with the window length and the treatment of
  alerts outside it agreed in advance. That protocol is **not implemented here**,
  and this task did not change the evaluator. Agreeing one is a prerequisite, and
  implementing it is separate work.

  *What no protocol can recover.* Onset-normalised **capture**
  (`lead / (fault − onset)`) is defined against the onset time. Confirmation-only
  labels do not support it under any matching convention, because the denominator
  does not exist. Report lead time in days against the agreed reference event and
  drop capture, rather than substituting a proxy denominator.
* Are fault-free periods actively verified? Unmatched alerts are only false
  alerts if annotation coverage is complete — `select_detector` says so in its
  own limitations.
* How many independent fault-bearing train groups exist?
  `assess_readiness` flags fewer than **5**; `SelectionPolicy` requires at least
  **3** inside the validation window alone.

---

## 2 · What works, what is configuration, what is code

### Works today, unchanged

| Capability | Where |
|---|---|
| Mapping suggestion from real column names, with fuzzy-match confidence flags | `suggest_mapping`, `scripts/bind_data.py` |
| Derivation of absent-but-computable columns (5 rules) | `DERIVATIONS` in `adapters/base.py` |
| Constant-fill of absent context, reported, with `context_supported=False` | `Adapter.FILL_DEFAULTS` |
| Dropping present-but-entirely-null columns so derivation can fire instead | `Adapter.apply` step 2b |
| Contract validation with actionable per-column errors | `contract.validate` |
| Read-only readiness report with separate pipeline and evaluation blockers | `assess_readiness`, `scripts/ps3_readiness.py` |
| Condition normalisation, daily aggregation, per-asset baselines, health index | `headway/pipeline.py` |
| Leakage-controlled detector selection and a separate untouched final test | `headway/detector_selection.py` |
| Replay metrics: detection, lead, capture, unmatched burden | `headway/evaluate/metrics.py` |
| Monitoring-availability accounting that keeps absent days in the denominator | `headway/outage_tolerant.py`, `headway/evidence.py` |
| End-to-end seam proof on a disguised export (mA, metres, text timestamps) | `scripts/bind_data.py --demo` |
| **Reviewed mapping configuration** — columns, unit scales, source timezone, timestamp format, boolean tokens; refuses drafts, unknown keys and invalid values | `headway/adapters/mapping.py` |
| **Canonical export** — Parquet plus a provenance sidecar, recoverable publication, input untouched | `scripts/bind_data.py --mapping --out` |
| **Path-alias refusal** — no two roles may be the same file; `--overwrite` never reaches a source | `refuse_aliases` in `scripts/bind_data.py` |
| **Declared timezone and timestamp format**, converted to naive Asia/Singapore, with offset-aware and naive values each keeping their own meaning | `Adapter.source_timezone`, `Adapter.timestamp_format` |
| **Declared boolean tokens** (`Y`/`N`), with unrecognised tokens raising rather than reading as False, and conflicting normalised tokens refused | `Adapter.boolean_tokens`, `headway/adapters/mapping.py` |
| **Sidecar bound to the artifact by hash**, verified before its claims are believed | `scripts/ps3_readiness.py` |

### Configuration — editing declared values, no new logic

* **A reviewed mapping file** — the primary route, and the one the export tool
  requires. JSON, validated by `headway/adapters/mapping.py`:

  ```json
  {
    "reviewed": true,
    "subsystem": "door",
    "columns": {"ts": "Event_Time", "asset_id": "EQUIPMENT_ID"},
    "unit_scales": {"peak_current_a": 0.001, "travel_mm": 1000.0},
    "source_timezone": "UTC",
    "timestamp_format": "%d/%m/%Y %H:%M:%S",
    "boolean_tokens": {"Y": true, "N": false},
    "reviewed_by": "name", "reviewed_at": "2026-09-18", "notes": "..."
  }
  ```

  `reviewed` must be `true` or the file is refused; a draft from the suggester
  carries `false`. Unknown top-level keys are an error, so a mistyped
  `unit_scale` cannot silently do nothing. Every target must be a real contract
  column, no source column may be claimed twice, scales must be finite and
  positive, and the timezone must be a real IANA name.

  `unit_scales` and `boolean_tokens` must be objects, or `null`, or absent —
  `false`, `0`, `""` and `[]` are rejected rather than read as "empty
  configuration". Tokens are matched after strip-and-lower, so `"Y"` and `" y "`
  are the same token: declaring both with **different** values is refused rather
  than resolved by file order, and redefining a token the adapter already knows
  (`yes`, `true`, `1`, …) is refused because it would invert every row using it.
* **`DOOR_MAP` / `BOGIE_MAP`** in `headway/adapters/nebulax.py`. The older
  in-code route, still there for `nebulax.load()`. Currently empty; `load()`
  raises a directed error until filled. Prefer the mapping file, which is
  hashed into the provenance record.
* **Identifier lookup**, if telemetry and fault records use different keys —
  a dict in `nebulax.py`.
* **Split dates** for selection (`reference_end`, `calibration_end`,
  `validation_end`) and **`SelectionPolicy`** gates. All are parameters; all are
  illustrative until reviewed against the real label density.

### Requires code — concrete, scoped, not yet written

1. **An episode builder** that turns the organiser's fault records into the
   `asset_id, train_id, onset_ts, fault_ts` episode frame — including whatever
   onset convention is agreed (§1). This is the piece most likely to be needed
   and is entirely absent.
2. **An alternative event-matching protocol**, if the labels turn out to be
   confirmation-only (§1). Not implemented; the current evaluator is
   onset-bounded.
3. **Bogie evaluation path** (§3). The selection runner refuses bogie outright.
4. **Waveform ingestion** (§3), only if raw traces exist.

Three items from the first audit are now **done**: writing the mapped frame,
declaring non-standard `fault_confirmed` tokens, and parameterising detector
selection for reviewed data. All are covered in §6.

---

## 3 · Ingestion paths

`read_any` dispatches on file extension: `.parquet`/`.pq`, `.xlsx`/`.xls`,
`.json`, and CSV for everything else.

### A · Cycle summaries — supported

The designed path. One row per cycle per asset, mapped to the contract, then
`HealthPipeline` → daily aggregate → detectors. This is what every existing test
and every build script exercises.

### B · Periodic snapshots or event logs — supported with a decision

Mechanically these bind the same way, but "one row per cycle" stops being true.
A snapshot stream aggregates to asset-days without complaint, and the daily
median then summarises a different population than a cycle median does. Workable;
must be stated rather than silently accepted. Duplicate `(asset_id, ts)` rows
will block readiness until the identity rule is settled.

### C · Daily summaries — supported, with capability loss

If the organiser supplies pre-aggregated daily rows, the per-cycle layer is gone:
no within-day dispersion, and no within-day load contrast, which is what the
fit-for-duty assessment is built on. The health index and detectors still run.

### D · Raw waveforms — **NOT SUPPORTED**

There is no waveform reader, no per-cycle array column handling, no resampling,
and no HDF5/TDMS/MAT support in `read_any`. `PS3_RESEARCH_AND_PLAN.md` §3 sets
out what would be tried if traces exist — duration-normalised shape, starting
peak, movement-phase charge, transient spikes, multiscale energy — as a *feature
experiment*, explicitly claiming no accuracy. Treat raw waveforms as new work
whose scope cannot be estimated before seeing the sample rate and file format.

### E · Door — supported end to end

`contract.DOOR` declares seven signals, primary `current_integral_as`, with
`travel_mm` oriented so that less is worse. Every build script, the duty
assessment, RUL and the whole dashboard assume door.

### F · Bogie — contract and pipeline only

Audited by probe rather than assumption: `HealthPipeline(subsystem="bogie")`
**fits and transforms** a bogie-shaped frame, producing five normalised levels
(`bearing_temp_rise_c`, `vibration_rms_g`, `vibration_kurtosis`,
`suspension_defl_mm`, `wheel_impact_g`). The generic machinery is real.

What does not exist:

* **No bogie tests.** `grep -rl bogie tests/` returns nothing.
* **No bogie synthetic generator** — `headway/synth/` contains only `doors.py`.
* **No bogie build script**; every `scripts/build_*.py` hardcodes `data/door_*`.
* **No duty semantics.** Peak-band duty restriction is a door-load concept; the
  probe produced no duty columns.
* Aspect thresholds, RUL threshold and detector choice are all door-derived. Do
  not transfer the door failure threshold to a bogie.

So: bogie data can be *bound and normalised*, and nothing beyond that is
demonstrated. Say exactly that.

---

## 4 · Leakage-safe evaluation protocol

This protocol is implemented, not proposed. `headway/detector_selection.py`
enforces most of it and refuses rather than degrades.

**Reference eligibility.** Baselines are fitted only on an explicit historical
reference window. A complete reference is *not* evidence the asset was healthy —
`headway/onboarding.py` requires a reviewer attestation, and
`ROBUSTNESS_RESULTS.md` records that an offset present from the first reference
day was absorbed invisibly (median residual 27.92 → 0.017 on identical telemetry).
Reference eligibility must come from maintenance or inspection records.

**Chronological splits.** Four ordered periods: reference → calibration →
validation → final test. `select_detector` requires
`calibration_start < calibration_end < validation_end` and raises otherwise.
`select_detector` filters to labels with `fault_ts <= validation_end`, but that
is a filter inside a function that has been handed every label. The reviewed-data
runner (§6, Step 5) enforces the boundary on **inputs** instead: selection opens
only a development label file, and the test label file is opened by the test
stage alone.

**Asset separation.** An episode that straddles the calibration/validation
boundary contaminates both. `select_detector` excludes *that asset's entire
validation rows* and the episode itself, and `assess_selected` repeats the
exclusion at the validation/test boundary. Preprocessing is additionally fitted
with the first train identity held out in our experiments.

**Threshold calibration.** One threshold per candidate, frozen at the
99th percentile of that candidate's own calibration scores — never shared across
models, whose score distributions differ. Candidates with fewer than five finite
calibration scores are disqualified rather than thresholded.

**Final testing.** `assess_selected` evaluates **only** the already chosen model
and cannot revise the winner. If nothing qualifies on validation, there is no
winner and no test — which is what happened on our synthetic data: zero complete
episodes fell inside the validation window after excluding three
boundary-crossing assets. Do not retune the boundaries after seeing that.

What earlier runs did **not** establish — see "Historical results affected" in
§6, Step 5. The legacy synthetic experiment loaded one combined label file and
passed all of it to `select_detector`; its test period had also been inspected
before. It is development evidence, not an untouched evaluation.

**Qualification gates** (`SelectionPolicy`, all illustrative until reviewed):
≥3 independent validation fault groups, ≥0.8 episode recall, ≤1.0 unmatched
alerts per asset-month, ≥0.9 score fraction, 2 alerts/day budget, 3-day cooldown.
`select_detector` computes score fraction and the per-asset-month rate over the
rows it is **given**. Given only observed rows, a day with no telemetry vanishes
from both denominators; the reviewed-data runner therefore aligns scores onto the
declared grid of expected asset-days first, so an absent day counts as
unavailable in the gate itself.

---

## 5 · Operational metrics

Reported by `DetectorResult.as_row()` under a frozen threshold, a daily alert
budget and an elapsed-time cooldown. The retrospective ranking mode spends a
budget using future score ranks and **is not a live policy** — the module says so
in its own docstring; only the replay mode is causal.

| Metric | Field | Depends on |
|---|---|---|
| Fault-event detection | `detected`, `missed`, `n_episodes`, `episode_recall` | an agreed event-matching convention. The current evaluator's convention is the onset-bounded window `[onset_ts, fault_ts)`, so it needs onset; a confirmation-only dataset needs a different agreed convention, which is not implemented |
| Warning lead time | `median_lead_d`, `worst_lead_d`, `mean_lead_d` | the same matching convention; measured to `fault_ts` |
| Fraction of the warning window used | `median_capture`, `worst_capture` | `lead / (fault − onset)` capped at 1 — **onset-normalised, so unavailable with confirmation-only labels under any convention** |
| Unmatched alert burden | `alarms`, `precision`, `false_alerts_per_asset_month` | complete fault annotation, or it is *unmatched*, not *false* |
| Monitoring availability | supported fraction of **expected** asset-days | `availability()` keeps absent days in the denominator |

Monitoring availability is reported separately from health, with an explicit
status per day (`supported`, `missing`, `not_available`, `stale`,
`outside_conditions`, `incomplete`, `warming_up`). A quiet detector during an
outage must never read as a healthy asset.

### RUL — separate, and conditional

Remaining-life estimation is **not** part of the detection deliverable and must
be reported separately. `ConformalRUL` learns its failure threshold *from the
episodes themselves*, projects log-linearly, and derives an empirical lower bound
by leave-one-group-out conformal calibration (`min_per_bin=20`,
`min_slope=0.25`). It therefore requires:

* both onset and confirmation times — the threshold is learned from the
  onset-to-fault trajectory, so a confirmation-only dataset does not support it
  without an agreed substitute for onset;
* enough independent fault-bearing groups to leave one out meaningfully;
* enough degradation history per episode to fit a slope at all.

If the organiser's failure histories do not meet that, report detection and
availability and say RUL was not evaluated. Do not substitute a point estimate
with no calibrated bound. Unseen-train RUL calibration is recorded as unresolved
in `PS3_RESEARCH_AND_PLAN.md`.

---

## 6 · Commands

PowerShell, from the repository root, using the project environment.

**Prove the seam before the data arrives** — disguises our own export with vendor
names, mA, metres and text timestamps, then binds it back:

```powershell
.\.venv\Scripts\python.exe -B scripts\bind_data.py --demo
```

### Step 1 — look at their file and draft a mapping

```powershell
.\.venv\Scripts\python.exe -B scripts\bind_data.py <their_file.csv> --subsystem door --suggest-out data\ps3_door_mapping.json
```

`--rows 50000` samples a large file first. The draft is written with
`"reviewed": false`, and **nothing will accept it in that state**. If the bind
itself cannot complete — most often unrecognised `fault_confirmed` tokens — the
tool reports the finding, still writes the draft, and exits non-zero.

### Step 2 — review the mapping

Open `data\ps3_door_mapping.json` and, for every entry:

* check the source column is the one you think it is, starting with those named
  in `notes` as low-confidence guesses;
* add `unit_scales` for anything not already in contract units
  (mA → A is `0.001`, m → mm is `1000.0`, ms → s is `0.001`);
* set `source_timezone` (the zone the source timestamps are in) and
  `timestamp_format` if the dates are ambiguous, such as day-first;
* declare `boolean_tokens` for label values outside
  `true/false/1/0/yes/no`, for example `{"Y": true, "N": false}`;
* record `reviewed_by` and `reviewed_at`, then set `"reviewed": true`.

Setting `reviewed` to true is a claim about the schema only. It is **not** an
attestation that units, identities or fault labels have been verified — those
are separate reviews, asserted in step 4, and the provenance record says so.

### Step 3 — export the canonical frame

```powershell
.\.venv\Scripts\python.exe -B scripts\bind_data.py <their_file.csv> --subsystem door --mapping data\ps3_door_mapping.json --out data\ps3_door_cycles.parquet
```

Writes `ps3_door_cycles.parquet` and `ps3_door_cycles.provenance.json`, or
neither. It refuses to overwrite either file unless `--overwrite` is given,
never modifies the input, and refuses `--out` without `--mapping`. The
provenance records the input and mapping hashes, the subsystem, the conversions
applied, which context columns were filled, the validation findings, row counts,
whether the export was **sampled**, and explicitly that no fault episodes, onset
dates or reference eligibility were produced.

### Step 4 — readiness

```powershell
.\.venv\Scripts\python.exe -B scripts\ps3_readiness.py data\ps3_door_cycles.parquet --episodes data\ps3_door_episodes.csv
```

The report carries a `source_provenance` block, but only after the sidecar has
been shown to describe **this** file: it records the exported Parquet's SHA-256
and its filename, and both are re-checked before any claim in it is believed. A
sidecar that is missing, malformed, swapped in from another export, or accurate
about a Parquet that has since been modified comes back as
`"verified": false` with its claims quarantined under `unverified_claims`, and
`UNVERIFIED PROVENANCE` on stderr. A sampled export also warns on stderr.

"Well formed" is checked field by field, before anything nested is read, so a
malformed sidecar produces a list of reasons rather than a traceback:

| Field | Required |
|---|---|
| `input`, `output`, `mapping`, `attestations` | present, and each a JSON object |
| `output.path` | a non-empty filename equal to the Parquet's — a matching hash alone is not enough |
| `output.sha256` | 64 lowercase hex characters, equal to the Parquet's actual hash |
| `input.complete_input` | a real `true` or `false`; the string `"false"` is rejected, so a sample cannot skip its warning |
| `input.rows_read` | `null` for a complete export, a non-negative integer for a sample; the two must agree |
| `input.sha256`, `mapping.canonical_sha256` | 64 lowercase hex characters |
| `context_supported` | `true`, `false` or `null` |
| `unit_conversions` | `null`, or an object of finite positive numbers |

`NaN` and `Infinity` are rejected as unreadable JSON. Every problem is reported,
not only the first, so one correction is not followed by another failed run.

Add `--units-verified`, `--identities-verified` and `--fault-labels-verified`
**only after** those reviews are actually done.

Bogie follows the same four steps with `--subsystem bogie`; the mapping file's
own `subsystem` must agree with the flag.

**Tests for the ingestion path:**

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests\test_mapping_export.py tests\test_adapters.py tests\test_data_readiness.py -q -p no:cacheprovider
```

### Step 5 — detector selection on the reviewed data

Door only. `scripts\select_ps3_detector.py --config` runs in two stages against a
JSON configuration in which **every** evaluation choice is declared; nothing is
inferred from fault outcomes or from where the telemetry starts. A bogie
configuration is refused: no bogie detector, threshold or episode semantics has
been validated, and door ones must not be transferred.

**Split the labels first — before anyone reads test-period labels:**

```powershell
.\.venv\Scripts\python.exe -B scripts\split_episode_labels.py data\ps3_door_episodes.csv --validation-end YYYY-MM-DD --development-out data\ps3_door_episodes_development.csv --test-out data\ps3_door_episodes_test.csv
```

An episode goes to the development file when its fault was confirmed on or
before `validation_end` (local midnight), otherwise to the test file. Rows are
copied verbatim. It refuses to overwrite either output, to write onto its input,
or to write both scopes to one file. Splitting a file that has already been read
end to end gives the right shape but **not** an untouched test set; say so in
`data_exposure`. If the organiser can deliver the two scopes separately, prefer
that and skip this step.

**The configuration** (`configs\ps3_synthetic_door.json` is a complete worked
example for the synthetic fleet). Paths are relative to the configuration file.

```json
{
  "run_kind": "external",
  "subsystem": "door",
  "telemetry": "../data/ps3_door_cycles.parquet",
  "development_episodes": "../data/ps3_door_episodes_development.csv",
  "test_episodes": "../data/ps3_door_episodes_test.csv",
  "output_dir": "../data/ps3_runs",
  "timezone": "Asia/Singapore",
  "boundaries": {
    "reference_start": "YYYY-MM-DD", "reference_end": "YYYY-MM-DD",
    "calibration_end": "YYYY-MM-DD", "validation_end": "YYYY-MM-DD",
    "test_end": "YYYY-MM-DD"
  },
  "expected_grid": "from_first_observation",
  "reference_eligibility": {"assets": ["..."], "declared_by": "...",
                            "basis": "what the inspection or maintenance record shows",
                            "evidence_id": "record identifier"},
  "attestations": {"units_verified": true, "identities_verified": true,
                   "fault_labels_verified": true, "reference_eligibility_verified": true,
                   "declared_by": "..."},
  "label_semantics": {"onset_ts": "what onset means in these records",
                      "fault_ts": "what the fault time means"},
  "data_exposure": {"test_period_previously_inspected": false, "declared_by": "..."},
  "policy": {"threshold_quantile": 0.99, "daily_budget": 2, "cooldown_days": 3.0,
             "minimum_fault_groups": 3, "minimum_recall": 0.8,
             "maximum_false_alerts_per_asset_month": 1.0, "minimum_score_fraction": 0.9},
  "smooth_days": 3,
  "min_reference_days_per_asset": 14
}
```

The `YYYY-MM-DD` and `...` values are placeholders to be filled from the review;
they are deliberately not example dates, because the splits must be chosen
without looking at where faults fall.

Rules the loader enforces: boundaries are strictly ordered midnights in
Asia/Singapore; every window is decided on the **availability clock** (a day
counts once its aggregate lands at the following midnight); the qualification
policy must be declared in full and is never defaulted; unknown keys are refused,
including the old single `episodes` key; the two label paths must differ; and
for `run_kind: external`, `reference_eligibility.evidence_id` is required —
eligibility rests on an independent inspection or maintenance record, never on
the absence of labels. `expected_grid` is one of:

| Rule | Expected asset-days |
|---|---|
| `from_first_observation` | each asset from its first observed day onward; later days with no telemetry stay expected |
| `from_reference_start` | every asset in the telemetry on every day of the evaluated window |

The loader handles the test-label path as text only, and never touches it.

**Select, then freeze:**

```powershell
.\.venv\Scripts\python.exe -B scripts\select_ps3_detector.py --config configs\ps3_door_external.json --stage select --run-id ps3-door-01
```

In order, the select stage:

1. creates `output_dir\runs\ps3-door-01` — refused if it already exists;
2. records `config.json` and `inputs.json` — telemetry, **development** label and
   provenance-sidecar hashes, plus code hashes — **before** preflight or fitting.
   The test label file is recorded by path only, as not opened;
3. runs **preflight** on development labels only and writes `preflight.json`. For
   `run_kind: external` it refuses on any of: canonical schema errors; unverified
   provenance; a sampled export; unsupported context; readiness data blockers;
   any missing human attestation; missing or blank `onset_ts` (with the reason
   that the current evaluator matches on `[onset_ts, fault_ts)` — onset is not
   invented and no other matching protocol is substituted); a development label
   confirmed after `validation_end`; eligible assets absent, or with fewer than
   `min_reference_days_per_asset` **valid** reference days; and labelled
   degradation overlapping the reference for an eligible asset. Labels can only
   contradict eligibility, never establish it;
4. fits preprocessing and **every** detector on the same declared reference —
   `reference_start < available_at <= reference_end`, eligible assets only — and
   only on asset-days that pass explicit **reference-quality** checks (below).
   It refuses if too few valid reference days remain after preprocessing, and
   records the span each fit actually used and how many days were excluded;
5. scores only rows that are **valid**, aligns the scores onto the expected grid,
   selects on validation only, then writes `scores.parquet` (aligned),
   `selection.json` and `frozen.json`, binding the selection, scores,
   configuration, development labels, telemetry, sidecar and code hashes.

A **no-qualifying-candidate** outcome is valid, exits 0, and is recorded as such.

**Validity, and why reference quality is its own check.**

| Purpose | A row passes when | Applied |
|---|---|---|
| Scoring | `data_quality_ok` and `context_supported` are both true | **before** any detector scores or smooths, so an invalid reading cannot enter the EWMA recursion or a later smoothing window |
| Reference fitting | every completeness column ≥ 0.9, `n_cycles` ≥ 5, health inputs complete, context supported | to cycles before preprocessing is fitted, and to daily rows before detectors are fitted |

The reference check deliberately does not reuse `data_quality_ok` or
`quality_ready`: after preprocessing, both include "available at or after the
fit", which is false for every reference date by construction.

The historical detector path (no validity columns) still masks invalid rows only
**after** scoring and smoothing, so an invalid day can move later scores there.
It is kept, unchanged, only because the legacy experiments below depend on it;
`D.run` refuses validity columns without an explicit reference so the two paths
cannot be mixed.

**Availability is the gate's denominator, not a side report.** The score fraction
used by `minimum_score_fraction`, the unmatched-alert rate, and the monitoring
availability in each report are all computed on the same expected asset-days,
with the same crossing-asset exclusions; the runner refuses to write a report if
the gate and the availability figure disagree. Two scored days out of five
expected is availability 0.4, and a 0.9 requirement fails.

**Final test, once:**

```powershell
.\.venv\Scripts\python.exe -B scripts\select_ps3_detector.py --config configs\ps3_door_external.json --stage test --run ps3-door-01
```

It re-hashes the configuration, `config.json`, `selection.json`,
`scores.parquet`, the telemetry, the development labels, the provenance sidecar
and the code, and **refuses** if any has changed or is missing since the freeze,
or if the freeze predates sidecar binding. It refuses if `test.json` already
exists. Only then, and only if a candidate qualified, does it open the test label
file: it refuses test labels confirmed on or before `validation_end`, records
their hash in `test.json`, and discloses any eligible asset whose test labels
show degradation starting inside the reference window. With no qualified
candidate the test labels are not opened at all.

What the reports contain: episode counts per phase; episodes and assets excluded
for crossing the calibration/validation boundary, the validation/test boundary,
and `test_end`; unmatched alerts alongside the per-asset-month rate; and
availability over expected asset-days. The test report states whether the test
period was previously inspected. It never calls a previously inspected period
unseen, and it notes that it cannot verify a declaration that it was not.

Nothing is promoted into the dashboard by either stage.

**The synthetic fleet** runs through the same path with
`configs\ps3_synthetic_door.json`, using `data\door_episodes_development.csv`
(3 episodes) and `data\door_episodes_test.csv` (6 episodes), split at
2026-06-24 by the helper above. That label file had been read in full long
before the split, so the split enforces the code boundary but the synthetic test
period is **not** untouched. It is `run_kind: synthetic`, so provenance,
readiness and attestation findings are recorded but not enforced.

**The legacy synthetic experiment** is kept unchanged so its published result in
`data\ps3_selection` stays reproducible. It never reads a configuration and is
never pointed at organiser data. It refuses to write into a directory that
already holds its artifacts:

```powershell
.\.venv\Scripts\python.exe -B scripts\select_ps3_detector.py --legacy-synthetic --out data\ps3_selection_repro
```

Re-verified on 13 September 2026 after the corrections below: its
`selection.json` and `test.json` are byte-identical to the published ones.

#### Historical results affected

The corrections change how results are **interpreted**; no historical artifact
has been rewritten or relabelled.

| Result | Label boundary | Denominator | Invalid rows | Status |
|---|---|---|---|---|
| `data\ps3_selection` (legacy synthetic selection) | Combined label file loaded and passed whole to `select_detector`, which filters internally; the synthetic test period had been inspected before | Observed rows only | Masked after smoothing; reference fitted without explicit quality checks | **Affected.** Development evidence only; never an untouched evaluation. Its outcome (no winner) does not depend on the denominator, because it failed on zero complete validation episodes |
| `data\door_tournament.parquet`, `data\tournament_replay.csv` (`scripts\run_tournament.py`) | Retrospective final-period comparison on previously inspected data | Observed rows only | Masked after smoothing (historical `D.run`); reference fitted without explicit quality checks | **Affected in interpretation.** Availability and false-alert rates exclude absent days; an invalid day could move nearby smoothed or EWMA scores. Not re-run, not relabelled |
| Robustness and outage experiments (`robustness.score_frozen`; `data\robustness_experiment`, `data\outage_experiment`) | Separate fixed-seed stress protocols | Aligned to expected asset-days | Masked before scoring | **Not affected** by item 1 or by the scoring order. **Partly affected** by the reference-quality check: their preprocessing and detectors are fitted on finite reference rows without the completeness or cycle-count check. Not re-run, so the size of any effect is not claimed |
| Reviewed-runner scratch verification `synth-verify` (session scratch directory, never published) | Selection read the **combined** synthetic label file; its `frozen.json` note "before any final-test label was read" is **inaccurate** | Observed rows | Masked after smoothing | **Superseded.** Kept as-is with a correction note beside it. Re-run as `synth-corrected` under the corrected path: preflight passed (3 checks waived as synthetic), all 2,400 reference asset-days valid, 0 complete validation episodes after 3 calibration-crossing assets, availability 1.0 on all 1,848 expected validation asset-days, **no candidate qualified**, test not evaluated and test labels not opened |

No reviewed external run has been made, so no real-data result is affected.

**Tests for selection:**

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests\test_ps3_run.py tests\test_detector_selection.py tests\test_detectors.py tests\test_normalise.py -p no:cacheprovider -rfEs --durations=15
```

## 7 · Blockers, in priority order

Ordered by what stops the next step, not by effort.

| # | Blocker | Why it stops us | Path |
|---|---|---|---|
| 1 | **Fault-label semantics, and the event-matching convention** | The current evaluator matches on `[onset_ts, fault_ts)`, so without onset it computes **neither detection nor lead time**; readiness blocks without onset. Confirmation-only labels need a different agreed convention, which is not implemented. Onset-normalised capture is unavailable either way. | Agree the convention with the organiser; then build the episode builder (§2, item 4) and, if the convention is not onset-based, implement it as separate work |
| 2 | **Row semantics — waveform, cycle, event, snapshot or daily** | Selects the ingestion path; one path (raw waveforms) is unsupported | Ask; §3 |
| 3 | **Identifier join between telemetry and fault records** | Readiness blocks if fault assets cannot all be joined | Ask; lookup dict in `nebulax.py` |
| 4 | **Units and timestamp timezone** | Derived channels are computed by multiplication across columns; nothing infers units | Ask; `unit_scales` |
| 7 | **Fault-group count** | Selection needs ≥3 independent groups in validation alone; readiness flags below 5. Our own synthetic run qualified *no* detector for exactly this reason | Ask about label density; adjust the protocol openly if the data is thin |
| 8 | **Context column availability** | A filled constant sets `context_supported=False`, which blocks readiness and forces an explicit decision | Ask; then decide and state it |
| 9 | **≥30 observed days** | Hard block in `assess_readiness` for the current reference configuration | Ask about the export span |
| 10 | **Bogie has no tests, generator, build script or duty semantics** | Binding and normalising is all that is demonstrated | Code, if bogie data arrives and is in scope |
| 11 | **Raw-waveform ingestion absent** | Whole path unsupported | Code, scope unknown before seeing the format |
| 12 | **Confirmation-only labels have no implemented matching protocol** | The evaluator is onset-bounded; an agreed alternative would be new work, and capture is unavailable regardless | Agree first, then code |

## What must not be claimed

* Any performance number on the organiser's data before it has been evaluated.
  Every figure in this repository is synthetic.
* That mapping is a fixed-duration exercise. Signal and label semantics decide
  which models are applicable at all, and those are unknown.
* That a contract PASS or a clean readiness report means the models are suitable.
  Both check schema and data quality. `assess_readiness` states this in its own
  `limitations` field.
* That unmatched alerts are false alerts, unless fault annotation coverage is
  confirmed complete.
* That the bogie contract is a bogie detector.
* That a stable reference window means a healthy asset.
