# Inspection evidence — demo acceptance checklist and walkthrough

For the NEBULA X presentation. Covers the **Inspection evidence** section in door
detail only; the rest of the dashboard has its own material.

Everything below is reproducible against the current build. Where a state cannot
occur in the demo telemetry, the row says so and names a **labelled test case**
instead of pretending a fleet door shows it.

## Two clocks, and they are never the same date

The dashboard replays telemetry days, but an assessment only exists once that
day's aggregate has landed — **the following midnight, Asia/Singapore**. So every
row below carries both:

* the **replay date** you select on the chart (a telemetry day), and
* the **assessment availability time** the meta line prints (that day + 1, 00:00 SGT).

Telemetry through **19 July** is assessed at **20 July 00:00 SGT**. Telemetry
through **29 July** — the last replay night — is assessed at **30 July 00:00
SGT**. Saying "the 19 July assessment" is wrong; it is the assessment *of* 19
July's telemetry, *available on* 20 July.

* Replay span: **2026-04-01 → 2026-07-29**, nights 0–119.
* Doors with evidence: **6 of 80** — `TRN006-DOOR-1`, `TRN013-DOOR-2`,
  `TRN021-DOOR-2`, `TRN025-DOOR-2`, `TRN034-DOOR-2`, `TRN038-DOOR-2`.
* Source artifact this build was computed from: `sha256:ce0c10e3c5068b88`.

## Before the room

From the repository root, using the project's virtual environment:

```
.venv\Scripts\python.exe -B scripts\rebuild.py
.venv\Scripts\python.exe -B scripts\check_artifacts.py
```

Add `--skip-js` to the second command **only** where Node is unavailable; it
prints a warning and omits the JavaScript probes, which must then be reported as
not run.

The rebuild must print, in order:

```
Wrote inspection_evidence_export.json: 12 record(s) for 6 door(s), 0 door(s) unavailable, 120 KiB.
Dashboard refreshed: 80 doors; explicit unknown states and window comparisons.
```

**Stop and rebuild if** the page build prints `Inspection evidence UNAVAILABLE —
rebuild required` or any line beginning `  - ` after the export step. Those are
coverage or identity failures, and the section will show a rebuild notice on the
day.

Open `ui\headway.html`. Navigate: **Status** → click a door id → Door detail →
scroll past the chart, Fit-for-duty and Repair verification.

---

## 1 · Current evidence

| | |
|---|---|
| **Where** | `TRN021-DOOR-2`, replay date **2026-07-29** (the default, last night) |
| **Assessment available** | 2026-07-30 00:00 SGT |
| **Also** | `TRN038-DOOR-2`, same replay date, for the no-change contrast |

**Expected on screen**

* Header: *Inspection evidence — read-only: explains the observations; does not
  identify a cause, change alerts or urgency, or authorise maintenance.*
* Meta line: `Assessed 2026-07-30 00:00 SGT · door TRN021-DOOR-2 · source
  artifact sha256:ce0c10e3c5068b88 · current for this date` — the last phrase in
  **green**. Note the date shown is the availability time, not the 29 July
  replay date on the chart.
* Section 1 lists five channels in plain names and physical units:
  Motor charge per cycle `+0.5699 A.s`, Cycle time `+0.2458 s`, Average current
  `+0.067 A`, Peak current `+0.6687 A`, Door travel `-4.2334 mm`.
* Section 2: *79 comparable doors of 79 considered* — every channel **not
  shared**, in green, with the peer median.
* Section 3: five explanation cards in fixed order, identical styling, headed by
  *"Listed in a fixed order. Neither the order nor the number of supporting
  points is a ranking, and no explanation here is a confirmed cause."*
* **Detailed calculations** and **Provenance and limits** both collapsed.
* On `TRN038-DOOR-2`: *"No change beyond the reporting threshold in the 5
  channels that could be assessed."*

**The presenter must not claim**

* That the door has a sensor fault, or a mechanical fault. The section names
  neither; it says which observations point where.
* That the top card is the most likely explanation. The order is fixed and
  carries no information.
* That five channels moving "confirms" wear. Independent channels moving does not
  exclude a simultaneous sensor problem — the card says so in words.
* That anything here changed the aspect, the countdown or the plan. It did not.
* That the assessment was made on 29 July. It was made on 29 July's telemetry and
  became available at 30 July 00:00 SGT.

---

## 2 · Stale evidence, and the replay clock

| | |
|---|---|
| **Where** | `TRN021-DOOR-2`; drag the chart to change the replay date |

`TRN021-DOOR-2` carries two assessments:

| Telemetry through | Assessment available | Replay night |
|---|---|---|
| 2026-07-19 | **2026-07-20 00:00 SGT** | 109 |
| 2026-07-29 | **2026-07-30 00:00 SGT** | 119 |

Four states, all from one door, all driven by the earlier assessment except the
last:

| Replay date selected | Meta line shows | Expected state |
|---|---|---|
| 2026-07-29 | `Assessed 2026-07-30 00:00 SGT` | `current for this date` (green) |
| 2026-07-22 | `Assessed 2026-07-20 00:00 SGT` | `not recomputed for 2026-07-22 — 3 days later` (amber) |
| 2026-07-27 | `Assessed 2026-07-20 00:00 SGT` | `stale — 8 days old, not recomputed for 2026-07-27` (red) |
| 2026-07-18 | — | *No inspection evidence had been produced for this door by 2026-07-18.* |

The age is counted in replay days between the assessed telemetry day and the
selected one: 19 July → 22 July is 3 days. The staleness threshold is 7 days. On
2026-07-29 the newer assessment supersedes the older automatically.

**The presenter must not claim**

* That the dashboard is live. It is a replay: an assessment of 19 July's
  telemetry is shown as such on every later date, never silently refreshed.
* That "stale" means the door got worse. It means nobody recomputed; the
  telemetry may be unchanged.
* That the 7-day threshold is calibrated. It is a display policy, stated as such.
* That the evidence existed on 19 July. It existed from 20 July 00:00 SGT — which
  is why 2026-07-18 shows nothing at all.

---

## 3 · Unknown channels

**Does not occur in the demo telemetry.** All twelve records assess all five
channels. Do not hunt for a fleet door that shows this.

| | |
|---|---|
| **Labelled test cases** | `tests/test_inspection_ui_js.py::test_all_channels_unknown_reports_insufficient_evidence` and `::test_mixed_coverage_limits_the_no_change_statement_to_assessed_channels` |
| **Illustrative record** | `data/inspection_evidence/ambiguous_missing_evidence.json` |

Run them with:

```
.venv\Scripts\python.exe -B -m pytest tests\test_inspection_ui_js.py -v
```

**Expected behaviour, if shown from the test fixtures**

* Nothing assessable → **"Insufficient evidence to assess change."** Never a
  no-change finding, and no "Assessed and steady" line.
* Mixed → *"No change beyond the reporting threshold in the 1 channel that could
  be assessed (4 not assessable)."* The steady list names only assessed channels.
* Each unassessable channel is listed as **not assessable** with its reason —
  never "not measured", because a reading may exist and simply fail the quality,
  reference or completeness requirements.

**The presenter must not claim**

* That a quiet section means a healthy door. Unknown is not "no change".
* That the fleet demo exercises this path. It does not; say it is a test case.
* That the illustrative JSON files describe real doors. They are fixtures built
  to exercise the rules, labelled as such in their own `index.json`, and are
  never attached to a fleet asset.

---

## 4 · A door with no supported assessment

**Does not occur in the demo telemetry** — the export reports `0 door(s)
unavailable`.

| | |
|---|---|
| **Labelled test case** | `tests/test_inspection_export.py::test_one_unassessable_door_does_not_invalidate_the_other` — builds a two-door frame, runs the real exporter, feeds it back through the page build |
| **Rendering test** | `tests/test_inspection_ui_js.py::test_an_unassessable_door_shows_its_own_reason_while_others_keep_evidence` |

```
.venv\Scripts\python.exe -B -m pytest tests\test_inspection_export.py -k unassessable -v
```

**Expected behaviour**

* The door shows **"No assessment available for this door. No day in the replay
  produced a supported daily aggregate for this door, so no change could be
  assessed."**
* Every other door keeps its evidence. One unassessable door does not blank the
  section fleet-wide.
* This wording is distinct from *"No inspection evidence for this door. Evidence
  is exported only for doors this dashboard surfaces."* — which is what
  `TRN001-DOOR-1` and the other 73 quiet doors show, and which you **can** show
  live.

**The presenter must not claim**

* That an unassessable door is a healthy door, or a faulty one. It is a
  monitoring gap.
* That the quiet-door message and the unassessable message mean the same thing.
  One door was never surfaced; the other was surfaced and could not be assessed.

---

## 5 · Source mismatch

**Demonstrate from the test suite, not by editing build artifacts.** The
temporary-source tests construct their own data directory and artifact, so they
exercise the guard without touching the demo export:

```
.venv\Scripts\python.exe -B -m pytest tests\test_inspection_export.py -k "source_artifact or changed_artifact or source_name" -v
```

* `test_a_different_source_artifact_is_refused` — the export names a hash the
  build does not have.
* `test_a_changed_artifact_invalidates_a_previously_good_export` — a good export,
  then the telemetry is rewritten underneath it.
* `test_a_different_source_name_is_refused` — the export names another file.

**Expected behaviour**

* `build_ui` returns the build-wide unavailable state: `unavailable: true`, zero
  records, and a problem naming both hashes —
  `source artifact hash differs: export 'sha256:…', this build 'sha256:…'`.
* When this happens during a real build, `build_ui.py` prints
  `Inspection evidence UNAVAILABLE — rebuild required:` followed by the problems,
  and the section renders **"Unavailable — rebuild required."** with the reason
  and each problem listed. No door shows evidence.
* This is the only **build-wide** failure. Per-door problems — an unverifiable
  unavailable claim, a conflicting entry, an unexplained omission, a foreign
  record — affect that door alone and leave the others intact.

**The presenter must not claim**

* That the check proves the evidence is correct. It **checks that recorded source
  identities match this build** — the export's, each record's, and the frame the
  page is being assembled from. It says nothing about whether the evidence itself
  is right.
* That the guard is unnecessary in production. It exists because a stale export
  would otherwise be shown as current.

---

## 6 · Unavailable reference views

Live on **every** door with evidence — this is the honest state of the demo, not
a failure to fix before the room.

| | |
|---|---|
| **Where** | any of the six doors; `TRN021-DOOR-2` at replay date 2026-07-29 |

**Expected on screen**

Section 4 reads: **Not available — no reference views supplied; the adaptation
effect is not visible**, above the standing explanation that the comparison would
report a baseline location difference in physical units and a scale ratio
separately.

The demo pipeline fits one fleet model and runs no onboarding, so there is no
adapted baseline to compare against the frozen one. Nothing is invented to fill
the gap: `modelVersion` is `null`, and "Provenance and limits" states that the
`sha256:` value identifies the **telemetry**, not a trained model.

**The presenter must not claim**

* That a missing reference comparison means the baseline is clean. It means the
  comparison was not possible.
* That the `sha256:` hash is a model version. It is a dataset hash, and the page
  says so.
* That reference contamination has been ruled out for these doors. It has not
  been tested — that requires both views.

---

## Standing claims to avoid, whatever is on screen

1. **No diagnostic accuracy.** Nothing in this repository measures whether the
   section's explanations are right. There is no ground-truth cause data. The
   synthetic fixtures were built to exercise the module's own rules.
2. **No probabilities.** There is no confidence, likelihood or percentage
   anywhere in the output, and the number of supporting points is not a score.
3. **No authority.** Seven safeguards are `False` in every record:
   root cause, alert suppression, urgency, deferral, service release, detector or
   threshold change, maintenance procedure.
4. **No real maintenance history.** The Repair verification block above it is
   simulated on synthetic telemetry, and says so.
5. **Synthetic data throughout.** The organiser dataset has not been received.
   Nothing here is evidence about an operating railway.
6. **The physical-independence claim is asserted, not measured.** That current
   features share one transducer is an engineering judgement from the feature
   contract; it needs checking against real vendor telemetry.

## Known cosmetic limitation

In section 3, observation text names measurement **families** in their raw form —
`motor_current`, `position`, `timing` — where channel names are rendered plainly.
Accepted as-is for the demo; if asked, they are the three physically distinct
measurement groups, and the page glossary is the feature contract.

## Verification status

### Reviewer's environment

**482 tests passed, no skips**, plus the full artifact checks and both JavaScript
checks — `check_artifacts.py` run without `--skip-js`, so the generated-script
syntax check, the lower-margin formatter probe and the inspection-renderer probe
all executed. This is the run to cite.

### This machine

**469 passed, 13 skipped.** The 13 skips are the Node behaviour tests; Node is
not installed here, so `check_artifacts.py` required `--skip-js` and its
JavaScript probes did not run. Nothing on this machine executed the rendered
JavaScript.

### Visual checks — what was and was not verified

Checked on 11 September 2026 at the default desktop width in the light theme, by
capturing **isolated sections**: preceding page content was hidden in the browser
so the section rendered at the top of the viewport without scrolling. What that
confirmed:

* the five sections in order, with right-aligned tabular figures and both
  `<details>` blocks collapsed;
* the four replay-clock states in their green / amber / red / muted treatments;
* both unknown-channel wordings;
* the unassessable-door notice.

Two limits on that:

* The **rebuild-required panel was reconstructed markup**, not the live branch:
  the page caches its payload at load, so that branch could not be re-entered in
  the browser. Its styling is therefore verified; the branch itself is covered by
  the terminal behaviour in scenario 5 and by
  `tests/test_inspection_ui_js.py::test_the_unavailable_state_replaces_all_evidence`.
* **The full unmodified-page layout remains unverified.** Every capture had
  preceding content hidden. How the section sits within a complete, untouched
  door-detail page — spacing against the chart, the duty card and Repair
  verification, and behaviour at other widths — has not been checked.
