# Inspection evidence — experimental module, 11 September 2026

Experimental. Synthetic only. **Not promoted, not wired into the dashboard**, and
no detector, threshold, aspect or alert is changed by it. Nothing here is evidence
of performance on an operating railway; the organiser dataset has not arrived.

`headway/inspection_evidence.py` answers one question about an already-flagged
door: **which observations support which explanations, and what evidence would
tell them apart?** It does not name a cause and does not produce a probability,
likelihood or confidence anywhere in its output.

Revised three times after review. Nine corrections are described below, each
with the behaviour it replaced. The backend is now **integrated into the
dashboard as a read-only section in door detail**; see the last part of this
document.

## What it produces

`collect_evidence(daily, asset_id=..., as_of=...)` returns one JSON-serialisable
record:

| Block | Contents |
|---|---|
| `observed_changes` | per channel: reference and recent medians, the change in the channel's own units, the reference scale, the change in reference sigma, the **quality basis** used, and a direction — or `status: unknown` with the reason |
| `cross_channel` | which channels shifted, unchanged and unknown; how many *physically independent* families that represents; which shifts are ambiguous; and whether they run in the wear direction |
| `peer_comparison` | comparability status, comparable peers, those excluded and why, **per-channel** peer counts and medians, and which channels are shared, not shared, or unknown |
| `reference_comparison` | each view's baseline **location and scale** with its `parameter_source`, the location offset in physical units, the scale ratio, matched-day counts, per-row provenance and `verified_model_identity` — or `quantifiable: false` with the reason |
| `quality_basis` | whether channel-level eligibility could be reconstructed at all, and which metadata blocked it |

`reference_comparison.quantifiable` is true only when provenance is verified
**and** at least one channel was actually quantified. Per-channel detail is
reported either way.
| `observations` | numbered, quotable statements — the only things an explanation is allowed to cite |
| `explanations` | five candidates in a **fixed, unranked order**, each with `supported_by` / `contradicted_by` observation ids |
| `missing_evidence` | what was not established, and why |
| `safeguards` | seven explicit `False` flags |

## Correction 1 — peer comparison fails closed

**Was:** absent context columns were filtered out of the comparability screen, so
a frame missing them admitted every peer unconditionally. A channel with no peer
evidence was reported the same way as a channel peers agreed had not moved.

**Now**, every one of these yields UNKNOWN comparability rather than a silent
pass, and no channel is described as *not shared* without evidence for that
channel:

* A required context column absent from the frame, or carrying no tolerance.
* Too few **supported** context observations, on the target or on a peer
  (`min_context_days`, default 3). A column is not support; supported readings
  are.
* Fewer than `min_overlap_days` (default 3) recent days overlapping the target's
  window. Contemporaneous means contemporaneous.
* Fewer than `min_peers` (default 3) comparable peers overall, **or** fewer than
  that with a supported change *in this channel*.

Duplicate asset-days are refused outright, for the target and for peers, since
they would double-count in every peer statistic. Peer counts are reported per
channel (`peers_with_observed_change`) and the observation text quotes that
number, not the size of the admitted pool — five admitted peers of whom three
carry the channel is reported as three.

`shared_channels`, `not_shared_channels` and `unknown_channels` are three
distinct lists. Unknown is not "no change".

## Correction 2 — reference comparison mathematics

**Was:** `adaptation_removed = fleet_index − asset_index`. The two indices are
standardised by different scales, so their difference is not a quantity in any
unit, and a difference arising purely from a scale change would have been read as
an absorbed offset.

**Now** the module recovers each view's baseline parameters from the affine
relation the pipeline actually applies,

```
residual = location + scale × index
```

by least squares over matched asset-days, and refuses to proceed unless the
relation holds to `affine_tolerance`. It then reports, per channel:

| Quantity | Meaning |
|---|---|
| `asset_reference` / `fleet_reference` | recovered `location`, `scale`, `unit`, `days_used`, `max_fit_error` |
| `location_offset_removed_by_adaptation` | `asset_location − fleet_location`, in **physical residual units** (A, A.s, s, mm) |
| `location_offset_in_fleet_scale` / `..._in_asset_scale` | the same offset on an explicitly named shared scale |
| `scale_ratio_asset_over_fleet` | reported **separately**; a scale change is not a location change |

Both baselines are preserved with their own parameters and provenance. The
comparison is refused, with `quantifiable: false` and a stated reason, when:

* only one view is supplied;
* the views fail the provenance checks of Correction 6 below;
* they share fewer than `min_matched_days` (default 5) asset-days;
* the underlying `res_*` values differ between views on matched days, meaning the
  two baselines were applied to different telemetry;
* the index does not vary, or the affine relation does not hold.

Contamination is supported **only** by a location offset at or above
`location_offset_sigma` on the shared scale. A scale-only difference produces a
`no_location_offset` observation that *contradicts* contamination. On the
`contaminated_reference` example the recovered offset is `+0.7938 A.s`
(+7.94 fleet sigma) with a scale ratio of `1.000` — the injected value was
`8 × 0.10 A.s`.

## Correction 3 — explanation text and ordering

**Was:** the mechanical statement asserted "and not on comparable peers" even
when no peer evidence existed; a lone shift beside four unobserved channels was
reported as an `isolated_channel_shift`; a multi-family shift was recorded as
*contradicting* a sensor explanation; and explanations were sorted by how many
observations supported them.

**Now:**

* Statements are assembled from clauses that appear **only** where the evidence
  for them exists. With `peer_ids=[]` the mechanical statement says "Peer
  evidence is not available (…), so a change common to other doors has not been
  excluded" and makes no claim about peers at all.
* `isolated_channel_shift` requires every other channel to have been *observed*.
  A single shift beside unknown channels sets `single_shifted_channel` instead,
  and the observation text says how many channels are unknown. Unknown channels
  are never counted as channels that did not move, and both the sensor and
  mechanical statements say they "neither support nor contradict this".
* A multi-family shift is an **absence of corroboration** for a
  single-transducer fault, not evidence against one — two problems can coexist.
  It no longer appears in the sensor explanation's `contradicted_by`, and the
  statement says so explicitly. Only a change shared across doors contradicts a
  fault in *this door's* sensor, and the explanations are named
  `..._on_this_door` to keep that scope visible.
* The five explanations are emitted in a **fixed order** in every record —
  sensor, mechanical, common influence, reference contamination,
  cannot-distinguish — regardless of support. Counting supporting observations
  cannot set diagnostic priority, because several derived statements can restate
  one underlying fact (a multi-family shift and its wear-direction check are two
  statements about the same set of channels).

## Correction 4 — quality flags against the real pipeline

The module claimed per-channel behaviour while gating on the pipeline's global
`data_quality_ok`. **Measured** on `HealthPipeline` output with the
`missing_channel` stress applied:

```
data_quality_ok True fraction: 0.0
   current_integral_as_completeness   mean=0.697   frac>=0.9 = 0.000
   cycle_duration_s_completeness      mean=1.000   frac>=0.9 = 1.000
   mean_current_a_completeness        mean=1.000   frac>=0.9 = 1.000
   peak_current_a_completeness        mean=1.000   frac>=0.9 = 1.000
   travel_mm_completeness             mean=1.000   frac>=0.9 = 1.000
```

One degraded channel made `data_quality_ok` False on **100%** of affected days
while four of the five channels were fully complete, so the module rejected every
channel. The claim in the docstring was false on real pipeline output.

## Correction 5 — the quality contract, not a reconstruction of it

**Was:** the module rebuilt `data_quality_ok`'s expression term by term whenever
completeness columns happened to be present. Two problems. It duplicated
constants and logic that belong to `features.to_daily`, so the two could drift
apart silently. And completeness columns alone are not evidence about *why* the
global flag is False — assuming it was the completeness term is a guess, and a
frame carrying completeness columns could override a global rejection that had
nothing to do with completeness.

**Now** the pipeline publishes the missing half of the contract. `features` owns:

```python
MIN_COMPLETENESS = 0.9
MIN_CYCLES = 5
READINESS_FLAG = "quality_ready"
```

`to_daily` emits `quality_ready = n_cycles >= MIN_CYCLES`, and
`HealthPipeline.transform` adds `health_inputs_complete` and the
preprocessing-readiness gate to **both** flags, so the published invariant is

```
data_quality_ok == quality_ready & (min over ALL channels' completeness >= MIN_COMPLETENESS)
```

`quality_ready` is exactly the channel-agnostic half. `onboarding` clears it
alongside `data_quality_ok` for a rejected or not-yet-available reference. The
module now imports `MIN_COMPLETENESS` and `READINESS_FLAG` rather than restating
them, and reconstructs channel eligibility **only** when the frame carries the
readiness flag *and* a completeness column for every declared channel. Otherwise
the global rejection stands and `quality_basis.blocking_metadata` names the
missing columns.

Three further guards, all fail-closed:

* A completeness value outside `[0, 1]`, or a readiness value that is not a real
  boolean, rejects the row.
* Rows where the published invariant does not hold — readiness true and every
  channel complete, yet `data_quality_ok` false — are dropped, and counted in
  `rows_rejected_for_inconsistent_quality_metadata`. Self-contradictory metadata
  is not resolved on a guess about which term failed.
* The onboarding reference safeguard applies on either basis.

The real-pipeline partial-channel example is preserved and still resolves the
four intact channels while `current_integral_as` stays unknown.

## Correction 6 — reference provenance

**Was:** the two views were compared on `preprocessing_fitted_at` read from the
**last row** of each. Two views that both carried nothing compared equal, so
absence of provenance on both sides was treated as agreement.

**Now:**

* A **non-empty, matching model identifier** is required — from
  `reference_model_versions[view]`, or a `normalisation_model_id` /
  `model_version` column. Missing on both sides is not agreement; it is a refusal.
* A fitting time is not an identity. `preprocessing_fitted_at` is still checked,
  **in addition to** the identifier, never instead of it.
* Both are validated across **every row used**, not the last: a value that is not
  constant within a view is itself a refusal.
* Without that evidence the views are still described — row counts, baseline
  source, onboarding status — but no operational adaptation effect is quantified.
* `allow_unverified_reference_demonstration=True` lets a caller see the algebra
  anyway. It lands in a separate `demonstration` block marked
  `status: "unverified_algebraic_demonstration"`, `verified: false`, carrying a
  warning that it must not be reported as an adaptation effect. Its observation
  is cited by **no** explanation.

**Parameters are now labelled by source.** Pass `reference_parameters` to supply
the baselines exported from the fitted pipeline
(`parameter_source: "exported"`); otherwise they are inferred from the published
index (`parameter_source: "inferred_from_index"`). An integration test fits a
real `HealthPipeline`, onboards an asset, and checks the inferred parameters
against `AssetBaseline.loc_` / `.scale_` for both views on every channel, and that
the exported path returns the same location offset.

## Correction 7 — scale validity

**Was:** a recovered scale was rejected only at `|scale| <= 1e-12`. Negating both
views' indices produced scales of `-1.2` and the like, passed every other check,
and flipped the sign of the whole comparison.

**Now** a baseline needs a finite location and a **strictly positive finite
scale**, on both the inferred and exported paths. A non-positive scale is
reported as unavailable with an explicit reason — it is rejected, not
reinterpreted as a different orientation convention.

## Correction 8 — every provenance source is validated

**Was:** a caller-supplied `reference_model_versions` entry short-circuited the
row-level scan entirely, so declaring an identifier excused the rows from being
checked. A view whose rows carried two different model identities was accepted
whenever the caller declared one.

**Now** `_view_identity` always scans every identity column and rejects:

* an identity column that is not constant across the rows used;
* two identity columns that disagree with each other within one view;
* any row-level identity that conflicts with the caller's declaration.

A declaration is reconciled with the rows, never substituted for them.

## Correction 9 — one admission gate before either parameter path

**Was:** the admission checks lived inside the affine recovery, so supplying
`reference_parameters` skipped all of them. Exported parameters produced a
"location offset" for a channel with zero quality-supported observations, and
the cross-view residual check ran on unmasked rows. Separately, a block in which
no channel could be quantified still reported `quantifiable: true`.

**Now** `_admit` runs first, for every channel, whichever parameter path follows.
It admits only asset-days that are **matched across both views, quality-supported
in both, and finite in both the residual and the index**, requires at least
`min_matched_days` of them, and checks that the underlying residuals agree on
exactly those days. Calendar overlap is not supported overlap: the
`disjoint_supported_days` example has every day matched on the calendar and none
supported in both views, and quantifies nothing.

Both parameter paths are then validated against the admitted observations:

```
max |residual - (location + scale x index)|
    <= affine_tolerance * max(1, |scale|, max|residual|)
```

with `affine_tolerance` defaulting to `1e-6`. The reference magnitude is taken
from the data so the tolerance means the same thing in millimetres and
ampere-seconds, and `max_fit_error` and `fit_tolerance` are reported alongside
every recovered parameter set. Exported parameters that fail this describe a
different baseline from the one that produced the index, and are rejected rather
than applied.

Where the gate fails but exported parameters were supplied, they are shown with
`status: "metadata_only"`, every offset `null`, and a reason saying they are not
evidence of an adaptation effect. Finally, `quantifiable` is now false whenever
no channel reached `recovered`, with a reason naming why each one did not.

## Reproducible examples

```bash
python scripts/inspection_evidence_examples.py
```

Writes sixteen records plus `index.json` to `data/inspection_evidence/`. The index
pairs each record with the effect that was injected; that label lives on the case
object and is **never** a column in the frame the module sees.

| Case | Injected | Explanations with any support |
|---|---|---|
| `isolated_sensor_offset` | offset on `current_integral_as` only | sensor, mechanical, cannot-distinguish |
| `coherent_multi_channel` | coherent mechanical change | sensor, mechanical, cannot-distinguish |
| `shared_peer_shift` | common influence across target and peers | sensor, common, cannot-distinguish |
| `contaminated_reference` | offset present from the first reference day | reference contamination |
| `ambiguous_missing_evidence` | sub-threshold move; four channels unsupported | cannot-distinguish |
| `sensor_and_mechanical_overlap` | offset **and** coherent mechanical change | sensor, mechanical, cannot-distinguish |
| `shared_shift_with_local_mechanical` | fleet-wide shift **and** local change | sensor, mechanical, common, cannot-distinguish |
| `quiet_control` | nothing | cannot-distinguish |
| `scale_only_difference` | adapted scale 3× the fleet scale | **none** |
| `views_cover_different_dates` | disjoint view coverage | cannot-distinguish |
| `incompatible_reference_provenance` | different preprocessing per view | cannot-distinguish |
| `unidentified_reference_model` | a real offset, but neither view names its model | cannot-distinguish |
| `negated_indices` | both views' indices multiplied by −1 | cannot-distinguish |
| `all_reference_rows_unsupported` | no quality-supported row in either view | cannot-distinguish |
| `disjoint_supported_days` | alternate days supported in each view | cannot-distinguish |
| `real_pipeline_missing_channel` | **real HealthPipeline** with 30% sample loss | sensor, mechanical, cannot-distinguish |

The right-hand column lists which explanations have *any* supporting observation.
It is not a verdict and not a ranking; several rows list three because three
explanations each have at least one observation pointing at them.

`real_pipeline_missing_channel` is the one case that is not handcrafted. It runs
the real pipeline over generated cycles: `current_integral_as` comes back
`unknown` and the four intact channels resolve normally, which is the behaviour
Correction 4 was made to produce.

## Limitations

These matter more than the tables above.

1. **The fixtures were built to exercise the rules the module implements.** They
   demonstrate that the code does what it documents. They measure nothing about
   real doors, and no diagnostic accuracy may be claimed from them. There is no
   ground-truth cause data in this repository against which to measure any.

2. **No output is a verdict.** The explanation order is fixed and carries no
   information. `supported_by` is a list of citations, not a score: two
   statements can restate one underlying fact, so its length is not a count of
   independent evidence.

3. **An isolated shift is not a sensor fault.** A genuinely single-channel
   mechanical effect produces the same pattern. The module says the change lacks
   independent corroboration; it cannot say why.

4. **Absence of corroboration is not exclusion, in either direction.** A
   coherent multi-channel change does not rule out a simultaneous sensor problem,
   and the `sensor_and_mechanical_overlap` example — which injects both — keeps
   both explanations supported.

5. **The thresholds are illustrative.** `shift_sigma=2.0`,
   `peer_share_sigma=1.0`, `location_offset_sigma=1.0`, `min_peers=3` and the
   context tolerances are transparent screening choices, not calibrated limits.
   Nothing has been tuned against outcomes, because there are no outcomes to tune
   against.

6. **Peer comparability is screened on three context means** within absolute
   tolerances, plus date overlap. Doors matched on temperature, load proxy and
   hour-of-day may still differ in ways that matter — position in the train, door
   type, service history.

7. **The physical families are asserted from the contract, not measured.**
   `FAMILY` and `DERIVED` encode an engineering claim about which door
   measurements share a transducer and signal path. That claim must be checked
   against real vendor telemetry before the independence argument carries weight.

8. **The reference comparison depends on the caller** supplying both views, and
   on the frozen fleet view being genuinely frozen. It cannot detect
   contamination on its own, and a recovered location offset cannot distinguish a
   contaminated reference from a genuine persistent build difference — only that
   the two baselines disagree, by how much, and in what units.

9. **A matching model identifier is a caller attestation, not authentication.**
   This code checks that both views name the same non-empty identifier and that
   the value is constant across every row used. It cannot verify that the
   identifier is truthful, any more than the onboarding provenance record can.

10. **Inferred baseline parameters are inferred.** Where
    `reference_parameters` is not supplied they are recovered from the published
    index by least squares. That is checked against a real fitted pipeline in the
    integration test, but supplying the exported parameters is strictly better
    and is the documented preference.

11. **Correlated observations.** Daily rows within a window are not independent.

12. **Change detection is a level comparison** between two robust medians. It
    does not model trend, seasonality or a changepoint, and a slow ramp entirely
    inside the reference window is invisible to it.

## Verification

`tests/test_inspection_evidence.py` **93 tests**, `tests/test_inspection_export.py`
**47 tests**, `tests/test_inspection_ui_js.py` **13 tests**. Full suite:
**469 passed, 13 skipped**.

Regressions reproducing each corrected defect:

* *Peer, fail closed* — a dropped context column, a context column with no
  tolerance, thin context support on peers and on the target, non-overlapping
  peer dates, duplicate asset-days, a channel every peer lacks (stays `unknown`,
  never `not shared`), and a channel three of five peers carry (reported as
  three).
* *Reference mathematics* — the location offset equals the injected offset in
  A.s; a 3× scale difference yields no contamination observation and an explicit
  contradiction; disjoint view dates, differing underlying residuals and a single
  view each produce `quantifiable: false` with a reason.
* *Statements* — with `peer_ids=[]` no statement mentions comparable peers; an
  unknown channel is described as unobserved rather than unchanged; a
  multi-family shift is absent from the sensor explanation's `contradicted_by`;
  the order is fixed across all fourteen cases.
* *Quality contract* — the real `HealthPipeline` frame with `data_quality_ok`
  False on every affected day still resolves the four intact channels; the global
  flag is used, and labelled, when completeness columns are absent; channel-level
  resolution does not bypass `health_inputs_complete`, `n_cycles` or a rejected
  onboarding reference.
* *Provenance* — mixed target-row identities are refused **both with and without**
  a `reference_model_versions` declaration; a declaration conflicting with
  constant rows, disagreeing identity columns, disjoint view dates and a missing
  identifier are each refused.
* *Admission gate* — unsupported rows and disjoint supported days quantify
  nothing; exported parameters do not skip the gate and come back
  `metadata_only`; inconsistent exported parameters are rejected on location or
  scale; valid exported ones reproduce the integration offsets.
* *Integration* — inferred parameters match a real fitted `HealthPipeline`'s
  exported `AssetBaseline` for both views on every resolved channel.
* *Source matching* — a matching export is accepted; a different hash, a
  different source name and a **previously good export whose artifact then
  changed** each produce the build-wide unavailable state with the reason named.
  A foreign record is dropped without condemning the rest, and a door whose only
  record is foreign becomes an omission. These run against a temporary data
  directory and a temporary artifact, so they test the check rather than today's
  files.
* *Per-door coverage* — an end-to-end test builds a two-door frame (one
  assessable and escalated, one never supported), writes it to a temporary
  parquet, runs the real exporter over it, and feeds the result back through
  `build_ui._inspection`: the assessable door **keeps its evidence** and the
  unassessable one carries `no_supported_assessment` with its reason. Alongside
  it: an unverifiable claim, an invented reason code, an unexplained omission,
  conflicting entries and an unexpected door are each rejected per-door while the
  valid door's evidence survives, and a source mismatch is still build-wide.
* *Door selection* — quiet, escalated, escalated-inside-the-window,
  duty-restricted, explicitly unknown, held, and the artifact-says-green /
  page-says-unknown case.
* *Rendered behaviour, under Node* — including the per-door unavailable state:
  an unassessable door shows its own reason while another door in the same
  payload keeps its evidence, a never-surfaced door keeps its distinct wording,
  and an unverified claim renders as rebuild-required. Also: the page's own
  renderer region is extracted
  from `ui/headway.html`, given a stub `DATA`, and executed: all-unknown wording,
  mixed-coverage wording limited to assessed channels, "not assessable" rather
  than "not measured", physical units with no raw column names, a record never
  rendering on another door, invisibility before its landing night, the
  not-recomputed and stale labels, the newest visible record winning, the
  unavailable state replacing everything, and hostile text being escaped rather
  than dropped.

**Two environment caveats, both still open here.**

Node is not installed on this machine, so the thirteen JavaScript behaviour tests
**skipped**, and `check_artifacts.py` needs `--skip-js` (which prints a warning)
to omit its syntax check, lower-margin probe and inspection-renderer probe.
Codex reports both JavaScript checks passing in its environment; that is not a
result this machine produced.

**Pixel verification remains outstanding.** Screenshots of the scrolled section
return blank in this browser pane. The section was verified live via the DOM and
`innerText` — five sections present, `<details>` collapsed, zero raw column
names, correct replay-clock labels, correct unknown wording — but how it *looks*
is unconfirmed.

---

# Dashboard integration

Shipped as a **read-only section inside door detail**. No new top-level page, no
change to alerts, aspects, urgency, the planner or repair verification, and no
change to any model or threshold. The light theme and the single-file build are
unchanged.

## Where the evidence comes from

`scripts/build_inspection_evidence.py` runs the real backend over
`data/door_deferral.parquet` — the same artifact the dashboard itself is built
from — and writes `data/inspection_evidence_export.json`, which
`scripts/build_ui.py` inlines. The static page cannot call Python, so this is the
same build-time export pattern already used for repair verification.

The handcrafted scenario fixtures are **not** part of this. They stay in
`data/inspection_evidence/`, are labelled illustrative in their own `index.json`,
and are never attached to a fleet door; `check_artifacts.py` asserts they do not
appear in the fleet export.

### Export strategy

| Decision | Why |
|---|---|
| **Which doors** — `build_ui.surfaced_assets`, imported, not re-implemented | 6 of 80 doors on this fleet. A door that never asks for attention does not need an evidence page. |
| **When** — at most two assessment times per door: its first escalation night and its last supported night | Recomputing on all 120 replay nights would multiply runtime and payload sixtyfold for no extra information; the evidence summarises a 5-day window against a 21-day reference. |
| **How much** — each record slimmed to rendered fields, floats rounded, the static suggestion catalogue hoisted out of every record | 12 records, **116 KiB**, about 2% of the page. |
| **Provenance** — `sourceArtifact` is `sha256:` of the telemetry frame; `modelVersion` is separate | A dataset hash identifies the frame, not the trained model, and is never labelled as one. No artifact in this build records a model identity, so `modelVersion` is `null` with `modelVersionReason` attached. Nothing is invented. |

### Door selection, reconciled with the page

The exporter does not have its own rule. `build_ui.display_rows` is the single
derivation of what the dashboard renders, and `build_ui.surfaced(rows)` is the
Status ∪ Review predicate transcribed from `app.js`:

```
Status:  aspect >= 2  or  aspect === -1  or  duty === 'off_peak_only'
Review:  aspect === 1, or held (aspect > 0 and aspect > raw),
         or aspect >= 2 / duty off_peak_only inside the 21-day window
```

This matters because the raw artifact and the rendered page disagree. When a
day's evidence is unsupported, `display_rows` forces `aspect` to `-1` and `duty`
to `not_assessed` — so a door the parquet records as green can be one the page
shows as unknown, and unknown is a Status condition. Selecting from the artifact
would have missed exactly those doors. A regression test builds that case: the
artifact says aspect 0, the display rows say `-1`, and the door is selected.

### Coverage: every surfaced door is explained

A door can be surfaced by Status or Review and still be impossible to assess —
if no day in the replay produced a supported daily aggregate, there is nothing to
compare. That is a fact about **that door**, and an earlier version got it wrong:
the exporter recorded such doors in a `skipped` map, `build_ui` ignored the map
while checking coverage, and one unassessable door therefore invalidated every
other door's evidence build-wide.

Coverage is now explicit. Every surfaced door must have **either** assessment
records **or** an entry in `unavailableDoors` carrying a reason code, and
`build_ui` re-checks that claim against the telemetry it is building from:

| Situation | Result |
|---|---|
| Records, identity matches | evidence kept |
| `reasonCode: no_supported_assessment`, and this build's frame agrees the door has zero supported nights | accepted; the page shows the specific reason |
| A reason code this build cannot verify | rejected → `unverified_claim`, rebuild required |
| Claim contradicted by the frame (it does have supported nights) | rejected → `unverified_claim`, with the count |
| Both records and an unavailable entry | rejected → `conflicting_entries`; neither is used |
| Surfaced, but neither assessed nor explained | rejected → `unexplained_omission` |
| A door this build does not surface | its entries are dropped and the problem recorded |

An export cannot skip a door out of the completeness check by inventing a reason:
`UNAVAILABLE_REASONS` is a closed set of codes `build_ui` knows how to verify.
Every one of these is **per-door** — the other doors keep their evidence, and
`doorProblems` records what happened. Only a source-artifact mismatch remains
build-wide.

The page renders the door's own reason: *"No assessment available for this door.
No day in the replay produced a supported daily aggregate for this door, so no
change could be assessed."* That is deliberately distinct from *"No inspection
evidence for this door. Evidence is exported only for doors this dashboard
surfaces."* — a door nobody needed to look at — and from *"No inspection evidence
had been produced for this door by …"* — an assessment that had not happened yet
on the selected replay night.

### Source matching, enforced where the page is built

An export is only evidence about the frame it was computed from. `build_ui`
checks this at the moment it assembles the page, not afterwards in a separate
script looking at whatever files happen to be on disk:

* the export must name the same source artifact as the frame being built, **by
  name and by hash**;
* **every record** must carry that same export identity, or it is dropped;
* the exported door set must equal the set these very rules surface.

Any mismatch produces an explicit unavailable state — `build_ui.main()` prints
the problems, and the section renders **"Unavailable — rebuild required"** with
the reason and each problem listed, instead of stale evidence shown as current.
Demonstrated end to end: editing `sourceArtifact` in the export and rebuilding
prints

```
Inspection evidence UNAVAILABLE — rebuild required:
  - source artifact hash differs: export 'sha256:0000000000000000', this build 'sha256:ce0c10e3c5068b88'
```

and the payload comes back with `unavailable: true` and zero records.

### The replay clock

Every record carries `visibleFrom`, the first replay night at which it could have
existed (`day + 1 day >= as_of`), computed by the same `_visible_from` the
verification export uses. The page then enforces three things:

* **No future evidence.** Before `visibleFrom` the section reads "No inspection
  evidence had been produced for this door by *date*."
* **No cross-asset substitution.** Records are matched on `assetId`; a door with
  no record at all reads "No inspection evidence for this door."
* **No silent carry-forward.** When the selected night is later than the
  assessment, the header says "not recomputed for *date* — N days later", and
  beyond `staleAfterDays` (7) it says "stale — N days old". Only an assessment
  made for the selected night reads "current for this date".

### Reference views

The demo pipeline fits one fleet model and runs no onboarding, so there is no
adapted view to compare against a frozen one. The backend returns
`quantifiable: false`, and section 4 displays its reason verbatim: *"no reference
views supplied; the adaptation effect is not visible."* No model identifier or
inspection provenance is invented to fill the gap, and a `metadata_only` result
can never render as a measured adaptation effect.

## What the section shows

1. **What changed** — plain channel names ("Motor charge per cycle"), the change
   in physical units (`+0.5699 A.s`), and the direction. The unknown states are
   worded precisely:
   * nothing assessable → **"Insufficient evidence to assess change."** — never a
     no-change finding;
   * mixed coverage → "No change beyond the reporting threshold in the *N*
     channels that could be assessed (*M* not assessable)", so a no-change
     statement only ever covers channels that were actually assessed, and
     "Assessed and steady" lists only those;
   * a channel whose reading exists but fails the quality, reference or
     completeness requirements is **not assessable**, not "not measured", with
     its reason shown.
   Medians, scales, day counts and the quality basis sit in a `<details>` block,
   collapsed by default.
2. **Peer evidence** — comparable-peer counts **per channel**, with *shared*,
   *not shared* and *unknown* as three distinct labels, plus the peer median.
   Where comparability is unresolved the whole section reads Unknown with the
   reason.
3. **Possible explanations** — all five, in the backend's fixed order, each with
   its statement, supporting observations and counter-evidence, followed by the
   missing-evidence list. Every card has identical styling, so *Cannot
   distinguish* carries the same visual weight as the rest. No percentages, no
   "most likely", no reordering.
4. **Reference comparison** — the baseline location difference in physical units
   and the scale ratio separately, with provenance and exported/inferred status —
   or the reason it cannot be quantified.
5. **Evidence that would help** — labelled illustrative and pending operator
   review, described as records and measurements to consider, not a procedure.

A "Provenance and limits" `<details>` block, also collapsed, carries the
limitation text, the quality basis, the window, the evidence hash and the export
time.

Canonical column names are rendered as plain names at the display layer only
(`iplain()`), applied after escaping. The stored record keeps the canonical names.

## Demo walkthrough

1. Rebuild: `python scripts/rebuild.py`. The chain now includes
   `build_inspection_evidence.py` between verification and the page.
2. Open `ui/headway.html`. **Status** lists the doors needing action.
3. Click a door id — for example **TRN021-DOOR-2** — to open Door detail.
4. Scroll past the chart, the fit-for-duty card and Repair verification. The
   **Inspection evidence** section is below them, headed "read-only — explains
   the observations; does not identify a cause, change alerts or urgency, or
   authorise maintenance".
5. The meta line reads *Assessed 2026-07-30 00:00 SGT · door TRN021-DOOR-2 ·
   source artifact sha256:… · current for this date*. "Provenance and limits"
   spells out that the hash identifies the telemetry, not a trained model, and
   that no model version was recorded.
6. Section 1 shows five channels moving together — motor charge `+0.5699 A.s`,
   cycle time `+0.2458 s`, peak current `+0.6687 A`, door travel `-4.2334 mm`.
   Section 2 reports 79 comparable doors, none sharing the change. Section 3
   therefore supports both "a measurement or sensor change" and "a mechanical
   change", and says in words that independent channels moving does not exclude a
   sensor problem.
7. **Drag the chart back in time.** At 2026-07-25 the header changes to "not
   recomputed for 2026-07-25 — 3 days later"; at 2026-07-28 it reads "stale — 9
   days old"; before 2026-07-09 the section reports that no evidence had been
   produced yet.
8. Open a door with no record — most of the fleet — and the section says so
   explicitly rather than showing anything favourable.
