# Review of the latest implementation

10 September 2026. Reviewed the current repository, implementation, tests and
technical notes. This review does not reconstruct Claude's private conversation.

## Work already present

- Peak-load fit-for-duty assessment, integrated into the daily pipeline and UI.
- Status, Review, Planner, Fleet and door-detail dashboard views.
- Local-browser planner ordering and deferral choices.
- Wear/load interaction, repair decay and late-record episodes in the simulator.
- Dependency-ordered rebuild script and refreshed submission/technical notes.

## Corrections made in this continuation

1. **Spurious load sensitivity across days.** The regression pooled cycle data
   across days although its explanation promised within-day contrast. Changes in
   daily wear and average load could generate a positive slope without any
   within-day load effect. Daily sufficient statistics are now centered before
   rolling aggregation, with one intercept per observed day in residual degrees
   of freedom. A deterministic regression case reproduces the confounding.
2. **Unknown evidence behind a retained escalation.** Aspect hysteresis can retain
   an old red/amber state while today's evidence is unavailable. Duty previously
   tested only the retained aspect, permitting an affirmative duty label. It now
   checks the prediction state first and reports not assessed without clearing
   the maintenance escalation.
3. **Unsupported operating conditions in the sensitivity history.** Unsupported
   cycles are now excluded from sufficient statistics so they cannot contribute
   to later duty assessments through the trailing window.
4. **Zero/one-episode simulator failure.** The late-episode placement always drew
   at least two dates. It now respects the configured episode count, including
   zero. Both cases have regression tests.
5. **Overstated wording.** The UI now says "Estimated lower margin" instead of
   "Safe time left". Technical notes no longer claim a single proven cause for
   the generalisation gap, a proven onboarding fix, or a 30-minute real-data
   integration time. The OLS t statistic is explicitly an uncalibrated diagnostic
   when cycles are correlated.
6. **Displayed margins rounded upward.** A lower margin of 1.96 days could be
   shown as at least 2 days. Both card and planner now round down to tenths;
   zero is labelled no positive margin. The artifact checker executes the actual
   card formatter on boundary examples and rejects overstated displays.
7. **Windows artifact-check encoding.** JavaScript was passed to Node using the
   system code page, which failed on Unicode labels. The checker now explicitly
   uses UTF-8 and bounds subprocess runtime.

## Remaining priorities

**Model evidence:** preserve a fixed experiment protocol for contaminated
onboarding references, missing observations and context shift, including error
comparisons on identical scored rows. The earlier onboarding experiment does not
establish reliable 90% coverage. New simulator mechanics also change the evidence
base; retain generator configuration and source/data hashes with future runs.

**Duty validity:** test with no wear/load interaction and with independent changes
in temperature, crowding and wear. Current positive demonstrations deliberately
include the interaction the method searches for. Correlated cycles can overstate
   the t diagnostic. Peak shifts also reuse the ordinary RUL calibration; the model
does not estimate how a timetable change alters failure time. A peak restriction
must not be presented as proof that off-peak operation is safe.

**Planner scope:** browser local storage is a convenience for ordering proposed
jobs. It is not a persistent audited deferral ledger: there is no shared decision
history, authenticated operator identity, recorded justification or immutable
snapshot of model evidence. These are separate implementation requirements.

**Simulation scope:** late-record episode placement is a demo design choice and
affects the chronological test distribution. Repair decay is generated after a
fault rather than driven by independently observed maintenance records. Neither
should be treated as operator evidence.

## Verification

The complete regression suite passed **206 tests**. Artifact checks passed for
9,600 asset-days and 80 doors, including unknown-duty states, window comparisons,
JavaScript syntax and conservative margin display. Rebuild and evaluation outputs
are tracked in `data/validation_report.json` and `data/duty_summary.json`.
No staging, commits or pushes were performed. Browser visual inspection is not
claimed by this review.

The rerun chronological duty diagnostic restricts all six future episodes before
the fault, with a median lead of 2.09 days and zero restrictions among 2,647
assessed healthy asset-days. Healthy days flagged sensitive fall from 56 to 45;
sensitivity alone is not a restriction. Entire-train RUL coverage remains 75.8%,
with 8.65-day point MAE among scored rows and 37.3% abstention. These fixes improve
correctness; they do not resolve the generalisation gap.
