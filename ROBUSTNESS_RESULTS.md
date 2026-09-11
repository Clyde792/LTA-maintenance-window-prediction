# Synthetic robustness experiments — 11 September 2026

These experiments test failure modes of the monitoring pipeline before the
organiser's dataset arrives. They are not evidence of real rail performance,
and do not select or change the dashboard's deployment model.

## Implemented protocol

Run `scripts/experiment_robustness.py`. Two fixed seeds (20260911, 20260912)
each generate 12 trains, two doors per train, 120 days and six fault episodes.
The first train by identity is excluded from all preprocessing and detector
fitting. Fit on the other trains' first 30 days; freeze the models and their
99th-percentile score thresholds before replaying after day 84. Threshold
calibration includes days 31–84 and is not claimed to be verified healthy.

Compare raw current, directional health index, Mahalanobis and Isolation Forest
under clean telemetry, 30% missing primary-channel samples, a +0.6 A.s sensor
offset, a +20°C context-input corruption, and every-third-day outages. Stress
affects the first three train identities, chosen without examining labels.
Models and thresholds remain identical across scenarios within a seed.

Scoring requires three supported, complete daily readings in a trailing
three-calendar-day window. Invalid inputs are excluded before smoothing.
Expected asset-days remain in the availability denominator during outages.
Whole-fleet and affected-asset replays each have their own two-alert daily
budget and three-day cooldown; counts from those populations are not additive.

## Findings

| Affected assets, both seeds | Observed result | Implication |
|---|---|---|
| Clean telemetry | Scores available for 94.4%–100% of expected asset-days | Even clean synthetic telemetry does not guarantee continuous monitoring. |
| 30% primary-channel sample loss | Scores available for 0%–0.46% of asset-days | Abstention avoids unsupported predictions but almost eliminates monitoring. |
| Every-third-day outage | No scores | Three consecutive daily observations never accumulate. |
| Unsupported context inputs | No scores | Context support guards operate; zero alerts must not be presented as good health. |
| +0.6 A.s sensor offset | Raw detector unmatched alerts rise from 0 to 3.47–8.06 per asset-month | Sensor problems can overwhelm a simple threshold. These alerts could still justify sensor inspection. |

In a separate reference-contamination test, both new-asset references were
accepted despite a constant offset from the start. For the exact same shifted
post-reference telemetry, the median primary-channel normalized residual fell
from 27.92 to 0.017 in one seed and 29.16 to 0.142 in the other when the offset
was included in the reference. A stable baseline is not proof of health.
This test cannot distinguish a sensor offset from actual mechanical wear.

The paired held-train RUL comparison had 18 eligible daily rows in one seed:
MAE changed from 5.13 to 3.67 days with onboarding. The other seed had zero
matched rows and provides no RUL comparison. Daily rows are correlated, not
18 independent failures. This comparison is retrospective: other trains'
later fault labels train the RUL model. It does not establish prospective
performance or calibrated uncertainty.

## Recommended next changes

1. **Evidence-status output.** Report mechanical health separately from
   monitoring availability: supported, incomplete, stale or outside the
   training conditions. Show the last supported assessment and missing channels.
   A quiet detector during an outage must not imply a healthy asset.
2. **Reference provenance.** Require an inspection/maintenance-backed eligible
   reference before operational onboarding. Preserve reference dates, reviewer
   and evidence identity. Keep an unadapted fleet comparison visible to expose
   what adaptation removed, without labelling that difference a confirmed fault.
3. **Evaluate an outage-tolerant detector alongside the strict smoother.** Test
   minimum observations, maximum time gap and freshness explicitly. Do not
   silently carry old scores forward. Compare fault capture, alert burden and
   availability on fresh seeds, then the official held-out data.
4. **Sensor-versus-mechanism evidence.** Compare current, duration and travel
   changes and whether neighboring doors share a shift. Describe competing
   explanations for inspection, rather than asserting a root cause without
   verified labels. Waveform features remain conditional on the supplied schema.

These are evidence-driven priorities, not implemented dashboard features.
The current experiment's outcomes must not be reused as an untouched test set
after tuning against them. No model is promoted from these results.

## Reproduce and inspect

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts/experiment_robustness.py
.\.venv\Scripts\python.exe -m pytest tests/test_robustness.py tests/test_detector_selection.py tests/test_data_readiness.py -q -p no:cacheprovider
```

Outputs: `data/robustness_experiment/protocol.json`, `comparison.csv`, and one
JSON report per seed. The protocol records script and scoring-module hashes.
The focused verification suite passes 24 tests; this is not a full-project
test-suite claim. No organiser data was used.
