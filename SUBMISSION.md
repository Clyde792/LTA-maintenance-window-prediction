# Headway — Time to act

Draft submission. Current demonstration uses synthetic door telemetry.

Headway helps an engineer move from a changing sensor signal to a reviewable
maintenance decision. It combines operating-condition normalisation, per-asset
reference behaviour, multiple health signals and remaining-life estimates to
compare the maintenance windows an engineer is actually considering.

Each door has an action, an estimated lower RUL margin where supported, a visible
data state, and the evidence behind the estimate. The window comparison can say
that a proposed delay exceeds the estimated margin. When the data cannot support
a countdown, it says so.

The distinctive workflow is a deferral decision that can be revisited as new
evidence arrives. Recording the operator's accepted window, override reason,
reassessment triggers and maintenance outcome is the next product extension;
that persistent decision-record workflow is not yet implemented.

## Implemented

- Canonical telemetry contract and adapter scaffolding.
- Training-frozen context normalisation and historical asset baselines.
- Directional multivariate health score, separate anomaly distance and channel contributions.
- Calendar-time trends and explicit aggregate availability timestamps.
- Empirical RUL bounds with train/asset holdout diagnostics.
- Maintenance-window comparison without unsupported failure percentages.
- Light dashboard with unknown states and decision-stability labels.
- Independent chronological and end-to-end held-out-train evaluation.

## Evidence and limits

Use the generated validation report for measured results. Earlier prototype
precision, warning-time, calibration and safety claims are superseded by the
corrected evaluation and must not be reused without verification.

There are nine synthetic fault episodes in the current full demonstration.
Thousands of telemetry rows are not thousands of independent failures. No real
fleet effectiveness, individual failure probability, safe deferral guarantee or
bogie prediction performance is claimed.

The maintenance thresholds are illustrative. Real deployment requires operator
policy, fault semantics, data quality checks and representative outcome validation.
