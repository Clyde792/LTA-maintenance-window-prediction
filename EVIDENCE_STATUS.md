# Monitoring evidence and onboarding references

The dashboard now reports telemetry support separately from mechanical health.
Each daily replay row has a monitoring status, incomplete channel names, and
the last supported assessment's availability time. Entire missing calendar
days remain visible. Unsupported rows cannot show a healthy recommendation,
countdown, maintenance-window assurance or duty suitability; an earlier
escalation remains held. Historical charts leave gaps for unsupported readings.

Statuses: supported, missing, not yet available, stale, outside training
conditions, incomplete, and collecting sufficient history. Staleness uses a
configurable one-day maximum age in `headway.evidence.evidence_status`.
The static dashboard assesses each row at its replay midnight, not the wall
clock; it does not claim to monitor a live feed. A supported assessment is not
proof of health or a calibrated RUL estimate.

Operational `onboard` now requires per-asset provenance:

```python
records = {asset_id: {
    "eligible": True,
    "reviewer": "Engineer identifier",
    "evidence_id": "Inspection or maintenance record identifier",
    "reference_start": reference_start,
    "reference_end": reference_end,  # exclusive end; covers the entire window
    "reviewed_at": review_time,
}}
adapted = onboard(pipe, reference, as_of=now, provenance=records)
```

Reviews must cover the reference window and be available by `as_of`. Reference
adaptation cannot support replay decisions before the review time. The returned
object preserves a copy of the provenance; transformed rows expose verification
status and evidence ID only after approval becomes available. Rejected assets
abstain. These fields are caller attestations: this code does not authenticate
reviewers or retrieve inspection documents.

Synthetic experiments explicitly pass `experimental=True`. This permits
unverified reference adaptation for research and sets `reference_verified=False`.
No historical inspection records are invented for the demo. Existing experiment
reports describe their original runs; changing this API does not regenerate or
retroactively validate those reports.

The change does not add a browser inspection-entry form, authentication, or a
live backend. Provenance is supplied through the Python onboarding API. The
existing dashboard source and generated single-file artifact retain their
light theme.
