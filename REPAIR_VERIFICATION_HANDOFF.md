# Repair verification: backend handoff

This feature is isolated from the dashboard work. Only the following new files
belong to this implementation:

- `headway/repair_verification.py`: reference builder, verification, SQLite store.
- `scripts/verify_repair.py`: JSON CLI for recording work and running assessments.
- `tests/test_repair_verification.py`: unit and CLI integration tests.
- This handoff document.

No dashboard, existing planner, model, generated fleet or Git state is changed.

## User workflow

1. An engineer records a unique job ID, asset, maintenance start/completion,
   recording time, operator, inspection finding and work performed.
2. Verification uses an explicitly reviewed healthy reference that was available
   before this maintenance and comes from the same frozen preprocessing model.
3. After a settling period, compare complete post-maintenance days against each
   reference channel separately. Channels cannot cancel one another out.
4. Display signal recovered, abnormality persists, or insufficient evidence.
5. Persistent abnormality atomically creates an open follow-up alongside its
   assessment. A later favourable result does not silently close that follow-up.

This checks telemetry consistency with a reference. It does not establish causal
repair effectiveness, confirm the diagnosis, retrain a model or release a train
to service. Existing maintenance escalations remain owned by their existing policy.

## Inputs and reference eligibility

Use daily **unsmoothed**, condition-normalised channels from one frozen pipeline,
for example `current_integral_as_hx`, `cycle_duration_s_hx`, `peak_current_a_hx`,
`mean_current_a_hx` and `travel_mm_hx`. Do not use a trailing smoothed signal:
its post-maintenance rows may include pre-maintenance observations.

Required columns: `asset_id`, `day`, `available_at`, boolean `data_quality_ok`,
boolean `context_supported`, each selected channel, and each selected operating
context. Example context columns are `ambient_temp_c_mean` and `load_proxy_mean`.
Use all operating contexts relevant to the comparison. Supported fleet context
alone is insufficient: post-maintenance daily context must also fall within the
reviewed reference's observed ranges. This conservative range check is not joint
distribution matching and does not establish peak-cycle comparability.

Select the healthy reference window explicitly; the builder does not infer that
quiet readings mean a healthy door. Supply a real inspection attestation in
`reviewed_by` and `reviewed_at`. Test identifiers in tests are synthetic fixtures,
not operator records. Supply a stable preprocessing/model version (preferably an
artifact hash). The caller is responsible for the model version's authenticity
and for causal upstream preprocessing; this module cannot detect relabelled or
retrospectively refitted inputs.

Naive timestamps are interpreted as Asia/Singapore. Stored times use UTC. `day`
must represent Singapore midnight; full-day availability cannot precede the next
midnight. Assessments use only rows available at `as_of`.

Default policy, all **illustrative and not operationally calibrated**:

| Rule | Default |
|---|---|
| Healthy reference | At least 14 consecutive valid days |
| Settling interval | One day after the first midnight at/after completion |
| Verification window | Latest three fully completed calendar days |
| Per-channel departure | Absolute deviation greater than 3 reference robust scales |
| Recovered | Every channel within the limit on all three days |
| Persistent | At least one channel beyond the limit on two or more days |
| Mixed evidence | Insufficient evidence; continue observation or inspect |

Scale is `1.4826 * median(abs(value - reference median))`. Degenerate reference
scales are rejected. These robust deviations are not probabilities; a threshold
of 3 is not a claimed Gaussian tail rate. Both higher and lower departures count,
because an unexpected low reading may indicate a sensor issue.

Incomplete/stale windows, unavailable data, unsupported conditions, changed
models and subsequent recorded maintenance yield insufficient evidence.

## Python integration

```python
from headway.repair_verification import RepairStore, make_reference

# Explicit caller-selected reference_daily contains healthy, reviewed rows only.
reference = make_reference(
    reference_daily, asset_id=asset_id,
    channels=["current_integral_as_hx", "cycle_duration_s_hx", "travel_mm_hx"],
    contexts=["ambient_temp_c_mean", "load_proxy_mean"],
    model_version=model_hash, reviewed_by=inspector, reviewed_at=review_timestamp,
)
store = RepairStore("data/repair_verification.sqlite")
store.record(
    job_id=job_id, asset_id=asset_id, started_at=started_at,
    completed_at=completed_at, recorded_at=recorded_at, operator=operator,
    finding=finding, work_performed=work_performed,
)
result = store.assess(job_id, reference, daily, as_of=assessment_time,
                      model_version=model_hash)
history = store.history(job_id)
open_followups = store.followups()
```

Persist the reference JSON returned by `make_reference`. The CLI reference command
creates that file exclusively and refuses to overwrite an existing one. Assessments
also store the complete reference snapshot. Each assessment has its own ID,
timestamp, per-channel deviations, observed-day count and an evidence hash when
the window is usable. Duplicate job IDs are rejected; a second repair on the same
asset requires a new ID. Database writes use transactions and a 30-second lock timeout.

The local SQLite store is durable, but not an authenticated or tamper-proof audit
system. Operator names are caller attestations, not verified identities. Follow-up
resolution, record corrections, backups and shared-server authorisation are future
work. There is deliberately no silent delete or automatic follow-up closure API.

## CLI

Run from the repository using the existing virtual environment. The reference
Parquet must already contain only the selected healthy reference period.

```powershell
.\.venv\Scripts\python.exe -B scripts/verify_repair.py --help
.\.venv\Scripts\python.exe -B scripts/verify_repair.py reference --daily healthy_reference.parquet --asset TRN001_D01 --channels current_integral_as_hx cycle_duration_s_hx travel_mm_hx --contexts ambient_temp_c_mean load_proxy_mean --model-version YOUR_MODEL_HASH --reviewed-by YOUR_INSPECTOR --reviewed-at YOUR_REVIEW_TIMESTAMP --output reviewed_reference.json
.\.venv\Scripts\python.exe -B scripts/verify_repair.py record --db data/repair_verification.sqlite --record maintenance_record.json
.\.venv\Scripts\python.exe -B scripts/verify_repair.py assess --db data/repair_verification.sqlite --job YOUR_JOB_ID --reference reviewed_reference.json --daily data/door_daily.parquet --as-of YOUR_ASSESSMENT_TIMESTAMP --model-version YOUR_MODEL_HASH
.\.venv\Scripts\python.exe -B scripts/verify_repair.py history --db data/repair_verification.sqlite --job YOUR_JOB_ID
.\.venv\Scripts\python.exe -B scripts/verify_repair.py followups --db data/repair_verification.sqlite
```

Replace illustrative IDs, paths and timestamps with actual inputs. Maintenance
JSON keys match `store.record` above. Commands emit JSON; error exits are nonzero.

## Dashboard contract for Claude

The static HTML cannot call Python/SQLite directly. Connect through a separately
authorised local/server API or export assessment JSON into the dashboard payload.
Until connected, do not claim the planner's Done button runs verification.

| Result field | Suggested display |
|---|---|
| `status=signal_recovered` | Signal recovered — inspection/release decision remains with engineer |
| `status=abnormality_persists` | Abnormality persists — follow-up review required |
| `status=insufficient_evidence` | Verification pending — show `reason` |
| `as_of`, `window_start`, `window_end` | Assessment time and observed period |
| `channels` | Per-channel repeated departures and peak deviation |
| `reference_id`, `model_version` | Evidence provenance in expandable details |
| `follow_up_required` | A review is required by this assessment |

Read `store.followups()` separately: an earlier open follow-up remains open even
when the latest assessment has `follow_up_required=false`. Do not turn a latest
favourable result into implicit closure or service clearance. `release_to_service`
is always false; the verification backend has no authority to release service.

## Verification

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_repair_verification.py -q -p no:cacheprovider
```

All 19 tests passed. Tests cover all three outcomes, missing/unsupported/late data, mixed channels,
reference integrity, model changes, future-data isolation, timezone equivalence,
SQLite persistence, repeat jobs, retained follow-ups and the full CLI round trip.
These tests establish software behaviour, not performance on real repair outcomes.
