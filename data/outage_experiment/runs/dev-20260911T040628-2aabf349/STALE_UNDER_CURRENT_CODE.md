# This run's freeze no longer binds to the current code

Status: **results stand as a record; the freeze can no longer be evaluated.**

After this run was produced, `scripts/experiment_outage_tolerance.py` was changed
to add freeze-integrity checks (re-derivation of the winner from the development
results, refusal to overwrite runs, and removal of an unverifiable blindness
claim). Those are bookkeeping fixes and change no measurement, but the freeze is
deliberately bound to a hash of the script, so:

```
$ python scripts/experiment_outage_tolerance.py --stage eval --run dev-20260911T040628-2aabf349
Refusing to evaluate a stale policy:
  - parameters or acceptance criteria differ from the frozen protocol
  - experiment_script changed since the policy was frozen
```

This is the guard working, not a fault. The protocol embeds the code hashes, so a
script change moves the canonical protocol hash too — hence both lines.

The expensive sweep was **not** re-run for a bookkeeping change. The numbers in
`dev_results.json`, `eval_results.json` and `OUTAGE_TOLERANCE_RESULTS.md` remain
the numbers this run produced, under the code hashed in `frozen_policy.json`.

To re-verify under the current code, produce a **new** run (the runner will not
overwrite this one):

```
python scripts/experiment_outage_tolerance.py --stage dev
python scripts/experiment_outage_tolerance.py --stage eval --run <new run id>
```

Note that any such evaluation is a reproduction on **already exposed** seeds
(20260915, 20260916) and must not be described as blind. See `frozen_policy.json`,
field `blinding`.
