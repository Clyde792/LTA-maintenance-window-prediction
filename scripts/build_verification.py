"""Export repair-verification evidence for the dashboard, and (optionally) seed a
separate demonstration store.

The static dashboard cannot call Python or SQLite, so this script reads the
verification stores at build time and writes a JSON snapshot that
scripts/build_ui.py inlines into the page.

TWO STORES, deliberately:

  data/repair_verification.sqlite       the operational store documented in
                                        REPAIR_VERIFICATION_HANDOFF.md. This
                                        script only ever READS it. It is never
                                        deleted, replaced or written to here.
  data/repair_verification_demo.sqlite  demonstration records only. `--seed-demo`
                                        rebuilds this file from scratch; nothing
                                        else is touched.

Everything seeded by `--seed-demo` is SIMULATED. The maintenance records describe
work that never happened, on synthetic telemetry, and their `reviewed_by` /
`operator` fields say so in words rather than naming a person. Nothing seeded here
should be read as a maintenance history or as an engineer's attestation.

Three cases are constructed so the dashboard can show all three outcomes. Rather
than hard-coding doors and dates, each case is SEARCHED for: candidate episodes
and assessment times are dry-run through `verify_repair` and the first
combination that actually produces the wanted status is committed. That keeps the
demonstration honest (the backend decides the outcome, not this script) and keeps
it working when the simulator is regenerated.
"""
import argparse,hashlib,json,os,sqlite3,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from headway.repair_verification import RepairStore,make_reference,verify_repair

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
STORE_DB=DATA/"repair_verification.sqlite"          # operational: read-only here
DEMO_DB=DATA/"repair_verification_demo.sqlite"      # demonstration: rebuilt on request
OUT=DATA/"verification_export.json"

CHANNELS=["current_integral_as_hx","cycle_duration_s_hx","peak_current_a_hx","travel_mm_hx"]
CONTEXTS=["ambient_temp_c_mean","load_proxy_mean"]

# Said in words, so it can never be mistaken for a real inspection signature.
SIMULATED="SIMULATED — no inspection took place (synthetic demonstration data)"
FINDING="SIMULATED finding — not a real inspection"
WORK="SIMULATED work record — no maintenance was performed"
# Longer than the 14-day policy minimum on purpose: the assessment rejects a
# post-maintenance window whose operating context falls outside the reference's
# OBSERVED range, so a short reference is self-defeating in a seasonal signal.
REF_DAYS=30

def model_version():
    """A real artifact hash of the preprocessing output the assessment reads."""
    return "sha256:"+hashlib.sha256((DATA/"door_daily.parquet").read_bytes()).hexdigest()[:16]

def usable(daily,asset):
    d=daily[(daily.asset_id==asset)&daily.data_quality_ok&daily.context_supported].sort_values("day")
    return d[d[CHANNELS+CONTEXTS].notna().all(axis=1)]

def healthy_reference(daily,asset,onset):
    """The last REF_DAYS consecutive usable days closing before degradation ONSET.

    Onset, not the maintenance date: the reference has to be genuinely healthy,
    and the module trusts the caller for that rather than inferring it from quiet
    telemetry.
    """
    d=usable(daily,asset)
    d=d[d.day<pd.Timestamp(onset).floor("D")]
    if len(d)<REF_DAYS: return None
    for start in range(len(d)-REF_DAYS,-1,-1):
        w=d.iloc[start:start+REF_DAYS]
        if w.day.diff().dropna().eq(pd.Timedelta(days=1)).all():
            return w
    return None

def attempt(daily,mv,asset,onset,completed_day,as_of_day):
    """Dry run: no store write, so a search can try many combinations."""
    ref_rows=healthy_reference(daily,asset,onset)
    if ref_rows is None: return None
    completed=pd.Timestamp(completed_day)+pd.Timedelta(hours=3)   # in the engineering window
    started=completed-pd.Timedelta(hours=2)
    reviewed_at=ref_rows.available_at.max()+pd.Timedelta(hours=1)
    if reviewed_at>=started: return None
    reference=make_reference(ref_rows,asset_id=asset,channels=CHANNELS,contexts=CONTEXTS,
        model_version=mv,reviewed_by=SIMULATED,reviewed_at=reviewed_at)
    record=dict(job_id="dry-run",asset_id=asset,started_at=started,completed_at=completed,
        recorded_at=completed+pd.Timedelta(minutes=30),operator=SIMULATED,
        finding=FINDING,work_performed=WORK)
    as_of=pd.Timestamp(as_of_day)+pd.Timedelta(hours=6)
    result=verify_repair(record,reference,daily,as_of=as_of,model_version=mv)
    return {"reference":reference,"record":record,"as_of":as_of,"result":result}

def search(daily,mv,eps,want,offsets,completed_offset):
    """First (episode, as_of) whose assessment actually returns `want`."""
    for ep in eps.itertuples():
        for off in offsets:
            completed=ep.fault_ts.floor("D")+pd.Timedelta(days=completed_offset)
            trial=attempt(daily,mv,ep.asset_id,ep.onset_ts,completed,
                          ep.fault_ts.floor("D")+pd.Timedelta(days=off))
            if trial and trial["result"]["status"]==want:
                return trial
    return None

def record_job(store,trial,job_id):
    r=trial["record"]
    store.record(job_id=job_id,asset_id=r["asset_id"],started_at=r["started_at"],
        completed_at=r["completed_at"],recorded_at=r["recorded_at"],operator=r["operator"],
        finding=r["finding"],work_performed=r["work_performed"])

def assess_job(store,trial,job_id,purpose,mv,daily):
    result=store.assess(job_id,trial["reference"],daily,as_of=trial["as_of"],model_version=mv)
    print(f"  {job_id}  {trial['record']['asset_id']}  {result['status']} — {result['reason'][:60]}")
    return result

def seed_demo(demo_db,daily,eps,mv,last):
    """Rebuild the DEMONSTRATION store. Only this path is ever deleted."""
    if demo_db.exists(): demo_db.unlink()
    store=RepairStore(demo_db)
    print(f"Seeding demonstration store {demo_db.name}:")
    purposes={}

    # Recovered: maintenance the day after the fault, assessed once the repair
    # decay has run out. Try progressively later assessment times.
    rec=search(daily,mv,eps,"signal_recovered",range(6,20),completed_offset=1)
    if rec:
        record_job(store,rec,"DEMO-RECOVERED-1")
        # Assess the SAME job early as well, while the repair is still settling.
        # That opens a follow-up, so the later recovered result demonstrates the
        # rule that a favourable assessment never silently closes an open review.
        early=None
        for off in range(3,int((rec["as_of"]-pd.Timestamp(rec["record"]["completed_at"])).days)+1):
            t=attempt(daily,mv,rec["record"]["asset_id"],
                      eps.set_index("asset_id").loc[rec["record"]["asset_id"],"onset_ts"],
                      rec["record"]["completed_at"].floor("D"),
                      rec["record"]["completed_at"].floor("D")+pd.Timedelta(days=off))
            if t and t["result"]["status"]=="abnormality_persists": early=t; break
        if early:
            r=assess_job(store,early,"DEMO-RECOVERED-1","",mv,daily)
            purposes[r["assessment_id"]]="Earlier assessment of the same job, while the repair was still settling. It opened a follow-up."
        r=assess_job(store,rec,"DEMO-RECOVERED-1","",mv,daily)
        purposes[r["assessment_id"]]=("Successful repair: every monitored channel is back inside the healthy reference. "
            "The follow-up opened by the earlier assessment stays open — only a person closes it.")

    # Persistent: "completed" mid-degradation, so the following days stay abnormal.
    per=None
    for back in (5,6,7,8,9,10):
        per=search(daily,mv,eps[::-1],"abnormality_persists",range(-back+4,1),completed_offset=-back)
        if per: break
    if per:
        record_job(store,per,"DEMO-PERSISTENT-1")
        r=assess_job(store,per,"DEMO-PERSISTENT-1","",mv,daily)
        purposes[r["assessment_id"]]="Abnormality persists: the signal did not return to the reference; a follow-up review is open."

    # Insufficient: assessed before the settling interval and three complete days.
    used={t["record"]["asset_id"] for t in (rec,per) if t}
    ins=None
    for ep in eps[::-1].itertuples():
        if ep.asset_id in used: continue
        t=attempt(daily,mv,ep.asset_id,ep.onset_ts,last-pd.Timedelta(days=1),last)
        if t and t["result"]["status"]=="insufficient_evidence":
            ins=t; break
    if ins:
        record_job(store,ins,"DEMO-INSUFFICIENT-1")
        r=assess_job(store,ins,"DEMO-INSUFFICIENT-1","",mv,daily)
        purposes[r["assessment_id"]]="Insufficient evidence: the settling interval and three complete days have not elapsed."

    missing=[w for w,t in (("signal_recovered",rec),("abnormality_persists",per),("insufficient_evidence",ins)) if not t]
    if missing:
        print("  WARNING: no demonstration case found for: "+", ".join(missing))
    return purposes

# A store must look like this before we read it. Opening an arbitrary database
# through RepairStore would CREATE these tables in it, which is a write.
EXPECTED_COLUMNS={
    "repair_records":{"job_id","asset_id","body"},
    "repair_assessments":{"id","job_id","body"},
    "repair_followups":{"job_id","assessment_id","state"},
}

def open_readonly(path):
    """Open a store with SQLite's own read-only mode.

    `mode=ro` makes writes impossible at the driver level, so a malformed or
    unrelated database cannot be altered by the act of exporting from it.
    """
    uri=Path(path).resolve().as_uri()+"?mode=ro"
    try:
        db=sqlite3.connect(uri,uri=True,timeout=30)
    except sqlite3.Error as exc:
        raise SystemExit(f"{path}: cannot open read-only ({exc})")
    try:
        names={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    except sqlite3.DatabaseError as exc:
        db.close()
        raise SystemExit(f"{path}: not a readable SQLite database ({exc})")
    missing=set(EXPECTED_COLUMNS)-names
    if missing:
        db.close()
        raise SystemExit(f"{path}: not a repair-verification store; missing table(s): {', '.join(sorted(missing))}")
    for table,cols in EXPECTED_COLUMNS.items():
        actual={r[1] for r in db.execute(f"PRAGMA table_info({table})")}
        if not cols<=actual:
            db.close()
            raise SystemExit(f"{path}: table {table} is missing column(s): {', '.join(sorted(cols-actual))}")
    return db

def read_store(path,demo,purposes):
    """Read a store WITHOUT creating or modifying it."""
    if not Path(path).exists():
        return [],{},[]
    db=open_readonly(path)
    try:
        assessments=[json.loads(r[0]) for r in db.execute("SELECT body FROM repair_assessments ORDER BY rowid")]
        records={r[0]:json.loads(r[1]) for r in db.execute("SELECT job_id,body FROM repair_records")}
        followups=[dict(zip(("job_id","assessment_id","state"),r))
                   for r in db.execute("SELECT job_id,assessment_id,state FROM repair_followups ORDER BY rowid")]
    finally:
        db.close()
    for a in assessments:
        a.pop("reference_snapshot",None)      # large; the id and hash carry provenance
        a["demo"]=demo
        if demo and purposes.get(a.get("assessment_id")):
            a["demo_purpose"]=purposes[a["assessment_id"]]
    for f in followups:
        f["demo"]=demo
    return assessments,records,followups

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store-db",default=STORE_DB,type=Path,
                    help="operational store; read only, never modified")
    ap.add_argument("--demo-db",default=DEMO_DB,type=Path,help="demonstration store")
    ap.add_argument("--out",default=OUT,type=Path,help="export written here")
    ap.add_argument("--seed-demo",action="store_true",
                    help="rebuild the demonstration store (deletes only --demo-db)")
    args=ap.parse_args(argv)

    # Refuse aliased paths BEFORE anything is deleted or written. Without this,
    # `--seed-demo --demo-db data/repair_verification.sqlite` would wipe the
    # operational store, which is the one thing this script must never do.
    seen={}
    for flag,p in (("--store-db",args.store_db),("--demo-db",args.demo_db),("--out",args.out)):
        key=os.path.normcase(str(Path(p).resolve()))
        if key in seen:
            raise SystemExit(f"{flag} and {seen[key]} resolve to the same path: {Path(p).resolve()}")
        seen[key]=flag

    daily=pd.read_parquet(DATA/"door_daily.parquet")
    eps=pd.read_csv(DATA/"door_episodes.csv",parse_dates=["onset_ts","fault_ts"]).sort_values("fault_ts")
    mv=model_version()

    purposes={}
    if args.seed_demo:
        purposes=seed_demo(args.demo_db,daily,eps,mv,daily.day.max())

    live_a,live_r,live_f=read_store(args.store_db,False,{})
    demo_a,demo_r,demo_f=read_store(args.demo_db,True,purposes)
    # Job ids are the join key the dashboard matches on, so a collision between
    # the two stores would silently hand one store's record to the other's
    # assessment. Namespacing would break the join; refuse instead.
    clash=sorted(set(live_r)&set(demo_r))
    if clash:
        raise SystemExit("job id(s) present in both the operational and demonstration stores: "
                         +", ".join(clash)+". Rename the demonstration job(s) or point --demo-db elsewhere.")
    records=dict(live_r); records.update(demo_r)
    export={
        "exportedAt":pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "modelVersion":mv,
        "sources":{"operational":str(args.store_db.name) if Path(args.store_db).exists() else None,
                   "demonstration":str(args.demo_db.name) if Path(args.demo_db).exists() else None},
        "hasOperationalRecords":bool(live_a),
        "note":("Simulated maintenance on synthetic telemetry. No inspection or repair took place; "
                "operator and reviewer fields are placeholders, not attestations."),
        "assessments":live_a+demo_a,
        "records":records,
        "followups":live_f+demo_f,
    }
    args.out.write_text(json.dumps(export,indent=2,default=str),encoding="utf-8")
    print(f"Wrote {args.out.name}: {len(export['assessments'])} assessments "
          f"({len(live_a)} operational, {len(demo_a)} demonstration), "
          f"{len(export['followups'])} open follow-up(s).")
    return 0

if __name__=="__main__": raise SystemExit(main())
