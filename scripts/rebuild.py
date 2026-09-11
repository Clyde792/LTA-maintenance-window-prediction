"""Rebuild artifacts in dependency order; fail immediately on a failed step."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    env=os.environ.copy()
    env["OPENBLAS_NUM_THREADS"]="1"
    env["OMP_NUM_THREADS"]="1"
    scripts=["build_health.py","build_rul.py","build_aspects.py","build_duty.py","build_deferral.py","build_verification.py","build_inspection_evidence.py","build_ui.py"]
    if "--validate" in sys.argv:
        scripts.extend(["validate_pipeline.py","run_tournament.py"])
    extra={"build_verification.py":["--seed-demo"]}   # seeds the DEMO store only
    for script in scripts:
        print(f"Running {script}",flush=True)
        subprocess.run([sys.executable,"-B","-u",str(ROOT/"scripts"/script)]+extra.get(script,[]),
                       cwd=ROOT,env=env,check=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
