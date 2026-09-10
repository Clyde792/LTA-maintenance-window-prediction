"""Rebuild artifacts in dependency order; fail immediately on a failed step."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    env=os.environ.copy()
    env["OPENBLAS_NUM_THREADS"]="1"
    env["OMP_NUM_THREADS"]="1"
    scripts=["build_health.py","build_rul.py","build_aspects.py","build_deferral.py","build_ui.py"]
    if "--validate" in sys.argv:
        scripts.extend(["validate_pipeline.py","run_tournament.py"])
    for script in scripts:
        print(f"Running {script}",flush=True)
        subprocess.run([sys.executable,"-B","-u",str(ROOT/"scripts"/script)],cwd=ROOT,env=env,check=True)
    return 0
if __name__=="__main__": raise SystemExit(main())
