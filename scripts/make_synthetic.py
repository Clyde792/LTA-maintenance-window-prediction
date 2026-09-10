"""Generate the synthetic door dataset and write it to data/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway import contract
from headway.synth.doors import SynthConfig, generate

OUT = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    cfg = SynthConfig()
    print(f"generating: {cfg.n_trains} trains x {cfg.doors_per_train} doors, {cfg.n_days} days ...")
    cycles, episodes = generate(cfg)

    report = contract.validate(cycles, "door")
    print(report)

    cycles.to_parquet(OUT / "door_cycles.parquet", index=False)
    episodes.to_csv(OUT / "door_episodes.csv", index=False)

    print(f"\nwrote {OUT / 'door_cycles.parquet'}  ({len(cycles):,} cycles)")
    print(f"wrote {OUT / 'door_episodes.csv'}  ({len(episodes)} episodes)")
    print("\nepisodes (synthetic ground truth, never shown to a model):")
    print(episodes[["asset_id", "fault_mode", "onset_ts", "fault_ts", "warning_days_available"]]
          .to_string(index=False, max_colwidth=22))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
