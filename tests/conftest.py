"""Shared fixtures. A small synthetic fleet keeps the suite fast (~seconds)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway.synth.doors import SynthConfig, generate


@pytest.fixture(scope="session")
def small_cfg() -> SynthConfig:
    """Small enough to be fast, large enough that the statistics still hold."""
    return SynthConfig(
        n_trains=8, doors_per_train=2, n_days=90,
        cycles_per_day=40, n_episodes=4, seed=7,
    )


@pytest.fixture(scope="session")
def synth(small_cfg):
    cycles, episodes = generate(small_cfg)
    return cycles, episodes


@pytest.fixture(scope="session")
def cycles(synth):
    return synth[0].copy()


@pytest.fixture(scope="session")
def episodes(synth):
    return synth[1].copy()
