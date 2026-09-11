import numpy as np
import pandas as pd
from headway.evidence import evidence_status


def row(**kwargs):
    return pd.Series(dict(available_at=pd.Timestamp("2026-01-02"),
        context_supported=True, data_quality_ok=True, health_index_smooth=2., **kwargs))


def test_support_is_causal_and_stale_is_not_healthy():
    r = row()
    assert evidence_status(r, as_of="2026-01-01")["status"] == "not_available"
    assert evidence_status(r, as_of="2026-01-02")["status"] == "supported"
    assert evidence_status(r, as_of="2026-01-04")["status"] == "stale"


def test_incomplete_channels_and_last_support_are_preserved():
    r = row(current_integral_as_completeness=.7)
    out = evidence_status(r, as_of="2026-01-02", last_supported="2026-01-01")
    assert out == dict(status="incomplete", missingChannels=["current_integral_as"], lastSupportedAt="2026-01-01T00:00:00")
    r["context_supported"] = False
    assert evidence_status(r, as_of="2026-01-02")["status"] == "outside_conditions"
    assert evidence_status(pd.Series(dtype=object), as_of="2026-01-02")["status"] == "missing"


def test_payload_retains_missing_calendar_days_and_masks_unsafe_countdowns(monkeypatch):
    from scripts import build_ui
    from headway.aspect import AspectPolicy
    monkeypatch.setattr(build_ui, "_verification", lambda _: None)
    d = pd.DataFrame([dict(row()), dict(row())])
    d["day"] = pd.to_datetime(["2026-01-01", "2026-01-03"])
    d["available_at"] = d.day + pd.Timedelta(days=1)
    d["asset_id"], d["train_id"] = "A", "T"
    d["aspect"], d["raw_aspect"] = 0, 0
    d["rul_lower"] = 10.
    d.loc[1,"data_quality_ok"] = False
    p = build_ui.build_payload(d, AspectPolicy())
    rows = p["assets"][0]["rows"]
    assert len(rows) == 3
    assert rows[1]["monitoring"]["status"] == "missing"
    assert rows[2]["aspect"] == -1 and rows[2]["margin"] is None
    assert set(rows[2]["windows"].values()) == {"unknown"}
    assert rows[2]["monitoring"]["lastSupportedAt"] == "2026-01-02T00:00:00"
