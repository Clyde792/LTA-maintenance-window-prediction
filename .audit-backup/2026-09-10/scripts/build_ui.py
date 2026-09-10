"""
Render the Headway UI from real pipeline output.

Emits a single self-contained HTML file - no server, no CDN, no build step -
with the fleet's actual aspect history embedded. That matters for the demo:
every number on screen came out of the pipeline, and the date scrubber replays
the real escalation of real episodes rather than a scripted animation.

Run:  .venv/Scripts/python.exe scripts/build_ui.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from headway.aspect import DISPLAY, MEANING, RECOMMENDATION, Aspect, AspectPolicy, confidence, why
from headway.deferral import DEFAULT_HORIZONS, RISK_CEILING

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "ui" / "headway.html"


def _clean(v) -> float | None:
    """JSON has no infinity. None means 'no failure path'."""
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), 2)


def build_payload(df: pd.DataFrame, policy: AspectPolicy) -> dict:
    days = sorted(df.day.unique())
    day_index = {d: i for i, d in enumerate(days)}
    n = len(days)

    assets = []
    for aid, g in df.groupby("asset_id", sort=True):
        g = g.sort_values("day")
        aspect = [0] * n
        margin: list[float | None] = [None] * n
        point: list[float | None] = [None] * n
        hi: list[float | None] = [None] * n
        slope: list[float | None] = [None] * n
        conf: list[str] = [""] * n
        ev: list[list[str]] = [[] for _ in range(n)]
        # One risk series per horizon, plus the latest safely-deferrable date.
        risk: dict[str, list[float | None]] = {
            f"{int(h)}": [None] * n for h in DEFAULT_HORIZONS}
        safe: list[float | None] = [None] * n

        for r in g.itertuples():
            i = day_index[r.day]
            aspect[i] = int(r.aspect)
            margin[i] = _clean(r.rul_lower)
            point[i] = _clean(r.rul_point)
            hi[i] = _clean(getattr(r, "health_index_smooth", np.nan))
            slope[i] = _clean(getattr(r, "health_index_slope", np.nan))
            for h in DEFAULT_HORIZONS:
                v = getattr(r, f"risk_{int(h)}d", np.nan)
                risk[f"{int(h)}"][i] = _clean(v)
            safe[i] = _clean(getattr(r, "safe_days_10pct", np.nan))
            if aspect[i] > 0:
                conf[i] = confidence(policy, r.rul_point, r.rul_lower)
                ev[i] = why(pd.Series(r._asdict()))

        assets.append({
            "id": aid,
            "train": g.train_id.iloc[0] if "train_id" in g else aid.split("-")[0],
            "aspect": aspect, "margin": margin, "point": point,
            "hi": hi, "slope": slope, "conf": conf, "ev": ev,
            "risk": risk, "safe": safe,
        })

    return {
        "days": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in days],
        "assets": assets,
        "labels": {str(int(a)): DISPLAY[a] for a in Aspect},
        "rec": {str(int(a)): RECOMMENDATION[a] for a in Aspect},
        "meaning": {str(int(a)): MEANING[a] for a in Aspect},
        "thresholds": {"withdraw": policy.withdraw_days,
                       "tonight": policy.tonight_days,
                       "plan": policy.plan_days},
        "horizons": [int(h) for h in DEFAULT_HORIZONS],
        "riskCeiling": RISK_CEILING,
    }


TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Headway &mdash; Fleet</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400..900;1,400..700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  /* ------------------------------------------------------------------ theme
     Light is the default: this is read on a depot screen and in daylight as
     often as at 01:00. Dark is kept because the product's working hours really
     are 00:00-05:30, and the toggle costs one attribute.

     The four aspects get FOUR distinct hues, not the two the real signalling
     colours would give. On a real signal head, PLAN and TONIGHT are both
     yellow and are told apart by the NUMBER of lamps lit. On a screen that
     reads as "two ambers that look the same", which is exactly the wrong
     ambiguity for the two most consequential states. So: distinct hue AND the
     lamp count, belt and braces.
  */
  :root{
    --ground:#F4F6F9; --panel:#FFFFFF; --panel2:#FAFBFC; --line:#DFE4EA;
    --line-strong:#C4CCD6;
    --text:#0E1621; --muted:#5A6675; --faint:#8A94A2;
    /* Lamp colours are vivid (they are dots, they need to read at 9px);
       text colours are darker so they clear 4.5:1 on white. Four separated
       hues - gold, orange, red - because PLAN and TONIGHT are the two most
       consequential states and must never be mistaken for each other. */
    --green:#07794F; --green-lamp:#10B981; --green-bg:#E7F6F0;
    --plan:#A16207;  --plan-lamp:#EAB308; --plan-bg:#FEF7E0;
    --tonight:#C2410C; --tonight-lamp:#F97316; --tonight-bg:#FEEEE2;
    --red:#B01212; --red-lamp:#EF4444; --red-bg:#FCEAEA;
    --accent:#1D4ED8; --accent-bg:#EAF0FE;
    --shadow:0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.04);
    /* Playfair Display throughout; IBM Plex Mono retained ONLY for figures,
       because margins, sigma values and dates line up in columns and a
       proportional serif breaks that alignment. */
    --disp:'Playfair Display',Georgia,'Times New Roman',serif;
    --body:'Playfair Display',Georgia,'Times New Roman',serif;
    --mono:'IBM Plex Mono',Consolas,monospace;
  }
  :root[data-theme="dark"]{
    --ground:#0A0F1A; --panel:#101726; --panel2:#0D1420; --line:#1E2A3D;
    --line-strong:#2C3A50;
    --text:#E6EDF7; --muted:#96A3B8; --faint:#5F6C80;
    --green:#34D399; --green-lamp:#34D399; --green-bg:#0E2A20;
    --plan:#FACC15;  --plan-lamp:#FACC15; --plan-bg:#2B2210;
    --tonight:#FB923C; --tonight-lamp:#FB923C; --tonight-bg:#2E1B0F;
    --red:#F87171; --red-lamp:#F87171; --red-bg:#2C1416;
    --accent:#8FB6E8; --accent-bg:#132033;
    --shadow:none;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--text);font-family:var(--body);
       font-size:16px;font-weight:500;line-height:1.6;padding:0 22px 90px}
  .wrap{max-width:1240px;margin:0 auto}

  /* Two-pane shell: a persistent selector on the left, one view at a time on
     the right. Splitting the views keeps each screen answering ONE question -
     what must I do tonight / how is the fleet / what is this door doing -
     rather than asking the reader to scroll past two answers to reach a third. */
  .app{display:grid;grid-template-columns:222px 1fr;gap:28px;align-items:start}
  .side{position:sticky;top:0;padding:22px 0 26px;display:flex;flex-direction:column;gap:20px}
  nav{display:flex;flex-direction:column;gap:3px}
  .nav-item{display:flex;justify-content:space-between;align-items:center;gap:10px;
            background:transparent;border:1px solid transparent;border-radius:8px;
            padding:10px 12px;cursor:pointer;width:100%;text-align:left;
            font-family:var(--body);font-size:15px;font-weight:500;color:var(--muted)}
  .nav-item:hover{background:var(--panel)}
  .nav-item[aria-current="page"]{background:var(--panel);border-color:var(--line);
            color:var(--text);font-weight:700;box-shadow:var(--shadow)}
  .nav-item:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .badge{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--muted);
         background:var(--panel2);border:1px solid var(--line);border-radius:999px;
         padding:1px 8px;font-variant-numeric:tabular-nums}
  .nav-item[aria-current="page"] .badge{border-color:var(--line-strong);color:var(--text)}
  .badge.hot{color:var(--tonight);border-color:var(--tonight)}

  .side .panelbox{background:var(--panel);border:1px solid var(--line);border-radius:9px;
                  padding:13px 14px;box-shadow:var(--shadow)}
  .side .datebig{font-family:var(--mono);font-size:16px;font-weight:600;
                 font-variant-numeric:tabular-nums;display:block;margin-bottom:9px}
  .side .row{display:flex;gap:7px;align-items:center;margin-top:9px}
  .side input[type=range]{width:100%;accent-color:var(--tonight)}
  main{padding-top:22px;min-height:70vh}
  .view{display:none} .view.on{display:block}
  @media(max-width:880px){
    .app{grid-template-columns:1fr;gap:8px}
    .side{position:static;padding-bottom:8px}
    nav{flex-direction:row;flex-wrap:wrap}
    .nav-item{width:auto}
    main{padding-top:6px}
  }

  /* ---------------------------------------------------------------- header */
  header{display:flex;flex-wrap:wrap;gap:18px;align-items:center;
         justify-content:space-between;padding:22px 0 16px}
  .brand{font-family:var(--disp);font-weight:700;font-size:25px;letter-spacing:.03em;
         display:flex;align-items:baseline}
  .brand .chev{color:var(--tonight);font-weight:500;font-size:.78em;margin:0 .05em}
  /* In the 222px sidebar the tagline cannot sit beside the wordmark without
     wrapping mid-phrase, so it takes its own line. Column direction would break
     the WORDMARK across lines instead ("HEAD / <> / WAY"), so wrap plus a
     full-width basis on the tagline is the right tool here. */
  .brand{flex-wrap:wrap;align-items:baseline;row-gap:3px}
  .brand small{flex:0 0 100%;font-family:var(--body);font-weight:500;font-size:10.5px;
               color:var(--faint);margin-left:0;letter-spacing:.14em;
               text-transform:uppercase;white-space:nowrap}
  .tools{display:flex;gap:8px;align-items:center}
  .btn{background:var(--panel);color:var(--text);border:1px solid var(--line);
       font-family:var(--mono);font-size:12px;padding:6px 12px;cursor:pointer;
       border-radius:6px;box-shadow:var(--shadow)}
  .btn:hover{border-color:var(--line-strong)}
  .btn:focus-visible{outline:2px solid var(--accent);outline-offset:2px}

  /* Aspect glyph. SHAPE carries the severity as well as colour, so the four
     states stay separable in greyscale, on a bad projector, or for a
     colour-blind viewer:

         ring  ->  half  ->  full  ->  octagon
       monitor    plan     tonight    withdraw

     The octagon is doing real work: it is the stop sign, and it means the one
     state that takes a train out of service never has to be read twice. */
  .ico.a0{color:var(--green-lamp)} .ico.a1{color:var(--plan-lamp)}
  .ico.a2{color:var(--tonight-lamp)} .ico.a3{color:var(--red-lamp)}

  /* ----------------------------------------------------------- filter chips */
  .filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px}
  .chip{display:flex;align-items:center;gap:8px;background:var(--panel);
        border:1px solid var(--line);border-radius:999px;padding:6px 14px;cursor:pointer;
        font-size:13px;box-shadow:var(--shadow)}
  .chip .n{font-family:var(--mono);font-weight:600;font-variant-numeric:tabular-nums}
  .chip .t{color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
  .chip:hover{border-color:var(--line-strong)}
  .chip[aria-pressed="true"]{border-color:var(--accent);background:var(--accent-bg)}
  .chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .chip.c0 .n{color:var(--green)} .chip.c1 .n{color:var(--plan)}
  .chip.c2 .n{color:var(--tonight)} .chip.c3 .n{color:var(--red)}

  /* ------------------------------------------------------------- date strip */
  .scrub{display:flex;align-items:center;gap:14px;background:var(--panel);
         border:1px solid var(--line);border-radius:8px;padding:12px 16px;
         margin:14px 0 4px;box-shadow:var(--shadow)}
  .scrub label{font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;
               text-transform:uppercase;color:var(--faint);white-space:nowrap}
  .scrub input[type=range]{flex:1;accent-color:var(--tonight);min-width:120px}
  .scrub .date{font-family:var(--mono);font-size:15px;font-weight:600;
               font-variant-numeric:tabular-nums;white-space:nowrap}

  h2{font-family:var(--disp);font-weight:600;font-size:18px;letter-spacing:.02em;
     margin:32px 0 3px}
  .sub{color:var(--muted);font-size:14.5px;margin:0 0 14px}

  /* ------------------------------------------------------------------ cards */
  .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
  .acard{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:0;overflow:hidden;box-shadow:var(--shadow)}
  .acard.a1{border-color:var(--plan)} .acard.a2{border-color:var(--tonight)}
  .acard.a3{border-color:var(--red)}
  .acard .body{padding:15px 17px 16px}
  .acard .top{display:flex;justify-content:space-between;align-items:center;gap:10px}
  .acard .vwrap{display:flex;align-items:center;gap:10px}
  .acard .verd{font-family:var(--disp);font-weight:700;font-size:15px;letter-spacing:.05em}
  .a1 .verd{color:var(--plan)} .a2 .verd{color:var(--tonight)} .a3 .verd{color:var(--red)}
  .acard .aid{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
  .acard .mean{color:var(--text);font-size:14.5px;margin:8px 0 12px}
  .acard.held{border-color:var(--line)}
  .acard.held .verd{color:var(--muted)}
  .chip-held{font-family:var(--mono);font-size:9px;letter-spacing:.12em;font-weight:600;
             border:1px solid var(--line-strong);color:var(--muted);padding:2px 7px;
             border-radius:999px;margin-left:2px}

  /* margin is the hero number - it is the thing being decided on */
  .nums{display:flex;gap:10px;margin-bottom:11px}
  .nums .cell{flex:1;background:var(--panel2);border:1px solid var(--line);
              border-radius:7px;padding:8px 10px}
  .nums .cell.hero{flex:1.25}
  .k{font-family:var(--mono);font-size:9px;letter-spacing:.13em;
     text-transform:uppercase;color:var(--faint);display:block;margin-bottom:2px}
  .v{font-family:var(--mono);font-size:18px;font-weight:600;font-variant-numeric:tabular-nums;
     line-height:1.2}
  .v.sm{font-size:13px;font-weight:500}
  .a1 .cell.hero .v{color:var(--plan)} .a2 .cell.hero .v{color:var(--tonight)}
  .a3 .cell.hero .v{color:var(--red)}
  .held .cell.hero .v{color:var(--muted)}
  /* the deferral ledger: a priced menu, not a verdict */
  .ledger{margin:2px 0 12px;border:1px solid var(--line);border-radius:7px;
          background:var(--panel2);overflow:hidden}
  .ledger .lhd{font-family:var(--mono);font-size:9px;letter-spacing:.13em;
               text-transform:uppercase;color:var(--faint);padding:8px 11px 6px}
  .ledger .lrow{display:flex;justify-content:space-between;align-items:baseline;
                gap:10px;padding:4px 11px;font-size:13px}
  .ledger .lrow:last-child{padding-bottom:9px}
  .ledger .lrow .w{color:var(--muted)}
  .ledger .lrow .p{font-family:var(--mono);font-weight:600;
                   font-variant-numeric:tabular-nums}
  .ledger .lrow.ok .p{color:var(--green)}
  .ledger .lrow.warn .p{color:var(--plan)}
  .ledger .lrow.hot .p{color:var(--red)}
  .ledger .lrow.unk .p{color:var(--faint)}
  .ledger .safe{border-top:1px solid var(--line);padding:7px 11px;font-size:12.5px;
                color:var(--text);background:var(--panel)}
  .ledger .safe b{font-family:var(--mono);font-variant-numeric:tabular-nums}

  .acard ul{margin:0;padding-left:17px}
  .acard li{font-size:13.5px;color:var(--muted);margin-bottom:4px}
  .acard li b{color:var(--text)}
  .empty{color:var(--muted);font-size:14.5px;background:var(--panel);
         border:1px dashed var(--line-strong);border-radius:10px;padding:26px;
         text-align:center;grid-column:1/-1}

  /* ------------------------------------------------------------------ fleet */
  .fleet{display:grid;grid-template-columns:repeat(auto-fill,minmax(104px,1fr));gap:8px}
  .tile{display:flex;align-items:center;gap:8px;background:var(--panel);
        border:1px solid var(--line);border-radius:8px;padding:8px 9px;cursor:pointer;
        text-align:left;box-shadow:var(--shadow);transition:border-color .12s,transform .12s}
  .tile:hover{border-color:var(--line-strong);transform:translateY(-1px)}
  .tile:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
  .tile.sel{border-color:var(--accent);background:var(--accent-bg)}
  .tile .meta{min-width:0}
  .tile .lab{font-family:var(--mono);font-size:10.5px;color:var(--muted);
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block}
  .tile .mrg{font-family:var(--mono);font-size:12px;font-weight:600;
             font-variant-numeric:tabular-nums;display:block;line-height:1.25}
  .tile.a0 .mrg{color:var(--faint);font-weight:400}
  .tile.a1{background:var(--plan-bg);border-color:var(--plan)}
  .tile.a1 .mrg{color:var(--plan)}
  .tile.a2{background:var(--tonight-bg);border-color:var(--tonight)}
  .tile.a2 .mrg{color:var(--tonight)}
  .tile.a3{background:var(--red-bg);border-color:var(--red)}
  .tile.a3 .mrg{color:var(--red)}

  /* ----------------------------------------------------------------- detail */
  .detail{background:var(--panel);border:1px solid var(--line);border-radius:10px;
          padding:18px 20px;margin-top:16px;box-shadow:var(--shadow)}
  .detail h3{font-family:var(--disp);font-weight:600;font-size:17px;margin:0 0 3px}
  .detail .note{color:var(--muted);font-size:13px;margin:0 0 14px}
  .detail .split{display:grid;grid-template-columns:1fr;gap:18px;align-items:start}
  @media(min-width:1180px){
    .detail .split{grid-template-columns:minmax(0,1.5fr) minmax(330px,1fr)}
  }
  .chart svg{display:block;width:100%;height:auto;overflow:visible}
  svg.ico{flex:none;display:inline-block;vertical-align:middle}
  .axis{display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px 14px;
        font-family:var(--mono);font-size:10px;color:var(--faint);margin-top:6px}
  .axis .mid{flex:1;text-align:center;min-width:150px}
</style>
</head>
<body>
<div class="wrap">
  <div class="app">
    <aside class="side">
      <div class="brand">HEAD<span class="chev">&lsaquo;&rsaquo;</span>WAY
        <small>Time to act</small></div>

      <nav id="nav" aria-label="Views">
        <button class="nav-item" data-view="tonight">Tonight's window
          <span class="badge" id="b-tonight">0</span></button>
        <button class="nav-item" data-view="fleet">Fleet
          <span class="badge" id="b-fleet">0</span></button>
        <button class="nav-item" data-view="asset">Door detail
          <span class="badge" id="b-asset">&mdash;</span></button>
      </nav>

      <div class="panelbox">
        <label class="k" for="day">As at</label>
        <span class="datebig" id="date"></span>
        <input type="range" id="day" min="0" value="0" step="1">
        <div class="row">
          <button class="btn" id="prev" aria-label="Previous day">&larr;</button>
          <button class="btn" id="next" aria-label="Next day">&rarr;</button>
        </div>
      </div>

      <button class="btn" id="theme" aria-label="Switch colour theme">&#9681; Dark</button>
    </aside>

    <main>
      <section class="view" id="v-tonight">
        <h2 style="margin-top:0">Tonight's window</h2>
        <p class="sub" id="tonight-sub"></p>
        <div class="cards" id="cards"></div>
      </section>

      <section class="view" id="v-fleet">
        <h2 style="margin-top:0">Fleet</h2>
        <p class="sub">Filter by aspect, then select a door to open its detail.</p>
        <div class="filters" id="filters"></div>
        <div class="fleet" id="fleet" style="margin-top:12px"></div>
      </section>

      <section class="view" id="v-asset">
        <h2 style="margin-top:0">Door detail</h2>
        <p class="sub" id="asset-sub">Select a door from the Fleet view.</p>
        <div class="detail" id="detail"></div>
      </section>
    </main>
  </div>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const N = DATA.days.length;
let day = N - 1, selected = null, filter = -1, view = 'tonight';   // filter -1 = all

const el = id => document.getElementById(id);
const fmt = v => v === null ? '—' : Math.round(v) + ' d';

// Open on the day the product is most legible - which is NOT the day with the
// most RED doors. On a heavily-RED day every ledger correctly reads HIGH at
// every horizon, which shows the reader nothing the aspect had not already
// said. Score days by how many doors carry an order whose ledger actually
// DISCRIMINATES: quotable at a short horizon, HIGH at a long one.
(function(){
  let best = N - 1, bestScore = -1;
  for (let i = 0; i < N; i++){
    let s = 0;
    for (const a of DATA.assets){
      if (!a.aspect[i] || !a.risk) continue;
      const near = a.risk['3'] ? a.risk['3'][i] : null;
      const far  = a.risk['28'] ? a.risk['28'][i] : null;
      const informative = near !== null && far !== null
        && near < DATA.riskCeiling && far > near;
      s += informative ? 3 : 1;      // an order still counts, just for less
    }
    if (s > bestScore){ bestScore = s; best = i; }
  }
  day = best;
})();

// Mirrors AspectPolicy.raw() - the aspect today's margin alone would justify.
function marginAspect(m){
  if (m === null) return 0;
  const t = DATA.thresholds;
  if (m < t.withdraw) return 3;
  if (m < t.tonight)  return 2;
  if (m < t.plan)     return 1;
  return 0;
}
const isHeld = a => a.aspect[day] > marginAspect(a.margin[day]);

// Shape-coded aspect glyph: ring, half, full, octagon. currentColor is set by
// the .a0-.a3 classes, so one markup path serves every context.
const GLYPH = [
  '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="2"/>',
  '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="2"/>'
    + '<path d="M7 2 A5 5 0 0 1 7 12 Z" fill="currentColor"/>',
  '<circle cx="7" cy="7" r="5.75" fill="currentColor"/>',
  '<polygon points="4.6,1 9.4,1 13,4.6 13,9.4 9.4,13 4.6,13 1,9.4 1,4.6" fill="currentColor"/>',
];
function head(asp, big){
  const n = big ? 18 : 14;
  return `<svg class="ico a${asp}" width="${n}" height="${n}" viewBox="0 0 14 14"
    style="width:${n}px;height:${n}px" aria-hidden="true">${GLYPH[asp]}</svg>`;
}

const ASPECTS = [0, 1, 2, 3];

function counts(i){
  const c = [0,0,0,0];
  for (const a of DATA.assets) c[a.aspect[i]]++;
  return c;
}

function renderFilters(){
  const c = counts(day);
  const items = [[-1, 'all doors', DATA.assets.length]].concat(
    ASPECTS.map(a => [a, DATA.rec[a], c[a]]));
  el('filters').innerHTML = items.map(([a, label, n]) =>
    `<button class="chip c${a}" data-f="${a}" aria-pressed="${filter===a}">
       ${a >= 0 ? head(a) : ''}<span class="n">${n}</span><span class="t">${label}</span>
     </button>`).join('');
}

// The ledger prices each alternative. Above the measured calibration ceiling
// we print HIGH rather than a percentage we are ~35 points out on - inventing
// precision is how a tool like this loses an engineer's trust permanently.
const WAIT_LABEL = {0: 'act tonight', 3: 'wait 3 days', 7: 'wait a week',
                    14: 'wait a fortnight', 28: 'wait a month'};

function fmtRisk(p){
  if (p === null || p === undefined) return {t: '—', c: 'unk'};
  if (p <= 0)                       return {t: '0%', c: 'ok'};
  if (p >= DATA.riskCeiling)        return {t: 'HIGH', c: 'hot'};
  if (p < 0.05)                     return {t: '<5%', c: 'ok'};
  return {t: Math.round(p * 100) + '%', c: p < 0.15 ? 'ok' : 'warn'};
}

function ledger(a){
  const i = day;
  if (!a.risk) return '';
  const rows = DATA.horizons.map(h => {
    const f = fmtRisk(a.risk[String(h)] ? a.risk[String(h)][i] : null);
    return `<div class="lrow ${f.c}"><span class="w">${WAIT_LABEL[h] || ('wait ' + h + ' days')}</span>
      <span class="p">${f.t}</span></div>`;
  }).join('');
  const s = a.safe ? a.safe[i] : null;
  const safe = (s === null || s === undefined) ? ''
    : `<div class="safe">Latest date under 10% risk: <b>${Math.round(s)} days</b></div>`;
  return `<div class="ledger"><div class="lhd">What does it cost to wait?</div>
    ${rows}${safe}</div>`;
}

function card(a){
  const i = day, asp = a.aspect[i], mAsp = marginAspect(a.margin[i]), held = asp > mAsp;
  const ev = (a.ev[i] || []).map(e => `<li>${e}</li>`).join('');
  return `<div class="acard a${asp}${held ? ' held' : ''}">
    <div class="body">
      <div class="top">
        <span class="vwrap">${head(asp)}
          <span class="verd">${DATA.labels[asp]} &middot; ${DATA.rec[asp]}</span>
          ${held ? '<span class="chip-held">HELD</span>' : ''}</span>
        <span class="aid">${a.id}</span>
      </div>
      <p class="mean">${DATA.meaning[asp]}</p>
      <div class="nums">
        <div class="cell hero"><span class="k">Margin &mdash; act within</span>
          <span class="v">${fmt(a.margin[i])}</span></div>
        <div class="cell"><span class="k">Confidence</span>
          <span class="v sm">${held ? 'held' : (a.conf[i] || '—')}</span></div>
        <div class="cell"><span class="k">Health</span>
          <span class="v sm">${a.hi[i] === null ? '—' : a.hi[i].toFixed(1) + ' σ'}</span></div>
      </div>
      ${ledger(a)}
      <ul>${held ? `<li><b>Held</b> from an earlier escalation &mdash; today's reading alone would be
        ${DATA.labels[mAsp]}. The order stands until the signal clears its dwell, or an engineer
        confirms the repair.</li>` : ''}${ev}</ul>
    </div>
  </div>`;
}

function renderCards(){
  const due = DATA.assets.filter(a => a.aspect[day] >= 2)
    .sort((x,y) => (isHeld(x) - isHeld(y)) || y.aspect[day] - x.aspect[day]
                || (x.margin[day] ?? 1e9) - (y.margin[day] ?? 1e9));
  const fresh = due.filter(a => !isHeld(a)).length;
  el('tonight-sub').textContent = due.length
    ? `${fresh} door${fresh===1?'':'s'} cannot wait for a later window`
      + (due.length > fresh ? `; ${due.length-fresh} held from an earlier escalation.` : '.')
    : 'Nothing requires tonight’s window.';
  el('cards').innerHTML = due.length ? due.map(card).join('')
    : `<div class="empty">All clear &mdash; the window is free for planned work.</div>`;
}

function renderFleet(){
  const rows = DATA.assets
    .map((a, idx) => ({a, idx}))
    .filter(({a}) => filter < 0 || a.aspect[day] === filter);
  el('fleet').innerHTML = rows.length ? rows.map(({a, idx}) => {
    const asp = a.aspect[day];
    return `<button class="tile a${asp}${selected===idx?' sel':''}" data-idx="${idx}"
      aria-label="${a.id}, ${DATA.labels[asp]}">
      ${head(asp)}
      <span class="meta"><span class="mrg">${asp ? fmt(a.margin[day]) : 'ok'}</span>
      <span class="lab">${a.id.replace('-DOOR-','·')}</span></span>
    </button>`;
  }).join('') : `<div class="empty">No doors at this aspect today.</div>`;
}

function spark(a){
  const W = 820, H = 190, padL = 40, padR = 14, padT = 16, padB = 30;
  const vals = a.hi.map(v => v === null ? 0 : v);
  const max = Math.max(6, ...vals), min = Math.min(0, ...vals);
  const x = i => padL + i * (W - padL - padR) / Math.max(N-1, 1);
  const y = v => H - padB - (v - min) * (H - padT - padB) / Math.max(max - min, 1e-9);
  const step = (W - padL - padR) / Math.max(N-1, 1);

  let path = '';
  vals.forEach((v,i) => { path += (i?'L':'M') + x(i).toFixed(1) + ' ' + y(v).toFixed(1) + ' '; });

  const COL = ['transparent','var(--plan)','var(--tonight)','var(--red)'];
  let band = '';
  a.aspect.forEach((s,i) => { if (s > 0)
    band += `<rect x="${x(i).toFixed(1)}" y="${H-padB+7}" width="${(step+0.6).toFixed(2)}"
             height="7" rx="1" fill="${COL[s]}"/>`; });

  let grid = '';
  for (const g of [0, max/2, max]) grid +=
    `<line x1="${padL}" y1="${y(g)}" x2="${W-padR}" y2="${y(g)}" stroke="var(--line)"/>
     <text x="${padL-7}" y="${y(g)+3.5}" text-anchor="end" fill="var(--faint)"
       font-family="var(--mono)" font-size="10">${g.toFixed(0)}</text>`;

  return `<svg viewBox="0 0 ${W} ${H}" role="img"
      aria-label="Health index history for ${a.id}, in sigma above its own baseline">
    ${grid}
    <path d="${path}" fill="none" stroke="var(--accent)" stroke-width="2"
      stroke-linejoin="round"/>
    ${band}
    <line x1="${x(day)}" y1="${padT-6}" x2="${x(day)}" y2="${H-padB}"
      stroke="var(--text)" stroke-width="1" stroke-dasharray="3 3" opacity=".55"/>
    <circle cx="${x(day)}" cy="${y(vals[day])}" r="4.5" fill="var(--panel)"
      stroke="var(--accent)" stroke-width="2.5"/>
    <text x="${padL-7}" y="${padT-2}" text-anchor="end" fill="var(--faint)"
      font-family="var(--mono)" font-size="9">&#963;</text>
  </svg>
  <div class="axis"><span>${DATA.days[0]}</span>
    <span class="mid">&sigma; above this door's own baseline</span>
    <span>${DATA.days[N-1]}</span></div>`;
}

function renderDetail(){
  if (selected === null){
    el('detail').innerHTML = `<p class="note" style="margin:0">No door selected.
      Open the Fleet view and choose one.</p>`;
    el('asset-sub').textContent = 'Select a door from the Fleet view.';
    return;
  }
  const a = DATA.assets[selected], asp = a.aspect[day];
  el('asset-sub').textContent =
    `${a.id} — ${DATA.labels[asp]} on ${DATA.days[day]}.`;
  el('detail').innerHTML = `<h3>${a.id}</h3>
    <p class="note">Full record. The dashed line is the selected day; the bar beneath the
      curve is the aspect in force on each day.</p>
    <div class="split"><div class="chart">${spark(a)}</div>
      <div>${asp > 0 ? card(a) : `<div class="empty">Healthy on ${DATA.days[day]} &mdash;
        no order outstanding.</div>`}</div></div>`;
}

function renderNav(){
  const c = counts(day);
  const due = c[2] + c[3];
  const bt = el('b-tonight');
  bt.textContent = due;
  bt.classList.toggle('hot', due > 0);
  el('b-fleet').textContent = DATA.assets.length;
  el('b-asset').textContent = selected === null ? '—'
    : DATA.assets[selected].id.replace('-DOOR-', '·');
  for (const b of document.querySelectorAll('.nav-item')){
    if (b.dataset.view === view) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  }
  for (const v of ['tonight','fleet','asset'])
    el('v-' + v).classList.toggle('on', v === view);
}

function render(){
  el('day').value = day;
  el('date').textContent = DATA.days[day];
  renderNav(); renderFilters(); renderCards(); renderFleet(); renderDetail();
}

el('day').max = N - 1;
el('day').addEventListener('input', e => { day = +e.target.value; render(); });
el('prev').addEventListener('click', () => { day = Math.max(0, day-1); render(); });
el('next').addEventListener('click', () => { day = Math.min(N-1, day+1); render(); });
el('filters').addEventListener('click', e => {
  const b = e.target.closest('.chip'); if (!b) return;
  const f = +b.dataset.f; filter = (filter === f) ? -1 : f; render();
});
el('fleet').addEventListener('click', e => {
  const t = e.target.closest('.tile'); if (!t) return;
  // Selecting a door is an intent to look at it, so go there rather than
  // leaving the reader to find the detail view on their own.
  selected = +t.dataset.idx; view = 'asset'; render();
});
el('nav').addEventListener('click', e => {
  const b = e.target.closest('.nav-item'); if (!b) return;
  view = b.dataset.view; render();
});
el('theme').addEventListener('click', () => {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark');
  el('theme').innerHTML = dark ? '◑ Dark' : '◐ Light';
});
render();
</script>
</body>
</html>
"""


def main() -> int:
    # Prefer the deferral frame - same rows, plus the priced alternatives.
    src = DATA / "door_deferral.parquet"
    if not src.exists():
        src = DATA / "door_aspects.parquet"
    df = pd.read_parquet(src)
    policy = AspectPolicy()
    payload = build_payload(df, policy)

    OUT.parent.mkdir(exist_ok=True)
    html = TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")

    size = OUT.stat().st_size / 1024
    c = df[df.day == df.day.max()].aspect.value_counts().reindex(range(4), fill_value=0)
    print(f"wrote {OUT}  ({size:.0f} KB, self-contained)")
    print(f"  {len(payload['assets'])} doors x {len(payload['days'])} days")
    print(f"  final day: {int(c[0])} monitor / {int(c[1])} plan / "
          f"{int(c[2])} tonight / {int(c[3])} withdraw")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
