/* Headway dashboard.
   Source lives here rather than inside a Python string so it can be read and
   reviewed; scripts/build_ui.py inlines it into one self-contained HTML file.

   Two ideas carry most of the weight:

   1. A JOB is the operator's object, not the model's. It has its own id, so one
      door can have several jobs over its life, and a fresh escalation after a
      completed repair raises a NEW candidate instead of being suppressed by the
      old one.
   2. Job state is EVENT-SOURCED. Nothing is overwritten; every action appends a
      timestamped event. The state of the plan on any given night is replayed
      from the events up to that night, so looking back at an earlier window
      shows what was actually outstanding then. */

const DATA = JSON.parse(document.getElementById('payload').textContent);
const N = DATA.days.length;
const W = DATA.window;

let day = N - 1;              // the night the Fleet slider / door chart is on
let planNight = N - 1;        // the night the Planner is planning
let selected = null, filter = 'all', view = 'status', prevView = 'status';
let dragging = false, query = '', focusKey = null;
const expanded = new Set();

const el = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dayIdx = d => DATA.days.indexOf(d);

/* ------------------------------------------------------------------ states */
const states = {valid:'', insufficient_history:'Not enough history yet',
  insufficient_data:'Missing or incomplete data', no_worsening_trend:'Not getting worse right now',
  elevated_no_trend:'High, but not trending', threshold_exceeded:'At or past the learned threshold',
  uncalibrated:'Not enough data to calibrate', outside_horizon:'More than 90 days out',
  outside_training_conditions:'Unusual conditions', stale_data:'Data is out of date',
  missing_observation:'No reading for this day', not_yet_available:'Not available yet'};
const label = s => s in states ? states[s] : String(s).replaceAll('_', ' ');
const held = r => r.aspect > 0 && r.aspect > r.raw;
const isFlagged = r => r.aspect >= 1 || r.duty === 'off_peak_only' || held(r);

function head(a) {
  const shape = a === -1 ? '<text x="7" y="11" text-anchor="middle" fill="currentColor" font-size="12">?</text>'
    : a === 3 ? '<polygon points="4,1 10,1 13,4 13,10 10,13 4,13 1,10 1,4" fill="currentColor"/>'
    : a === 2 ? '<circle cx="7" cy="7" r="6" fill="currentColor"/>'
    : a === 1 ? '<path d="M7 1 A6 6 0 0 1 7 13 Z" fill="currentColor"/><circle cx="7" cy="7" r="6" fill="none" stroke="currentColor"/>'
    : '<circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" stroke-width="2"/>';
  return '<svg class="ico a' + a + '" width="16" height="16" viewBox="0 0 14 14" aria-hidden="true">' + shape + '</svg>';
}

/* ------------------------------------------------------- time and margins */
/* The daily aggregate for telemetry day D lands at D+1 00:00; the engineering
   window opens startHour after that. Every hour figure below is measured from
   that landing instant, which is also where the RUL lower bound starts. */
const hoursToWindow = (fromNight, targetNight) => (targetNight - fromNight) * 24 + W.startHour;

/* Conservative rounding happens HERE, at the last step, so no caller can
   reintroduce it by pre-rounding. `dir` is 'down' for anything that must not be
   overstated (the margin itself, slack in hand) and 'up' for anything that must
   not be understated (a wait, an overrun). Sub-6-hour values keep minutes, so
   flooring 1.96 h gives "1 h 57 min" rather than a misleading "1 hour". */
function fmtHours(h, dir) {
  const r = dir === 'up' ? Math.ceil : Math.floor;
  if (h < 1) { const m = Math.max(0, r(h * 60)); return m + (m === 1 ? ' minute' : ' minutes'); }
  if (h < 6) {
    const total = Math.max(0, r(h * 60)), hh = Math.floor(total / 60), mm = total % 60;
    return hh + ' h' + (mm ? ' ' + mm + ' min' : '');
  }
  if (h < 48) { const n = r(h); return n + (n === 1 ? ' hour' : ' hours'); }
  const d = r(h / 24 * 10) / 10; return d.toFixed(1) + ' days';
}

/* Compare the estimated lower margin against a specific window opening.
   Margin is rounded DOWN and any overrun is rounded UP, so neither figure can
   flatter the bound. */
function windowFit(r, fromNight, targetNight) {
  if (!Number.isFinite(r.margin)) return {kind: 'unk', text: 'no countdown', detail: 'No countdown available — ' + (label(r.state) || 'evidence unavailable').toLowerCase() + '.'};
  const opens = hoursToWindow(fromNight, targetNight);
  if (r.margin <= 0) return {kind: 'bad', text: 'no positive margin',
    detail: 'No positive margin established. The next window opens in ' + fmtHours(opens, 'up') + '.'};
  const m = r.margin * 24, slack = m - opens;
  const marginTxt = 'estimated lower margin ' + fmtHours(m, 'down');
  return slack >= 0
    ? {kind: 'ok', text: 'within margin', detail: 'Next window opens in ' + fmtHours(opens, 'up') + '; ' + marginTxt + '; within margin by ' + fmtHours(slack, 'down') + '.'}
    : {kind: 'bad', text: 'exceeds margin', detail: 'Next window opens in ' + fmtHours(opens, 'up') + '; ' + marginTxt + '; exceeds margin by ' + fmtHours(-slack, 'up') + '.'};
}

/* ------------------------------------------------------------ job storage */
const JOBKEY = 'headway.jobs.v2';
let saveNote = '';

function loadStore() {
  try { const s = JSON.parse(localStorage.getItem(JOBKEY) || '{}'); return Array.isArray(s.jobs) ? s : {jobs: []}; }
  catch (e) { return {jobs: []}; }
}
/* Write, then read back, so a blocked or full store is reported rather than
   silently dropped. */
function saveStore(s) {
  try {
    localStorage.setItem(JOBKEY, JSON.stringify(s));
    const back = localStorage.getItem(JOBKEY);
    saveNote = back && JSON.parse(back).jobs.length === s.jobs.length ? 'Saved' : 'Not saved — storage rejected the change';
  } catch (e) { saveNote = 'Not saved — browser storage is unavailable'; }
}
const newId = () => 'j' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
const jobsFor = id => loadStore().jobs.filter(j => j.asset === id);

function addEvent(jobId, ev) {
  const s = loadStore(), j = s.jobs.find(x => x.id === jobId);
  if (!j) return;
  j.events.push(Object.assign({at: new Date().toISOString()}, ev));
  saveStore(s);
}
function createJob(asset, night) {
  const s = loadStore(), id = newId();
  const seq = s.jobs.length;
  s.jobs.push({id: id, asset: asset, seq: seq, events: [{type: 'planned', night: DATA.days[night],
    window: DATA.days[night], order: seq, at: new Date().toISOString()}]});
  saveStore(s);
  return id;
}

/* Replay a job's events up to `nightIdx`. EVERYTHING that can change is an
   event, including priority and estimated duration, so an earlier night
   reconstructs the plan and its allocation exactly as they stood then. */
function stateAsOf(job, nightIdx) {
  let state = null, window = null, workDone = false, verified = false, note = '';
  let order = job.seq || 0, minutes = W.defaultJobMinutes;
  for (const e of job.events) {
    const i = dayIdx(e.night);
    if (i < 0 || i > nightIdx) continue;
    if (e.type === 'planned' || e.type === 'rescheduled') { state = 'planned'; window = e.window; if (Number.isFinite(e.order)) order = e.order; }
    else if (e.type === 'work_done') { workDone = true; state = 'work_done'; if (e.note) note = e.note; }
    else if (e.type === 'verified') { verified = true; state = 'done'; }
    else if (e.type === 'removed') { state = 'removed'; }
    else if (e.type === 'reopened') { state = 'planned'; workDone = false; verified = false; window = e.window; }
    else if (e.type === 'reordered') { if (Number.isFinite(e.order)) order = e.order; }
    else if (e.type === 'duration') { if (Number.isFinite(e.minutes)) minutes = e.minutes; }
  }
  return {state: state, window: window, workDone: workDone, verified: verified, note: note,
          order: order, minutes: minutes};
}
const openStates = ['planned', 'work_done'];
function jobsAsOf(nightIdx) {
  return loadStore().jobs.map(j => Object.assign({job: j}, stateAsOf(j, nightIdx))).filter(x => x.state);
}
/* Left the outstanding list — verified OR removed. Drives the "closed tonight"
   group, which should show both. */
function closedAt(job, nightIdx) {
  let idx = -1;
  for (const e of job.events) {
    const i = dayIdx(e.night);
    if (i < 0 || i > nightIdx) continue;
    if (e.type === 'verified' || e.type === 'removed') idx = i;
    if (e.type === 'reopened' || e.type === 'planned') idx = -1;
  }
  return idx;
}
/* Actually RESOLVED — verified only. Removing a job says "not this window", not
   "fixed", so a removal must never suppress a door that is still flagged. */
function resolvedAt(job, nightIdx) {
  let idx = -1;
  for (const e of job.events) {
    const i = dayIdx(e.night);
    if (i < 0 || i > nightIdx) continue;
    if (e.type === 'verified') idx = i;
    if (e.type === 'reopened' || e.type === 'planned') idx = -1;
  }
  return idx;
}

/* ------------------------------------------------- escalations and candidates */
/* The contiguous run of flagged days ending at `nightIdx`. A job that closed
   before this run began belongs to a previous problem and must not suppress
   the current one. */
function escalationStart(asset, nightIdx) {
  let i = nightIdx;
  while (i > 0 && isFlagged(asset.rows[i - 1])) i--;
  return i;
}
function openJobFor(assetId, nightIdx) {
  const m = jobsAsOf(nightIdx).find(x => x.job.asset === assetId && openStates.indexOf(x.state) >= 0);
  return m || null;
}
function candidates(nightIdx) {
  return DATA.assets.filter(a => {
    if (!isFlagged(a.rows[nightIdx])) return false;
    if (openJobFor(a.id, nightIdx)) return false;
    const start = escalationStart(a, nightIdx);
    // suppressed only by work VERIFIED during THIS escalation; a removed job
    // returns the door here, because the problem did not go away
    return !jobsFor(a.id).some(j => resolvedAt(j, nightIdx) >= start);
  });
}

/* --------------------------------------------------------- job presentation */
function jobStatus(assetId, nightIdx) {
  const open = openJobFor(assetId, nightIdx);
  if (open) return open.state === 'work_done'
    ? {cls: 'wk', text: 'awaiting return to service', job: open}
    : {cls: 'pl', text: 'planned ' + open.window + (dayIdx(open.window) < nightIdx ? ' (carried)' : ''), job: open};
  const a = DATA.assets.find(x => x.id === assetId);
  const start = a ? escalationStart(a, nightIdx) : 0;
  // Only a VERIFIED repair is a status. A repair that predates the current
  // escalation is history, and a removal leaves the door simply unplanned.
  const res = jobsFor(assetId).map(j => resolvedAt(j, nightIdx)).filter(i => i >= 0).sort((x, y) => y - x)[0];
  if (res !== undefined && res >= start) return {cls: 'dn', text: 'completed ' + DATA.days[res], job: null};
  return {cls: 'no', text: 'not planned', job: null};
}

/* ----------------------------------------------------------------- rows */
function reasonText(r) {
  if (r.monitoring && r.monitoring.status !== 'supported') return monitoringLabel(r);
  const s = label(r.state);
  if (s) return s;
  return (r.evidence && r.evidence[0]) || '—';
}
function monitoringLabel(r) {
  const names = {supported: 'Telemetry supported', missing: 'Telemetry missing', stale: 'Assessment stale',
    not_available: 'Telemetry not yet available', outside_conditions: 'Unfamiliar operating conditions',
    incomplete: 'Incomplete telemetry', warming_up: 'Collecting sufficient history'};
  return names[r.monitoring && r.monitoring.status] || 'Monitoring status unavailable';
}
function monitoringDetail(r) {
  const m = r.monitoring || {};
  return '<p class="fitline unk"><strong>' + esc(monitoringLabel(r)) + '</strong> · Last supported assessment: '
    + esc(m.lastSupportedAt ? m.lastSupportedAt.replace('T', ' ') + ' SGT' : 'none available')
    + (m.missingChannels && m.missingChannels.length ? ' · Incomplete channels: ' + esc(m.missingChannels.map(c => DATA.signalNames[c] || c).join(', ')) : '')
    + '. Telemetry support does not establish mechanical health.</p>';
}
function rowFor(a, di, opts) {
  // Job status is read as of THIS row's assessment date, never the planner's
  // browsing date, or Status would report an earlier night's plan.
  const r = a.rows[di], js = jobStatus(a.id, di);
  const fit = windowFit(r, di, di);
  const open = expanded.has(a.id);
  const act = js.job ? '' : '<button class="btn prim" data-job="add:' + esc(a.id) + '" data-fk="add-' + esc(a.id) + '">Add to plan</button>';
  return '<article class="row a' + r.aspect + (open ? ' open' : '') + '">'
    + '<div class="rmain">'
    + '<span class="rid">' + head(r.aspect) + '<button class="link" data-open="' + esc(a.id) + '" data-fk="id-' + esc(a.id) + '">' + esc(a.id) + '</button></span>'
    + '<span class="ract">' + esc(DATA.rec[r.aspect]) + (held(r) ? ' <span class="tag">held</span>' : '') + '</span>'
    + '<span class="rwhy">' + esc(reasonText(r)) + '</span>'
    + '<span class="rfit ' + fit.kind + '">' + esc(fit.text) + '</span>'
    + '<span class="rjob ' + js.cls + '">' + esc(js.text) + '<span class="vline">' + verificationChip(a.id, di) + '</span></span>'
    + '<span class="rdo">' + act
    + '<button class="btn xp" data-xp="' + esc(a.id) + '" data-fk="xp-' + esc(a.id) + '" aria-expanded="' + open + '">' + (open ? 'Less' : 'More') + '</button></span>'
    + '</div>' + (open ? detailFor(a, di, fit) : '') + '</article>';
}

function detailFor(a, di, fit) {
  const r = a.rows[di];
  // Written without spaces around '=': scripts/check_artifacts.py extracts this
  // exact expression and probes it for margins that must never round upward.
  const safe=Number.isFinite(r.margin)?(r.margin<=0?'no positive margin':r.margin<1?'under 1 day':'≥ '+(Math.floor(r.margin*10)/10).toFixed(1)+' days'):'—';
  const wear = Number.isFinite(r.hi) ? r.hi.toFixed(0) : '—';
  const thr = Number.isFinite(DATA.threshold) ? DATA.threshold.toFixed(0) : '—';
  return '<div class="rdet">'
    + monitoringDetail(r)
    + '<p class="mean">' + esc(DATA.meaning[r.aspect]) + '</p>'
    + '<div class="nums"><div class="cell hero"><span class="k">Estimated lower margin</span><span class="v">' + safe + '</span></div>'
    + '<div class="cell"><span class="k">Wear now</span><span class="v sm">' + wear + '</span></div>'
    + '<div class="cell"><span class="k">Learned threshold</span><span class="v sm">' + thr + '</span></div></div>'
    + '<p class="fitline ' + fit.kind + '">' + esc(fit.detail) + '</p>'
    + duty(r) + ledger(r)
    + (r.evidence && r.evidence.length ? '<ul>' + r.evidence.map(e => '<li>' + esc(e) + '</li>').join('') + '</ul>' : '')
    + verificationBlock(a.id, di)
    + '<button class="btn" data-open="' + esc(a.id) + '">Open door detail →</button></div>';
}

const windowText = {within_margin: ['Within estimated margin', 'ok'], exceeds_margin: ['Exceeds estimated margin', 'hot'],
  no_positive_margin: ['No positive margin established', 'warn'], threshold_exceeded: ['Learned threshold reached', 'hot'],
  unknown: ['Cannot assess', 'unk']};
function ledger(r) {
  const rows = [0, 3, 7, 14, 28].map(h => {
    const w = windowText[r.windows[String(h)]] || windowText.unknown;
    return '<div class="lrow ' + w[1] + '"><span class="w">' + (h === 0 ? 'Now' : '+' + h + ' days') + '</span><span class="p">' + w[0] + '</span></div>';
  }).join('');
  return '<div class="ledger"><div class="lhd">Can the fix wait?</div>' + rows + '</div>';
}
function duty(r) {
  const d = DATA.duty;
  const v = {off_peak_only: 'OFF-PEAK ONLY', withdraw: 'WITHDRAW', full_service: 'no peak restriction', not_assessed: 'not assessed'}[r.duty] || esc(r.duty);
  let out = '<div class="duty d-' + esc(r.duty) + '"><span class="dk">Fit for duty</span><span class="dv">' + v + '</span>';
  if (r.duty === 'off_peak_only') {
    out += '<span class="db">avoid ' + esc(d.bands) + '</span>';
    const p = [];
    if (Number.isFinite(r.peakHi)) p.push('at peak ' + r.peakHi.toFixed(0));
    if (Number.isFinite(r.hi)) p.push('typical ' + r.hi.toFixed(0));
    if (Number.isFinite(r.offHi)) p.push('off-peak ' + r.offHi.toFixed(0));
    if (Number.isFinite(DATA.threshold)) p.push('learned threshold ' + DATA.threshold.toFixed(0));
    if (p.length) out += '<span class="ds">wear: ' + esc(p.join(' · ')) + '</span>';
  }
  return out + '</div>';
}

/* ------------------------------------------------------ repair verification
   A build-time snapshot of headway/repair_verification.py. It is EVIDENCE about
   telemetry, not a service decision: `release_to_service` is always false and
   returning a door to traffic stays an operator action recorded in the planner.
   Nothing is shown before its own `as_of`, so the replay never reveals a result
   before it existed. */
const VER = DATA.verification;
const VSTATUS = {
  signal_recovered: ['Signal recovered', 'ok'],
  abnormality_persists: ['Abnormality persists', 'bad'],
  insufficient_evidence: ['Insufficient evidence', 'unk']
};
const visible = (a, nightIdx) => a.visibleFrom !== null && a.visibleFrom <= nightIdx;
const byNewest = (x, y) => (x.as_of < y.as_of ? 1 : -1);

/* Door-level history: every assessment for this asset, whichever job raised it.
   Each one names its job when displayed, because a door can carry several. */
function assessmentsFor(assetId, nightIdx) {
  if (!VER) return [];
  return VER.assessments.filter(a => a.asset_id === assetId && visible(a, nightIdx)).sort(byNewest);
}
/* Job-level: BOTH ids must match. A new repair must never inherit an older
   repair's verdict just because it is the same door. */
function assessmentsForJob(jobId, assetId, nightIdx) {
  if (!VER) return [];
  return VER.assessments
    .filter(a => a.job_id === jobId && a.asset_id === assetId && visible(a, nightIdx))
    .sort(byNewest);
}
/* An open follow-up survives a later favourable assessment: it is keyed by the
   job that raised it and is only closed by a person, never by a newer result. */
function followupsFrom(list) {
  if (!VER) return [];
  const seen = new Set(list.map(a => a.assessment_id));
  return VER.followups.filter(f => seen.has(f.assessment_id) && f.state === 'open')
    .map(f => Object.assign({}, f, {record: VER.records[f.job_id]}));
}
const followupsFor = (assetId, nightIdx) => followupsFrom(assessmentsFor(assetId, nightIdx));

function chipFrom(list, open, extra) {
  const a = list[0];
  if (!a) return '<span class="vchip none">Not assessed</span>' + (extra || '');
  const s = VSTATUS[a.status] || ['Unknown', 'unk'];
  return '<span class="vchip ' + s[1] + '" title="job ' + esc(a.job_id) + '">' + s[0] + '</span>'
    + (open ? '<span class="vchip bad">follow-up open</span>' : '') + (extra || '');
}
/* For a door row: the door's own latest assessment, labelled with its job. */
function verificationChip(assetId, nightIdx) {
  const list = assessmentsFor(assetId, nightIdx);
  return chipFrom(list, followupsFrom(list).length, '');
}
/* For a planner job: strictly this job's assessments. Any other work on the same
   door is offered separately, never folded into this job's verdict. */
function verificationChipForJob(jobId, assetId, nightIdx) {
  const mine = assessmentsForJob(jobId, assetId, nightIdx);
  const others = assessmentsFor(assetId, nightIdx).filter(a => a.job_id !== jobId).length;
  const aside = (!mine.length && others)
    ? '<span class="vchip none">' + others + ' earlier assessment' + (others > 1 ? 's' : '') + ' on this door</span>'
    : '';
  return chipFrom(mine, followupsFrom(mine).length, aside);
}
function channelRows(a) {
  return Object.keys(a.channels || {}).map(c => {
    const ch = a.channels[c], n = DATA.signalNames[c.replace(/_hx$/, '')] || c.replace(/_hx$/, '').replaceAll('_', ' ');
    return '<li><span>' + esc(n) + '</span><b>' + ch.max_absolute_deviation.toFixed(1) + '× scale · '
      + ch.abnormal_days + (ch.abnormal_days === 1 ? ' abnormal day' : ' abnormal days') + '</b></li>';
  }).join('');
}
/* The store keeps UTC. Every timestamp on screen is converted to Asia/Singapore
   and labelled SGT — truncating the ISO string instead would silently show an
   assessment made at 12 July 06:00 SGT as 11 July 22:00. */
const SGT_OPTS = {timeZone: 'Asia/Singapore', year: 'numeric', month: '2-digit', day: '2-digit'};
const SGT_DT = new Intl.DateTimeFormat('en-GB', Object.assign({hour: '2-digit', minute: '2-digit', hourCycle: 'h23'}, SGT_OPTS));
const SGT_D = new Intl.DateTimeFormat('en-GB', SGT_OPTS);
function sgtParts(fmt, t) {
  const d = new Date(t);
  if (!t || isNaN(d.getTime())) return null;
  return Object.fromEntries(fmt.formatToParts(d).map(p => [p.type, p.value]));
}
function when(t) {
  const p = sgtParts(SGT_DT, t);
  return p ? p.year + '-' + p.month + '-' + p.day + ' ' + p.hour + ':' + p.minute + ' SGT' : String(t || 'unknown');
}
function whenDate(t) {
  const p = sgtParts(SGT_D, t);
  return p ? p.year + '-' + p.month + '-' + p.day : String(t || 'unknown');
}
function verificationBlock(assetId, nightIdx) {
  if (!VER) return '';
  const list = assessmentsFor(assetId, nightIdx), open = followupsFor(assetId, nightIdx);
  let h = '<div class="verif"><div class="vhd">Repair verification <span class="vsub">telemetry evidence — does not authorise return to service</span></div>';
  if (!list.length) {
    h += '<p class="vnone">Not assessed. No verification assessment for this door was available on ' + DATA.days[nightIdx] + '.</p>';
  } else {
    for (const a of list) {
      const s = VSTATUS[a.status] || ['Unknown', 'unk'];
      const rec = VER.records[a.job_id] || {};
      h += '<div class="vitem ' + s[1] + '">'
        + '<div class="vtop"><b>' + s[0] + '</b><span class="vjob">job ' + esc(a.job_id) + '</span></div>'
        + '<p class="vwhy">' + esc(a.reason) + '</p>'
        // An insufficient-evidence result can return before a window is even
        // established, so every field here is optional except the reason.
        + '<ul class="vmeta">'
        + '<li>maintenance completed ' + esc(when(rec.completed_at)) + '</li>'
        + '<li>assessed ' + esc(when(a.as_of)) + '</li>'
        + (a.window_start && a.window_end
            ? '<li>window ' + esc(whenDate(a.window_start)) + ' → ' + esc(whenDate(a.window_end))
              + ' SGT, ' + (a.observed_days || 0) + ' complete days observed</li>'
            : '<li>no complete observation window yet</li>')
        + '<li>reference ' + esc(String(a.reference_id || '—').slice(0, 12)) + ' · model ' + esc(String(a.model_version || '—').slice(0, 20)) + '</li>'
        + '</ul>'
        + (Object.keys(a.channels || {}).length ? '<ul class="drv vch">' + channelRows(a) + '</ul>' : '')
        + (a.demo_purpose ? '<p class="vdemo">' + esc(a.demo_purpose) + '</p>' : '')
        + '</div>';
    }
  }
  for (const f of open) {
    h += '<p class="vfollow">Open follow-up from job ' + esc(f.job_id)
      + ' — a later favourable assessment does not close it. Closure is a human decision.</p>';
  }
  h += '<p class="vfoot">' + esc(VER.note) + ' Verification snapshot exported '
    + esc(when(VER.exportedAt)) + ' — refresh requires rebuild.</p></div>';
  return h;
}

/* ------------------------------------------------- inspection evidence (read only)
   Build-time evidence from headway.inspection_evidence, computed on THIS build's
   telemetry. It explains which observations support which explanations; it never
   names a cause, carries no probability, and changes no alert, aspect, urgency or
   service status. Everything below is display only.

   Three rules the replay clock imposes, all enforced here:
     * a record is invisible before its own `visibleFrom` night;
     * a record is only ever shown on the door named by its own `assetId`;
     * an older assessment is labelled as not recomputed, never presented as
       current for the selected date. */
const INSP = DATA.inspection || null;
const IEXP_TITLE = {
  sensor_or_measurement_change_on_this_door: 'A measurement or sensor change on this door',
  mechanical_change_on_this_door: 'A mechanical change on this door',
  common_influence_across_doors: 'Something affecting several doors',
  reference_contamination: 'The comparison baseline itself',
  cannot_distinguish: 'Cannot distinguish'
};
const chName = c => DATA.signalNames[c] || String(c).replaceAll('_', ' ');
/* The record keeps canonical column names; the page shows the plain ones. A 1:1
   rename of known channels applied AFTER escaping — channel names contain no
   HTML-special characters, so this cannot reintroduce markup. */
const ICHAN = Object.keys(DATA.signalNames).sort((a, b) => b.length - a.length);
function iplain(text) {
  let out = esc(text);
  for (const c of ICHAN) out = out.split(c).join(DATA.signalNames[c]);
  return out;
}
const idir = d => d === 'increase' ? 'higher' : d === 'decrease' ? 'lower' : d;

/* Records for THIS door only, that had landed by this replay night. */
function inspectionFor(assetId, nightIdx) {
  if (!INSP) return [];
  return (INSP.records || []).filter(r =>
    r.assetId === assetId && r.visibleFrom !== null && r.visibleFrom <= nightIdx);
}
function latestInspection(assetId, nightIdx) {
  const list = inspectionFor(assetId, nightIdx);
  return list.length ? list.reduce((a, b) => (b.visibleFrom > a.visibleFrom ? b : a)) : null;
}

function ichanged(r) {
  const ch = r.changes || {};
  const names = Object.keys(ch);
  const assessed = names.filter(c => ch[c].status === 'observed');
  const moved = (r.cross.shifted || []).map(c => {
    const v = ch[c] || {};
    const amount = Number.isFinite(v.change) ? (v.change > 0 ? '+' : '') + v.change + ' ' + esc(v.unit) : 'unknown';
    return '<li><span>' + esc(chName(c)) + '</span><b>' + esc(amount) + '</b>'
      + '<em>' + esc(idir(v.direction)) + ' than its own recent history</em></li>';
  }).join('');
  /* Only channels that were actually assessed can be called steady. */
  const steady = (r.cross.unchanged || []).map(c => esc(chName(c))).join(', ');
  /* A reading may exist and still fail the quality, reference or completeness
     requirements. That is "not assessable", not "not measured". */
  const unknown = (r.cross.unknown || []).map(c =>
    '<li><span>' + esc(chName(c)) + '</span><b>not assessable</b><em>'
    + iplain((r.changes[c] || {}).reason || 'no supported reading') + '</em></li>').join('');
  const detail = Object.keys(ch).map(c => {
    const v = ch[c];
    return '<li><span>' + esc(chName(c)) + '</span><b>'
      + (v.status === 'observed'
          ? esc(v.recentMedian + ' vs ' + v.referenceMedian + ' ' + v.unit + ' · ' + v.sigma + '× scale')
          : esc(v.status))
      + '</b><em>' + v.recentDays + ' recent / ' + v.referenceDays + ' reference days · '
      + esc(v.qualityBasis) + '</em></li>';
  }).join('');
  /* With nothing assessable there is no finding to report - not a quiet one. */
  const headline = moved ? '<ul class="ich">' + moved + '</ul>'
    : !assessed.length ? '<p class="inone">Insufficient evidence to assess change.</p>'
    : '<p class="inone">No change beyond the reporting threshold in the '
      + assessed.length + ' channel' + (assessed.length === 1 ? '' : 's')
      + ' that could be assessed'
      + (assessed.length < names.length ? ' (' + (names.length - assessed.length) + ' not assessable)' : '')
      + '.</p>';
  return '<div class="isec"><h4>1 · What changed</h4>' + headline
    + (steady ? '<p class="ihint">Assessed and steady: ' + steady + '.</p>' : '')
    + (unknown ? '<p class="ihint">Not assessable — a reading may exist but did not meet the '
                 + 'quality, reference or completeness requirements, so these neither support '
                 + 'nor contradict anything:</p><ul class="ich iunk">' + unknown + '</ul>' : '')
    + '<details class="idet"><summary>Detailed calculations</summary><ul class="ich">'
    + detail + '</ul></details></div>';
}

function ipeers(r) {
  const p = r.peers || {}, ch = p.channels || {};
  let head;
  if (p.status !== 'resolved') {
    head = '<p class="iunknown"><b>Unknown</b> — ' + iplain(p.reason || 'no comparable peer evidence') + '</p>';
  } else {
    head = '<p class="ihint">' + p.comparable + ' comparable door' + (p.comparable === 1 ? '' : 's')
      + ' of ' + p.considered + ' considered, matched on operating conditions and dates.</p>';
  }
  const rows = Object.keys(ch).map(c => {
    const v = ch[c];
    const label = v.status !== 'observed' ? ['unknown', 'iu']
      : v.shared ? ['shared with peers', 'is'] : ['not shared', 'ins'];
    const count = v.peers + ' peer' + (v.peers === 1 ? '' : 's') + ' with evidence';
    return '<li><span>' + esc(chName(c)) + '</span><b class="' + label[1] + '">' + esc(label[0]) + '</b>'
      + '<em>' + esc(count) + (v.status !== 'observed' && v.reason ? ' · ' + iplain(v.reason) : '')
      + (Number.isFinite(v.median) ? ' · peer median ' + v.median + '× scale' : '') + '</em></li>';
  }).join('');
  return '<div class="isec"><h4>2 · Peer evidence</h4>' + head
    + '<ul class="ich ipeer">' + rows + '</ul>'
    + '<p class="ihint">A change shared with comparable doors points to a common influence. '
    + 'It is not proof that this door’s sensor failed.</p></div>';
}

function iexplain(r) {
  const obs = {};
  for (const o of r.observations || []) obs[o.id] = o.observation;
  const cards = (r.explanations || []).map(e => {
    const pts = (e.supported_by || []).map(i => '<li>' + iplain(obs[i] || i) + '</li>').join('');
    const against = (e.contradicted_by || []).map(i => '<li>' + iplain(obs[i] || i) + '</li>').join('');
    return '<div class="iexp"><div class="iname">' + esc(IEXP_TITLE[e.explanation] || e.explanation) + '</div>'
      + '<p>' + iplain(e.statement) + '</p>'
      + (pts ? '<p class="ik">Supporting observations</p><ul class="ipt">' + pts + '</ul>'
             : '<p class="ik inone">No supporting observation.</p>')
      + (against ? '<p class="ik">Counter-evidence</p><ul class="ipt iagainst">' + against + '</ul>' : '')
      + '</div>';
  }).join('');
  const missing = (r.missingEvidence || []).map(m => '<li>' + iplain(m) + '</li>').join('');
  return '<div class="isec"><h4>3 · Possible explanations</h4>'
    + '<p class="ihint">Listed in a fixed order. Neither the order nor the number of supporting '
    + 'points is a ranking, and no explanation here is a confirmed cause.</p>'
    + '<div class="iexps">' + cards + '</div>'
    + (missing ? '<p class="ik">Missing evidence</p><ul class="ipt imiss">' + missing + '</ul>' : '')
    + '</div>';
}

function ireference(r) {
  const ref = r.reference || {};
  let body;
  if (!ref.quantifiable) {
    body = '<p class="iunknown"><b>Not available</b> — ' + iplain(ref.reason || 'no reference views supplied') + '</p>';
  } else {
    const rows = Object.keys(ref.channels || {}).map(c => {
      const v = ref.channels[c];
      if (v.status !== 'recovered') {
        const tag = v.status === 'metadata_only' ? 'baseline metadata only — not a measured effect' : 'not available';
        return '<li><span>' + esc(chName(c)) + '</span><b class="iu">' + esc(tag) + '</b>'
          + '<em>' + iplain(v.reason || '') + '</em></li>';
      }
      return '<li><span>' + esc(chName(c)) + '</span><b>'
        + esc(v.locationOffset + ' ' + v.unit + ' baseline difference') + '</b>'
        + '<em>scale ratio ' + esc(String(v.scaleRatio)) + ' · ' + esc(v.parameterSource || 'unknown source') + '</em></li>';
    }).join('');
    body = '<ul class="ich">' + rows + '</ul>'
      + '<p class="ihint">Model identity verified: ' + (ref.verifiedModelIdentity ? 'yes' : 'no')
      + ' · parameters ' + esc(ref.parameterSource || 'unknown') + '.</p>';
  }
  return '<div class="isec"><h4>4 · Reference comparison</h4>' + body
    + '<p class="ihint">Compares this door’s own adapted baseline with the frozen fleet baseline: '
    + 'a level difference in physical units and a scale ratio, reported separately.</p></div>';
}

function isuggest(r) {
  const cat = (INSP && INSP.suggestions) || {};
  const wanted = (r.explanations || []).filter(e => (e.supported_by || []).length).map(e => e.explanation);
  if (wanted.indexOf('cannot_distinguish') < 0) wanted.push('cannot_distinguish');
  const seen = {}, items = [];
  for (const name of wanted) {
    for (const s of cat[name] || []) { if (!seen[s]) { seen[s] = 1; items.push(s); } }
  }
  return '<div class="isec"><h4>5 · Evidence that would help</h4>'
    + '<p class="ihint">' + esc(r.suggestionsStatus || 'illustrative, pending operator review')
    + '. These are records and measurements to consider collecting, not a maintenance instruction.</p>'
    + '<ul class="ipt">' + items.map(s => '<li>' + esc(s) + '</li>').join('') + '</ul></div>';
}

function inspectionBlock(assetId, nightIdx) {
  if (!INSP) return '';
  let h = '<div class="insp"><div class="ihd">Inspection evidence '
    + '<span class="isub">read-only — explains the observations; does not identify a cause, '
    + 'change alerts or urgency, or authorise maintenance</span></div>';
  /* The page build refused the export: it was computed from different telemetry,
     or its records do not carry its identity. Nothing is shown rather than
     something stale. */
  if (INSP.unavailable) {
    return h + '<p class="iunknown"><b>Unavailable — rebuild required.</b> '
      + iplain(INSP.reason || 'the exported evidence does not match this build') + '</p>'
      + '<ul class="ipt">' + (INSP.problems || []).map(p => '<li>' + iplain(p) + '</li>').join('')
      + '</ul></div>';
  }
  const r = latestInspection(assetId, nightIdx);
  if (!r) {
    /* A surfaced door with no assessment carries its own reason, verified at
       build time. It is not the same as a door nobody looked at, and not the
       same as an assessment that had not happened yet on this replay night. */
    const claim = (INSP.unavailableDoors || {})[assetId];
    const any = INSP.records.some(x => x.assetId === assetId);
    h += (claim
      ? '<p class="iunknown"><b>No assessment available for this door.</b> ' + iplain(claim.reason) + '</p>'
      : '<p class="inone">' + (any
          ? 'No inspection evidence had been produced for this door by ' + esc(DATA.days[nightIdx]) + '.'
          : 'No inspection evidence for this door. Evidence is exported only for doors this dashboard surfaces.')
        + '</p>')
      + '</div>';
    return h;
  }
  const gap = nightIdx - r.visibleFrom;
  const age = gap === 0 ? '<span class="iok">current for this date</span>'
    : gap > (INSP.staleAfterDays || 7)
      ? '<span class="istale">stale — ' + gap + ' days old, not recomputed for ' + esc(DATA.days[nightIdx]) + '</span>'
      : '<span class="iage">not recomputed for ' + esc(DATA.days[nightIdx]) + ' — ' + gap + ' day'
        + (gap === 1 ? '' : 's') + ' later</span>';
  h += '<p class="imeta">Assessed ' + esc(when(r.asOf)) + ' · door ' + esc(r.assetId)
    + ' · source artifact ' + esc(String(r.sourceArtifact || '—').slice(0, 24))
    + ' · ' + age + '</p>'
    + ichanged(r) + ipeers(r) + iexplain(r) + ireference(r) + isuggest(r)
    + '<details class="idet"><summary>Provenance and limits</summary><ul class="ipt">'
    + '<li>' + esc(r.limitation) + '</li>'
    + '<li>Quality basis: ' + esc(r.qualityBasis)
    + (r.qualityBlocking && r.qualityBlocking.length ? ' — ' + esc(r.qualityBlocking.join('; ')) : '') + '</li>'
    + '<li>Window ' + esc(whenDate(r.window.reference_start)) + ' → ' + esc(whenDate(r.window.recent_end)) + ' SGT</li>'
    + '<li>Evidence hash ' + esc(String(r.evidenceHash || '').slice(0, 16)) + '</li>'
    /* A telemetry hash is not a model identity, and is not shown as one. */
    + '<li>Source artifact ' + esc(String(r.sourceArtifact || 'unknown')) + ' — a hash of the '
    + 'telemetry this evidence was computed from, not of a trained model.</li>'
    + '<li>Model version: ' + (r.modelVersion
        ? esc(String(r.modelVersion))
        : 'not recorded — ' + esc(String(r.modelVersionReason || 'no model identity available')))
    + '</li>'
    + '<li>Exported ' + esc(when(INSP.exportedAt)) + ' from ' + esc(INSP.source) + ' — refresh requires rebuild.</li>'
    + '</ul></details></div>';
  return h;
}

/* --------------------------------------------------------------- filtering */
const matches = a => !query || a.id.toLowerCase().indexOf(query) >= 0 || String(a.train || '').toLowerCase().indexOf(query) >= 0;

/* ------------------------------------------------------------ Status/Review */
function statusAssets() {
  return DATA.assets.filter(a => { const r = a.rows[N - 1]; return r.aspect >= 2 || r.aspect === -1 || r.duty === 'off_peak_only'; });
}
const RVWIN = 21;
function recentPeak(a) { let b = {aspect: -9, i: -1}; for (let i = Math.max(0, N - RVWIN); i < N; i++) { const x = a.rows[i]; if (x.aspect > b.aspect) b = {aspect: x.aspect, i: i}; } return b; }
function reviewAssets() {
  const s = new Set(statusAssets().map(a => a.id));
  return DATA.assets.filter(a => {
    if (s.has(a.id)) return false;
    const r = a.rows[N - 1];
    if (r.aspect === 1 || held(r)) return true;
    for (let i = Math.max(0, N - RVWIN); i < N; i++) { const x = a.rows[i]; if (x.aspect >= 2 || x.duty === 'off_peak_only') return true; }
    return false;
  });
}
function renderStatus() {
  const rows = statusAssets().filter(matches).sort((x, y) => y.rows[N - 1].aspect - x.rows[N - 1].aspect);
  const restr = rows.filter(a => a.rows[N - 1].duty === 'off_peak_only').length;
  const unplanned = rows.filter(a => !openJobFor(a.id, N - 1)).length;
  el('status-sub').textContent = rows.length
    ? (rows.length === 1 ? '1 door needs' : rows.length + ' doors need') + ' action now'
      + (restr ? ', ' + restr + ' off-peak only until fixed' : '') + (unplanned ? ' · ' + unplanned + ' not yet in a plan.' : '.')
    : (query ? 'No doors match “' + query + '”.' : 'Nothing needs action right now.');
  el('status-list').innerHTML = rows.length ? headerRow() + rows.map(a => rowFor(a, N - 1)).join('') : emptyBox('Nothing needs action right now.');
}
function renderReview() {
  const rows = reviewAssets().filter(matches).sort((x, y) => y.rows[N - 1].aspect - x.rows[N - 1].aspect || recentPeak(y).aspect - recentPeak(x).aspect);
  el('review-sub').textContent = rows.length
    ? (rows.length === 1 ? '1 door' : rows.length + ' doors') + ' flagged in the last ' + RVWIN + ' days.'
    : (query ? 'No doors match “' + query + '”.' : 'Nothing flagged recently.');
  el('review-list').innerHTML = rows.length ? headerRow() + rows.map(a => {
    const pk = recentPeak(a), cur = a.rows[N - 1];
    const note = cur.aspect === 1 ? 'Plan a maintenance slot.'
      : pk.aspect >= 2 ? 'Reached ' + esc(DATA.rec[pk.aspect]) + ' on ' + DATA.days[pk.i] + ' — confirm the repair held.'
      : 'Flagged recently — review.';
    return '<p class="rvnote">' + note + '</p>' + rowFor(a, N - 1);
  }).join('') : emptyBox('Nothing flagged recently.');
}
const headerRow = () => '<div class="rhead"><span>Door</span><span>Action</span><span>Reason</span><span>Next window</span><span>Job</span><span></span></div>';
const emptyBox = t => '<div class="empty">' + esc(t) + '</div>';

/* ----------------------------------------------------------------- Planner */
function plannedJobs(nightIdx) {
  return jobsAsOf(nightIdx).filter(x => openStates.indexOf(x.state) >= 0 && dayIdx(x.window) <= nightIdx)
    .sort((a, b) => a.order - b.order || dayIdx(a.window) - dayIdx(b.window));
}
function closedThisNight(nightIdx) {
  return loadStore().jobs.map(j => ({job: j, i: closedAt(j, nightIdx), st: stateAsOf(j, nightIdx)})).filter(x => x.i === nightIdx);
}
const assetOf = id => DATA.assets.find(a => a.id === id);

function renderPlanner() {
  el('pl-when').textContent = DATA.days[planNight];
  // The night's workload is a property of the plan, not of what is on screen:
  // allocation and rollover are computed over EVERY planned job, and the search
  // filters the display only.
  const all = plannedJobs(planNight);
  const cap = W.minutes * W.crew;
  let used = 0;
  const rolls = new Set(), pos = new Map();
  all.forEach((x, i) => { pos.set(x.job.id, i); used += x.minutes; if (used > cap) rolls.add(x.job.id); });
  el('pl-cap').textContent = used + ' of ' + cap + ' minutes allocated · ' + W.crew + (W.crew === 1 ? ' crew' : ' crews');
  el('capfill').style.width = Math.min(100, cap ? used / cap * 100 : 0) + '%';
  el('capfill').className = used > cap ? 'over' : '';

  const closedAll = closedThisNight(planNight);
  const candAll = candidates(planNight);
  const plan = all.filter(x => matches(assetOf(x.job.asset)));
  const closed = closedAll.filter(x => matches(assetOf(x.job.asset)));
  const cand = candAll.filter(matches);

  const carried = all.filter(x => dayIdx(x.window) < planNight).length;
  const bits = [];
  if (all.length) bits.push(all.length + (all.length > 1 ? ' jobs' : ' job') + ' in this window');
  if (carried) bits.push(carried + ' carried over');
  if (rolls.size) bits.push(rolls.size + ' would roll over');
  if (closedAll.length) bits.push(closedAll.length + ' closed tonight');
  let sub = bits.length ? bits.join(' · ') + '.' : 'Nothing planned for this night.';
  if (query) sub += ' Showing ' + (plan.length + closed.length + cand.length) + ' of '
    + (all.length + closedAll.length + candAll.length) + ' matching “' + query + '”; the allocation above covers the whole plan.';
  el('plan-sub').textContent = sub;

  let h = '';
  h += all.length
    ? '<div class="psec">In this window</div>' + (plan.length ? plan.map(x => jobRow(x, pos.get(x.job.id), all.length, rolls.has(x.job.id))).join('') : emptyBox('No planned job matches the search.'))
    : emptyBox('Nothing planned for this night. Add a flagged door below, or from Status.');
  if (closed.length) h += '<div class="psec">Closed this night</div>' + closed.map(closedRow).join('');
  if (cand.length) h += '<div class="psec">Flagged, not planned — ' + cand.length + (query && candAll.length !== cand.length ? ' of ' + candAll.length : '') + '</div>' + cand.map(candRow).join('');
  el('plan-list').innerHTML = h;
}

function jobRow(x, i, n, rolls) {
  const a = assetOf(x.job.asset), r = a.rows[planNight];
  const wi = dayIdx(x.window), late = planNight - wi;
  const fit = windowFit(r, planNight, rolls ? planNight + 1 : planNight);
  let notes = '';
  if (late > 0) notes += '<span class="ptag">carried from ' + x.window + ' · ' + late + (late > 1 ? ' nights' : ' night') + ' late</span>';
  if (held(r)) notes += '<span class="ptag held">held from an earlier escalation; evidence today: ' + esc(DATA.rec[r.raw]) + '</span>';
  notes += '<span class="pnote ' + fit.kind + '">' + (rolls ? 'Past capacity — would roll to the next window. ' : '') + esc(fit.detail) + '</span>';
  notes += '<span class="vline">Telemetry check: ' + verificationChipForJob(x.job.id, a.id, planNight) + '</span>';
  // Deliberately "returned to service", not "verified": this is a human signing
  // the door back into traffic. Telemetry verification that the repair actually
  // held is a separate assessment (headway/repair_verification.py) and is not
  // wired to this button.
  const acts = x.state === 'work_done'
    ? '<button class="btn prim" data-job="verify:' + esc(x.job.id) + '" data-fk="v-' + x.job.id + '">Returned to service</button>'
      + '<button class="btn" data-job="reopen:' + esc(x.job.id) + '">Reopen</button>'
    : '<button class="btn prim" data-job="work:' + esc(x.job.id) + '" data-fk="w-' + x.job.id + '">Work done</button>'
      + '<button class="btn" data-job="rm:' + esc(x.job.id) + '" aria-label="Remove from plan">Remove</button>';
  return '<div class="pjob' + (rolls ? ' rolls' : '') + (x.state === 'work_done' ? ' wk' : '') + '"><span class="pnum">' + (i + 1) + '</span>' + head(r.aspect)
    + '<span class="pmeta"><button class="link" data-open="' + esc(a.id) + '" data-fk="id-' + esc(a.id) + '">' + esc(a.id) + '</button>'
    + '<span class="psub">' + esc(DATA.rec[r.aspect]) + (x.state === 'work_done' ? ' · work done, awaiting return to service' : '') + '</span>' + notes + '</span>'
    + '<span class="pdur"><label class="vh" for="d-' + x.job.id + '">Estimated minutes</label>'
    + '<input id="d-' + x.job.id + '" class="dur" type="number" min="5" max="480" step="5" value="' + x.minutes + '" data-dur="' + esc(x.job.id) + '" data-fk="d-' + x.job.id + '"><span class="unit">min</span></span>'
    + '<span class="pact"><button class="btn" data-pmv="up:' + esc(x.job.id) + '"' + (i === 0 ? ' disabled' : '') + ' aria-label="Higher priority">↑</button>'
    + '<button class="btn" data-pmv="dn:' + esc(x.job.id) + '"' + (i === n - 1 ? ' disabled' : '') + ' aria-label="Lower priority">↓</button>'
    + acts + '</span></div>';
}
function closedRow(x) {
  const a = DATA.assets.find(z => z.id === x.job.asset);
  const done = x.st.state === 'done';
  return '<div class="pjob done"><span class="pnum">' + (done ? '✓' : '✗') + '</span>' + head(a.rows[planNight].aspect)
    + '<span class="pmeta"><button class="link" data-open="' + esc(a.id) + '" data-fk="id-' + esc(a.id) + '">' + esc(a.id) + '</button>'
    + '<span class="psub">' + (done ? 'work done, returned to service by an operator' : 'removed from the plan') + '</span></span>'
    + '<span class="pact"><button class="btn" data-job="reopen:' + esc(x.job.id) + '">' + (done ? 'Reopen' : 'Undo removal') + '</button></span></div>';
}
function candRow(a) {
  const r = a.rows[planNight], fit = windowFit(r, planNight, planNight);
  const prior = jobsFor(a.id).length;
  return '<div class="pjob out"><span class="pnum">○</span>' + head(r.aspect)
    + '<span class="pmeta"><button class="link" data-open="' + esc(a.id) + '" data-fk="id-' + esc(a.id) + '">' + esc(a.id) + '</button>'
    + '<span class="psub">' + esc(DATA.rec[r.aspect]) + ' · ' + esc(fit.text) + '</span>'
    + (held(r) ? '<span class="ptag held">held from an earlier escalation</span>' : '')
    + (prior ? '<span class="ptag">new escalation · ' + prior + ' earlier job' + (prior > 1 ? 's' : '') + ' on this door</span>' : '') + '</span>'
    + '<span class="pact"><button class="btn prim" data-job="add:' + esc(a.id) + '" data-fk="add-' + esc(a.id) + '">Add to plan</button></span></div>';
}

/* Priority and duration are dated events like everything else, so replaying an
   earlier night reproduces that night's order and allocation rather than
   today's. A move rewrites the whole visible order as one dated batch, which
   keeps the ranks unambiguous. */
function planMove(jobId, d) {
  const plan = plannedJobs(planNight), i = plan.findIndex(x => x.job.id === jobId), k = i + d;
  if (i < 0 || k < 0 || k >= plan.length) return;
  const order = plan.map(x => x.job.id);
  const t = order[i]; order[i] = order[k]; order[k] = t;
  const s = loadStore(), night = DATA.days[planNight], at = new Date().toISOString();
  order.forEach((id, n) => {
    const j = s.jobs.find(z => z.id === id);
    if (j) j.events.push({type: 'reordered', night: night, order: n, at: at});
  });
  saveStore(s);
}
function setDuration(jobId, mins) {
  const m = Math.max(5, Math.min(480, Math.round(mins / 5) * 5));
  addEvent(jobId, {type: 'duration', night: DATA.days[planNight], minutes: m});
}

/* ------------------------------------------------------------------ Fleet */
function counts() { const c = {'-1': 0, 0: 0, 1: 0, 2: 0, 3: 0}; for (const a of DATA.assets) c[a.rows[day].aspect]++; return c; }
function renderFleet() {
  el('fday').max = N - 1; el('fday').value = day; el('fdate').textContent = DATA.days[day];
  const c = counts();
  const items = [['all', 'All', DATA.assets.length], [3, 'Withdraw', c[3]], [2, 'Tonight', c[2]], [1, 'Plan', c[1]],
    [0, 'Monitor', c[0]], [-1, 'Check data', c[-1]], ['duty', 'Off-peak only', DATA.assets.filter(a => a.rows[day].duty === 'off_peak_only').length]];
  el('filters').innerHTML = items.map(it => '<button class="chip' + (it[0] === 'duty' ? ' cd' : '') + '" data-f="' + it[0] + '" aria-pressed="' + (String(filter) === String(it[0])) + '"><span class="n">' + it[2] + '</span><span class="t">' + esc(it[1]) + '</span></button>').join('');
  const list = DATA.assets.filter(matches).filter(a => filter === 'all' || (filter === 'duty' ? a.rows[day].duty === 'off_peak_only' : a.rows[day].aspect === Number(filter)));
  el('fleet').innerHTML = list.length ? list.map(a => {
    const r = a.rows[day];
    return '<button class="tile a' + r.aspect + '" data-open="' + esc(a.id) + '" aria-label="' + esc(a.id + ', ' + DATA.rec[r.aspect]) + '">' + head(r.aspect)
      + '<span class="meta"><span class="mrg">' + (r.aspect === -1 ? 'Check data' : r.aspect === 0 ? 'OK' : Number.isFinite(r.margin) ? (r.margin <= 0 ? 'no margin' : (Math.floor(r.margin * 10) / 10).toFixed(1) + ' d') : '—') + '</span>'
      + '<span class="lab">' + esc(a.id) + '</span>' + (r.duty === 'off_peak_only' ? '<span class="dm">off-peak only</span>' : '') + '</span></button>';
  }).join('') : emptyBox('No doors match “' + query + '”.');
}

/* ------------------------------------------------------------ Door detail */
const CW = 920, CH = 230, ML = 40, MR = 16, MT = 12, MB = 24;
const cx = i => ML + i * (CW - ML - MR) / Math.max(N - 1, 1);
function chartBlock(a) {
  const vals = a.rows.map(r => r.hi), finite = vals.filter(Number.isFinite);
  const nav = '<div class="daynav"><button class="btn" id="dprev" aria-label="Previous day">‹</button><span id="dwhen">' + DATA.days[day] + '</span><button class="btn" id="dnext" aria-label="Next day">›</button><span class="hint">drag the chart to change date</span></div>';
  if (!finite.length) return '<p class="sub">No wear readings for this door.</p>' + nav;
  const lo = Math.min(0, ...finite), fl = Number.isFinite(DATA.threshold) ? DATA.threshold : 0, hi = Math.max(6, fl * 1.06, ...finite);
  const cy = v => CH - MB - (v - lo) * (CH - MT - MB) / (hi - lo);
  let path = '', pen = false;
  vals.forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } path += (pen ? 'L' : 'M') + cx(i).toFixed(1) + ' ' + cy(v).toFixed(1) + ' '; pen = true; });
  const flL = fl ? '<line x1="' + cx(0) + '" x2="' + cx(N - 1) + '" y1="' + cy(fl).toFixed(1) + '" y2="' + cy(fl).toFixed(1) + '" stroke="var(--red)" stroke-dasharray="4 4" opacity=".6"/><text x="' + (cx(N - 1) - 2) + '" y="' + (cy(fl) - 4).toFixed(1) + '" text-anchor="end" fill="var(--red)" font-size="11">learned threshold</text>' : '';
  const mk = cx(day).toFixed(1);
  return '<div class="chart"><svg id="dchart" viewBox="0 0 ' + CW + ' ' + CH + '" preserveAspectRatio="none" role="img" aria-label="Wear level over time; drag to change date">'
    + flL + '<path d="' + path + '" fill="none" stroke="var(--accent)" stroke-width="2"/>'
    + '<line id="dmark" x1="' + mk + '" x2="' + mk + '" y1="' + MT + '" y2="' + (CH - MB) + '" stroke="var(--text)" stroke-width="1.5"/></svg></div>' + nav;
}
function extra(a) {
  const r = a.rows[day];
  const cs = Object.entries(r.contributions).filter(kv => Number.isFinite(kv[1])).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
  const base = r.baseline === 'asset_reference' ? 'Compared against this door’s own early history.'
    : r.baseline === 'fleet_fallback' ? 'Compared against the fleet average (own history too short).' : '';
  const drv = cs.length ? '<p class="k dk2">What is pushing the wear reading</p><ul class="drv">' + cs.map(kv => {
    const n = DATA.signalNames[kv[0].replace(/_hx$/, '')] || kv[0].replace(/_hx$/, '').replaceAll('_', ' ');
    return '<li><span>' + esc(n) + '</span><b>' + (kv[1] > 0 ? '+' : '') + kv[1].toFixed(1) + '</b></li>';
  }).join('') + '</ul>' : '';
  return (base ? '<p class="hint">' + base + '</p>' : '') + drv + inspectionBlock(a.id, day);
}
function renderDetail() {
  const a = DATA.assets.find(x => x.id === selected);
  if (!a) { el('detail').innerHTML = '<p class="sub">Pick a door from Status, Review or Fleet.</p>'; return; }
  el('detail').innerHTML = '<h3>' + esc(a.id) + '</h3>' + chartBlock(a) + '<div id="dbody"></div>';
  el('dbody').innerHTML = detailFor(a, day, windowFit(a.rows[day], day, day)) + extra(a);
  wireChart(a);
}
function updateDay(a) {
  day = Math.max(0, Math.min(N - 1, day));
  const m = el('dmark'); if (m) { const gx = cx(day).toFixed(1); m.setAttribute('x1', gx); m.setAttribute('x2', gx); }
  const w = el('dwhen'); if (w) w.textContent = DATA.days[day];
  el('dbody').innerHTML = detailFor(a, day, windowFit(a.rows[day], day, day)) + extra(a);
}
function seek(a, ev) {
  const svg = el('dchart'); if (!svg) return;
  const r = svg.getBoundingClientRect();
  const xCW = (ev.clientX - r.left) / r.width * CW, step = (CW - ML - MR) / Math.max(N - 1, 1);
  const i = Math.max(0, Math.min(N - 1, Math.round((xCW - ML) / step)));
  if (i !== day) { day = i; updateDay(a); }
}
function wireChart(a) {
  const svg = el('dchart');
  if (svg) {
    svg.addEventListener('pointerdown', e => { dragging = true; try { svg.setPointerCapture(e.pointerId); } catch (_) {} seek(a, e); });
    svg.addEventListener('pointermove', e => { if (dragging) seek(a, e); });
    svg.addEventListener('pointerup', () => { dragging = false; });
    svg.addEventListener('pointercancel', () => { dragging = false; });
  }
  const p = el('dprev'), n = el('dnext');
  if (p) p.onclick = () => { if (day > 0) { day--; updateDay(a); } };
  if (n) n.onclick = () => { if (day < N - 1) { day++; updateDay(a); } };
}

/* ------------------------------------------------------------------ render */
function render() {
  el('b-status').textContent = statusAssets().length;
  el('b-review').textContent = reviewAssets().length;
  el('b-planner').textContent = plannedJobs(N - 1).length;
  el('b-fleet').textContent = DATA.assets.length;
  el('replay').textContent = 'Replay — observations through ' + DATA.replay.through + '; daily aggregate available ' + DATA.replay.availableAt + '.';
  el('save').textContent = saveNote;
  el('save').className = 'save' + (saveNote.indexOf('Not saved') === 0 ? ' bad' : saveNote ? ' ok' : '');
  for (const v of ['status', 'review', 'planner', 'fleet', 'asset']) el('v-' + v).classList.toggle('on', v === view);
  document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-current', b.dataset.view === view ? 'page' : 'false'));
  el('back').hidden = view !== 'asset';
  if (view === 'status') renderStatus();
  else if (view === 'review') renderReview();
  else if (view === 'planner') renderPlanner();
  else if (view === 'fleet') renderFleet();
  else renderDetail();
  // Scope to the visible section: hidden views keep their last markup, and the
  // same focus key can appear there, where focus() is a silent no-op.
  if (focusKey) {
    const scope = el('v-' + view) || document;
    const t = scope.querySelector('[data-fk="' + focusKey + '"]');
    if (t) t.focus();
    focusKey = null;
  }
}

/* ------------------------------------------------------------------ events */
el('fday').addEventListener('input', e => { day = +e.target.value; render(); });
el('q').addEventListener('input', e => { query = e.target.value.trim().toLowerCase(); render(); });
document.addEventListener('change', e => {
  const d = e.target.closest('[data-dur]');
  if (d) { focusKey = d.dataset.fk; setDuration(d.dataset.dur, Number(d.value)); render(); }
});
document.addEventListener('click', e => {
  const mv = e.target.closest('[data-pmv]');
  if (mv) { const a = mv.dataset.pmv.split(':'); planMove(a.slice(1).join(':'), a[0] === 'up' ? -1 : 1); render(); return; }

  const jb = e.target.closest('[data-job]');
  if (jb) {
    const s = jb.dataset.job, k = s.indexOf(':'), verb = s.slice(0, k), id = s.slice(k + 1);
    // The button that was clicked usually disappears with the state change, so
    // park focus on the door name, which survives every transition.
    const asset = verb === 'add' ? id : (loadStore().jobs.find(j => j.id === id) || {}).asset;
    focusKey = asset ? 'id-' + asset : (jb.dataset.fk || null);
    const night = DATA.days[view === 'planner' ? planNight : N - 1];
    if (verb === 'add') { if (view !== 'planner') planNight = N - 1; createJob(id, view === 'planner' ? planNight : N - 1); }
    else if (verb === 'work') addEvent(id, {type: 'work_done', night: night});
    else if (verb === 'verify') addEvent(id, {type: 'verified', night: night});
    else if (verb === 'rm') addEvent(id, {type: 'removed', night: night});
    else if (verb === 'reopen') addEvent(id, {type: 'reopened', night: night, window: night});
    render(); return;
  }

  const xp = e.target.closest('[data-xp]');
  if (xp) { const id = xp.dataset.xp; expanded.has(id) ? expanded.delete(id) : expanded.add(id); focusKey = xp.dataset.fk; render(); return; }

  if (e.target.id === 'pl-prev') { planNight = Math.max(0, planNight - 1); render(); return; }
  if (e.target.id === 'pl-next') { planNight = Math.min(N - 1, planNight + 1); render(); return; }
  if (e.target.id === 'pl-clear') {
    if (confirm('Clear every stored job, on all nights? This cannot be undone.')) {
      try { localStorage.removeItem(JOBKEY); saveNote = 'Cleared'; } catch (x) { saveNote = 'Not saved — browser storage is unavailable'; }
      render();
    }
    return;
  }
  const o = e.target.closest('[data-open]');
  if (o) { if (view !== 'asset') prevView = view; if (prevView !== 'fleet') day = prevView === 'planner' ? planNight : N - 1; selected = o.dataset.open; view = 'asset'; render(); window.scrollTo(0, 0); return; }
  if (e.target.id === 'back') { view = prevView; render(); window.scrollTo(0, 0); return; }
  const f = e.target.closest('[data-f]'); if (f) { filter = f.dataset.f; render(); return; }
  const nav = e.target.closest('[data-view]'); if (nav) { view = nav.dataset.view; render(); window.scrollTo(0, 0); }
});
el('theme').onclick = () => {
  const d = document.documentElement.dataset.theme === 'dark';
  document.documentElement.dataset.theme = d ? 'light' : 'dark';
  el('theme').textContent = d ? 'Dark theme' : 'Light theme';
};
render();
