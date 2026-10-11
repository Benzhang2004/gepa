// Async Pipeline Dashboard. Two sources feed one renderer:
//   live:   poll status.json (the engine's snapshot, exact) and keep a history of it;
//   replay: parse events.jsonl and compute the same snapshot at any time t (resources approximate: the log
//           records what bodies took, not what budgets have left).
// Model code first (also usable from node for a self-test), DOM code after.

function parseEvents(text) {
  const ev = [];
  for (const line of text.split("\n")) { const s = line.trim(); if (!s) continue; try { ev.push(JSON.parse(s)); } catch (e) { /* skip */ } }
  return ev;
}

// Build a replayable model from events: fires with ready/fire/complete times, item ends, edge transitions.
function buildReplay(ev) {
  const m = { topology: null, stages: [], resources: [], fires: [], ends: [], items: new Map(), t0: 0, tEnd: 0, trans: [], stop: null };
  const open = new Map();
  for (const e of ev) {
    if (e.kind === "run_start") { m.stages = e.stages || []; m.resources = e.resources || []; m.topology = e.topology || null; m.t0 = e.t; }
    else if (e.kind === "item") m.items.set(e.item, { id: e.item, parent: e.parent, created: e.t, path: [], end: null });
    else if (e.kind === "fire") {
      const f = { stage: e.stage, item: e.item, t0: e.t, t1: null, demand: e.demand || {} };
      m.fires.push(f); open.set(e.stage + "|" + e.item, f);
      if (!m.items.has(e.item)) m.items.set(e.item, { id: e.item, parent: null, created: e.t, path: [], end: null });
      m.items.get(e.item).path.push(f);
    } else if (e.kind === "complete") { const f = open.get(e.stage + "|" + e.item); if (f) { f.t1 = e.t; open.delete(e.stage + "|" + e.item); } }
    else if (e.kind === "end") { m.ends.push(e); const it = m.items.get(e.item); if (it) it.end = e; }
    else if (e.kind === "stop") m.stop = e;
    m.tEnd = Math.max(m.tEnd, e.t || 0);
  }
  for (const f of m.fires) if (f.t1 == null) f.t1 = m.tEnd;
  const comps = m.fires.map(f => ({ t: f.t1, stage: f.stage, item: f.item })).sort((a, b) => a.t - b.t);
  const latestCompBefore = (t, notItem) => { let lo = 0, hi = comps.length - 1, best = null; while (lo <= hi) { const mid = (lo + hi) >> 1; if (comps[mid].t <= t) { best = mid; lo = mid + 1; } else hi = mid - 1; } for (let k = best; k != null && k >= 0 && t - comps[k].t <= 0.01; k--) if (comps[k].item !== notItem) return comps[k]; return null; };
  for (const it of m.items.values()) {
    it.path.sort((a, b) => a.t0 - b.t0);
    it.readItem = it.path.length > 20 && it.path.every(f => f.stage === it.path[0].stage);
    let ready = it.created;
    for (const f of it.path) { f.ready = ready; ready = f.t1; }
    if (it.readItem) continue;
    for (let i = 1; i < it.path.length; i++) if (it.path[i - 1].stage !== it.path[i].stage) m.trans.push({ src: it.path[i - 1].stage, dst: it.path[i].stage, t: it.path[i].t0 });
    if (it.path.length) {
      let src = null;
      if (it.parent != null) { const p = m.items.get(it.parent); if (p) { const f = p.path.find(f => f.t0 <= it.created && f.t1 >= it.created) || [...p.path].reverse().find(f => f.t1 <= it.created); if (f) src = f.stage; } }
      else { const c = latestCompBefore(it.created, it.id); if (c) src = c.stage; }
      if (src) m.trans.push({ src, dst: it.path[0].stage, t: it.path[0].t0 });
    }
    if (it.end && it.path.length) it.lastStage = it.path[it.path.length - 1].stage;
  }
  m.trans.sort((a, b) => a.t - b.t);
  if (!m.topology) m.topology = inferTopology(m);
  return m;
}

function inferTopology(m) {
  const counts = new Map();
  for (const tr of m.trans) { const k = tr.src + "->" + tr.dst; counts.set(k, (counts.get(k) || 0) + 1); }
  const edges = [...counts.entries()].map(([k, n]) => { const [src, dst] = k.split("->"); return { src, dst, label: null, n }; });
  const readStages = new Set(); for (const it of m.items.values()) if (it.readItem) readStages.add(it.path[0].stage);
  const incoming = new Set(edges.map(e => e.dst));
  const sources = [...readStages, ...m.stages.filter(s => !incoming.has(s) && !readStages.has(s))];
  const order = [], seen = new Set(), onPath = new Set();
  const visit = s => { if (seen.has(s)) return; seen.add(s); onPath.add(s); order.push(s); for (const e of edges.filter(e => e.src === s).sort((x, y) => y.n - x.n)) if (!onPath.has(e.dst)) visit(e.dst); onPath.delete(s); };
  for (const s of sources) visit(s); for (const s of m.stages) visit(s);
  const demand = {}; for (const f of m.fires) { demand[f.stage] = demand[f.stage] || {}; for (const r of Object.keys(f.demand)) demand[f.stage][r] = f.demand[r]; }
  return {
    stages: order.map(name => ({ name, mode: readStages.has(name) ? "read" : "consume", kind: "unknown", workers: null, max_workers: null, max_queue: null, demand: demand[name] || {}, hold: [] })),
    edges: edges.map(e => ({ src: e.src, dst: e.dst, label: null })),
    resources: Object.fromEntries(m.resources.map(r => [r, { cap: null }])),
    inferred: true,
  };
}

// The engine's snapshot shape, computed from the replay model at time t (relative seconds).
function snapshotAt(m, t) {
  const T = m.t0 + t;
  const stages = {};
  for (const st of m.topology.stages) stages[st.name] = { waiting: 0, in_flight: 0, workers: st.max_workers, completed: 0 };
  const inflightUnits = {}, taken = {};
  for (const r of m.resources) { inflightUnits[r] = 0; taken[r] = 0; }
  for (const f of m.fires) {
    const s = stages[f.stage]; if (!s) continue;
    if (f.t0 <= T && f.t1 > T) { s.in_flight++; for (const [r, u] of Object.entries(f.demand)) if (r in inflightUnits) inflightUnits[r] += u; }
    if (f.t1 <= T) s.completed++;
    if (f.ready != null && f.ready <= T && f.t0 > T) s.waiting++;
    if (f.t0 <= T) for (const [r, u] of Object.entries(f.demand)) if (r in taken) taken[r] += u;
  }
  // a read place's content: items delivered into it (ended "accepted") up to t, plus the seed if it was injected
  for (const st of m.topology.stages) if (st.mode === "read") { let n = 0; for (const e of m.ends) if (e.t <= T && e.outcome === "accepted") n++; stages[st.name].waiting = n; }
  const edges = {};
  for (const tr of m.trans) {
    if (tr.t > T) break;
    const k = tr.src + "->" + tr.dst; const e = edges[k] || (edges[k] = { src: tr.src, dst: tr.dst, label: null, count: 0, per_min: 0 });
    e.count++; if (T - tr.t <= 60) e.per_min++;
  }
  const terminals = {}, outcomes = {};
  for (const e of m.ends) { if (e.t > T) continue; outcomes[e.outcome] = (outcomes[e.outcome] || 0) + 1; const it = m.items.get(e.item); if (it && it.lastStage && !["accepted", "fanned_out", "nothing"].includes(e.outcome)) { const k = it.lastStage + ":" + e.outcome; terminals[k] = (terminals[k] || 0) + 1; } }
  const resources = {};
  for (const r of m.resources) { const cap = (m.topology.resources[r] || {}).cap ?? null; resources[r] = { available: cap != null ? cap - inflightUnits[r] : null, cap, in_flight: inflightUnits[r], taken: taken[r], used: null, refused: null, approx: true }; }
  return { t, topology: m.topology, stopping: !!(m.stop && m.stop.t <= T), stop_reason: m.stop && m.stop.t <= T ? m.stop.reason : null, stages, edges, terminals, outcomes, resources };
}

if (typeof module !== "undefined") module.exports = { parseEvents, buildReplay, snapshotAt, inferTopology };

// ----------------------------------------------------------------------------------------------- DOM
if (typeof document !== "undefined") (function () {
  const $ = id => document.getElementById(id);
  const PALETTE = ["#2f6fed", "#e0792b", "#2a9d6f", "#c43d6b", "#7b5bd6", "#b58a1e", "#1f9fb8", "#8b6a4a", "#d14a2a", "#4f7a28", "#5c6bc0", "#00897b"];
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (v, d = 1) => v == null ? "–" : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : Number.isInteger(v) ? String(v) : Math.abs(v) < 1 ? Number(v).toPrecision(2) : Number(v).toFixed(d);
  const fmtMin = s => (s / 60).toFixed(1) + " min";

  let mode = "live", replay = null, timer = null, playing = null;
  const history = { t: [], snaps: [] }; // live history of snapshots (bounded)
  const scaleMax = {}; // running max per place for the fill bars

  // ---------------------------------------------------------------- figure
  function renderFigure(snap) {
    const topo = snap.topology, order = topo.stages.map(s => s.name), n = order.length;
    const top = Math.ceil(n / 2), colW = 262, PW = 26, BW = 150, H = 74, GAP = colW - PW - BW;
    const W = 210 + Math.max(top, n - top) * colW + 70;
    const yTop = 100, yRes = 300, yBot = 490, FH = 640;
    const colX = i => 200 + i * colW;
    const pos = {};
    order.forEach((s, i) => { if (i < top) pos[s] = { x: colX(i), y: yTop, row: "top" }; else { const j = i - top; pos[s] = { x: colX(top - 1 - j), y: yBot, row: "bot" }; } });
    const bodyMid = s => pos[s].x + PW + BW / 2;
    let svg = `<svg viewBox="0 0 ${W} ${FH}" width="100%" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Pipeline figure">`;
    svg += `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="currentColor"/></marker></defs>`;
    // resources band
    const resNames = Object.keys(topo.resources);
    const bandL = 230, bandR = W - 160, rW = Math.min(240, (bandR - bandL - 20) / Math.max(1, resNames.length));
    const resPos = {}; resNames.forEach((r, i) => { resPos[r] = { x: bandL + 10 + rW / 2 + i * rW, y: yRes }; });
    svg += `<rect x="${bandL}" y="${yRes - 46}" width="${bandR - bandL}" height="92" rx="8" fill="none" stroke="var(--line)" stroke-dasharray="5 4"/>`;
    svg += `<text x="${bandL + 10}" y="${yRes - 52}" class="ts">engine resources: what is left of each cap</text>`;
    // demand lines: from the body's edge facing the band to the ellipse's near edge; faint, bold on hover of the stage
    for (const st of topo.stages) for (const r of Object.keys(st.demand || {})) {
      if (!resPos[r]) continue; const p = pos[st.name], q = resPos[r], up = p.row === "top";
      svg += `<line data-stage="${st.name}" x1="${bodyMid(st.name)}" y1="${up ? p.y + H : p.y}" x2="${q.x}" y2="${q.y + (up ? -36 : 36)}" class="demand"/>`;
    }
    // edges
    const edgeStats = snap.edges || {};
    const statFor = (src, dst, label) => edgeStats[`${src}->${dst}` + (label ? ":" + label : "")] || Object.values(edgeStats).find(e => e.src === src && e.dst === dst) || { count: 0, per_min: 0 };
    const idx = Object.fromEntries(order.map((s, i) => [s, i]));
    const lab = (st, e) => `${fmt(st.per_min, 0)}/min · ${st.count}`;
    const tag = (e, x, y, anchor = "middle") => (e.label ? `<text x="${x}" y="${y}" text-anchor="${anchor}" class="ts">${e.label}</text>` : "");
    for (const e of topo.edges) {
      if (!(e.src in pos) || !(e.dst in pos)) continue;
      const st = statFor(e.src, e.dst, e.label), hot = st.per_min > 0 ? " hot" : "";
      const a = pos[e.src], b = pos[e.dst];
      const aR = a.x + PW + BW, ay = a.y + H / 2, by = b.y + H / 2;
      if (idx[e.dst] === idx[e.src] + 1 && a.row === b.row && a.row === "top") {
        svg += `<line x1="${aR}" y1="${ay}" x2="${b.x - 2}" y2="${by}" class="edge${hot}" marker-end="url(#ah)"/><text x="${aR + GAP / 2}" y="${ay - 8}" text-anchor="middle" class="tm">${lab(st, e)}</text>` + tag(e, aR + GAP / 2, ay + 18);
      } else if (idx[e.dst] === idx[e.src] + 1 && a.row === b.row) {
        const bR = b.x + PW + BW;
        svg += `<line x1="${a.x}" y1="${ay}" x2="${bR + 2}" y2="${by}" class="edge${hot}" marker-end="url(#ah)"/><text x="${bR + GAP / 2}" y="${ay - 8}" text-anchor="middle" class="tm">${lab(st, e)}</text>` + tag(e, bR + GAP / 2, ay + 18);
      } else if (idx[e.dst] === idx[e.src] + 1) { // top row end -> bottom row start, down the right side
        const xr = aR + 34;
        svg += `<path d="M${aR},${ay} H${xr} V${by} H${b.x + PW + BW + 2}" class="edge${hot}" marker-end="url(#ah)"/><text x="${xr + 8}" y="${(ay + by) / 2 + 4}" class="tm">${lab(st, e)}</text>` + tag(e, xr + 8, (ay + by) / 2 + 20, "start");
      } else if (idx[e.dst] <= idx[e.src]) { // the loop back, around the left side
        const xl = Math.min(a.x, b.x) - 44;
        svg += `<path d="M${a.x},${ay} H${xl} V${by} H${b.x - 2}" class="edge${hot}" marker-end="url(#ah)"/><text x="${xl - 8}" y="${(ay + by) / 2 + 4}" text-anchor="end" class="tm">${lab(st, e)}</text>` + tag(e, xl - 8, (ay + by) / 2 + 20, "end");
      } else {
        const lift = a.row === "top" ? -44 : 44;
        svg += `<path d="M${aR},${ay} C${aR},${ay + lift} ${b.x},${by + lift} ${b.x - 2},${by}" class="edge${hot}" marker-end="url(#ah)"/><text x="${(aR + b.x) / 2}" y="${ay + lift * 0.8}" text-anchor="middle" class="tm">${lab(st, e)}</text>` + tag(e, (aR + b.x) / 2, ay + lift * 0.8 + 14);
      }
    }
    // terminals: where items end, above the top row and below the bottom row
    const termsBy = {};
    for (const [k, v] of Object.entries(snap.terminals || {})) { const [s, o] = k.split(":"); (termsBy[s] = termsBy[s] || []).push([o, v]); }
    for (const [s, list] of Object.entries(termsBy)) {
      if (!(s in pos)) continue; const p = pos[s], up = p.row === "top";
      list.sort((x, y) => y[1] - x[1]).forEach(([o, v], k) => {
        const cx = p.x + PW + BW - 16, cy = up ? p.y - 44 - k * 22 : p.y + H + 44 + k * 22;
        svg += `<line x1="${cx}" y1="${up ? p.y : p.y + H}" x2="${cx}" y2="${cy + (up ? 9 : -9)}" class="edge" marker-end="url(#ah)"/>`;
        svg += `<circle cx="${cx}" cy="${cy}" r="7" class="term"/><text x="${cx + 12}" y="${cy + 4}" class="ts">${o} ${v}</text>`;
      });
    }
    // stages: place box + body box, demand printed inside the body
    for (const st of topo.stages) {
      const p = pos[st.name], s = snap.stages[st.name] || { waiting: 0, in_flight: 0, workers: st.max_workers, completed: 0 };
      const isPool = st.mode === "read";
      scaleMax[st.name] = Math.max(scaleMax[st.name] || 1, s.waiting, st.max_queue || 0);
      const fillH = Math.min(1, s.waiting / scaleMax[st.name]) * (H - 4);
      svg += `<g class="stage" data-stage="${st.name}">`;
      svg += `<rect x="${p.x}" y="${p.y}" width="${PW}" height="${H}" class="place" fill="${isPool ? "var(--pool)" : "var(--queue)"}"/>`;
      svg += `<rect x="${p.x + 2}" y="${p.y + H - 2 - fillH}" width="${PW - 4}" height="${fillH}" class="fill"/>`;
      svg += `<text x="${p.x + PW / 2}" y="${p.y + H / 2 + 4}" text-anchor="middle" class="tm">${s.waiting}</text>`;
      const workers = s.workers ?? st.max_workers; const bodyFill = st.kind === "head" ? "var(--head)" : st.kind === "worker" ? "var(--worker)" : "var(--queue)";
      svg += `<rect x="${p.x + PW}" y="${p.y}" width="${BW}" height="${H}" class="body" fill="${bodyFill}"/>`;
      if (workers) { const bw = Math.min(1, s.in_flight / workers) * (BW - 4); svg += `<rect x="${p.x + PW + 2}" y="${p.y + H - 5}" width="${bw}" height="3" class="fill"/>`; }
      svg += `<text x="${bodyMid(st.name)}" y="${p.y + 17}" text-anchor="middle" class="t">${st.name}</text>`;
      const line2 = isPool && s.lineages != null ? `${s.lineages}${st.max_in_flight ? "/" + st.max_in_flight : ""} alive` : `${s.in_flight}${workers ? "/" + workers : ""} running`;
      svg += `<text x="${bodyMid(st.name)}" y="${p.y + 32}" text-anchor="middle" class="tm">${line2}</text>`;
      const dems = Object.entries(st.demand || {}).map(([r, u]) => `${r} ${fmt(u, 2)}`);
      if (!dems.length) svg += `<text x="${bodyMid(st.name)}" y="${p.y + 48}" text-anchor="middle" class="ts">no demand</text>`;
      else dems.slice(0, 2).forEach((d, k) => { svg += `<text x="${bodyMid(st.name)}" y="${p.y + 48 + k * 13}" text-anchor="middle" class="ts">${d}</text>`; });
      svg += `<text x="${p.x + PW / 2}" y="${p.row === "top" ? p.y - 7 : p.y + H + 14}" text-anchor="middle" class="ts">${isPool ? "pool" : "queue"}</text>`;
      svg += `</g>`;
    }
    // resources
    for (const r of resNames) {
      const p = resPos[r], v = (snap.resources || {})[r] || {}; const cap = v.cap ?? null, left = v.available;
      svg += `<ellipse cx="${p.x}" cy="${p.y}" rx="${rW / 2 - 10}" ry="34" fill="${cap != null && v.used != null && v.used > 0 ? "var(--budget)" : "var(--bandwidth)"}" stroke="var(--fg)" stroke-width="1.2"/>`;
      svg += `<text x="${p.x}" y="${p.y - 12}" text-anchor="middle" class="t">${r}</text>`;
      if (cap != null && left != null) {
        const bw = rW - 70, fr = Math.max(0, Math.min(1, left / cap));
        svg += `<rect x="${p.x - bw / 2}" y="${p.y - 3}" width="${bw}" height="7" class="usedbar"/><rect x="${p.x - bw / 2}" y="${p.y - 3}" width="${bw * fr}" height="7" class="leftbar"/>`;
        svg += `<text x="${p.x}" y="${p.y + 20}" text-anchor="middle" class="ts">${fmt(left)} of ${fmt(cap)} left</text>`;
      } else if (left != null) svg += `<text x="${p.x}" y="${p.y + 10}" text-anchor="middle" class="ts">${fmt(left)} left</text>`;
      else if (v.used != null && v.used > 0) svg += `<text x="${p.x}" y="${p.y + 6}" text-anchor="middle" class="ts">${fmt(v.used)} spent, no cap</text><text x="${p.x}" y="${p.y + 20}" text-anchor="middle" class="ts">${fmt(v.in_flight)} in flight</text>`;
      else svg += `<text x="${p.x}" y="${p.y + 10}" text-anchor="middle" class="ts">${fmt(v.in_flight)} in flight</text>`;
    }
    svg += "</svg>";
    $("figure").innerHTML = svg;
    for (const g of $("figure").querySelectorAll("g.stage")) {
      const name = g.dataset.stage, lines = $("figure").querySelectorAll(`line.demand[data-stage="${name}"]`);
      g.addEventListener("mouseenter", () => lines.forEach(l => l.classList.add("hot")));
      g.addEventListener("mouseleave", () => lines.forEach(l => l.classList.remove("hot")));
    }
  }

  function renderSummary(snap) {
    const chips = [];
    const chip = (k, v) => chips.push(`<div class="chip"><span class="k">${k}</span><span class="v">${v}</span></div>`);
    chip("time", fmtMin(snap.t)); chip("state", snap.stop_reason || (snap.stopping ? "stopping" : "running"));
    const alive = Object.values(snap.stages).reduce((a, s) => a + s.in_flight, 0); chip("bodies running", alive);
    for (const [o, v] of Object.entries(snap.outcomes || {}).sort((a, b) => b[1] - a[1])) chip(o, v);
    $("summary").innerHTML = chips.join("");
  }

  function renderTables(snap) {
    $("edgeRows").innerHTML = Object.values(snap.edges || {}).sort((a, b) => b.count - a.count).map(e => `<tr><td>${e.src} → ${e.dst}${e.label ? " <em>" + e.label + "</em>" : ""}</td><td class="num">${e.count}</td><td class="num">${e.per_min}</td></tr>`).join("");
    $("resRows").innerHTML = Object.entries(snap.resources || {}).map(([r, v]) => `<tr><td>${r}${v.approx ? " <em>(replay, approx.)</em>" : ""}</td><td class="num">${fmt(v.available)}</td><td class="num">${fmt(v.cap)}</td><td class="num">${fmt(v.in_flight)}</td><td class="num">${fmt(v.taken)}</td><td class="num">${fmt(v.used)}</td><td class="num">${fmt(v.refused, 0)}</td></tr>`).join("");
  }

  // ---------------------------------------------------------------- timelines over a history of snapshots
  function setupCanvas(c, h) { const dpr = window.devicePixelRatio || 1, w = c.clientWidth || 800; c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); c.style.height = h + "px"; const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); return [g, w, h]; }
  function panel(c, title, ts, lines) {
    const [g, w, h] = setupCanvas(c, 150), pad = { l: 46, r: 12, t: 22, b: 22 };
    const tmin = ts[0] || 0, tmax = Math.max(ts[ts.length - 1] || 1, tmin + 1);
    let ymax = 1; for (const l of lines) for (const v of l.arr) if (v != null) ymax = Math.max(ymax, v);
    ymax = Math.ceil(ymax * 1.08);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = css("--line"); g.fillStyle = css("--fg2"); g.font = "11px " + css("--mono"); g.lineWidth = 1;
    const span = (tmax - tmin) / 60, stepMin = span > 240 ? 60 : span > 90 ? 20 : span > 30 ? 10 : span > 10 ? 2 : span > 3 ? 1 : 0.5;
    for (let t = Math.ceil(tmin / 60 / stepMin) * stepMin; t <= tmax / 60; t += stepMin) { const X = pad.l + (t * 60 - tmin) / (tmax - tmin) * (w - pad.l - pad.r); g.beginPath(); g.moveTo(X, pad.t); g.lineTo(X, h - pad.b); g.stroke(); g.fillText(t + " min", X + 3, h - pad.b + 14); }
    g.fillStyle = css("--fg"); g.font = "600 12px " + css("--sans"); g.fillText(title, pad.l, 14);
    g.fillStyle = css("--fg2"); g.font = "11px " + css("--mono"); g.textAlign = "right"; g.fillText(String(ymax), pad.l - 6, pad.t + 10); g.fillText("0", pad.l - 6, h - pad.b); g.textAlign = "left";
    const X = t => pad.l + (t - tmin) / (tmax - tmin) * (w - pad.l - pad.r), Y = v => h - pad.b - v / ymax * (h - pad.t - pad.b);
    for (const l of lines) { g.beginPath(); let started = false; for (let i = 0; i < ts.length; i++) { const v = l.arr[i]; if (v == null) continue; if (!started) { g.moveTo(X(ts[i]), Y(v)); started = true; } else g.lineTo(X(ts[i]), Y(v)); } g.strokeStyle = l.col; g.lineWidth = 1.5; g.stroke(); }
    let lx = pad.l + 110; for (const l of lines) { g.fillStyle = l.col; g.fillRect(lx, 6, 10, 10); g.fillStyle = css("--fg2"); g.fillText(l.label, lx + 14, 15); lx += 14 + g.measureText(l.label).width + 16; }
  }
  function renderTimelines(hist) {
    const host = $("timelines"); host.innerHTML = "";
    if (!hist.snaps.length) return;
    const topo = hist.snaps[hist.snaps.length - 1].topology, stages = topo.stages.map(s => s.name), color = Object.fromEntries(stages.map((s, i) => [s, PALETTE[i % PALETTE.length]]));
    const mk = () => { const c = document.createElement("canvas"); c.className = "panel"; host.appendChild(c); return c; };
    const ts = hist.t;
    panel(mk(), "Place occupancy (items waiting; a pool shows its content)", ts, stages.map(s => ({ arr: hist.snaps.map(x => (x.stages[s] || {}).waiting ?? null), col: color[s], label: s })));
    panel(mk(), "Bodies running per stage", ts, stages.map(s => ({ arr: hist.snaps.map(x => (x.stages[s] || {}).in_flight ?? null), col: color[s], label: s })));
    const edgeKeys = [...new Set(hist.snaps.flatMap(x => Object.keys(x.edges || {})))].slice(0, 12);
    panel(mk(), "Edge rate, items per minute (trailing minute)", ts, edgeKeys.map((k, i) => ({ arr: hist.snaps.map(x => (x.edges[k] || {}).per_min ?? 0), col: PALETTE[i % PALETTE.length], label: k })));
    for (const r of Object.keys(topo.resources)) {
      const lines = [{ arr: hist.snaps.map(x => (x.resources[r] || {}).in_flight ?? null), col: css("--accent"), label: "in flight" }];
      if (hist.snaps.some(x => (x.resources[r] || {}).available != null)) lines.push({ arr: hist.snaps.map(x => (x.resources[r] || {}).available ?? null), col: css("--good"), label: "left" });
      panel(mk(), "Resource " + r, ts, lines);
    }
  }

  // ---------------------------------------------------------------- sources
  function render(snap, hist) { renderSummary(snap); renderFigure(snap); renderTables(snap); renderTimelines(hist); }
  function setStatus(t) { $("status").textContent = t; }

  async function pollLive() {
    try {
      const r = await fetch("status.json", { cache: "no-store" }); if (!r.ok) throw new Error(r.status + " " + r.statusText);
      const snap = await r.json();
      if (!history.t.length || snap.t !== history.t[history.t.length - 1]) { history.t.push(snap.t); history.snaps.push(snap); if (history.t.length > 3000) { history.t.shift(); history.snaps.shift(); } }
      render(snap, history); setStatus("live · snapshot at " + fmtMin(snap.t) + (snap.stop_reason ? " · run ended: " + snap.stop_reason : ""));
    } catch (e) { setStatus("live: " + e.message + " (serve a run directory with python -m gepa.async_stages.dashboard, or switch to replay)"); }
  }
  function startLive() { stopTimers(); mode = "live"; $("scrubWrap").hidden = true; pollLive(); timer = setInterval(pollLive, Math.max(1, Number($("pollSec").value)) * 1000); }
  function stopTimers() { if (timer) clearInterval(timer); timer = null; if (playing) clearInterval(playing); playing = null; }

  let replayHist = null;
  function loadReplayText(text, name) {
    replay = buildReplay(parseEvents(text)); $("runName").textContent = name;
    const wall = replay.tEnd - replay.t0, N = 400;
    replayHist = { t: [], snaps: [] };
    for (let i = 0; i <= N; i++) { const t = wall * i / N; replayHist.t.push(t); replayHist.snaps.push(snapshotAt(replay, t)); }
    $("scrubWrap").hidden = false; $("scrub").value = 1000; showReplayAt(1);
    setStatus("replay · " + replay.fires.length.toLocaleString() + " fires · " + fmtMin(wall) + (replay.topology.inferred ? " · topology inferred from the item flow (older log)" : ""));
  }
  function showReplayAt(frac) {
    const wall = replay.tEnd - replay.t0, t = wall * frac, snap = snapshotAt(replay, t);
    const k = Math.round(frac * (replayHist.t.length - 1));
    render(snap, { t: replayHist.t.slice(0, k + 1), snaps: replayHist.snaps.slice(0, k + 1) });
    $("scrubLabel").textContent = fmtMin(t) + " of " + fmtMin(wall);
  }
  async function startReplay() {
    stopTimers(); mode = "replay";
    try { const r = await fetch("events.jsonl", { cache: "no-store" }); if (!r.ok) throw new Error(r.status); loadReplayText(await r.text(), "events.jsonl"); }
    catch (e) { setStatus("replay: no events.jsonl served (" + e.message + "); open a log file instead"); $("scrubWrap").hidden = true; }
  }

  $("source").addEventListener("change", e => (e.target.value === "live" ? startLive() : startReplay()));
  $("pollSec").addEventListener("change", () => { if (mode === "live") startLive(); });
  $("fileInput").addEventListener("change", e => { const f = e.target.files[0]; if (!f) return; stopTimers(); mode = "replay"; $("source").value = "replay"; const rd = new FileReader(); rd.onload = () => loadReplayText(rd.result, f.name); rd.readAsText(f); });
  $("scrub").addEventListener("input", e => { if (replay) showReplayAt(Number(e.target.value) / 1000); });
  $("play").addEventListener("click", () => {
    if (playing) { clearInterval(playing); playing = null; $("play").textContent = "play"; return; }
    if (!replay) return; $("play").textContent = "pause"; let v = Number($("scrub").value); if (v >= 1000) v = 0;
    playing = setInterval(() => { v = Math.min(1000, v + 4); $("scrub").value = v; showReplayAt(v / 1000); if (v >= 1000) { clearInterval(playing); playing = null; $("play").textContent = "play"; } }, 120);
  });
  window.addEventListener("resize", () => { if (mode === "live" && history.snaps.length) renderTimelines(history); else if (replay) showReplayAt(Number($("scrub").value) / 1000); });

  $("runName").textContent = location.host ? location.host : "";
  startLive();
})();
