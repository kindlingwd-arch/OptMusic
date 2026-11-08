/* OptMusic 前端：钢琴窗、播放、试听对比、校验、导出 */
"use strict";

// ---------------------------------------------------------------- 常量与状态
const LO = 21, HI = 108;                         // A0 ~ C8
const KB = 64, RULER = 26, CHORD = 22, WAVE = 60;
const TOP = RULER + CHORD + WAVE;
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const isBlack = p => [1, 3, 6, 8, 10].includes(p % 12);
const pname = p => NAMES[p % 12] + (Math.floor(p / 12) - 1);
const COLORS = { ok: "#3fb68b", offpitch: "#f0a030", suspect: "#e5484d", silent: "#6b7280", unchecked: "#5b8def" };

const S = {
  project: null, duration: 0,
  audioBuf: null, peaks: null,
  notes: [], nextId: 1,
  grid: { bpm: 120, beats_per_bar: 4, origin: 0, beat_times: [], confidence: 0 },
  key: null, chords: [],
  curve: null,
  engine: null, activation: null, heatCanvas: null,
  view: { pps: 120, scrollX: 0, rowH: 14, scrollY: 0 },
  sel: new Set(),
  checks: {},
  playing: false, playhead: 0,
  vstBuf: null, vstDirty: true,
  plugin: { name: "内置合成器", version: 0 },
  undo: [], redo: [],
};

const $ = id => document.getElementById(id);
const canvas = $("roll"), g = canvas.getContext("2d");
let W = 0, H = 0, dpr = 1;

// ---------------------------------------------------------------- 工具
function status(msg, err = false) { const el = $("status"); el.textContent = msg; el.classList.toggle("err", err); }
function busy(text) { $("busy").hidden = !text; if (text) $("busyText").textContent = text; }

async function api(path, body, raw = false) {
  const opt = body instanceof FormData ? { method: "POST", body }
    : body !== undefined ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch (_) { }
    throw new Error(msg);
  }
  return raw ? r : r.json();
}

const beatLen = () => 60 / S.grid.bpm;
const barLen = () => beatLen() * S.grid.beats_per_bar;
const snapBeats = () => parseFloat($("selSnap").value);
function snapT(t, force = false) {
  const sb = snapBeats();
  if (!sb && !force) return t;
  const gl = (sb || 0.25) * beatLen();
  return Math.round((t - S.grid.origin) / gl) * gl + S.grid.origin;
}
function fmtBar(t) {
  const bt = (t - S.grid.origin) / beatLen();
  const bpb = S.grid.beats_per_bar;
  const bar = Math.floor(bt / bpb) + 1;
  const beat = Math.floor(((bt % bpb) + bpb) % bpb) + 1;
  const tick = Math.floor((((bt % 1) + 1) % 1) * 480);
  return `${bar}.${beat}.${String(tick).padStart(3, "0")}`;
}
function fmtSec(t) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${String(m).padStart(2, "0")}:${s.toFixed(3).padStart(6, "0")}`;
}
const tx = t => KB + (t - S.view.scrollX) * S.view.pps;
const xt = x => (x - KB) / S.view.pps + S.view.scrollX;
const py = p => TOP + (HI - p) * S.view.rowH - S.view.scrollY;
const yp = y => HI - Math.floor((y - TOP + S.view.scrollY) / S.view.rowH);
const centsOf = n => n.measured == null ? null : Math.round((n.measured - n.pitch) * 100);

// ---------------------------------------------------------------- 撤销
function snapshot() { return JSON.stringify(S.notes); }
function pushUndo() { S.undo.push(snapshot()); if (S.undo.length > 200) S.undo.shift(); S.redo = []; }
function restore(json) { S.notes = JSON.parse(json); S.sel.clear(); notesChanged(); }
function undo() { if (!S.undo.length) return; S.redo.push(snapshot()); restore(S.undo.pop()); }
function redo() { if (!S.redo.length) return; S.undo.push(snapshot()); restore(S.redo.pop()); }

let harmonyTimer = null;
function notesChanged(ids) {
  S.vstDirty = true;
  if (ids) ids.forEach(id => delete S.checks[id]);
  updateNoteInfo(); updateCheckSummary(); draw();
  clearTimeout(harmonyTimer);
  harmonyTimer = setTimeout(refreshHarmony, 600);
}

// ---------------------------------------------------------------- 画布
function resize() {
  dpr = window.devicePixelRatio || 1;
  W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  clampScroll(); draw();
}
function clampScroll() {
  const maxY = Math.max(0, (HI - LO + 1) * S.view.rowH - (H - TOP));
  S.view.scrollY = Math.min(Math.max(0, S.view.scrollY), maxY);
  const maxX = Math.max(0, (S.duration || 10) - (W - KB) / S.view.pps * 0.5);
  S.view.scrollX = Math.min(Math.max(Math.min(0, S.grid.origin), S.view.scrollX), maxX);
}

let drawQueued = false;
function draw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; render(); });
}

function render() {
  const { rowH } = S.view;
  g.clearRect(0, 0, W, H);
  g.fillStyle = "#16181d"; g.fillRect(0, 0, W, H);
  const t0 = xt(KB), t1 = xt(W);

  // ---- 钢琴窗背景
  g.save();
  g.beginPath(); g.rect(KB, TOP, W - KB, H - TOP); g.clip();
  for (let p = HI; p >= LO; p--) {
    const y = py(p);
    if (y > H || y + rowH < TOP) continue;
    g.fillStyle = isBlack(p) ? "#1a1d23" : "#20242c";
    g.fillRect(KB, y, W - KB, rowH);
    if (p % 12 === 0) { g.fillStyle = "#3a404d"; g.fillRect(KB, y + rowH - 1, W - KB, 1); }
  }
  drawGridLines(TOP, H, t0, t1);
  drawHeat(t0, t1);

  // ---- 音符
  g.font = "11px Segoe UI, Microsoft YaHei";
  for (const n of S.notes) {
    if (n.end < t0 || n.start > t1) continue;
    const x0 = tx(n.start), x1 = tx(n.end), y = py(n.pitch);
    if (y > H || y + rowH < TOP) continue;
    const st = S.checks[n.id]?.status || "unchecked";
    const sel = S.sel.has(n.id);
    g.globalAlpha = 0.45 + 0.55 * (n.velocity / 127);
    g.fillStyle = COLORS[st];
    roundRect(x0, y + 1, Math.max(2, x1 - x0), rowH - 2, 3); g.fill();
    g.globalAlpha = 1;
    if (sel) { g.strokeStyle = "#fff"; g.lineWidth = 1.5; roundRect(x0, y + 1, Math.max(2, x1 - x0), rowH - 2, 3); g.stroke(); }
    // 实测音高偏移标记（Melodyne 风格的音准线）
    const c = centsOf(n);
    if (c != null && Math.abs(c) >= 5) {
      const yc = y + rowH / 2 - (c / 100) * rowH;
      g.strokeStyle = "rgba(255,255,255,0.8)"; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x0 + 2, yc); g.lineTo(x1 - 2, yc); g.stroke();
    }
    if (x1 - x0 > 30 && rowH >= 11) {
      g.fillStyle = "rgba(0,0,0,0.75)";
      g.fillText(pname(n.pitch) + (c != null && Math.abs(c) >= 5 ? ` ${c > 0 ? "+" : ""}${c}` : ""), x0 + 4, y + rowH - 3);
    }
  }
  // ---- 音高曲线（画在音符之上，便于对照音准）
  if (S.curve) {
    const hop = S.curve.hop_sec, m = S.curve.midi;
    const i0 = Math.max(0, Math.floor(t0 / hop)), i1 = Math.min(m.length - 1, Math.ceil(t1 / hop));
    g.strokeStyle = "rgba(255,255,255,0.85)"; g.lineWidth = 1.3;
    g.beginPath();
    let pen = false;
    for (let i = i0; i <= i1; i++) {
      const v = m[i];
      if (v == null) { pen = false; continue; }
      const x = tx(i * hop), y = TOP + (HI - v) * rowH - S.view.scrollY + rowH / 2;
      if (pen) g.lineTo(x, y); else g.moveTo(x, y);
      pen = true;
    }
    g.stroke();
  }

  if (drag?.mode === "marquee") {
    g.strokeStyle = "#9bb8ff"; g.fillStyle = "rgba(91,141,239,0.15)";
    const r = marqueeRect();
    g.fillRect(r.x, r.y, r.w, r.h); g.strokeRect(r.x, r.y, r.w, r.h);
  }
  g.restore();

  drawTopStrips(t0, t1);
  drawKeyboard();

  // ---- 播放头
  const xp = tx(S.playhead);
  if (xp >= KB && xp <= W) { g.fillStyle = "#ffdd57"; g.fillRect(xp, 0, 1.5, H); }
}

function drawGridLines(yTop, yBot, t0, t1) {
  const bl = beatLen(), bpb = S.grid.beats_per_bar, pps = S.view.pps;
  const sb = snapBeats() || 0.25;
  const sub = bl * sb * pps >= 7 ? sb : (bl * pps >= 7 ? 1 : bpb);
  const step = bl * sub;
  let k = Math.floor((t0 - S.grid.origin) / step);
  for (let t = S.grid.origin + k * step; t <= t1; t += step, k++) {
    const beats = Math.round((t - S.grid.origin) / bl * 1000) / 1000;
    const isBar = Math.abs(beats / bpb - Math.round(beats / bpb)) < 1e-6;
    const isBeat = Math.abs(beats - Math.round(beats)) < 1e-6;
    g.fillStyle = isBar ? "#4a5263" : isBeat ? "#30353f" : "#262a32";
    g.fillRect(Math.round(tx(t)), yTop, 1, yBot - yTop);
  }
}

function drawTopStrips(t0, t1) {
  const bl = beatLen(), bpb = S.grid.beats_per_bar;
  // 标尺
  g.fillStyle = "#1e2128"; g.fillRect(0, 0, W, RULER);
  g.save(); g.beginPath(); g.rect(KB, 0, W - KB, TOP); g.clip();
  const barPx = bl * bpb * S.view.pps;
  const every = barPx < 28 ? Math.ceil(28 / barPx) : 1;
  let b = Math.floor((t0 - S.grid.origin) / (bl * bpb));
  g.font = "11px Consolas, monospace";
  for (let t = S.grid.origin + b * bl * bpb; t <= t1; t += bl * bpb, b++) {
    const x = Math.round(tx(t));
    g.fillStyle = "#596173"; g.fillRect(x, 8, 1, RULER - 8);
    if (b % every === 0) { g.fillStyle = "#c9ced8"; g.fillText(String(b + 1), x + 3, 13); }
    for (let k = 1; k < bpb && bl * S.view.pps > 6; k++) {
      g.fillStyle = "#3d4350"; g.fillRect(Math.round(tx(t + k * bl)), RULER - 7, 1, 7);
    }
  }
  // 检测到的原始拍点（橙色小三角），用于核对网格是否对齐
  g.fillStyle = "rgba(240,160,48,0.8)";
  for (const bt of S.grid.beat_times || []) {
    if (bt < t0 || bt > t1) continue;
    const x = tx(bt);
    g.beginPath(); g.moveTo(x - 3, RULER - 6); g.lineTo(x + 3, RULER - 6); g.lineTo(x, RULER - 1); g.fill();
  }

  // 和弦轨
  g.fillStyle = "#1b1e24"; g.fillRect(KB, RULER, W - KB, CHORD);
  g.font = "12px Segoe UI, Microsoft YaHei";
  for (const c of S.chords) {
    if (c.end < t0 || c.start > t1) continue;
    const x0 = tx(c.start), x1 = tx(c.end);
    g.fillStyle = c.name === "N" ? "#22262e" : "#2c3550";
    g.fillRect(x0 + 1, RULER + 2, Math.max(1, x1 - x0 - 2), CHORD - 4);
    if (c.name !== "N" && x1 - x0 > 18) {
      g.fillStyle = "#cfd8ff"; g.fillText(c.name, x0 + 4, RULER + CHORD - 7);
    }
  }

  // 波形
  const wy = RULER + CHORD;
  g.fillStyle = "#14161a"; g.fillRect(KB, wy, W - KB, WAVE);
  drawGridLines(wy, wy + WAVE, t0, t1);
  if (S.peaks) {
    const { min, max, block, sr } = S.peaks;
    const mid = wy + WAVE / 2, amp = WAVE / 2 - 3;
    g.fillStyle = "#4f7bd0";
    for (let x = KB; x < W; x++) {
      const a = Math.floor(xt(x) * sr / block), z = Math.max(a + 1, Math.floor(xt(x + 1) * sr / block));
      if (z < 0 || a >= min.length) continue;
      let lo = 0, hi = 0;
      for (let i = Math.max(0, a); i < Math.min(z, min.length); i++) { if (min[i] < lo) lo = min[i]; if (max[i] > hi) hi = max[i]; }
      g.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
    }
  }
  g.restore();
  g.fillStyle = "#2f3440"; g.fillRect(0, TOP - 1, W, 1); g.fillRect(0, RULER, W, 1); g.fillRect(0, RULER + CHORD, W, 1);
  // 左上角信息
  g.fillStyle = "#1e2128"; g.fillRect(0, 0, KB, TOP);
  g.fillStyle = "#8a919e"; g.font = "11px Segoe UI, Microsoft YaHei";
  g.fillText("小节", 6, 16); g.fillText("和弦", 6, RULER + 15); g.fillText("波形", 6, RULER + CHORD + 34);
}

function drawKeyboard() {
  const { rowH } = S.view;
  g.save(); g.beginPath(); g.rect(0, TOP, KB, H - TOP); g.clip();
  g.font = "10px Segoe UI";
  for (let p = HI; p >= LO; p--) {
    const y = py(p);
    if (y > H || y + rowH < TOP) continue;
    g.fillStyle = isBlack(p) ? "#2b2f37" : "#e9ecf1";
    g.fillRect(0, y, KB - 1, rowH - (isBlack(p) ? 0 : 1));
    if (p % 12 === 0 || (rowH >= 13 && !isBlack(p))) {
      g.fillStyle = p % 12 === 0 ? "#222" : "#888";
      g.fillText(pname(p), 4, y + rowH - 3);
    }
  }
  g.restore();
  g.fillStyle = "#2f3440"; g.fillRect(KB - 1, TOP, 1, H - TOP);
}

function roundRect(x, y, w, h, r) {
  g.beginPath();
  g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

// ---------------------------------------------------------------- 鼠标交互
let drag = null;
function noteAt(x, y) {
  for (let i = S.notes.length - 1; i >= 0; i--) {
    const n = S.notes[i], x0 = tx(n.start), x1 = tx(n.end), y0 = py(n.pitch);
    if (x >= x0 && x <= x1 && y >= y0 && y <= y0 + S.view.rowH) return n;
  }
  return null;
}
function marqueeRect() {
  const x = Math.min(drag.x0, drag.x), y = Math.min(drag.y0, drag.y);
  return { x, y, w: Math.abs(drag.x - drag.x0), h: Math.abs(drag.y - drag.y0) };
}
function pos(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

canvas.addEventListener("mousedown", e => {
  const [x, y] = pos(e);
  if (y < TOP && x >= KB) { // 标尺 / 和弦 / 波形：定位播放头
    seek(Math.max(0, xt(x))); drag = { mode: "seek" }; return;
  }
  if (x < KB && y >= TOP) { auditionPitch(yp(y)); return; }
  if (y < TOP) return;
  const n = noteAt(x, y);
  if (n) {
    if (e.shiftKey || e.ctrlKey) { S.sel.has(n.id) ? S.sel.delete(n.id) : S.sel.add(n.id); }
    else if (!S.sel.has(n.id)) { S.sel.clear(); S.sel.add(n.id); }
    const edge = tx(n.end) - x < 6 && tx(n.end) - tx(n.start) > 10;
    drag = {
      mode: edge ? "resize" : "move", note: n, x0: x, y0: y, moved: false, before: snapshot(),
      orig: [...S.sel].map(id => { const m = S.notes.find(q => q.id === id); return { m, start: m.start, end: m.end, pitch: m.pitch }; }),
    };
    updateNoteInfo(); draw();
  } else {
    if (!e.shiftKey) S.sel.clear();
    drag = { mode: "marquee", x0: x, y0: y, x, y, base: new Set(S.sel) };
    updateNoteInfo(); draw();
  }
});

window.addEventListener("mousemove", e => {
  const [x, y] = pos(e);
  if (!drag) {
    const n = y >= TOP && x >= KB ? noteAt(x, y) : null;
    canvas.style.cursor = n && tx(n.end) - x < 6 ? "ew-resize" : n ? "grab" : y < TOP && x >= KB ? "text" : "default";
    return;
  }
  if (drag.mode === "seek") { seek(Math.max(0, xt(x))); return; }
  if (drag.mode === "marquee") {
    drag.x = x; drag.y = y;
    const r = marqueeRect();
    S.sel = new Set(drag.base);
    for (const n of S.notes) {
      const x0 = tx(n.start), x1 = tx(n.end), y0 = py(n.pitch);
      if (x1 >= r.x && x0 <= r.x + r.w && y0 + S.view.rowH >= r.y && y0 <= r.y + r.h) S.sel.add(n.id);
    }
    updateNoteInfo(); draw(); return;
  }
  const dx = x - drag.x0, dy = y - drag.y0;
  if (!drag.moved && Math.abs(dx) < 3 && Math.abs(dy) < 3) return;
  drag.moved = true;
  const dt = dx / S.view.pps;
  const prim = drag.orig.find(o => o.m === drag.note);
  if (drag.mode === "move") {
    const dp = -Math.round(dy / S.view.rowH);
    const ns = e.altKey ? prim.start + dt : snapT(prim.start + dt);
    const shift = ns - prim.start;
    for (const o of drag.orig) {
      o.m.start = Math.max(0, o.start + shift); o.m.end = o.m.start + (o.end - o.start);
      o.m.pitch = Math.min(HI, Math.max(LO, o.pitch + dp));
    }
  } else {
    const ne = e.altKey ? prim.end + dt : snapT(prim.end + dt);
    const delta = ne - prim.end;
    for (const o of drag.orig) o.m.end = Math.max(o.m.start + 0.03, o.end + delta);
  }
  updateNoteInfo(); draw();
});

window.addEventListener("mouseup", () => {
  if (!drag) return;
  const d = drag; drag = null;
  if ((d.mode === "move" || d.mode === "resize") && d.moved) {
    S.undo.push(d.before); S.redo = [];
    notesChanged(d.orig.map(o => o.m.id));
  } else if (d.mode === "move" && !d.moved) {
    const mode = $("selClickAud").value;
    if (mode !== "off") audition(d.note, mode);
  }
  draw();
});

canvas.addEventListener("dblclick", e => {
  const [x, y] = pos(e);
  if (y < TOP || x < KB || noteAt(x, y)) return;
  pushUndo();
  const len = (snapBeats() || 1) * beatLen();
  const start = snapT(xt(x), true);
  const n = { id: S.nextId++, start, end: start + len, pitch: yp(y), velocity: 100, measured: null };
  S.notes.push(n); S.notes.sort((a, b) => a.start - b.start);
  S.sel = new Set([n.id]);
  notesChanged([n.id]);
});

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const [x] = pos(e);
  if (e.ctrlKey) {
    const t = xt(x);
    S.view.pps = Math.min(2000, Math.max(8, S.view.pps * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
    S.view.scrollX = t - (x - KB) / S.view.pps;
  } else if (e.altKey) {
    S.view.rowH = Math.min(30, Math.max(6, S.view.rowH + (e.deltaY < 0 ? 1 : -1)));
  } else if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
    S.view.scrollX += (e.shiftKey ? e.deltaY : e.deltaX) / S.view.pps;
  } else {
    S.view.scrollY += e.deltaY;
  }
  clampScroll(); draw();
}, { passive: false });

window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const selNotes = S.notes.filter(n => S.sel.has(n.id));
  const k = e.key;
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "Home") { seek(0); }
  else if ((k === "Delete" || k === "Backspace") && selNotes.length) {
    pushUndo(); S.notes = S.notes.filter(n => !S.sel.has(n.id)); S.sel.clear(); notesChanged();
  } else if ((k === "ArrowUp" || k === "ArrowDown") && selNotes.length) {
    e.preventDefault(); pushUndo();
    const d = (k === "ArrowUp" ? 1 : -1) * (e.shiftKey ? 12 : 1);
    selNotes.forEach(n => n.pitch = Math.min(HI, Math.max(LO, n.pitch + d)));
    notesChanged(selNotes.map(n => n.id));
    if ($("selClickAud").value !== "off" && selNotes.length === 1) audition(selNotes[0], "vst");
  } else if ((k === "ArrowLeft" || k === "ArrowRight") && selNotes.length) {
    e.preventDefault(); pushUndo();
    const d = (k === "ArrowRight" ? 1 : -1) * (snapBeats() || 0.25) * beatLen();
    selNotes.forEach(n => { const l = n.end - n.start; n.start = Math.max(0, n.start + d); n.end = n.start + l; });
    notesChanged(selNotes.map(n => n.id));
  } else if (e.ctrlKey && k.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); }
  else if (e.ctrlKey && k.toLowerCase() === "y") { e.preventDefault(); redo(); }
  else if (e.ctrlKey && k.toLowerCase() === "a") { e.preventDefault(); S.sel = new Set(S.notes.map(n => n.id)); updateNoteInfo(); draw(); }
  else if (k.toLowerCase() === "q" && !e.ctrlKey) { quantize(); }
  else if (["1", "2", "3", "4"].includes(k) && selNotes.length === 1) {
    audition(selNotes[0], { 1: "orig", 2: "vst", 3: "ab", 4: "both" }[k]);
  }
});

// ---------------------------------------------------------------- 音频
let actx = null, gOrig = null, gVst = null;
function ensureCtx() {
  if (!actx) {
    actx = new AudioContext();
    gOrig = actx.createGain(); gOrig.connect(actx.destination);
    gVst = actx.createGain(); gVst.connect(actx.destination);
    applyVolumes();
  }
  if (actx.state === "suspended") actx.resume();
  return actx;
}
function applyVolumes() {
  if (!gOrig) return;
  gOrig.gain.value = parseFloat($("volOrig").value);
  gVst.gain.value = parseFloat($("volVst").value);
}
function normalize(buf, peak = 0.7) {
  let m = 0;
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > m) m = a; }
  }
  if (m < 1e-5) return buf;
  const k = peak / m;
  for (let c = 0; c < buf.numberOfChannels; c++) { const d = buf.getChannelData(c); for (let i = 0; i < d.length; i++) d[i] *= k; }
  return buf;
}

function computePeaks(buf) {
  const block = 256, n = Math.ceil(buf.length / block);
  const min = new Float32Array(n), max = new Float32Array(n);
  const chans = [...Array(buf.numberOfChannels).keys()].map(c => buf.getChannelData(c));
  for (let b = 0; b < n; b++) {
    let lo = 0, hi = 0;
    for (let i = b * block; i < Math.min(buf.length, (b + 1) * block); i++) {
      let v = 0; for (const d of chans) v += d[i]; v /= chans.length;
      if (v < lo) lo = v; if (v > hi) hi = v;
    }
    min[b] = lo; max[b] = hi;
  }
  let m = 0; for (let b = 0; b < n; b++) m = Math.max(m, max[b], -min[b]);
  if (m > 0) for (let b = 0; b < n; b++) { min[b] /= m; max[b] /= m; }
  return { min, max, block, sr: buf.sampleRate };
}

// 单音试听
const noteCache = new Map();
async function vstNoteBuffer(pitch, velocity, dur) {
  const key = `${S.plugin.version}|${pitch}|${velocity}|${dur.toFixed(2)}`;
  if (noteCache.has(key)) return noteCache.get(key);
  const r = await api("/api/render/note", { pitch, velocity, duration: dur }, true);
  const buf = normalize(await ensureCtx().decodeAudioData(await r.arrayBuffer()));
  noteCache.set(key, buf);
  return buf;
}
let auditionSrcs = [];
function stopAudition() { auditionSrcs.forEach(s => { try { s.stop(); } catch (_) { } }); auditionSrcs = []; }
function playBuf(buf, gain, when, offset = 0, dur) {
  const src = actx.createBufferSource(); src.buffer = buf; src.connect(gain);
  dur == null ? src.start(when, offset) : src.start(when, offset, dur);
  auditionSrcs.push(src);
}
async function audition(n, mode) {
  ensureCtx(); stopAudition();
  const dur = Math.min(8, n.end - n.start);
  try {
    const now = actx.currentTime + 0.03;
    const needVst = mode !== "orig";
    const vb = needVst ? await vstNoteBuffer(n.pitch, n.velocity, Math.max(0.15, dur)) : null;
    const t = actx.currentTime + 0.03;
    if (mode === "orig" && S.audioBuf) playBuf(S.audioBuf, gOrig, now, n.start, dur);
    if (mode === "vst") playBuf(vb, gVst, t);
    if (mode === "ab") { if (S.audioBuf) playBuf(S.audioBuf, gOrig, t, n.start, dur); playBuf(vb, gVst, t + dur + 0.3); }
    if (mode === "both") { if (S.audioBuf) playBuf(S.audioBuf, gOrig, t, n.start, dur); playBuf(vb, gVst, t); }
  } catch (err) { status("试听失败：" + err.message, true); }
}
async function auditionPitch(p) {
  ensureCtx(); stopAudition();
  try { playBuf(await vstNoteBuffer(p, 100, 0.6), gVst, actx.currentTime + 0.02); }
  catch (err) { status("试听失败：" + err.message, true); }
}

// 走带
let transport = [];
let playT0 = 0, playPos0 = 0;
async function renderVstTrack() {
  if (!S.notes.length) return null;
  busy("正在用 " + S.plugin.name + " 渲染 MIDI 轨…");
  try {
    const r = await api("/api/render/sequence", { notes: S.notes.map(n => ({ pitch: n.pitch, velocity: n.velocity, start: n.start, end: n.end })) }, true);
    S.vstBuf = normalize(await ensureCtx().decodeAudioData(await r.arrayBuffer()));
    S.vstDirty = false;
  } finally { busy(null); }
  return S.vstBuf;
}
async function play() {
  ensureCtx(); stopAudition();
  const mon = $("selMonitor").value;
  try {
    if (mon !== "orig" && (S.vstDirty || !S.vstBuf)) await renderVstTrack();
  } catch (err) { status("VST 渲染失败：" + err.message, true); return; }
  const start = actx.currentTime + 0.06, pos0 = S.playhead;
  const add = src => { transport.push(src); return src; };
  if (mon !== "vst" && S.audioBuf && pos0 < S.audioBuf.duration) {
    const s = add(actx.createBufferSource()); s.buffer = S.audioBuf; s.connect(gOrig); s.start(start, pos0);
  }
  if (mon !== "orig" && S.vstBuf && pos0 < S.vstBuf.duration) {
    const s = add(actx.createBufferSource()); s.buffer = S.vstBuf; s.connect(gVst); s.start(start, pos0);
  }
  if ($("chkClick").checked) {
    const bl = beatLen(), end = Math.max(S.duration, pos0) + 1;
    let k = Math.ceil((pos0 - S.grid.origin) / bl - 1e-6);
    for (let t = S.grid.origin + k * bl; t < end; t += bl, k++) {
      const o = add(actx.createOscillator()), gg = actx.createGain();
      const down = ((k % S.grid.beats_per_bar) + S.grid.beats_per_bar) % S.grid.beats_per_bar === 0;
      o.frequency.value = down ? 1760 : 1100;
      const at = start + (t - pos0);
      gg.gain.setValueAtTime(0.0001, at); gg.gain.exponentialRampToValueAtTime(0.4, at + 0.002);
      gg.gain.exponentialRampToValueAtTime(0.0001, at + 0.05);
      o.connect(gg); gg.connect(actx.destination); o.start(at); o.stop(at + 0.06);
    }
  }
  playT0 = start; playPos0 = pos0; S.playing = true;
  $("btnPlay").textContent = "■";
  requestAnimationFrame(tick);
}
function stop() {
  transport.forEach(s => { try { s.stop(); } catch (_) { } }); transport = [];
  S.playing = false; $("btnPlay").textContent = "▶";
}
function togglePlay() { S.playing ? stop() : play(); }
function tick() {
  if (!S.playing) return;
  S.playhead = Math.max(playPos0, playPos0 + actx.currentTime - playT0);
  const end = Math.max(S.duration, S.vstBuf?.duration || 0);
  if (S.playhead > end) { stop(); }
  const x = tx(S.playhead);
  if (x > W - 60 || x < KB) { S.view.scrollX = S.playhead - (W - KB) / S.view.pps * 0.1; clampScroll(); }
  updateTime(); draw();
  requestAnimationFrame(tick);
}
function seek(t) {
  const was = S.playing; if (was) stop();
  S.playhead = t; updateTime(); draw();
  if (was) play();
}
function updateTime() { $("timeBar").textContent = fmtBar(S.playhead); $("timeSec").textContent = fmtSec(S.playhead); }

// ---------------------------------------------------------------- 项目：打开 / 分析
$("btnOpen").onclick = () => $("fileInput").click();
$("fileInput").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  stop();
  busy("上传并解码音频…");
  try {
    const fd = new FormData(); fd.append("file", f);
    const info = await api("/api/upload", fd);
    S.project = info.id; S.duration = info.duration;
    S.audioBuf = await ensureCtx().decodeAudioData(await f.arrayBuffer());
    S.duration = S.audioBuf.duration;
    S.peaks = computePeaks(S.audioBuf);
    S.notes = []; S.checks = {}; S.sel.clear(); S.curve = null; S.chords = []; S.key = null;
    S.engine = null; setActivation(null); $("polySection").hidden = true;
    S.undo = []; S.redo = []; S.vstBuf = null; S.vstDirty = true; S.playhead = 0;
    S.view.scrollX = 0; S.view.pps = Math.max(20, (W - KB) / Math.min(S.duration, 20));
    $("fileName").textContent = f.name; $("btnAnalyze").disabled = false;
    updateTime(); updateGridUI(); clampScroll(); draw();
    status(`已加载 ${f.name}（${S.duration.toFixed(1)} 秒）。点击“分析”开始识别。`);
    await analyze();
  } catch (err) { status("打开失败：" + err.message, true); }
  finally { busy(null); e.target.value = ""; }
};

async function analyze() {
  if (!S.project) return;
  stop();
  busy("正在识别音高、切分音符、检测 BPM 与小节…（长音频可能需要几十秒）");
  try {
    const meter = $("selMeter").value;
    const res = await api("/api/analyze", { id: S.project, mode: $("selMode").value,
      beats_per_bar: meter ? parseInt(meter) : null, poly_params: polyParams() });
    pushUndo();
    S.notes = res.notes.map(toNote);
    S.nextId = S.notes.length + 1;
    S.engine = res.engine;
    setActivation(res.activation);
    $("polySection").hidden = res.engine !== "poly";
    S.grid = res.grid; S.curve = res.pitch_curve; S.key = res.key; S.chords = res.chords || [];
    S.checks = {}; S.sel.clear(); S.vstDirty = true;
    if (S.notes.length) {
      const ps = S.notes.map(n => n.pitch).sort((a, b) => a - b);
      const mid = ps[Math.floor(ps.length / 2)];
      S.view.scrollY = (HI - mid) * S.view.rowH - (H - TOP) / 2;
    }
    clampScroll(); updateGridUI(); updateNoteInfo(); updateCheckSummary(); draw();
    const eng = res.engine === "poly" ? "复音引擎" : "单音引擎";
    const why = res.polyphony != null ? `（自动：同时发声约 ${res.polyphony} 个音级）` : "";
    status(`${eng}${why}识别完成：${S.notes.length} 个音符，${S.grid.bpm.toFixed(2)} BPM，${S.grid.beats_per_bar} 拍/小节` +
      (S.key ? `，调性 ${S.key.name}` : "") + `，用时 ${res.elapsed}s`);
  } catch (err) { status("分析失败：" + err.message, true); }
  finally { busy(null); }
}
$("btnAnalyze").onclick = analyze;

function toNote(n) {
  return { id: n.id, start: n.start, end: n.end, pitch: n.pitch, velocity: n.velocity,
    measured: n.pitch + (n.cents || 0) / 100, confidence: n.confidence };
}

// ---- 复音参数面板
const POLY_UI = [ // [输入框 id, 参数名, 显示格式]
  ["ppOnset", "onset_thresh", v => v.toFixed(2)],
  ["ppFrame", "frame_thresh", v => v.toFixed(2)],
  ["ppMin", "min_note_ms", v => v.toFixed(0)],
  ["ppReattack", "reattack", v => v.toFixed(2)],
  ["ppGhost", "ghost_ratio", v => v.toFixed(2)],
];
let polyDefaults = {};
function polyParams() {
  const p = {};
  POLY_UI.forEach(([id, key]) => { if ($(id).value !== "") p[key] = parseFloat($(id).value); });
  return p;
}
function setPolyUI(params) {
  POLY_UI.forEach(([id, key, fmt]) => {
    if (params[key] == null) return;
    $(id).value = params[key];
    $(id).nextElementSibling.textContent = fmt(parseFloat(params[key]));
  });
}
let polyTimer = null;
POLY_UI.forEach(([id, , fmt]) => {
  $(id).addEventListener("input", () => {
    $(id).nextElementSibling.textContent = fmt(parseFloat($(id).value));
    clearTimeout(polyTimer); polyTimer = setTimeout(retranscribe, 250);
  });
});
$("btnPolyReset").onclick = () => { setPolyUI(polyDefaults); retranscribe(); };
$("chkHeat").onchange = draw;

async function retranscribe() {
  if (!S.project || S.engine !== "poly") return;
  try {
    const r = await api("/api/retranscribe", { id: S.project, poly_params: polyParams() });
    pushUndo();
    S.notes = r.notes.map(toNote); S.nextId = S.notes.length + 1;
    S.sel.clear(); S.checks = {};
    $("polyStats").textContent = `模型 ${r.stats.raw} → 合并 ${r.stats.after_merge} → 去泛音 ${r.stats.final}`;
    notesChanged();
    status(`已按新参数重新生成：${S.notes.length} 个音符（${r.elapsed}s）`);
  } catch (err) { status("重新生成失败：" + err.message, true); }
}

// ---- 模型热度图：每个音高“正在响”的概率（背景），帮助判断漏音 / 多音
function setActivation(act) {
  S.activation = null; S.heatCanvas = null;
  if (!act) return;
  const bin = atob(act.data), n = act.frames, b = act.bins;
  const cv = document.createElement("canvas"); cv.width = n; cv.height = b;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(n, b);
  for (let t = 0; t < n; t++) {
    for (let k = 0; k < b; k++) {
      const v = bin.charCodeAt(t * b + k);
      const o = ((b - 1 - k) * n + t) * 4;       // 高音在上
      img.data[o] = 255; img.data[o + 1] = 170; img.data[o + 2] = 60;
      img.data[o + 3] = v < 45 ? 0 : Math.min(255, (v - 45) * 1.6);   // 概率 < 0.18 不画
    }
  }
  ctx.putImageData(img, 0, 0);
  S.activation = act; S.heatCanvas = cv;
}
function drawHeat(t0, t1) {
  const a = S.activation;
  if (!a || !S.heatCanvas || !$("chkHeat").checked) return;
  const off = a.t0 || 0;
  const f0 = Math.max(0, Math.floor((t0 - off) * a.fps)), f1 = Math.min(a.frames, Math.ceil((t1 - off) * a.fps) + 1);
  if (f1 <= f0) return;
  const topPitch = a.lo + a.bins - 1;
  g.save();
  g.globalAlpha = 0.8;
  g.imageSmoothingEnabled = true;
  g.drawImage(S.heatCanvas, f0, 0, f1 - f0, a.bins,
    tx(off + f0 / a.fps), py(topPitch), (f1 - f0) / a.fps * S.view.pps, a.bins * S.view.rowH);
  g.restore();
}

async function refreshHarmony() {
  if (!S.notes.length) { S.chords = []; draw(); return; }
  try {
    const r = await api("/api/harmony", { id: S.project, notes: S.notes, bpm: S.grid.bpm, beats_per_bar: S.grid.beats_per_bar, origin: S.grid.origin });
    S.key = r.key; S.chords = r.chords; updateGridUI(); draw();
  } catch (err) { status("和弦识别失败：" + err.message, true); }
}

// ---------------------------------------------------------------- 节奏网格
function updateGridUI() {
  $("inBpm").value = S.grid.bpm.toFixed(2);
  $("inBpb").value = S.grid.beats_per_bar;
  $("gridConf").textContent = S.grid.confidence ? `稳定度 ${(S.grid.confidence * 100).toFixed(0)}%` : "";
  const off = (-S.grid.origin / beatLen());
  $("gridInfo").textContent =
    (S.key ? `调性：${S.key.name}（${S.key.signature_text}，r=${S.key.confidence}）\n` : "") +
    `音频起点位于第 1 小节第 ${(off + 1).toFixed(2)} 拍`;
  $("gridInfo").style.whiteSpace = "pre-line";
}
function setGrid(patch) {
  Object.assign(S.grid, patch);
  const bl = barLen();
  while (S.grid.origin > 1e-9) S.grid.origin -= bl;
  while (S.grid.origin <= -bl) S.grid.origin += bl;
  updateGridUI(); updateTime(); clampScroll(); draw();
  clearTimeout(harmonyTimer); harmonyTimer = setTimeout(refreshHarmony, 400);
}
$("inBpm").onchange = e => { const v = parseFloat(e.target.value); if (v >= 20 && v <= 400) setGrid({ bpm: v }); };
$("inBpb").onchange = e => { const v = parseInt(e.target.value); if (v >= 1 && v <= 16) setGrid({ beats_per_bar: v }); };
$("btnBpmHalf").onclick = () => setGrid({ bpm: S.grid.bpm / 2 });
$("btnBpmDouble").onclick = () => setGrid({ bpm: S.grid.bpm * 2 });
$("btnDownPrev").onclick = () => setGrid({ origin: S.grid.origin - beatLen() });
$("btnDownNext").onclick = () => setGrid({ origin: S.grid.origin + beatLen() });
$("btnSetDownbeat").onclick = () => {
  const n = S.notes.find(q => S.sel.has(q.id));
  if (!n) return status("请先选中一个音符", true);
  setGrid({ origin: n.start });
};

// ---------------------------------------------------------------- 音符信息 / 试听按钮
function updateNoteInfo() {
  const sel = S.notes.filter(n => S.sel.has(n.id));
  const el = $("noteInfo");
  if (!sel.length) { el.textContent = "点击钢琴窗中的音符"; return; }
  if (sel.length > 1) { el.textContent = `已选中 ${sel.length} 个音符`; return; }
  const n = sel[0], c = centsOf(n), ck = S.checks[n.id];
  const stText = { ok: "✔ 正确", offpitch: "⚠ 音准偏差", suspect: "✖ 疑似错音", silent: "— 无声" };
  el.textContent =
    `音高  ${pname(n.pitch)} (MIDI ${n.pitch})\n` +
    `位置  ${fmtBar(n.start)}   ${fmtSec(n.start)}\n` +
    `时长  ${((n.end - n.start) / beatLen()).toFixed(2)} 拍 / ${(n.end - n.start).toFixed(3)}s\n` +
    `力度  ${n.velocity}    实测偏差 ${c == null ? "—" : (c > 0 ? "+" : "") + c + " 音分"}\n` +
    `校验  ${ck ? stText[ck.status] + (ck.match != null ? `  相似度 ${ck.match}` : "") + (ck.suggest ? `\n建议  ${pname(ck.suggest)}` : "") : "未校验"}`;
}
function selOne() { return S.notes.find(n => S.sel.has(n.id)); }
$("btnAudOrig").onclick = () => selOne() && audition(selOne(), "orig");
$("btnAudVst").onclick = () => selOne() && audition(selOne(), "vst");
$("btnAudAB").onclick = () => selOne() && audition(selOne(), "ab");
$("btnAudBoth").onclick = () => selOne() && audition(selOne(), "both");

// ---------------------------------------------------------------- 校验
async function runCheck(notes) {
  if (!S.project || !notes.length) return;
  stop();
  const tol = parseFloat($("inTol").value) || 30;
  const chunk = 30;
  try {
    for (let i = 0; i < notes.length; i += chunk) {
      busy(`用 ${S.plugin.name} 逐音对比中… ${i}/${notes.length}`);
      const part = notes.slice(i, i + chunk).map(n => ({ id: n.id, pitch: n.pitch, start: n.start, end: n.end, cents: centsOf(n) }));
      const r = await api("/api/check", { id: S.project, notes: part, cents_tolerance: tol });
      r.results.forEach(x => S.checks[x.id] = x);
      updateCheckSummary(); draw();
    }
    const c = countChecks();
    status(`校验完成：正确 ${c.ok}，音准偏 ${c.offpitch}，疑似错音 ${c.suspect}`);
  } catch (err) { status("校验失败：" + err.message, true); }
  finally { busy(null); updateNoteInfo(); }
}
function countChecks() {
  const c = { ok: 0, offpitch: 0, suspect: 0, silent: 0, unchecked: 0 };
  S.notes.forEach(n => c[S.checks[n.id]?.status || "unchecked"]++);
  return c;
}
function updateCheckSummary() {
  const c = countChecks();
  $("checkSummary").innerHTML =
    `<span class="dot unchecked"></span>未校验 <b>${c.unchecked}</b>` +
    `<span class="dot ok"></span>正确 <b>${c.ok}</b>` +
    `<span class="dot offpitch"></span>音准偏 <b>${c.offpitch}</b>` +
    `<span class="dot suspect"></span>疑似错音 <b>${c.suspect}</b>`;
}
$("btnCheck").onclick = () => runCheck(S.notes);
$("btnCheckSel").onclick = () => runCheck(S.notes.filter(n => S.sel.has(n.id)));
$("btnNextIssue").onclick = () => {
  const cur = selOne();
  const issues = S.notes.filter(n => ["suspect", "offpitch"].includes(S.checks[n.id]?.status));
  if (!issues.length) return status("没有发现问题音符");
  const next = issues.find(n => !cur || n.start > cur.start + 1e-6) || issues[0];
  S.sel = new Set([next.id]);
  S.view.scrollX = next.start - (W - KB) / S.view.pps * 0.3;
  S.view.scrollY = (HI - next.pitch) * S.view.rowH - (H - TOP) / 2;
  clampScroll(); updateNoteInfo(); draw();
};
$("btnApplySuggest").onclick = () => {
  const ns = S.notes.filter(n => S.sel.has(n.id) && S.checks[n.id]?.suggest);
  if (!ns.length) return status("选中的音符没有建议音高", true);
  pushUndo(); ns.forEach(n => n.pitch = S.checks[n.id].suggest);
  notesChanged(ns.map(n => n.id));
};

// ---------------------------------------------------------------- 编辑 / 导出
function quantize() {
  const targets = S.sel.size ? S.notes.filter(n => S.sel.has(n.id)) : S.notes;
  if (!targets.length) return;
  pushUndo();
  const gl = (snapBeats() || 0.25) * beatLen();
  targets.forEach(n => {
    n.start = Math.max(0, snapT(n.start, true));
    n.end = Math.max(n.start + gl, snapT(n.end, true));
  });
  notesChanged(targets.map(n => n.id));
  status(`已量化 ${targets.length} 个音符`);
}
$("btnQuantize").onclick = quantize;
$("btnUndo").onclick = undo;
$("btnRedo").onclick = redo;
$("btnExport").onclick = async () => {
  if (!S.notes.length) return status("没有可导出的音符", true);
  try {
    const name = ($("fileName").textContent || "OptMusic").replace(/\.[^.]+$/, "");
    const r = await api("/api/export/midi", { notes: S.notes, bpm: S.grid.bpm, beats_per_bar: S.grid.beats_per_bar,
      origin: S.grid.origin, align: $("selAlign").value, name }, true);
    const blob = await r.blob();
    if (window.optmusic?.saveFile) {
      // 桌面版（Electron）：系统“另存为”对话框
      const saved = await window.optmusic.saveFile(name + ".mid", new Uint8Array(await blob.arrayBuffer()));
      status(saved ? "已导出 " + saved : "已取消导出");
    } else {
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = name + ".mid"; a.click();
      status("已导出 " + name + ".mid");
    }
  } catch (err) { status("导出失败：" + err.message, true); }
};

// ---------------------------------------------------------------- 插件
async function loadPluginList() {
  const r = await api("/api/plugins");
  const sel = $("selPlugin"); sel.innerHTML = "";
  r.plugins.forEach(p => { const o = document.createElement("option"); o.value = p.path; o.textContent = p.name; sel.appendChild(o); });
  sel.value = r.current.path;
  setPluginInfo(r.current);
}
function setPluginInfo(info) {
  if (info.version !== S.plugin.version) { S.vstDirty = true; noteCache.clear(); }
  S.plugin = info;
  $("pluginStatus").textContent = "当前：" + info.name + (info.editor_open ? "（插件界面打开中）" : "");
  $("btnEditor").disabled = info.is_builtin;
}
$("btnLoadPlugin").onclick = async () => {
  const path = $("selPlugin").value;
  busy("正在加载插件 " + $("selPlugin").selectedOptions[0].textContent + "…（采样音源可能需要较长时间）");
  try {
    const info = await api("/api/plugins/load", { path });
    setPluginInfo(info); S.checks = {}; updateCheckSummary(); draw();
    info.warning ? status("⚠ " + info.warning, true) : status("已加载 " + S.plugin.name);
  }
  catch (err) { status("加载失败：" + err.message, true); }
  finally { busy(null); }
};
$("btnEditor").onclick = async () => {
  try {
    setPluginInfo(await api("/api/plugins/editor", {}));
    status("插件界面已打开。调整音色后关闭插件窗口即可继续渲染。");
    const poll = setInterval(async () => {
      const info = await api("/api/plugins/current");
      setPluginInfo(info);
      if (!info.editor_open) { clearInterval(poll); status("插件界面已关闭，音色已更新"); }
    }, 1000);
  } catch (err) { status(err.message, true); }
};
$("btnRescan").onclick = () => loadPluginList().catch(err => status(err.message, true));

// ---------------------------------------------------------------- 其他 UI
$("btnPlay").onclick = togglePlay;
$("btnHome").onclick = () => seek(0);
$("volOrig").oninput = applyVolumes;
$("volVst").oninput = applyVolumes;
$("selSnap").onchange = draw;
$("selMonitor").onchange = () => { if (S.playing) { stop(); play(); } };

new ResizeObserver(resize).observe($("rollWrap"));
resize();
loadPluginList().catch(err => status("插件列表获取失败：" + err.message, true));
api("/api/capabilities").then(c => {
  if (!c.poly) { const o = $("selMode").querySelector('option[value="poly"]'); o.disabled = true; o.textContent += "（未安装）"; }
  if (c.poly_defaults) { polyDefaults = c.poly_defaults; setPolyUI(polyDefaults); }
}).catch(() => { });
S.view.scrollY = (HI - 72) * S.view.rowH;
updateGridUI(); updateCheckSummary();
