// app.js - the playable Ungroup web app: home/lobby, game screen (renderer + panel), end screen.
//
// Modes
//   host    created a room: runs the engine (session.Host) and serves clients over net.js
//   client  joined a room from a link (#r=CODE[&local]): renders session.Client.view()
//   watch   bots only, no network (session.Host with a null transport), spectator camera
//
// Hash parameters: r=CODE (join), local (BroadcastChannel transport instead of WebRTC), n=NAME,
// watch (start a bots-only game at once; optional bots=bail,loyal,... preset=life seed=N).
// Test hook: window.__app is the app state (mode, frame, frames, me, host, client, lobby, feed ...).

import { createRenderer, PALETTE_CSS, WORLD_PX } from './render.js';
import { PRESETS } from './engine.js';
import { createTransport } from './net.js';
import { Host, Client, RESTART_DELAY_MS } from './session.js';

const $ = (id) => document.getElementById(id);
const BOT_TYPES = ['solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge'];
const RES = ['A', 'B', 'C', 'D'];
const AGENT_MODEL = 'models/v10_100.onnx';
const FEED_MAX = 60;

const S = {
  screen: 'home', mode: null, kind: null, room: null, transport: null, host: null, client: null,
  renderer: null, meta: null, cfg: null, names: [], seats: [], me: -1, round: 0,
  frame: null, frames: 0, feed: [], feedDirty: false, groups: new Map(),
  endMsg: null, endAt: 0, lobby: null, arena: false, agentOk: false,
  keys: new Set(), pointer: null, joinable: false, leaveHeld: false, intentVal: 0, intentUntil: 0,
  lastSent: null, panelAt: 0, name: '', conn: '', toasts: [], lastCrown: new Map(), myStun: false, myGroup: '',
};
window.__app = S;

// ------------------------------------------------------------------------------------------- helpers
function hashParams() { return new URLSearchParams(location.hash.replace(/^#/, '')); }
function randomCode(n = 6) { const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; let s = ''; for (let i = 0; i < n; i++) s += A[Math.floor(Math.random() * A.length)]; return s; }
function show(name) {
  S.screen = name;
  for (const id of ['home', 'lobby', 'game']) $(id).classList.toggle('active', id === name);
  if (name === 'game') { ensureRenderer(); requestAnimationFrame(() => S.renderer && S.renderer.resize()); }
}
function setConn(text, cls = '') { S.conn = text; const el = $('conn'); el.textContent = text; el.className = cls; }
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
function myId() { return S.transport ? S.transport.id : 'host'; }
function pname(i) { return S.names[i] != null ? S.names[i] : 'player ' + i; }
function seatLabel(s) { return !s ? '' : s.type === 'bot' ? (s.bot || s.name) + ' bot' : s.type === 'agent' ? 'agent' : 'human'; }
function progressOf(frame, i) {
  const need = S.meta && S.meta.needs ? S.meta.needs[i] : null;
  const p = frame.players[i];
  if (!need || !p) return 0;
  let s = 0;
  for (let t = 0; t < 4; t++) s += Math.min(p.banked[t] / need[t], 1);
  return s / 4;
}
function bodyOf(frame, i) { for (const b of frame.bodies) if (b.m.indexOf(i) >= 0) return b; return null; }
function ordinal(k) { return k + (k === 1 ? 'st' : k === 2 ? 'nd' : k === 3 ? 'rd' : 'th'); }
// short message over the canvas for things that happen to the local player (banked, merged, leaving, hit ...)
function toast(text, cls = '', ms = 3000) {
  const box = $('toasts');
  const el = document.createElement('div');
  el.className = cls; el.textContent = text;
  box.appendChild(el);
  while (box.children.length > 4) box.removeChild(box.firstChild);
  setTimeout(() => el.classList.add('fade'), ms);
  setTimeout(() => { if (el.parentNode) el.parentNode.removeChild(el); }, ms + 450);
}
function resName(k) { return ['blue', 'yellow', 'orange', 'red'][k & 3]; }
function amountText(a) { const parts = []; a.forEach((v, k) => { if (v >= 0.5) parts.push(`${v.toFixed(0)} ${resName(k)}`); }); return parts.join(', ') || 'nothing'; }

// ------------------------------------------------------------------------------------------ renderer
function ensureRenderer() {
  if (S.renderer) return S.renderer;
  const r = createRenderer($('c'), {
    fontUrl: 'assets/monogram.ttf',
    dottedBackgroundUrl: 'assets/dotted_background.png',
    minePatternUrl: 'assets/mine_pattern.png',
    sparkUrl: 'assets/spark.png',
    letterUrls: ['assets/a_letter.png', 'assets/m_letter.png', 'assets/e_letter.png', 'assets/n_letter.png'],
  });
  r.ready.then(() => { S.rendererReady = true; }).catch((e) => console.warn('renderer assets', e));
  S.renderer = r;
  return r;
}

// -------------------------------------------------------------------------------------------- events
function describe(e) {
  const nm = (i) => pname(i);
  const sum = (a) => a.reduce((x, y) => x + y, 0).toFixed(1);
  switch (e.kind) {
    case 'merge': return { tag: 'merge', text: `${e.a.map(nm).join(' + ')} joined ${e.b.map(nm).join(' + ')} (group of ${e.size})` };
    case 'leave_start': return { tag: 'leave', text: `${nm(e.player)} starts leaving` };
    case 'leave_cancel': return { tag: 'leave', text: `${nm(e.player)} stays after all` };
    case 'leave': return { tag: 'leave', text: `${nm(e.player)} left with ${sum(e.share)} units` };
    case 'bank': return { tag: 'bank', text: `${nm(e.player)} banked ${sum(e.amount)}` + (e.group.length > 1 ? ` (group of ${e.group.length})` : '') };
    case 'spill': if (!e.units) return null; return { tag: 'spill', text: `${e.a.map(nm).join('+')} hit ${e.b.map(nm).join('+')}: ${e.units} units spilled` };
    case 'mine_dead': return { tag: 'mine', text: `mine ${e.mine} (${RES[S.meta && S.meta.mine_type ? S.meta.mine_type[e.mine] : 0]}) swallowed by the edge` };
    case 'crown': return e.group && e.group.length > 1 ? { tag: 'crown', text: `${nm(e.player)} holds the crown of ${e.group.map(nm).join(' + ')}` } : null;
    case 'win': return { tag: 'win', text: `${nm(e.player)} wins` };
    case 'timeout': return { tag: 'win', text: e.player !== undefined && e.player >= 0 ? `time limit: ${nm(e.player)} leads and wins` : 'time limit reached' };
    default: return null;
  }
}
function toastFor(e) {
  const me = S.me;
  if (me < 0) return;
  const nm = (i) => pname(i);
  switch (e.kind) {
    case 'merge': {
      const inA = e.a.indexOf(me) >= 0, inB = e.b.indexOf(me) >= 0;
      if (!inA && !inB) return;
      const others = (inA ? e.b : e.a).map(nm).join(' + ');
      toast(`you merged with ${others} - group of ${e.size}, shared pool`, 'merge', 3500); return;
    }
    case 'leave_start': if (e.player === me) toast('leaving the group... keep holding L', 'leave', 1500); return;
    case 'leave_cancel': if (e.player === me) toast('leave cancelled', 'leave', 1500); return;
    case 'leave': if (e.player === me) toast(`you left with ${amountText(e.share)}`, 'leave', 3500); return;
    case 'bank': {
      if (e.player === me) { toast(`banked ${amountText(e.amount)} at your pad`, 'bank', 3000); return; }
      if (e.group && e.group.indexOf(me) >= 0) toast(`${nm(e.player)} banked the group's pool at their pad`, 'bank', 3000);
      return;
    }
    case 'spill': {
      const hit = e.a.indexOf(me) >= 0 || e.b.indexOf(me) >= 0;
      if (hit && e.units > 0) toast(`collision! ${e.units} units spilled on the floor - pick them up`, 'spill', 3000);
      return;
    }
    case 'crown': {
      if (!e.group || e.group.length < 2 || e.group.indexOf(me) < 0) return;
      if (e.player === me) toast('you hold the crown: only your pad banks for the group', 'crown', 3500);
      else toast(`${nm(e.player)} holds the crown: only their pad banks`, 'crown', 3500);
      return;
    }
    case 'win': toast(e.player === me ? 'you win!' : `${nm(e.player)} completed all four needs`, 'win', 6000); return;
    case 'timeout': toast('time is up - highest progress wins', 'win', 6000); return;
    default: return;
  }
}
function ingestEvents(events) {
  if (!events || !events.length) return;
  for (const e of events) {
    // the engine re-announces the crown on every bank; only a change of head is news
    if (e.kind === 'crown' && e.group) {
      const key = e.group.slice().sort((x, y) => x - y).join(',');
      if (S.lastCrown.get(key) === e.player) continue;
      S.lastCrown.set(key, e.player);
    }
    toastFor(e);
    const d = describe(e);
    if (!d) continue;
    S.feed.push({ t: e.t, tag: d.tag, text: d.text });
  }
  if (S.feed.length > FEED_MAX) S.feed.splice(0, S.feed.length - FEED_MAX);
  S.feedDirty = true;
}

// --------------------------------------------------------------------------------------------- panel
function updatePanel(frame) {
  const cfg = S.cfg || {};
  const t = frame.t;
  $('ptitle').textContent = S.mode === 'watch' ? 'watching bots' : 'room ' + S.room;
  const who = S.me >= 0 ? `you: ${pname(S.me)} (seat ${S.me + 1})` : 'spectator';
  $('pinfo').textContent = `round ${S.round} · ${who} · ${S.conn || S.kind || 'no network'} · t ${t.toFixed(0)}s`;

  // players
  const rows = [];
  const n = frame.players.length;
  for (let i = 0; i < n; i++) {
    const p = frame.players[i];
    const b = bodyOf(frame, i);
    const prog = progressOf(frame, i);
    const tags = [];
    if (b && b.m.length > 1) {
      const others = b.m.filter((j) => j !== i).map(pname).join(', ');
      tags.push(`<span class="dim">with ${esc(others)}</span>`);
      if (cfg.crown && b.head === i) tags.push('<span class="tag crown">crown</span>');
    } else tags.push('<span class="dim">solo</span>');
    if (p.join) tags.push('<span class="tag join">joinable</span>');
    if (p.leaving >= 0) tags.push(`<span class="tag leave">leaving ${p.leaving.toFixed(1)}s</span>`);
    if (p.brand > 0) tags.push(`<span class="tag brand">branded ${Math.ceil(p.brand)}s</span>`);
    if (b && b.stun > 0) tags.push('<span class="tag stun">stunned</span>');
    if (p.cd > 0) tags.push(`<span class="tag">cooldown ${Math.ceil(p.cd)}s</span>`);
    const banked = p.banked.map((v, k) => `<span style="color:${PALETTE_CSS[k]}">${Math.floor(v)}</span>`).join('/');
    const pool = b ? b.pool.reduce((x, y) => x + y, 0) : 0;
    if (pool >= 0.5) tags.push(`<span class="dim">pool ${pool.toFixed(0)}</span>`);
    rows.push(`<li class="${i === S.me ? 'me' : ''}">
      <div class="top"><span class="sw" style="background:${PALETTE_CSS[p.intent & 3]}" title="intent ${RES[p.intent & 3]}"></span>
        <span class="nm">${esc(pname(i))}</span><span class="tag">${esc(seatLabel(S.seats[i]))}</span><span class="pct">${Math.round(prog * 100)}%</span></div>
      <div class="bar"><i class="${prog >= 0.999 ? 'done' : ''}" style="width:${(prog * 100).toFixed(1)}%"></i></div>
      <div class="sub"><span>${banked}</span>${tags.join(' ')}</div></li>`);
  }
  $('players').innerHTML = rows.join('');

  // status strip on the canvas: what state am I in right now
  const st = $('status');
  if (S.me >= 0 && frame.players[S.me]) {
    const p = frame.players[S.me], b = bodyOf(frame, S.me);
    const parts = [];
    if (b && b.m.length > 1) {
      const others = b.m.filter((j) => j !== S.me).map(pname).join(', ');
      parts.push(`<span class="group">in a group with ${esc(others)}</span>`);
      if (cfg.crown) parts.push(b.head === S.me ? '<span class="crown">you hold the crown - only your pad banks</span>' : `<span class="crown">crown: ${esc(pname(b.head))} - only their pad banks</span>`);
    }
    if (p.leaving >= 0) parts.push(`<span class="leave">leaving in ${p.leaving.toFixed(1)} s - keep holding L</span>`);
    else if (S.leaveHeld && (!b || b.m.length < 2)) parts.push('<span class="cd">not in a group - nothing to leave</span>');
    if (p.join) parts.push('<span class="join">joinable - touching bodies merge with you</span>');
    if (b && b.stun > 0) parts.push('<span class="stun">stunned</span>');
    if (p.brand > 0) parts.push(`<span class="brand">branded ${Math.ceil(p.brand)} s - others see you betrayed</span>`);
    if (p.cd > 0) parts.push(`<span class="cd">join cooldown ${Math.ceil(p.cd)} s</span>`);
    st.innerHTML = parts.join('');
    st.classList.toggle('hidden', !parts.length);
    const stunned = !!(b && b.stun > 0);
    if (stunned && !S.myStun) toast('stunned', 'spill', 1000);
    S.myStun = stunned;
  } else st.classList.add('hidden');

  // alliances: groups of 2+ with the time since they formed
  const seen = new Set();
  for (const b of frame.bodies) {
    if (b.m.length < 2) continue;
    const key = b.m.slice().sort((x, y) => x - y).join(',');
    seen.add(key);
    if (!S.groups.has(key)) S.groups.set(key, t);
  }
  for (const k of Array.from(S.groups.keys())) if (!seen.has(k)) S.groups.delete(k);
  const al = [];
  for (const [key, since] of S.groups) al.push(`<li>${key.split(',').map((i) => esc(pname(+i))).join(' + ')}<span class="dur">${Math.max(0, t - since).toFixed(0)}s</span></li>`);
  $('alliances').innerHTML = al.length ? al.join('') : '<li class="dim">none</li>';

  // feed
  if (S.feedDirty) {
    S.feedDirty = false;
    const feed = $('feed');
    feed.innerHTML = S.feed.slice().reverse().map((e) => `<li class="${e.tag}"><span class="t">${e.t.toFixed(0)}s</span>${esc(e.text)}</li>`).join('');
  }
  for (const el of document.querySelectorAll('#ctrl .play')) el.classList.toggle('hidden', S.me < 0);
  $('keys').classList.toggle('hidden', S.me < 0);
  $('bJoin').textContent = 'join: ' + (S.joinable ? 'on' : 'off');
  $('bJoin').classList.toggle('on', S.joinable);
  $('bLeave').classList.toggle('on', S.leaveHeld);
  $('bCam').classList.toggle('on', S.arena);
  const myIntent = S.me >= 0 && frame.players[S.me] ? (frame.players[S.me].intent & 3) : -1;
  for (const b of document.querySelectorAll('#ctrl .int')) b.classList.toggle('on', (+b.dataset.intent - 1) === myIntent);
}

// --------------------------------------------------------------------------------------------- input
function myScreenPos() {
  if (S.me < 0 || !S.frame || !S.renderer) return null;
  const b = bodyOf(S.frame, S.me);
  if (!b) return null;
  const c = $('c');
  const w = c.clientWidth, h = c.clientHeight;
  const cam = S.renderer.camera;
  const per = S.arena ? Math.min(w, h) / 2.12 : WORLD_PX * (cam.zoom || 1);
  return { x: w / 2 + (b.x - cam.x) * per, y: h / 2 + (b.y - cam.y) * per };
}
function computeDir() {
  let dx = 0, dy = 0;
  const k = S.keys;
  if (k.has('KeyW') || k.has('ArrowUp')) dy -= 1;
  if (k.has('KeyS') || k.has('ArrowDown')) dy += 1;
  if (k.has('KeyA') || k.has('ArrowLeft')) dx -= 1;
  if (k.has('KeyD') || k.has('ArrowRight')) dx += 1;
  if (dx === 0 && dy === 0 && S.pointer) {
    const p = myScreenPos();
    if (p) { dx = S.pointer.x - p.x; dy = S.pointer.y - p.y; if (Math.hypot(dx, dy) < 10) { dx = 0; dy = 0; } }
  }
  const n = Math.hypot(dx, dy);
  if (n > 0) { dx /= n; dy /= n; }
  return [Math.round(dx * 1000) / 1000, Math.round(dy * 1000) / 1000];
}
function pushInput(now, force = false) {
  if (S.me < 0 || S.screen !== 'game') return;
  const dir = computeDir();
  const intent = now < S.intentUntil ? S.intentVal : 0;
  const inp = { dir, join: S.joinable ? 1 : 0, leave: S.leaveHeld ? 1 : 0, intent };
  const l = S.lastSent;
  if (!force && l && l.dir[0] === dir[0] && l.dir[1] === dir[1] && l.join === inp.join && l.leave === inp.leave && l.intent === inp.intent) return;
  S.lastSent = inp;
  if (S.mode === 'client' && S.client) S.client.sendInput(inp);
  else if (S.host) S.host.setInput(inp);
}
function declareIntent(k) { S.intentVal = k; S.intentUntil = performance.now() + 300; pushInput(performance.now()); }
function toggleJoin() { S.joinable = !S.joinable; pushInput(performance.now()); if (S.me >= 0) toast(S.joinable ? 'joinable on: touch another joinable body to merge' : 'joinable off: nobody can merge with you', 'merge', 2000); }
function setLeave(v) {
  S.leaveHeld = !!v; pushInput(performance.now());
  if (v && S.me >= 0 && S.frame) { const b = bodyOf(S.frame, S.me); if (!b || b.m.length < 2) toast('you are not in a group - nothing to leave', 'hint', 1500); }
}

function bindInput() {
  const typing = (e) => e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA');
  addEventListener('keydown', (e) => {
    if (S.screen !== 'game' || typing(e)) return;
    if (e.repeat) { if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault(); return; }
    switch (e.code) {
      case 'KeyW': case 'KeyA': case 'KeyS': case 'KeyD': case 'ArrowUp': case 'ArrowDown': case 'ArrowLeft': case 'ArrowRight':
        S.keys.add(e.code); e.preventDefault(); break;
      case 'KeyJ': case 'Space': toggleJoin(); e.preventDefault(); break;
      case 'KeyL': setLeave(true); break;
      case 'Digit1': case 'Digit2': case 'Digit3': case 'Digit4': declareIntent(+e.code.slice(5)); break;
      case 'KeyC': S.arena = !S.arena; break;
      default: return;
    }
    pushInput(performance.now());
  });
  addEventListener('keyup', (e) => {
    if (e.code === 'KeyL') setLeave(false);
    if (S.keys.delete(e.code)) pushInput(performance.now());
  });
  addEventListener('blur', () => { S.keys.clear(); S.pointer = null; if (S.leaveHeld) setLeave(false); pushInput(performance.now()); });
  const c = $('c');
  const pos = (e) => { const r = c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  c.addEventListener('pointerdown', (e) => { if (S.me < 0) return; S.pointer = pos(e); try { c.setPointerCapture(e.pointerId); } catch (_) { /* ignore */ } pushInput(performance.now()); e.preventDefault(); });
  c.addEventListener('pointermove', (e) => { if (S.pointer) { S.pointer = pos(e); pushInput(performance.now()); } });
  const up = () => { if (S.pointer) { S.pointer = null; pushInput(performance.now()); } };
  c.addEventListener('pointerup', up);
  c.addEventListener('pointercancel', up);
  c.addEventListener('contextmenu', (e) => e.preventDefault());
  $('bJoin').addEventListener('click', toggleJoin);
  $('bLeave').addEventListener('pointerdown', (e) => { setLeave(true); e.preventDefault(); });
  for (const ev of ['pointerup', 'pointercancel', 'pointerleave']) $('bLeave').addEventListener(ev, () => setLeave(false));
  for (const b of document.querySelectorAll('#ctrl .int')) b.addEventListener('click', () => declareIntent(+b.dataset.intent));
  $('bCam').addEventListener('click', () => { S.arena = !S.arena; });
  $('bQuit').addEventListener('click', quit);
  $('endLeave').addEventListener('click', quit);
  $('hostLeftHome').addEventListener('click', quit);
}

// ---------------------------------------------------------------------------------------------- loop
function loop(now) {
  requestAnimationFrame(loop);
  if (S.screen !== 'game') return;
  if (S.endMsg) {
    const cd = $('countdown');
    if (S.lobby && S.lobby.finished) cd.textContent = 'series over';
    else cd.textContent = 'next round in ' + Math.max(0, Math.ceil((S.endAt + RESTART_DELAY_MS - now) / 1000)) + ' s';
  }
  let frame;
  if (S.mode === 'client') {
    frame = S.client ? S.client.view(now) : null;
    if (frame) { S.frame = frame; S.frames++; ingestEvents(frame.events); }
  } else frame = S.frame;
  if (!frame || !S.meta) return;
  if (S.pointer) pushInput(now);   // the body moves under a held pointer, so the direction changes
  S.renderer.draw({ frame, meta: S.meta, cfg: S.cfg, me: S.me, names: S.names, time: now / 1000, camera: (S.arena || S.me < 0) ? { mode: 'arena' } : undefined });
  if (now - S.panelAt > 150) { S.panelAt = now; updatePanel(frame); }
}

// ----------------------------------------------------------------------------------------- transport
async function makeTransport(kind, room) {
  let t = null;
  if (kind === 'rtc') {
    setConn('connecting (rtc)...', 'warn');
    try {
      t = await createTransport({ kind: 'rtc', room });
      S.kind = 'rtc';
    } catch (e) {
      console.warn('rtc transport failed', e);
      S.rtcFallback = 'rtc unavailable, ';
    }
  }
  if (!t) { t = await createTransport({ kind: 'local', room }); S.kind = 'local'; }
  const status = () => {
    const n = t.peers().length;
    const via = S.kind === 'rtc' ? 'rtc via ' + (t.strategy || t.strategyName || 'p2p') : (S.rtcFallback || '') + 'local (this browser)';
    setConn(`${via} · ${n} peer${n === 1 ? '' : 's'}`, S.kind === 'rtc' && n === 0 ? 'warn' : 'ok');
  };
  t.onPeer(status); t.onLeave(status); status();
  return t;
}

function linkFor(room) {
  const base = location.href.split('#')[0];
  return base + '#r=' + room + (S.kind === 'local' ? '&local' : '');
}

// ---------------------------------------------------------------------------------------------- lobby
function renderLobby(l) {
  $('roomCode').textContent = S.room || '-';
  $('lobbyKind').textContent = S.mode === 'host' ? '(you are the host)' : S.mode === 'client' ? '(joined)' : '';
  $('link').value = S.room ? linkFor(S.room) : '';
  $('linkHelp').textContent = S.mode === 'host'
    ? (S.kind === 'local' ? 'This room is local to this browser: open the link in another tab of this browser to add a player. Peer-to-peer rooms (network: rtc) work across machines.'
      : 'Send this link to the people you want to play with. Anyone who opens it takes a free human seat; the rest spectate.')
    : 'Wait for the host to start. Take a free human seat by being here first.';
  const seats = $('seats');
  if (!l) {
    $('lobbySettings').textContent = '';
    seats.innerHTML = '<li class="empty"><span class="name">waiting for the host...</span></li>';
    $('notice').textContent = '';
    $('spectators').textContent = '';
    $('start').classList.add('hidden');
    return;
  }
  const st = l.settings;
  $('lobbySettings').textContent = `preset ${st.preset} · seed ${st.seed} · round ${st.overrides && st.overrides.time_limit ? st.overrides.time_limit : 240} s · rounds ${st.rounds || 'endless'}`;
  const me = myId();
  seats.innerHTML = l.seats.map((s, i) => {
    const mine = s.peer === me;
    const empty = s.type === 'human' && !s.peer;
    const name = empty ? 'empty - waiting for a player' : s.name || seatLabel(s);
    return `<li class="${mine ? 'me' : ''} ${empty ? 'empty' : ''}"><span class="idx">${i + 1}</span><span class="name">${esc(name)}${mine ? ' (you)' : ''}</span><span class="type">${esc(seatLabel(s))}${s.ready ? ' ✓' : ''}</span></li>`;
  }).join('');
  const sp = l.spectators || [];
  $('spectators').textContent = sp.length ? `spectators: ${sp.length}` : '';
  $('notice').textContent = l.notice || '';
  const startBtn = $('start');
  const empty = l.seats.filter((s) => s.type === 'human' && !s.peer).length;
  $('startHelp').textContent = S.mode !== 'host' ? '' : l.running ? 'a round is running' : empty > 0
    ? `${empty} human seat${empty === 1 ? '' : 's'} still empty - you can start anyway; empty seats sit idle until someone opens the link.`
    : 'Everyone is seated - start when ready.';
  startBtn.classList.toggle('hidden', S.mode !== 'host');
  startBtn.disabled = !!l.running;
  startBtn.textContent = l.running ? 'running...' : l.round > 0 ? 'Start next round' : 'Start';
}

// ------------------------------------------------------------------------------------ host / client
function wireHost(host) {
  host.onLobby((l) => { S.lobby = l; renderLobby(l); });
  host.onStart((msg) => beginRound(msg, host.seatOf(host.id)));
  host.onFrame((f) => { S.frame = f; S.frames++; ingestEvents(f.events); });
  host.onEnd((msg) => showEnd(msg));
}
function agentLoader(host) {
  return async (url) => {
    const m = await import('./agent.js');
    const a = await m.loadAgent(url);
    if (host.game) host.game.decideEvery = a.decideEvery;
    // session.Host calls act() for every agent seat in one tick; an onnxruntime session runs one inference
    // at a time, so the calls are queued (and the observation copied, the engine reuses its buffer).
    const act = a.act.bind(a);
    let queue = Promise.resolve();
    a.act = (obs, maskInfo) => { const o = obs.slice(); const r = queue.then(() => act(o, maskInfo)); queue = r.catch(() => {}); return r; };
    setConn(S.conn + ' · agent loaded', 'ok');
    return a;
  };
}

async function createRoom(settings, kind) {
  const room = randomCode();
  const t = await makeTransport(kind, room);
  S.mode = 'host'; S.room = room; S.transport = t;
  const host = new Host(t, settings);
  host._loadAgent = agentLoader(host);   // sets game.decideEvery to the model's cadence once loaded
  S.host = host;
  wireHost(host);
  S.lobby = host.lobby();
  history.replaceState(null, '', linkFor(room));
  renderLobby(S.lobby);
  show('lobby');
}

async function joinRoom(room, kind) {
  const t = await makeTransport(kind, room);
  S.mode = 'client'; S.room = room; S.transport = t;
  // a per-tab token (kept across a refresh) lets the session reclaim this seat instead of spectating
  let token = null;
  try { token = sessionStorage.getItem('ungroup-token-' + room); } catch (_) { /* ignore */ }
  const c = new Client(t, { name: S.name, token });
  try { sessionStorage.setItem('ungroup-token-' + room, c.token); } catch (_) { /* ignore */ }
  S.client = c;
  c.onLobby((l) => { S.lobby = l; renderLobby(l); if (S.screen === 'home') show('lobby'); });
  c.onStart((msg) => beginRound(msg, c.seat));
  c.onEnd((msg) => showEnd(msg));
  c.onHostLeft(() => { $('hostLeft').classList.remove('hidden'); setConn('host left', 'err'); });
  c.onHostBack(() => { $('hostLeft').classList.add('hidden'); setConn(`${S.kind} · ${t.peers().length} peers`, 'ok'); });
  renderLobby(null);
  show('lobby');
  setTimeout(() => { if (S.client === c && !c.lobby) $('notice').textContent = `no host found in room ${room} yet - is the host online, and did you use the host's link?`; }, 8000);
}

function watchBots(settings) {
  const host = new Host(null, Object.assign({ humans: 0 }, settings));
  S.mode = 'watch'; S.kind = null; S.room = null; S.host = host;
  setConn('bots only, no network', 'ok');
  wireHost(host);
  host.start();
}

function beginRound(msg, me) {
  S.meta = msg.meta; S.cfg = msg.cfg; S.names = msg.names; S.seats = msg.seats; S.round = msg.round;
  S.me = me == null ? -1 : me;
  S.frame = null; S.feed = []; S.feedDirty = true; S.groups = new Map(); S.endMsg = null;
  S.joinable = false; S.leaveHeld = false; S.keys.clear(); S.pointer = null; S.lastSent = null;
  S.lastCrown = new Map(); S.myStun = false;
  $('toasts').innerHTML = ''; $('status').classList.add('hidden');
  $('end').classList.add('hidden');
  ensureRenderer().camera.init = false;
  show('game');
  pushInput(performance.now(), true);
  if (S.me >= 0) {
    const need = S.meta && S.meta.needs ? S.meta.needs[S.me] : null;
    const prim = need ? need.indexOf(Math.max(...need)) : -1;
    toast(`round ${S.round}: sit on a mine to fill your pool, then bring it home to your pad (the ring labelled home)`, 'hint', 7000);
    if (prim >= 0) setTimeout(() => { if (S.frame && !S.endMsg) toast(`you mostly need ${resName(prim)} (${need[prim]}) - press ${prim + 1} to declare it`, 'hint', 6000); }, 4000);
  } else toast('spectating - C toggles the camera', 'hint', 4000);
}

function showEnd(msg) {
  S.endMsg = msg; S.endAt = performance.now();
  const names = msg.names || S.names;
  const w = msg.winner;
  $('endTitle').textContent = w >= 0 ? `${names[w] || 'player ' + w} wins` : 'time limit reached';
  $('endSub').textContent = msg.timeoutWin ? 'time limit: highest progress wins' : w >= 0 ? 'completed all four needs' : '';
  const order = msg.progress.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  $('endtab').innerHTML = '<tr><th>#</th><th>player</th><th></th><th>progress</th></tr>' + order.map(([p, i], k) =>
    `<tr class="${i === w ? 'win' : ''} ${i === S.me ? 'me' : ''}"><td>${k + 1}</td><td>${esc(names[i] || 'player ' + i)} <span class="tag">${esc(seatLabel(S.seats[i]))}</span></td><td><span class="bar"><i style="width:${(p * 100).toFixed(0)}%"></i></span></td><td>${(p * 100).toFixed(0)}%</td></tr>`).join('');
  const myRank = order.findIndex(([, i]) => i === S.me);
  $('endMe').textContent = S.me >= 0 && myRank >= 0 ? (w === S.me ? 'you won this round' : `you finished ${ordinal(myRank + 1)} of ${order.length} at ${(msg.progress[S.me] * 100).toFixed(0)}% of your needs`) : '';
  $('countdown').textContent = '';
  $('toasts').innerHTML = ''; $('status').classList.add('hidden');
  $('end').classList.remove('hidden');
  S.feedDirty = true;
}

function quit() {
  try { if (S.host) S.host.close(); } catch (e) { console.warn(e); }
  try { if (S.client) S.client.close(); } catch (e) { console.warn(e); }
  try { if (S.transport && !S.transport.closed) S.transport.close(); } catch (e) { console.warn(e); }
  Object.assign(S, { mode: null, kind: null, room: null, transport: null, host: null, client: null, meta: null, cfg: null, names: [], seats: [], me: -1, frame: null, feed: [], groups: new Map(), endMsg: null, lobby: null, lastSent: null });
  $('end').classList.add('hidden'); $('hostLeft').classList.add('hidden');
  history.replaceState(null, '', location.href.split('#')[0]);
  setConn('');
  show('home');
}

// ----------------------------------------------------------------------------------------------- home
function readSettings() {
  const bots = [];
  for (const el of document.querySelectorAll('.bot')) { const k = Math.max(0, Math.min(8, el.value | 0)); for (let i = 0; i < k; i++) bots.push(el.dataset.bot); }
  const humans = Math.max(1, Math.min(8, $('humans').value | 0));
  const agents = S.agentOk ? Math.max(0, Math.min(8, $('agents').value | 0)) : 0;
  const seed = Math.max(0, $('seed').value | 0);
  const timeLimit = Math.max(30, Math.min(900, $('timeLimit').value | 0));
  const rounds = Math.max(0, $('rounds').value | 0);
  return { humans, bots, agents, preset: $('preset').value, overrides: { time_limit: timeLimit }, seed, rounds, name: S.name, agentUrl: AGENT_MODEL };
}
function readName() {
  const v = $('name').value.trim().slice(0, 24);
  if (v) { S.name = v; try { localStorage.setItem('ungroup-name', v); } catch (_) { /* ignore */ } }
  return S.name;
}
// A host page can pin the transport with <meta name="ungroup-transport" content="local"> (for sandboxes that
// block WebSocket signalling); otherwise peer-to-peer over http(s), same-browser channels over file:.
function pinnedKind() { const m = document.querySelector('meta[name="ungroup-transport"]'); return m && m.content === 'local' ? 'local' : null; }
function rtcAllowed() { return /^https?:$/.test(location.protocol) && !pinnedKind(); }
function defaultKind() { return rtcAllowed() ? 'rtc' : 'local'; }

async function main() {
  const h = hashParams();
  for (const k of Object.keys(PRESETS)) { const o = document.createElement('option'); o.value = k; o.textContent = k; $('preset').appendChild(o); }
  $('preset').value = h.get('preset') && PRESETS[h.get('preset')] ? h.get('preset') : 'life';
  let stored = '';
  try { stored = localStorage.getItem('ungroup-name') || ''; } catch (_) { /* ignore */ }
  S.name = h.get('n') || stored || ('player-' + randomCode(4).toLowerCase());
  $('name').value = S.name;
  const kindNote = () => { $('kindNote').textContent = $('kind').value === 'rtc' ? 'share the link with anyone; peers connect directly (no server)' : 'the link only works in other tabs of this browser'; };
  $('kind').addEventListener('change', kindNote);
  $('kind').value = h.has('local') ? 'local' : defaultKind();
  if (!rtcAllowed()) { $('kind').value = 'local'; $('kind').querySelector('[value=rtc]').disabled = true; }
  $('joinLocal').checked = h.has('local') || !rtcAllowed();
  kindNote();
  if (h.get('seed')) $('seed').value = h.get('seed');
  bindInput();
  addEventListener('resize', () => S.renderer && S.renderer.resize());
  addEventListener('hashchange', () => location.reload());   // a new #r= / #watch link in the same tab starts over
  addEventListener('beforeunload', () => { try { if (S.host) S.host.close(); else if (S.client) S.client.close(); } catch (_) { /* ignore */ } });
  requestAnimationFrame(loop);

  // agent seats need a model next to the page (fetch fails on file://)
  fetch(AGENT_MODEL.replace(/\.onnx$/, '.json'), { method: 'HEAD' }).then((r) => {
    S.agentOk = r.ok;
    $('agents').disabled = !r.ok;
    $('agentNote').textContent = r.ok ? 'model: ' + AGENT_MODEL + ' (onnxruntime from the CDN)' : 'no model found';
  }).catch(() => { $('agentNote').textContent = 'no model (agents need an http(s) host)'; });

  $('create').addEventListener('click', async () => {
    $('createErr').textContent = '';
    readName();
    const st = readSettings();
    try { await createRoom(st, $('kind').value); } catch (e) { console.error(e); $('createErr').textContent = 'could not create the room: ' + e.message; }
  });
  $('watch').addEventListener('click', () => {
    const st = readSettings();
    if (!st.bots.length) st.bots = ['bail', 'bail', 'loyal', 'loyal', 'solo', 'rammer'];
    watchBots({ bots: st.bots, agents: 0, preset: st.preset, overrides: st.overrides, seed: st.seed, rounds: st.rounds });
  });
  $('join').addEventListener('click', async () => {
    readName();
    const code = $('joinCode').value.trim().toUpperCase();
    if (!code) return;
    try { await joinRoom(code, $('joinLocal').checked ? 'local' : defaultKind()); } catch (e) { console.error(e); alert('could not join: ' + e.message); }
  });
  $('copy').addEventListener('click', async () => {
    const link = $('link').value;
    try { await navigator.clipboard.writeText(link); $('copied').textContent = 'copied'; }
    catch (_) { $('link').select(); try { document.execCommand('copy'); $('copied').textContent = 'copied'; } catch (e) { $('copied').textContent = 'select and copy the link'; } }
    setTimeout(() => { $('copied').textContent = ''; }, 2000);
  });
  $('start').addEventListener('click', () => { if (S.host) S.host.start(); });
  $('leaveLobby').addEventListener('click', quit);

  if (h.has('watch')) {
    const bots = (h.get('bots') || 'bail,bail,loyal,loyal,solo,rammer').split(',').filter((b) => BOT_TYPES.includes(b));
    watchBots({ bots: bots.length ? bots : ['solo'], agents: 0, preset: $('preset').value, overrides: { time_limit: +(h.get('time') || 240) }, seed: +(h.get('seed') || 0), rounds: 0 });
  } else if (h.get('r')) {
    $('joinCode').value = h.get('r').toUpperCase();
    try { await joinRoom(h.get('r').toUpperCase(), h.has('local') ? 'local' : defaultKind()); }
    catch (e) { console.error(e); setConn('join failed: ' + e.message, 'err'); }
  }
}

main().catch((e) => { console.error(e); setConn('app error: ' + e.message, 'err'); });
