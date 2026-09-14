// session.js - host-authoritative multiplayer on top of net.js.
//
// Protocol (transport channels):
//   'hello' {name, token?, ready?}         joining peer -> host (sent to every peer; only the host answers). token is a
//                                          per-browser-tab secret the client keeps across a page refresh: a hello carrying
//                                          the token of an occupied seat takes that seat over (the old peer id is dropped).
//   'lobby' {lobby}                        host -> all, on every lobby change and at least every LOBBY_BEAT_MS while no
//                                          round runs (the client's liveness signal between rounds);
//                                          lobby = {settings, seats, host, running, round, notice, spectators, ...}
//   'start' {meta, names, seats, settings, cfg, tick0, round}
//   'input' {tick, dir:[dx,dy], join, leave, intent, move?}   client -> host, 20 Hz when changed, keepalive every 500 ms
//   'snap'  {tick, round, frame, ev0, prev}  host -> all every 2 ticks (15 Hz); frame.events = events since the previous
//                                          snapshot, ev0 = sequence number of the first of them, prev = the events of the
//                                          previous EVENT_REPLAY snapshots (so a lost snapshot loses no event; the client
//                                          de-duplicates by sequence number)
//   'end'   {winner, timeoutWin, progress:[..], round, names}
// The host runs the engine at 30 Hz with a fixed-step accumulator (catch-up capped at MAX_CATCHUP ticks per
// timer fire, so a throttled background tab slows the game instead of freezing it), applies each client's
// latest input to that seat every tick, runs bots and agents itself, restarts a round 8 s after it ends
// keeping the lobby, and turns the seat of a peer that leaves into a bot of the first bot type (the seat
// remembers the leaver's token, so a refreshed tab gets the same seat back). A seat whose client falls silent
// for INPUT_TIMEOUT_MS (no input, no keepalive) stops moving until the client speaks again. A peer that
// arrives after start becomes a spectator unless a human seat is free (including a seat handed to a bot
// when its player left); a spectator's next hello (setReady/setName) takes a seat that has freed up since.
// When the host closes its transport every client sees onHostLeft; a client
// that hears nothing from the host for HOST_TIMEOUT_MS reports the same (a crashed tab on a transport whose
// own leave detection takes longer), and onHostBack fires if the host turns out to be alive after all.
//
// Inputs from the network are sanitised: directions are finite and at most unit length, intent is 0..4,
// move is 0..MOVE_CLASSES-1 (anything else means "steer by dir").
//
// The engine import below is the single swap point: anything exporting { Game, preset, MOVE_CLASSES } with the
// engine.js interface (setSeats, setInput, tick, frame, meta, takeEvents, observe, progress) works here.
import { Game, preset, MOVE_CLASSES } from './engine.js';   // ENGINE IMPORT (swap for a stub if needed)
import { randomId } from './net.js';

export const TICK_HZ = 30;
export const SNAP_EVERY = 2;              // ticks per snapshot -> 15 Hz
export const MAX_CATCHUP = 10;            // ticks per timer fire
export const RESTART_DELAY_MS = 8000;
export const INPUT_PERIOD_MS = 50;        // 20 Hz
export const INPUT_KEEPALIVE_MS = 500;
export const INPUT_TIMEOUT_MS = 1000;     // a silent client's seat stops after this
export const LOBBY_BEAT_MS = 1000;        // host liveness between rounds
export const HOST_TIMEOUT_MS = 1500;      // client: nothing from the host for this long -> host left
export const EVENT_REPLAY = 3;            // snapshots whose events are repeated in the next one
export const SNAP_MAX_BYTES = 16384;      // wire budget for one snapshot (Trystero chunks above 16 KB)
export const VIEW_LAG_MS = 100;           // nominal distance behind the newest snapshot
export const VIEW_MAX_EXTRAP_MS = 150;    // dead-reckoning bound when the buffer runs dry
export const VIEW_JITTER_K = 3;           // target lag = lag + K * mean |arrival jitter|
export const VIEW_MAX_ADAPT_MS = 400;     // cap on the jitter allowance
export const VIEW_RATE_SLEW = 0.3;        // playback speed range 0.7x .. 1.3x while re-centring
export const VIEW_RATE_GAIN_S = 1.0;      // seconds of lag error for full slew
export const VIEW_RESYNC_S = 1.0;         // further behind than this: jump (once) instead of speeding up
export const VIEW_MAX_BUF = 120;          // snapshots kept (8 s)

const HOST_ID_LOCAL = 'host';

function defaultTimers() {
  return {
    now: () => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now()),
    setInterval: (fn, ms) => setInterval(fn, ms),
    clearInterval: (h) => clearInterval(h),
  };
}

function bodyKey(b) { return b.m.join(','); }
function lerp(a, b, k) { return a + (b - a) * k; }

// Linear interpolation of two snapshot frames (a older, b newer) at alpha in [0, 1]. Bodies are matched
// by member set; a body that merged or split between the two frames takes the newer frame's values.
export function interpolateFrame(a, b, alpha) {
  const k = Math.max(0, Math.min(1, alpha));
  const byKey = new Map();
  for (const ba of a.bodies) byKey.set(bodyKey(ba), ba);
  const bodies = b.bodies.map((bb) => {
    const ba = byKey.get(bodyKey(bb));
    if (!ba) return { ...bb, m: bb.m.slice(), pool: bb.pool.slice() };
    return {
      m: bb.m.slice(), x: lerp(ba.x, bb.x, k), y: lerp(ba.y, bb.y, k), vx: lerp(ba.vx, bb.vx, k), vy: lerp(ba.vy, bb.vy, k),
      stun: lerp(ba.stun, bb.stun, k), pool: (k < 0.5 ? ba.pool : bb.pool).slice(), head: bb.head,
    };
  });
  const src = k < 0.5 ? a : b;
  const mines = b.mines.map((v, i) => lerp(a.mines[i] !== undefined ? a.mines[i] : v, v, k));
  const players = src.players.map((p, i) => {
    const pa = a.players[i], pb = b.players[i];
    return { ...p, banked: p.banked.slice(), dir: pa && pb ? [lerp(pa.dir[0], pb.dir[0], k), lerp(pa.dir[1], pb.dir[1], k)] : p.dir.slice() };
  });
  return { t: lerp(a.t, b.t, k), R: lerp(a.R, b.R, k), bodies, mines, alive: b.alive.slice(), picks: src.picks.map((q) => q.slice()), players, events: [] };
}

// Dead-reckoning: advance body positions by their velocity for dt seconds (bounded by the caller).
export function extrapolateFrame(f, dt) {
  return {
    t: f.t + dt, R: f.R,
    bodies: f.bodies.map((b) => ({ ...b, m: b.m.slice(), pool: b.pool.slice(), x: b.x + b.vx * dt, y: b.y + b.vy * dt, stun: Math.max(0, b.stun - dt) })),
    mines: f.mines.slice(), alive: f.alive.slice(), picks: f.picks.map((q) => q.slice()),
    players: f.players.map((p) => ({ ...p, banked: p.banked.slice(), dir: p.dir.slice() })), events: [],
  };
}

function cloneFrame(f) { return { ...f, bodies: f.bodies.map((b) => ({ ...b, m: b.m.slice(), pool: b.pool.slice() })), mines: f.mines.slice(), alive: f.alive.slice(), picks: f.picks.map((q) => q.slice()), players: f.players.map((p) => ({ ...p, banked: p.banked.slice(), dir: p.dir.slice() })), events: [] }; }

// A frame is trusted only if it has the schema's arrays (a garbled snapshot is dropped, not rendered).
function validFrame(f) {
  return !!f && typeof f === 'object' && Number.isFinite(f.t) && Number.isFinite(f.R) && Array.isArray(f.bodies) && Array.isArray(f.mines)
    && Array.isArray(f.alive) && Array.isArray(f.picks) && Array.isArray(f.players)
    && f.bodies.every((b) => b && Array.isArray(b.m) && Array.isArray(b.pool) && Number.isFinite(b.x) && Number.isFinite(b.y) && Number.isFinite(b.vx) && Number.isFinite(b.vy))
    && f.players.every((p) => p && Array.isArray(p.banked) && Array.isArray(p.dir));
}

function finite(v) { v = Number(v); return Number.isFinite(v) ? v : 0; }
// Sanitise an input from the network (or the local UI). dir: finite, at most unit length, 3 decimals;
// intent 0..4; move only if it is a legal class (0 stop, 1..8 compass, 9 dir, 10.. macro targets).
export function normInput(input) {
  const src = input && typeof input === 'object' ? input : {};
  let dx = finite(Array.isArray(src.dir) ? src.dir[0] : 0), dy = finite(Array.isArray(src.dir) ? src.dir[1] : 0);
  const n = Math.hypot(dx, dy);
  if (n > 1) { dx /= n; dy /= n; }
  dx = Math.round(dx * 1000) / 1000; dy = Math.round(dy * 1000) / 1000;
  const intent = Math.max(0, Math.min(4, finite(src.intent) | 0));
  const out = { dir: [dx, dy], join: src.join ? 1 : 0, leave: src.leave ? 1 : 0, intent };
  if (src.move !== undefined && src.move !== null) { const mv = finite(src.move) | 0; if (mv >= 0 && mv < MOVE_CLASSES) out.move = mv; }
  return out;
}
function sameInput(a, b) {
  return !!a && !!b && a.dir[0] === b.dir[0] && a.dir[1] === b.dir[1] && a.join === b.join && a.leave === b.leave && a.intent === b.intent && a.move === b.move;
}
function cleanName(v, fallback) { return typeof v === 'string' && v.trim() ? v.trim().slice(0, 24) : fallback; }
function cleanToken(v) { return typeof v === 'string' && v.length >= 4 && v.length <= 64 ? v : null; }

// ================================================================================================ Host
// settings = { humans, bots: [seat names], agents, preset, overrides, seed, rounds, name?, agentUrl?, hostPlays? }
// opts     = { now, setInterval, clearInterval, manual (no timers; call pump(nowMs) yourself), agent ({act}), loadAgent(url) }
export class Host {
  constructor(transport, settings = {}, opts = {}) {
    this.transport = transport || null;
    this.id = transport ? transport.id : HOST_ID_LOCAL;
    const timers = defaultTimers();
    this.now = opts.now || timers.now;
    this._setInterval = opts.setInterval || timers.setInterval;
    this._clearInterval = opts.clearInterval || timers.clearInterval;
    this.manual = !!opts.manual;
    this.settings = {
      humans: settings.humans | 0, bots: (settings.bots || []).slice(), agents: settings.agents | 0,
      preset: settings.preset || 'legacy', overrides: Object.assign({}, settings.overrides || {}),
      seed: settings.seed || 0, rounds: settings.rounds | 0, name: settings.name || 'Host', agentUrl: settings.agentUrl || null,
      hostPlays: settings.hostPlays !== undefined ? !!settings.hostPlays : true,
    };
    if (!this.settings.seed) this.settings.seed = Math.floor(Math.random() * 0x7fffffff) + 1;
    this.agent = opts.agent || null;
    this._loadAgent = opts.loadAgent || null;
    this.peerNames = new Map();
    this.peerTokens = new Map();
    this.spectators = new Set();     // insertion order = arrival order = promotion order
    this.seats = [];
    this.game = null;
    this.cfg = null;
    this.running = false;
    this.finished = false;
    this.round = 0;
    this.restartAt = null;
    this.tick = 0;
    this.tick0 = 0;
    this.acc = 0;
    this.lastPump = 0;
    this.dtMs = 1000 / TICK_HZ;
    this.pending = [];
    this.inputAt = [];
    this.inputStale = [];
    this.seatMacro = [];
    this.notice = '';
    this._frameCbs = []; this._endCbs = []; this._startCbs = []; this._lobbyCbs = [];
    this._timer = null;
    this.snapCount = 0;
    this.evSeq = 0;
    this._evRing = [];
    this._lastLobbySent = -Infinity;
    this._buildSeats();
    if (this.transport) {
      this.transport.on('hello', (obj, from) => this._hello(obj, from));
      this.transport.on('input', (obj, from) => this._input(obj, from));
      this.transport.onPeer((id) => { this._sendLobby(id); });
      this.transport.onLeave((id) => this._peerLeft(id));
    }
    if (!this.manual) this._timer = this._setInterval(() => this.pump(this.now()), Math.floor(this.dtMs));
  }

  // ----------------------------------------------------------------- lobby
  _buildSeats() {
    const s = this.settings;
    const seats = [];
    for (let i = 0; i < s.humans; i++) seats.push({ type: 'human', name: '', peer: null, ready: false, token: null });
    for (const b of s.bots) seats.push({ type: 'bot', bot: b, name: b, peer: null, ready: true, token: null });
    for (let i = 0; i < s.agents; i++) seats.push({ type: 'agent', name: 'agent', peer: null, ready: true, token: null });
    if (seats.length === 0) seats.push({ type: 'bot', bot: 'solo', name: 'solo', peer: null, ready: true, token: null });
    this.seats = seats;
    if (s.humans > 0 && s.hostPlays) { seats[0].peer = this.id; seats[0].name = s.name; seats[0].ready = true; }
    this.pending = new Array(seats.length).fill(null);
    this.inputAt = new Array(seats.length).fill(0);
    this.inputStale = new Array(seats.length).fill(0);
    this.seatMacro = new Array(seats.length).fill(-1);
  }
  get n() { return this.seats.length; }
  get botFallback() { return this.settings.bots[0] || 'solo'; }
  seatOf(peer) { return this.seats.findIndex((s) => s.peer === peer); }
  freeHumanSeat() { return this.seats.findIndex((s) => (s.type === 'human' && s.peer === null) || (s.type === 'bot' && s.handover)); }
  // the seat a returning tab (same token) should get: its own seat, even if a stale peer id still sits there
  _seatOfToken(token) { return token ? this.seats.findIndex((s) => s.token === token && ((s.type === 'human' && s.peer !== null) || (s.type === 'bot' && s.handover))) : -1; }

  lobby() {
    return {
      settings: JSON.parse(JSON.stringify(this.settings)),
      seats: this.seats.map((s) => ({ type: s.type, name: s.name, peer: s.peer, ready: s.ready, bot: s.bot || null })),
      host: this.id, running: this.running, round: this.round, notice: this.notice, finished: this.finished,
      restartAt: this.restartAt, spectators: Array.from(this.spectators),
    };
  }
  onLobby(cb) { this._lobbyCbs.push(cb); }
  onStart(cb) { this._startCbs.push(cb); }
  onFrame(cb) { this._frameCbs.push(cb); }
  onEnd(cb) { this._endCbs.push(cb); }

  _sendLobby(toId = null) {
    const msg = { lobby: this.lobby() };
    if (this.transport) this.transport.send('lobby', msg, toId);
    if (toId === null) { this._lastLobbySent = this.now(); for (const cb of this._lobbyCbs) cb(msg.lobby); }
  }
  // Change settings from the lobby (only between rounds); rebuilds the seat list keeping joined peers.
  updateSettings(patch) {
    if (this.running) throw new Error('cannot change settings while a round runs');
    const joined = this.seats.filter((s) => s.type === 'human' && s.peer && s.peer !== this.id).map((s) => ({ peer: s.peer, name: s.name, token: s.token }));
    Object.assign(this.settings, patch);
    if (patch.bots) this.settings.bots = patch.bots.slice();
    if (patch.overrides) this.settings.overrides = Object.assign({}, patch.overrides);
    this._buildSeats();
    for (const j of joined) { const k = this.freeHumanSeat(); if (k >= 0) this._assign(k, j.peer, j.name, j.token); else this.spectators.add(j.peer); }
    this.finished = false;
    this._sendLobby();
  }

  _assign(k, peer, name, token = null) {
    const s = this.seats[k];
    const old = s.peer;
    s.type = 'human'; s.peer = peer; s.name = name; s.ready = true; s.token = token || null; delete s.handover; delete s.bot;
    this.spectators.delete(peer);
    if (old !== null && old !== peer) { this.peerNames.delete(old); this.peerTokens.delete(old); }
    this.pending[k] = null;
    this.inputAt[k] = this.now();
    this.inputStale[k] = 0;
    this.seatMacro[k] = -1;
    if (this.running && this.game) {
      this.game.setSeats(this._seatTypes());
      this.game.setInput(k, { move: 0, dir: [0, 0], join: 0, leave: 0, intent: 0 });
    }
  }
  _hello(obj, from) {
    const name = cleanName(obj && obj.name, from.slice(0, 6));
    const token = cleanToken(obj && obj.token);
    this.peerNames.set(from, name);
    if (token) this.peerTokens.set(from, token);
    let k = this.seatOf(from);
    if (k >= 0) { this.seats[k].name = name; if (token) this.seats[k].token = token; if (obj && obj.ready !== undefined) this.seats[k].ready = !!obj.ready; }
    else {
      const mine = this._seatOfToken(token);
      if (mine >= 0) { this._assign(mine, from, name, token); this.notice = name + ' is back in seat ' + (mine + 1); }
      else {
        k = this.freeHumanSeat();
        if (k >= 0) { this._assign(k, from, name, token); this.notice = name + ' joined seat ' + (k + 1); }
        else { this.spectators.add(from); this.notice = name + ' is spectating'; }
      }
    }
    this._sendLobby();
    if (this.running) this.transport.send('start', this._startMsg(), from);
  }

  _peerLeft(id) {
    const name = this.peerNames.get(id) || id.slice(0, 6);
    this.peerNames.delete(id);
    this.peerTokens.delete(id);
    this.spectators.delete(id);
    const k = this.seatOf(id);
    if (k >= 0) {
      const s = this.seats[k];
      s.type = 'bot'; s.bot = this.botFallback; s.name = s.bot; s.peer = null; s.ready = true; s.handover = true;   // s.token is kept for a refresh
      this.pending[k] = null;
      if (this.running && this.game) { this.game.setSeats(this._seatTypes()); this.game.setInput(k, { move: 0, dir: [0, 0], join: 0, leave: 0, intent: 0 }); }
      this.notice = name + ' left; seat ' + (k + 1) + ' is now a ' + s.bot + ' bot';
    } else this.notice = name + ' left';
    this._sendLobby();
  }

  _seatTypes() {
    return this.seats.map((s) => {
      if (s.type === 'bot') return s.bot;
      if (s.type === 'agent') return this.agent ? 'agent' : 'solo';   // an agent seat without a loaded agent plays as a solo bot
      return 'human';
    });
  }
  names() { return this.seats.map((s, i) => s.name || (s.type === 'human' ? 'seat ' + (i + 1) : s.type)); }

  // ----------------------------------------------------------------- rounds
  _startMsg() {
    return { meta: this.game.meta(), names: this.names(), seats: this.lobby().seats, settings: JSON.parse(JSON.stringify(this.settings)), cfg: Object.assign({}, this.cfg), tick0: this.tick0, round: this.round };
  }

  start() {
    if (this.running) return;
    this.finished = false;
    if (this.settings.agents > 0 && !this.agent && !this._agentTried) {
      this._agentTried = true;
      const url = this.settings.agentUrl;
      const loader = this._loadAgent || ((u) => import('./agent.js').then((m) => m.loadAgent(u)));
      Promise.resolve().then(() => loader(url)).then((a) => { this.agent = a; if (this.game) this.game.setSeats(this._seatTypes()); })
        .catch((e) => { console.warn('agent unavailable, agent seats play as solo bots:', e && e.message); });
    }
    this._startRound(this.now());
  }

  _startRound(now) {
    const s = this.settings;
    const seed = s.seed + this.round;
    this.cfg = preset(s.preset, Object.assign({}, s.overrides, { n_players: this.n }));
    if (!this.game) this.game = new Game(this.cfg, seed); else this.game.reset(seed, false);
    this.game.setSeats(this._seatTypes());
    this.round++;
    this.running = true;
    this.restartAt = null;
    this.tick = 0;
    this.acc = 0;
    this.lastPump = now;
    this.tick0 = now;
    this.evSeq = 0;
    this._evRing = [];
    this.pending.fill(null);
    this.inputStale.fill(0);
    this.seatMacro.fill(-1);
    for (let k = 0; k < this.n; k++) this.inputAt[k] = now;
    this.notice = 'round ' + this.round + ' started';
    const msg = this._startMsg();
    if (this.transport) this.transport.send('start', msg);
    for (const cb of this._startCbs) cb(msg);
    this._sendLobby();
  }

  // Fixed-step loop body. Call with the current time in ms (the timer does this; tests drive it by hand).
  pump(now) {
    if (!this.running) {
      if (this.restartAt !== null && now >= this.restartAt) { this._startRound(now); return; }
      // liveness for the clients while no snapshots flow
      if (this.transport && now - this._lastLobbySent >= LOBBY_BEAT_MS) { this._lastLobbySent = now; this.transport.send('lobby', { lobby: this.lobby() }); }
      return;
    }
    this.acc += now - this.lastPump;
    this.lastPump = now;
    let n = Math.floor(this.acc / this.dtMs);
    if (n > MAX_CATCHUP) { n = MAX_CATCHUP; this.acc = 0; } else this.acc -= n * this.dtMs;
    for (let k = 0; k < n && this.running; k++) this._tick(now);
  }

  _applyPending(now) {
    const g = this.game;
    for (let i = 0; i < this.n; i++) {
      const s = this.seats[i];
      // a remote seat whose client fell silent (no input, no keepalive) stops; join/intent are kept
      if (s.type === 'human' && s.peer !== null && s.peer !== this.id && !this.inputStale[i] && now - this.inputAt[i] > INPUT_TIMEOUT_MS) {
        this.inputStale[i] = 1;
        this.pending[i] = null;
        this.seatMacro[i] = -1;
        g.setInput(i, { move: 0, dir: [0, 0], leave: 0 });
        continue;
      }
      const p = this.pending[i];
      if (!p) continue;
      let move;
      if (p.move !== undefined) move = p.move;
      else if (this.seatMacro[i] >= 0 || (p.dir[0] === 0 && p.dir[1] === 0)) {
        // steering by hand after a macro target: the macro must be cancelled first (move 0), the direction
        // itself is applied on the next tick
        move = 0;
        if (p.dir[0] !== 0 || p.dir[1] !== 0) { this.seatMacro[i] = -1; g.setInput(i, { move: 0, dir: [0, 0], join: p.join, leave: p.leave, intent: p.intent }); continue; }
      } else move = 9;
      this.pending[i] = null;
      this.seatMacro[i] = move >= 10 ? move : -1;
      g.setInput(i, { move, dir: p.dir, join: p.join, leave: p.leave, intent: p.intent });
    }
  }

  _agentsDecide() {
    if (!this.agent) return;
    const g = this.game;
    if (g.tickCount % g.decideEvery !== 0) return;
    for (let i = 0; i < this.n; i++) {
      if (this.seats[i].type !== 'agent') continue;
      let r;
      try { r = this.agent.act(g.observe(i), { seat: i, nPlayers: this.n, nMines: this.cfg.n_mines }); } catch (e) { console.warn('agent act failed', e); continue; }
      const apply = (a) => { if (a && this.running && g === this.game) g.setInput(i, { move: a[0] | 0, join: a[1] | 0, leave: a[2] | 0, intent: a[3] | 0 }); };
      if (r && typeof r.then === 'function') r.then(apply).catch((e) => console.warn('agent act failed', e)); else apply(r);
    }
  }

  _tick(now) {
    const g = this.game;
    this._applyPending(now);
    this._agentsDecide();
    const evStart = g.events.length;
    g.tick();
    this.tick = g.tickCount;
    const done = g.done;
    if (this._frameCbs.length) {
      const f = g.frame();
      f.events = g.events.slice(evStart);
      for (const cb of this._frameCbs) cb(f, this.tick);
    }
    if (this.tick % SNAP_EVERY === 0 || done) {
      const f = g.frame();
      const evs = g.takeEvents();
      f.events = evs;
      this.snapCount++;
      if (this.transport) {
        const prev = this._evRing.length ? [].concat(...this._evRing) : [];
        this.transport.send('snap', { tick: this.tick, round: this.round, frame: f, ev0: this.evSeq, prev });
      }
      this.evSeq += evs.length;
      this._evRing.push(evs);
      if (this._evRing.length > EVENT_REPLAY) this._evRing.shift();
    }
    if (done) this._end(now);
  }

  _end(now) {
    const g = this.game;
    this.running = false;
    const progress = [];
    for (let i = 0; i < this.n; i++) progress.push(g.progress(i));
    const msg = { winner: g.winner, timeoutWin: g.timeoutWin, progress, round: this.round, names: this.names() };
    if (this.transport) this.transport.send('end', msg);
    for (const cb of this._endCbs) cb(msg);
    if (this.settings.rounds > 0 && this.round >= this.settings.rounds) { this.finished = true; this.restartAt = null; this.notice = 'series over'; }
    else { this.restartAt = now + RESTART_DELAY_MS; this.notice = 'next round in ' + (RESTART_DELAY_MS / 1000) + ' s'; }
    this._sendLobby();
  }

  // The host's own seat input (same shape as a client's sendInput). Never times out (no keepalive needed).
  setInput(input) {
    const k = this.seatOf(this.id);
    if (k >= 0) this.pending[k] = normInput(input);
  }
  _input(obj, from) {
    const k = this.seatOf(from);
    if (k < 0 || this.seats[k].type !== 'human') return;
    this.pending[k] = normInput(obj);
    this.inputAt[k] = this.now();
    this.inputStale[k] = 0;
  }

  frame() { return this.game ? this.game.frame() : null; }
  meta() { return this.game ? this.game.meta() : null; }

  // Stop the round loop and any pending restart; the lobby stays and start() works again. The timer keeps
  // running (it only heartbeats the lobby while no round runs), so clients do not report the host gone.
  stop() {
    this.running = false;
    this.restartAt = null;
    if (this.transport && !this.transport.closed) { this.notice = 'stopped'; this._sendLobby(); }
  }
  // Leave: stop, drop the timer and close the transport so every peer sees the host go.
  close() {
    this.stop();
    if (this._timer !== null) { this._clearInterval(this._timer); this._timer = null; }
    if (this.transport) this.transport.close();
  }
}

// ============================================================================================== Client
// opts = { name, token, now, setInterval, clearInterval, manual, lag (ms, default 100), maxExtrap (ms, default 150) }
// token: keep it in sessionStorage and pass it back after a refresh to get the same seat back.
export class Client {
  constructor(transport, opts = {}) {
    this.transport = transport;
    this.id = transport.id;
    this.name = opts.name || ('player-' + transport.id.slice(0, 4));
    this.token = cleanToken(opts.token) || randomId(16);
    const timers = defaultTimers();
    this.now = opts.now || timers.now;
    this._setInterval = opts.setInterval || timers.setInterval;
    this._clearInterval = opts.clearInterval || timers.clearInterval;
    this.manual = !!opts.manual;
    this.lag = opts.lag !== undefined ? opts.lag : VIEW_LAG_MS;
    this.maxExtrap = opts.maxExtrap !== undefined ? opts.maxExtrap : VIEW_MAX_EXTRAP_MS;
    this.lobby = null;
    this.startMsg = null;
    this.endMsg = null;
    this.hostId = null;
    this.hostLeft = false;
    this.seat = -1;
    this.spectator = true;
    this.buf = [];
    this.offset = null;        // recvMs - t*1000 (EMA)
    this.jitter = 0;           // EMA of |offset deviation| in ms
    this.latestTick = 0;
    this.snapCount = 0;
    this.eventsLost = 0;       // events that no replay window could recover
    this.resyncs = 0;          // forward jumps of the view clock
    this.ended = false;
    this._clockT = null;       // the view's render time (s, host timeline)
    this._clockNow = 0;
    this._evNext = 0;
    this._evRound = null;
    this._orphan = [];
    this._lastHostMsg = null;
    this._input = null;
    this._sentInput = null;
    this._dirty = false;
    this._lastSent = -Infinity;
    this._lobbyCbs = []; this._startCbs = []; this._snapCbs = []; this._endCbs = []; this._hostLeftCbs = []; this._hostBackCbs = [];
    transport.on('lobby', (obj, from) => this._onLobby(obj, from));
    transport.on('start', (obj, from) => this._onStart(obj, from));
    transport.on('snap', (obj, from) => this._onSnap(obj, from));
    transport.on('end', (obj, from) => this._onEnd(obj, from));
    transport.onLeave((id) => { if (id === this.hostId) this._hostGone(); });
    // Say hello to every peer (existing ones are replayed by onPeer); only the host answers.
    transport.onPeer((id) => transport.send('hello', this._helloMsg(), id));
    this._timer = this.manual ? null : this._setInterval(() => this.pump(this.now()), INPUT_PERIOD_MS);
  }

  onLobby(cb) { this._lobbyCbs.push(cb); }
  onStart(cb) { this._startCbs.push(cb); }
  onSnapshot(cb) { this._snapCbs.push(cb); }
  onEnd(cb) { this._endCbs.push(cb); }
  onHostLeft(cb) { this._hostLeftCbs.push(cb); }
  onHostBack(cb) { this._hostBackCbs.push(cb); }

  _helloMsg(extra) { return Object.assign({ name: this.name, token: this.token }, extra || {}); }
  setName(name) { this.name = name; if (this.hostId) this.transport.send('hello', this._helloMsg(), this.hostId); }
  setReady(ready) { if (this.hostId) this.transport.send('hello', this._helloMsg({ ready: !!ready }), this.hostId); }

  _seatFrom(seats) {
    this.seat = Array.isArray(seats) ? seats.findIndex((s) => s && s.peer === this.id) : -1;
    this.spectator = this.seat < 0;
  }
  // Every message from the host is a liveness signal; a host reported gone by the silence timer that speaks
  // again is taken back.
  _hostSeen(from) {
    this._lastHostMsg = this.now();
    if (this.hostLeft && from === this.hostId) { this.hostLeft = false; for (const cb of this._hostBackCbs) cb(from); }
  }
  _hostGone() {
    if (this.hostLeft) return;
    this.hostLeft = true;
    for (const cb of this._hostLeftCbs) cb(this.hostId);
  }
  _fromHost(from) { return !this.hostId || from === this.hostId; }
  _onLobby(obj, from) {
    if (!obj || !obj.lobby || typeof obj.lobby !== 'object') return;
    if (this.hostId && from !== this.hostId && obj.lobby.host !== this.hostId) {
      if (!this.hostLeft) return;   // ignore a second host while ours is alive
      // our host is gone: adopt the new one
      this.hostId = null; this.hostLeft = false; this.buf = []; this._clockT = null;
      for (const cb of this._hostBackCbs) cb(from);
    }
    this.hostId = obj.lobby.host || from;
    this._hostSeen(from);
    this.lobby = obj.lobby;
    this._seatFrom(obj.lobby.seats);
    for (const cb of this._lobbyCbs) cb(obj.lobby);
  }
  _onStart(obj, from) {
    if (!this._fromHost(from)) return;
    if (!obj || typeof obj !== 'object' || !obj.meta || !Array.isArray(obj.seats)) return;
    if (!this.hostId) this.hostId = from;
    this._hostSeen(from);
    this.startMsg = obj;
    this.endMsg = null;
    this.ended = false;
    this.buf = [];
    this.offset = null;
    this.jitter = 0;
    this.latestTick = 0;
    this._clockT = null;
    this._evNext = 0;
    this._evRound = obj.round !== undefined ? obj.round : null;
    this._orphan = [];
    this._seatFrom(obj.seats);
    for (const cb of this._startCbs) cb(obj);
  }
  _onSnap(obj, from) {
    if (!this._fromHost(from)) return;
    if (!obj || !validFrame(obj.frame) || !Number.isFinite(obj.tick)) return;
    this._hostSeen(from);
    const recv = this.now();
    const t = obj.frame.t;
    const off = recv - t * 1000;
    if (this.offset === null) this.offset = off;
    else {
      const dev = Math.abs(off - this.offset);
      this.jitter += (dev - this.jitter) * (dev > this.jitter ? 0.1 : 0.2);
      this.offset += (off - this.offset) * 0.2;
    }
    // events: de-duplicate by sequence number; prev holds the previous snapshots' events again so a
    // dropped snapshot loses nothing
    if (obj.round !== undefined && obj.round !== this._evRound) { this._evRound = obj.round; this._evNext = 0; }
    const evs = Array.isArray(obj.frame.events) ? obj.frame.events : [];
    let fresh = evs;
    if (Number.isFinite(obj.ev0)) {
      const prev = Array.isArray(obj.prev) ? obj.prev : [];
      const all = prev.length ? prev.concat(evs) : evs;
      const base = obj.ev0 - prev.length;
      if (this._evNext < base) { this.eventsLost += base - this._evNext; this._evNext = base; }
      fresh = all.slice(Math.max(0, this._evNext - base));
      if (base + all.length > this._evNext) this._evNext = base + all.length;
    }
    const snap = { tick: obj.tick, t, frame: obj.frame, recv, emitted: false, events: fresh };
    // keep the buffer sorted by t (a late-arriving older snapshot is inserted, not appended)
    let k = this.buf.length;
    while (k > 0 && this.buf[k - 1].t > t) k--;
    if (k > 0 && this.buf[k - 1].t === t) { if (fresh.length) this._orphan.push(...fresh); return; }   // duplicate
    this.buf.splice(k, 0, snap);
    while (this.buf.length > VIEW_MAX_BUF) { const old = this.buf.shift(); if (!old.emitted) this._orphan.push(...old.events); }
    if (obj.tick > this.latestTick) this.latestTick = obj.tick;
    this.snapCount++;
    for (const cb of this._snapCbs) cb(obj);
  }
  _onEnd(obj, from) {
    if (!this._fromHost(from)) return;
    if (!obj || typeof obj !== 'object') return;
    this._hostSeen(from);
    this.endMsg = obj;
    this.ended = true;
    for (const cb of this._endCbs) cb(obj);
  }

  // ----------------------------------------------------------------- inputs
  // input = { dir:[dx,dy], join, leave, intent, move? } - stored, sent at 20 Hz when changed (+ keepalive).
  sendInput(input) {
    const ni = normInput(input);
    if (!sameInput(ni, this._input)) { this._input = ni; this._dirty = true; }
    this.pump(this.now());
  }
  pump(now) {
    if (this.hostId && !this.hostLeft && this._lastHostMsg !== null && now - this._lastHostMsg > HOST_TIMEOUT_MS) this._hostGone();
    if (this.spectator || !this.hostId || this.hostLeft || !this._input) return;
    const since = now - this._lastSent;
    if ((this._dirty && since >= INPUT_PERIOD_MS) || since >= INPUT_KEEPALIVE_MS) {
      const msg = Object.assign({ tick: this.latestTick }, this._input, { dir: this._input.dir.slice() });
      this.transport.send('input', msg, this.hostId);
      this._sentInput = this._input;
      this._dirty = false;
      this._lastSent = now;
    }
  }

  // ----------------------------------------------------------------- view
  // The distance the view keeps behind the host's newest state: the nominal lag plus an allowance for the
  // measured arrival jitter (so late snapshots still land ahead of the render time).
  targetLag() { return this.lag + Math.min(VIEW_MAX_ADAPT_MS, VIEW_JITTER_K * this.jitter); }

  // Interpolated frame for rendering at nowMs. The render clock runs on the host's timeline at (nearly) real
  // time, targetLag() behind where the host's newest snapshot is expected to be now; it speeds up or slows
  // down by at most VIEW_RATE_SLEW to re-centre (never stepping backwards), jumps forward only when it has
  // fallen more than VIEW_RESYNC_S behind (a hidden tab), and dead-reckons by velocity for at most maxExtrap
  // ms when the buffer runs dry. Each snapshot's events are returned exactly once, when the render time
  // passes that snapshot.
  view(nowMs = this.now()) {
    const buf = this.buf;
    if (!buf.length || this.offset === null) return null;
    const newest = buf[buf.length - 1];
    const target = this.targetLag() / 1000;
    const maxEx = this.ended ? 0 : this.maxExtrap / 1000;
    const expected = (nowMs - this.offset) / 1000;      // where the host's stream should be by now
    let renderT;
    if (this._clockT === null) renderT = expected - target;   // may sit before buf[0].t: buf[0] is shown until the clock reaches it
    else {
      const dt = Math.min(Math.max(0, nowMs - this._clockNow), 500) / 1000;
      // error of the clock once advanced at real time (measuring before the advance would leave the
      // equilibrium dt short of the target, i.e. dependent on how often view() is called)
      const err = expected - (this._clockT + dt) - target;   // > 0: behind the target, speed up
      if (err > VIEW_RESYNC_S) { renderT = expected - target; this.resyncs++; }
      else renderT = this._clockT + dt * (1 + Math.max(-VIEW_RATE_SLEW, Math.min(VIEW_RATE_SLEW, err / VIEW_RATE_GAIN_S)));
    }
    if (renderT > newest.t + maxEx) renderT = newest.t + maxEx;
    if (this._clockT !== null && renderT < this._clockT) renderT = this._clockT;
    this._clockT = renderT; this._clockNow = nowMs;
    let frame;
    if (renderT >= newest.t) {
      const dt = renderT - newest.t;
      frame = dt > 0 ? extrapolateFrame(newest.frame, dt) : cloneFrame(newest.frame);
    } else if (renderT <= buf[0].t) {
      frame = cloneFrame(buf[0].frame);
    } else {
      let i = buf.length - 2;
      while (i > 0 && buf[i].t > renderT) i--;
      const a = buf[i], b = buf[i + 1];
      const span = b.t - a.t;
      frame = interpolateFrame(a.frame, b.frame, span > 0 ? (renderT - a.t) / span : 1);
    }
    frame.t = renderT;
    const events = this._orphan.length ? this._orphan.splice(0) : [];
    for (const s of buf) if (!s.emitted && s.t <= renderT + 1e-9) { s.emitted = true; if (s.events.length) events.push(...s.events); }
    frame.events = events;
    frame.tick = newest.tick;
    while (buf.length > 2 && buf[1].t <= renderT && buf[0].emitted) buf.shift();
    return frame;
  }

  close() {
    if (this._timer !== null) { this._clearInterval(this._timer); this._timer = null; }
    this.transport.close();
  }
}
