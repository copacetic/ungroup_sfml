// engine.js - browser port of the Ungroup rules core (rl/native/ungroup.cpp is the single source of
// truth; this file keeps its function names and order so the two can be read side by side).
//
// Rules (v2 + the v3 package, see the header of ungroup.cpp):
// - Circles that touch merge into one group body when EVERY member of both bodies is joinable, nobody
//   is on a join cooldown and no pair is on a partner cooldown.
// - Group speed = base / sqrt(n) times the mean direction of the pushing members.
// - Mines yield mine_rate * n^mine_exp per second while a body touches them (not while stunned).
// - Nothing counts until banked: touching a member's home pad banks the whole pool to them (under
//   the crown rule only the head's pad pays).
// - Leaving is a HELD action with a timer; the leaver takes a share weighted toward its intent
//   minus a forfeit, is ejected, its joinable flag is forced off and it gets cooldowns.
// - Hard collisions spill carried units onto the floor as pickups and stun both bodies.
// - The arena shrinks after shrink_start; mines outside it die; pads slide inward.
// - A player wins by banking its full need vector; at the time limit the highest progress wins.
//
// Numerics: all arithmetic is IEEE double exactly as in the C++ (built with -std=c++17, so no FMA
// contraction); Math.sqrt is correctly rounded like std::sqrt. Math.sin/cos/atan2/pow may differ
// from glibc by one ulp on rare inputs, so long trajectories are not guaranteed bit-identical to
// the C++ core, but reset() (needs, pads, intents, mine layout) reproduces the C++ draws exactly
// through rng.js, and every rule is otherwise a literal transcription.

import { RNG, roundHalfEven } from './rng.js';

export const TYPES = 4;
export const MAX_SLOTS = 8;      // other-player slots in the observation (nearest others)
export const MAX_PLAYERS = 32;   // hard cap on seats (pairwise history arrays)
export const N_PICKUPS_OBS = 4;
export const N_DR = 4;           // randomised constants exposed in the observation
// Macro movement actions (move >= MACRO_BASE): the core steers toward a target every tick until the next
// decision. 10..17 = mine m, 18 = own pad, 19 = the group head's pad (own pad when solo), 20..23 = the k-th
// nearest other player's body (the observation's slot order). Legacy moves 0..9 still work.
export const MACRO_BASE = 10;
export const MACRO_MINES = 8;
export const MACRO_BODIES = 4;
export const MOVE_CLASSES = MACRO_BASE + MACRO_MINES + 2 + MACRO_BODIES;  // 24
const PI = Math.PI;
const NEVER = -1e9;

// Config: same fields, order and defaults as the Python Config dataclass (rl/ungroup/native.py) and
// cfg_fill() in the C++.
export const CONFIG_DEFAULTS = Object.freeze({
  n_players: 6, n_mines: 8,
  dt: 1.0 / 30.0, time_limit: 240.0, base_speed: 0.45, vel_lerp: 6.0, solo_radius: 0.045,
  mine_radius: 0.08, mine_cap: 30.0, mine_regen: 0.5, mine_rate: 0.12, mine_exp: 2.0,
  pad_radius: 0.06,
  need_primary: 18, need_secondary: 6,
  leave_time: 2.0, spill_min_speed: 0.40, spill_k: 6.0,
  spill_max: 6,
  pickup_ttl: 8.0,
  max_pickups: 64,
  shrink_start: 0.5, final_radius: 0.5, restitution: 0.5,
  max_group: 6,
  join_cooldown: 3.0, partner_cooldown: 10.0, leave_forfeit: 0.15, intent_weight: 3.0, stun_time: 1.0,
  leave_hold: 1.0,
  group_bank_bonus: 0.15,
  rammer_stun_mult: 2.5,
  carried_shaping: 2.0, win_bonus: 10.0, lose_penalty: 2.0, relative_reward: 0.5,
  // v3 package (all off = legacy rules)
  crown: 0,
  head_vest: 10.0,
  brand: 0,
  brand_min: 4.0, brand_base: 20.0, brand_per_unit: 2.0, brand_max: 60.0,
  bloom_rate: 0.0,
  seed_rate: 0.0,
  seed_floor: 0.0,
  seed_range: 0.5,
  bloom_cap: 0.0,
  persist: 0,
  ledger_decay: 0.5,
  grudge_window: 500.0,
  obs_legacy: 0,
  head_steer: 1.0,   // weight of the head's push direction in the group's mean (the crown steers when > 1)
  bank_round: 0,     // 1 = banked amounts are whole units (the remainder stays in the pool)
});
export const CFG_FIELDS = Object.freeze(Object.keys(CONFIG_DEFAULTS));
export const CFG_LEN = CFG_FIELDS.length;  // 55
const INT_FIELDS = new Set(['n_players', 'n_mines', 'need_primary', 'need_secondary', 'spill_max', 'max_pickups',
  'max_group', 'crown', 'brand', 'persist', 'obs_legacy', 'bank_round']);

// Named rule sets (rl/ungroup/native.py PRESETS).
export const PRESETS = Object.freeze({
  legacy: {},
  crown: { crown: 1 },
  bloom: { bloom_rate: 0.1, seed_rate: 0.15, seed_floor: 0.05, bloom_cap: 8.0 },
  life: { crown: 1, brand: 1, bloom_rate: 0.1, seed_rate: 0.15, seed_floor: 0.05, bloom_cap: 8.0, head_steer: 2.0, bank_round: 1 },
  series: { crown: 1, brand: 1, bloom_rate: 0.1, seed_rate: 0.15, seed_floor: 0.05, bloom_cap: 8.0, head_steer: 2.0, bank_round: 1, persist: 1 },
});

// Config for a named rule set; bloom_cap scales with the lobby (K = n_players + 2) unless overridden.
export function preset(name, overrides = {}) {
  if (!(name in PRESETS)) throw new Error('unknown preset ' + name);
  const kw = Object.assign({}, PRESETS[name], overrides);
  if ((kw.bloom_rate || 0) > 0 && !('bloom_cap' in overrides)) kw.bloom_cap = (kw.n_players ?? 6) + 2;
  return normalizeConfig(kw);
}

// Fill missing fields from the defaults and coerce integer fields (cfg_fill casts them to int).
export function normalizeConfig(cfg = {}) {
  const out = {};
  for (const k of CFG_FIELDS) {
    let v = cfg[k] !== undefined ? Number(cfg[k]) : CONFIG_DEFAULTS[k];
    if (INT_FIELDS.has(k)) v = Math.trunc(v);
    out[k] = v;
  }
  return out;
}

export function configToArray(cfg) { return CFG_FIELDS.map((k) => cfg[k]); }
export function configFromArray(arr) {
  const o = {};
  CFG_FIELDS.forEach((k, i) => { o[k] = arr[i]; });
  return normalizeConfig(o);
}

export const SEATS = ['policy', 'snapshot', 'solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge'];
export const SEAT_EXTERNAL = 0, SEAT_EXTERNAL2 = 1, SEAT_SOLO = 2, SEAT_BAIL = 3, SEAT_LOYAL = 4, SEAT_KIDNAP = 5,
  SEAT_RAMMER = 6, SEAT_GRUDGE = 7;
// 'human' and 'agent' seats are external at the engine level (their actions come from setInput).
export function seatType(nameOrType) {
  if (typeof nameOrType === 'number') return nameOrType | 0;
  if (nameOrType === 'human' || nameOrType === 'agent') return SEAT_EXTERNAL;
  const k = SEATS.indexOf(nameOrType);
  if (k < 0) throw new Error('unknown seat ' + nameOrType);
  return k;
}

// ---- number formatting matching the C++ frame/meta JSON (printf %.nf)
const r1 = (x) => Number(x.toFixed(1)), r2 = (x) => Number(x.toFixed(2)), r3 = (x) => Number(x.toFixed(3)), r6 = (x) => Number(x.toFixed(6));

class Body {
  constructor(members) {
    this.members = members;
    this.x = 0; this.y = 0; this.vx = 0; this.vy = 0;
    this.pool = new Float64Array(TYPES);
    this.stun = 0;   // seconds of stun remaining
    this.head = -1;  // crown: the member whose pad the pool pays out at (always the sole member of a solo body)
  }
  n() { return this.members.length; }
  poolTotal() { return this.pool[0] + this.pool[1] + this.pool[2] + this.pool[3]; }
}

class Player {
  constructor() {
    this.need = new Float64Array(TYPES);
    this.banked = new Float64Array(TYPES);
    this.padAngle = 0;
    this.intent = 0;
    this.joinable = false;
    this.dx = 0; this.dy = 0;
    this.leaveTimer = -1.0;      // seconds remaining, < 0 when not leaving
    this.leaveLastReq = -1e9;    // time of the last leave=1 decision
    this.macro = -1;             // current macro movement target (>= MACRO_BASE) or -1 for direct steering
    this.joinCooldown = 0;       // seconds until this player may merge again
    this.groupSince = 0;         // time the current membership started (t)
    this.lastBankT = 0;
  }
}

class Stats {
  constructor() {
    this.group = 0;
    this.steps = 0; this.merges = 0; this.leaves = 0; this.spills = 0; this.banks = 0; this.group_banks = 0;
    this.remerge_fast = 0; this.cancels = 0;
    this.alliances = 0; this.alliances_long = 0; this.fair_banks = 0; this.crowns = 0;
    this.alliance_dur = 0;
    this.units_taken = 0;  // units carried away by leavers
  }
}

export function directionToMove(x, y) {
  const n = Math.sqrt(x * x + y * y);
  if (n < 1e-6) return 0;
  const ang = Math.atan2(y, x);
  let idx = roundHalfEven(ang / (2 * PI / 8));
  idx = ((idx % 8) + 8) % 8;
  return idx + 1;
}

export class Game {
  // cfg: a plain object (missing fields take CONFIG_DEFAULTS); seed: number, bigint or decimal string.
  constructor(cfg = {}, seed = 1) {
    this.cfg = normalizeConfig(cfg);
    if (this.cfg.n_players > MAX_PLAYERS) this.cfg.n_players = MAX_PLAYERS;
    if (this.cfg.n_players < 1) this.cfg.n_players = 1;
    this.decideEvery = 6;
    this.rng = new RNG(0n);
    this.seedBig = 0n;
    this.seed = 0;
    this.nextSeed = 1n;
    this.t = 0;
    this.done = false;
    this.winner = -1;
    this.timeoutWin = false;
    this.R = 1.0;
    this.players = [];
    this.bodies = [];
    this.mineX = new Float64Array(0); this.mineY = new Float64Array(0);
    this.mineType = new Int32Array(0);
    this.mineStock = new Float64Array(0);
    this._mineNext = new Float64Array(0);
    this.mineAlive = new Uint8Array(0);
    this.mineNbr = [];      // bloom: ring neighbours within seed_range
    this.picks = [];        // {x, y, type, ttl}
    this.events = [];
    this.prevPot = new Float64Array(0);
    this.rewards = new Float64Array(0);
    this.seats = new Int32Array(this.cfg.n_players);
    this.seatNames = new Array(this.cfg.n_players).fill('policy');
    this.stats = new Stats();
    const P2 = MAX_PLAYERS * MAX_PLAYERS;
    // pairwise history (indexed [i * MAX_PLAYERS + j])
    this.comemberTime = new Float64Array(P2);
    this.tookFrom = new Float64Array(P2);      // units j took when leaving a body containing i
    this.bankedWhile = new Float64Array(P2);   // units banked to j while i was a member
    this.lastLeftMe = new Float64Array(P2);    // t at which j last left a body containing i (NEVER)
    this.partnerCd = new Float64Array(P2);     // t until which i may not merge with j
    this.pairSince = new Float64Array(P2);     // start time of the current co-membership (-1 none)
    this.pairEnded = new Float64Array(P2);     // end time of the last co-membership (-1 none)
    this.brand = new Float64Array(MAX_PLAYERS);          // public betrayal mark, seconds remaining
    this.lastUnjustLeave = new Float64Array(MAX_PLAYERS).fill(NEVER);
    this.roundsPlayed = 0;
    this.tickCount = 0;
    // external inputs (persist until changed)
    this.inputs = [];
    this.inputDirty = new Uint8Array(MAX_PLAYERS);
    for (let i = 0; i < MAX_PLAYERS; i++) this.inputs.push({ dir: [0, 0], move: 0, join: 0, leave: 0, intent: 0 });
    // scratch (no per-tick allocations in the hot loops)
    this._act = new Int32Array(MAX_PLAYERS * 4);
    this._bodyOf = new Int32Array(MAX_PLAYERS);
    this._ordIdx = new Int32Array(MAX_PLAYERS);
    this._ordDist = new Float64Array(MAX_PLAYERS);
    this._tgt = { x: 0, y: 0 };
    this._botTgt = { x: 0, y: 0 };  // the last bot's movement target (for macro labels)
    this._botHas = false;
    this._needed = new Uint8Array(TYPES);
    this._needed2 = new Uint8Array(TYPES);
    this._weights = new Float64Array(TYPES);
    this._share = new Float64Array(TYPES);
    this._counts = new Int32Array(TYPES);
    this._probs = new Float64Array(TYPES);
    this._amount = new Float64Array(TYPES);
    this._keep = new Uint8Array(0);
    this.reset(seed);
  }

  // ------------------------------------------------------------ helpers
  uniform(a, b) { return this.rng.uniform(a, b); }
  radius(n) { return this.cfg.solo_radius * Math.sqrt(n); }
  needTotal(i) { const p = this.players[i]; return p.need[0] + p.need[1] + p.need[2] + p.need[3]; }
  mineK() { return this.cfg.bloom_cap > 0 ? this.cfg.bloom_cap : this.cfg.mine_cap; }

  padPos(i, out) {
    const r = Math.max(this.R - this.cfg.pad_radius - 0.02, 0.1);
    const a = this.players[i].padAngle;
    out.x = r * Math.cos(a); out.y = r * Math.sin(a);
    return out;
  }
  padX(i) { return Math.max(this.R - this.cfg.pad_radius - 0.02, 0.1) * Math.cos(this.players[i].padAngle); }
  padY(i) { return Math.max(this.R - this.cfg.pad_radius - 0.02, 0.1) * Math.sin(this.players[i].padAngle); }

  bodyIndex(player) { return this._bodyOf[player]; }
  _rebuildIndex() {
    const bo = this._bodyOf;
    for (let k = 0; k < this.bodies.length; k++) {
      const m = this.bodies[k].members;
      for (let q = 0; q < m.length; q++) bo[m[q]] = k;
    }
  }

  progress(i) {
    const p = this.players[i];
    let s = 0;
    for (let t = 0; t < TYPES; t++) s += Math.min(p.banked[t] / p.need[t], 1.0);
    return s / TYPES;
  }
  bankedTotal(i) { const p = this.players[i]; return p.banked[0] + p.banked[1] + p.banked[2] + p.banked[3]; }

  event(obj) { obj.t = r2(this.t); this.events.push(obj); }
  static nums(v, n, prec) { const out = new Array(n); for (let k = 0; k < n; k++) out[k] = Number(v[k].toFixed(prec)); return out; }

  // ------------------------------------------------------------ seats and inputs
  // names: array of seat names ('policy', 'snapshot', 'solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge',
  // 'human', 'agent') or seat-type integers.
  setSeats(namesOrTypes) {
    const n = this.cfg.n_players;
    for (let i = 0; i < n; i++) {
      const v = namesOrTypes[i] ?? 'policy';
      this.seats[i] = seatType(v);
      this.seatNames[i] = typeof v === 'number' ? SEATS[v] : v;
    }
  }

  // Inputs for external seats persist until changed. move: 0 stop, 1..8 compass, 9 keep the continuous
  // direction `dir` (humans), >= 10 macro targets. join 0/1, leave 0/1 (held), intent 0 keep or 1..4.
  setInput(i, input) {
    const s = this.inputs[i];
    if (input.dir) { s.dir[0] = input.dir[0]; s.dir[1] = input.dir[1]; }
    if (input.move !== undefined) s.move = input.move | 0;
    if (input.join !== undefined) s.join = input.join ? 1 : 0;
    if (input.leave !== undefined) s.leave = input.leave ? 1 : 0;
    if (input.intent !== undefined) s.intent = input.intent | 0;
    this.inputDirty[i] = 1;
  }

  // ------------------------------------------------------------ reset
  sampleCfg() { /* domain randomisation is not used in the browser: cfg is the base cfg */ }

  reset(sd, fresh = false) {
    if (fresh) this.roundsPlayed = 0;
    const cfg = this.cfg;
    const tPrev = this.t;
    const keep = cfg.persist > 0 && this.roundsPlayed > 0;
    let seedBig = sd === undefined || sd === null || sd === 0 ? 0n : BigInt.asUintN(64, BigInt(typeof sd === 'number' ? Math.trunc(sd) : sd));
    if (seedBig === 0n) seedBig = this.nextSeed++;
    this.seedBig = seedBig;
    this.seed = Number(seedBig);
    this.rng.seed(seedBig);
    this.sampleCfg();
    this.t = 0; this.done = false; this.winner = -1; this.timeoutWin = false; this.R = 1.0;
    this.tickCount = 0;
    this.events.length = 0;
    this.stats = new Stats();
    const n = cfg.n_players;
    this.players = [];
    for (let i = 0; i < n; i++) this.players.push(new Player());
    const rot = this.uniform(0, 2 * PI);
    for (let i = 0; i < n; i++) {
      const primary = this.rng.uniformInt(0, TYPES - 1);
      const p = this.players[i];
      for (let t = 0; t < TYPES; t++) { p.need[t] = cfg.need_secondary; p.banked[t] = 0; }
      p.need[primary] = cfg.need_primary;
      p.padAngle = rot + 2 * PI * i / n;
      p.intent = this.rng.uniformInt(0, TYPES - 1);
    }
    const M = cfg.n_mines;
    this.mineX = new Float64Array(M); this.mineY = new Float64Array(M);
    this.mineType = new Int32Array(M);
    this.mineStock = new Float64Array(M).fill(this.mineK());
    this._mineNext = new Float64Array(M);
    this.mineAlive = new Uint8Array(M).fill(1);
    const mrot = this.uniform(0, 2 * PI);
    for (let m = 0; m < M; m++) {
      const ring = (m % 2 === 0) ? 0.62 : 0.35;
      const ang = mrot + 2 * PI * m / M;
      this.mineX[m] = ring * Math.cos(ang); this.mineY[m] = ring * Math.sin(ang);
      this.mineType[m] = (M >= 2 * TYPES) ? Math.trunc(m / 2) % TYPES : m % TYPES;
    }
    this.mineNbr = [];
    for (let m = 0; m < M; m++) {
      const nb = [];
      for (let j = 0; j < M; j++) {
        if (j !== m && Math.hypot(this.mineX[m] - this.mineX[j], this.mineY[m] - this.mineY[j]) < cfg.seed_range) nb.push(j);
      }
      this.mineNbr.push(nb);
    }
    this.bodies = [];
    for (let i = 0; i < n; i++) {
      const b = new Body([i]);
      b.head = i;
      b.x = 0.72 * Math.cos(this.players[i].padAngle); b.y = 0.72 * Math.sin(this.players[i].padAngle);
      this.bodies.push(b);
    }
    this._rebuildIndex();
    this.picks = [];
    for (let i = 0; i < MAX_PLAYERS; i++) {
      if (keep) {
        // the ledger survives: unit counts decay, timestamps shift so "seconds since" keeps counting
        if (this.lastUnjustLeave[i] > NEVER / 2) this.lastUnjustLeave[i] -= tPrev;
      } else { this.brand[i] = 0; this.lastUnjustLeave[i] = NEVER; }
      for (let j = 0; j < MAX_PLAYERS; j++) {
        const ij = i * MAX_PLAYERS + j;
        if (keep) {
          this.comemberTime[ij] *= cfg.ledger_decay; this.tookFrom[ij] *= cfg.ledger_decay; this.bankedWhile[ij] *= cfg.ledger_decay;
          if (this.lastLeftMe[ij] > NEVER / 2) this.lastLeftMe[ij] -= tPrev;
          this.partnerCd[ij] -= tPrev;
        } else {
          this.comemberTime[ij] = 0; this.tookFrom[ij] = 0; this.bankedWhile[ij] = 0;
          this.lastLeftMe[ij] = NEVER; this.partnerCd[ij] = -1;
        }
        this.pairSince[ij] = -1; this.pairEnded[ij] = -1;
      }
    }
    this.roundsPlayed++;
    this.prevPot = new Float64Array(n);
    this.rewards = new Float64Array(n);
    this.potentials(this.prevPot);
    if (this.seats.length !== n) { this.seats = new Int32Array(n); this.seatNames = new Array(n).fill('policy'); }
    for (let i = 0; i < n; i++) {
      const s = this.inputs[i];
      s.dir[0] = 0; s.dir[1] = 0; s.move = 0; s.join = 0; s.leave = 0; s.intent = 0;
      this.inputDirty[i] = 0;
    }
  }

  // ------------------------------------------------------------ actions

  // apply_actions in the C++ loops over every seat; the per-seat body is independent, so it is
  // factored out to let external seats apply as soon as their input changes.
  applyAction(i, move, joinable, leave, intent) {
    const cfg = this.cfg;
    const p = this.players[i];
    if (move === 0) { p.dx = 0; p.dy = 0; p.macro = -1; }
    else if (move === 9) { /* keep the direction set through setDirection (human seats) */ }
    else if (move >= MACRO_BASE) { p.macro = move; this.steerMacro(i); }
    else { const a = 2 * PI * (move - 1) / 8; p.dx = Math.cos(a); p.dy = Math.sin(a); p.macro = -1; }
    p.joinable = joinable !== 0;
    const bi = this.bodyIndex(i);
    const n = this.bodies[bi].n();
    if (intent > 0 && n === 1) p.intent = intent - 1;  // locked while grouped
    if (leave && n > 1) {
      p.leaveLastReq = this.t;
      if (p.leaveTimer < 0) { p.leaveTimer = cfg.leave_time; this.event({ kind: 'leave_start', player: i }); }
    } else if (!leave && p.leaveTimer >= 0 && this.t - p.leaveLastReq > cfg.leave_hold) {
      p.leaveTimer = -1.0;
      this.stats.cancels++;
      this.event({ kind: 'leave_cancel', player: i });
    }
  }

  applyActions(act) {
    for (let i = 0; i < this.cfg.n_players; i++) this.applyAction(i, act[i * 4], act[i * 4 + 1], act[i * 4 + 2], act[i * 4 + 3]);
  }

  setDirection(i, dx, dy) {
    const n = Math.sqrt(dx * dx + dy * dy);
    const p = this.players[i];
    if (n < 1e-6) { p.dx = 0; p.dy = 0; } else { p.dx = dx / n; p.dy = dy / n; }
  }

  // ------------------------------------------------------------ macro targets

  // Other players sorted by body distance from player i (stable on index), the observation's slot
  // order. Fills this._ordIdx and returns the count (n - 1). Insertion sort in index order keeps ties stable.
  nearestOthers(i) {
    const n = this.cfg.n_players;
    const b = this.bodies[this._bodyOf[i]];
    const idx = this._ordIdx, dist = this._ordDist;
    let cnt = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const bj = this.bodies[this._bodyOf[j]];
      const d = Math.sqrt((bj.x - b.x) * (bj.x - b.x) + (bj.y - b.y) * (bj.y - b.y));
      let k = cnt;
      while (k > 0 && dist[k - 1] > d) { dist[k] = dist[k - 1]; idx[k] = idx[k - 1]; k--; }
      dist[k] = d; idx[k] = j;
      cnt++;
    }
    return cnt;
  }

  macroTarget(i, m, out) {
    let k = m - MACRO_BASE;
    if (k < 0) return false;
    if (k < MACRO_MINES) { if (k >= this.cfg.n_mines) return false; out.x = this.mineX[k]; out.y = this.mineY[k]; return true; }
    k -= MACRO_MINES;
    if (k === 0) { this.padPos(i, out); return true; }
    if (k === 1) { const b = this.bodies[this._bodyOf[i]]; this.padPos(b.n() > 1 && b.head >= 0 ? b.head : i, out); return true; }
    k -= 2;
    if (k < MACRO_BODIES) {
      const cnt = this.nearestOthers(i);
      if (k >= cnt) return false;
      const bo = this.bodies[this._bodyOf[this._ordIdx[k]]];
      out.x = bo.x; out.y = bo.y;
      return true;
    }
    return false;
  }

  steerMacro(i) {
    const p = this.players[i];
    if (p.macro < 0) return;
    const tgt = this._tgt;
    if (!this.macroTarget(i, p.macro, tgt)) { p.dx = 0; p.dy = 0; return; }
    const b = this.bodies[this._bodyOf[i]];
    const dx = tgt.x - b.x, dy = tgt.y - b.y;
    const n = Math.sqrt(dx * dx + dy * dy);
    if (n < 1e-6) { p.dx = 0; p.dy = 0; } else { p.dx = dx * (1.0 / n); p.dy = dy * (1.0 / n); }
  }

  steerMacros() { for (let i = 0; i < this.cfg.n_players; i++) if (this.players[i].macro >= 0) this.steerMacro(i); }

  // The macro class a scripted bot's target corresponds to (DAgger labels for macro policies).
  macroLabel(i) {
    if (!this._botHas) return 0;
    const tx = this._botTgt.x, ty = this._botTgt.y;
    const nm = Math.min(this.cfg.n_mines, MACRO_MINES);
    for (let m = 0; m < nm; m++) if (Math.hypot(tx - this.mineX[m], ty - this.mineY[m]) < 1e-9) return MACRO_BASE + m;
    if (Math.hypot(tx - this.padX(i), ty - this.padY(i)) < 1e-9) return MACRO_BASE + MACRO_MINES;
    const b = this.bodies[this._bodyOf[i]];
    if (b.n() > 1 && b.head >= 0 && Math.hypot(tx - this.padX(b.head), ty - this.padY(b.head)) < 1e-9) return MACRO_BASE + MACRO_MINES + 1;
    for (const j of b.members) if (j !== i && Math.hypot(tx - this.padX(j), ty - this.padY(j)) < 1e-9) return MACRO_BASE + MACRO_MINES + 1;  // a partner's pad: closest class
    const cnt = this.nearestOthers(i);
    let best = -1;
    let bd = 0.15;
    for (let k = 0; k < Math.min(cnt, MACRO_BODIES); k++) {
      const bo = this.bodies[this._bodyOf[this._ordIdx[k]]];
      const d = Math.hypot(tx - bo.x, ty - bo.y);
      if (d < bd) { bd = d; best = k; }
    }
    if (best >= 0) return MACRO_BASE + MACRO_MINES + 2 + best;
    // anything else (a pickup, a predicted position): the nearest mine to the target
    let bm = -1, bmd = 1e9;
    for (let m = 0; m < nm; m++) { const d = Math.hypot(tx - this.mineX[m], ty - this.mineY[m]); if (d < bmd) { bmd = d; bm = m; } }
    return bm >= 0 && bmd < 0.2 ? MACRO_BASE + bm : 0;
  }

  // --------------------------------------------------------------- tick

  // One physics tick. Bot seats decide every decideEvery ticks (like ugb_step); external seats apply
  // their stored input at decision ticks and immediately after setInput.
  tick() {
    if (this.done) return;
    const cfg = this.cfg;
    const n = cfg.n_players;
    const decision = this.tickCount % this.decideEvery === 0;
    const act = this._act;
    for (let i = 0; i < n; i++) {
      if (this.seats[i] >= SEAT_SOLO) {
        if (!decision) continue;
        this._botAction(this.seats[i], i, act, i * 4);
        this.applyAction(i, act[i * 4], act[i * 4 + 1], act[i * 4 + 2], act[i * 4 + 3]);
      } else if (decision || this.inputDirty[i]) {
        const s = this.inputs[i];
        if (s.move === 9) this.setDirection(i, s.dir[0], s.dir[1]);
        this.applyAction(i, s.move, s.join, s.leave, s.intent);
        this.inputDirty[i] = 0;
      }
    }
    this._tickCore();
    this.tickCount++;
    if (this.tickCount % this.decideEvery === 0 || this.done) this._endStep();
  }

  // The C++ tick() body.
  _tickCore() {
    const cfg = this.cfg;
    this.updateArena();
    this.steerMacros();
    this.moveBodies();
    this.updateTimers();
    this.updateLeaving();
    this.collideBodies();
    this.collideMines();
    this.bank();
    this.collectPickups();
    this.regen();
    this.t += cfg.dt;
    const n = cfg.n_players;
    for (let i = 0; i < n; i++) {
      const p = this.players[i];
      let win = true;
      for (let k = 0; k < TYPES; k++) if (p.banked[k] < p.need[k]) { win = false; break; }
      if (win) { this.done = true; this.winner = i; this.event({ kind: 'win', player: i }); break; }
    }
    if (!this.done && this.t >= cfg.time_limit - 1e-9) {
      this.done = true;
      let best = 0;
      for (let i = 1; i < n; i++) {
        const pi = this.progress(i), pb = this.progress(best);
        if (pi > pb + 1e-12 || (Math.abs(pi - pb) <= 1e-12 && this.bankedTotal(i) > this.bankedTotal(best))) best = i;
      }
      this.winner = best;
      this.timeoutWin = true;
      this.event({ kind: 'timeout', player: best });
    }
  }

  // Per-decision accounting (the tail of ugb_step): rewards, mean group size, closing alliances at the end.
  _endStep() {
    const cfg = this.cfg, n = cfg.n_players;
    const pot = this.rewards;  // reuse: rewards = pot - prevPot, written in place
    this.potentials(pot);
    for (let i = 0; i < n; i++) { const r = pot[i] - this.prevPot[i]; this.prevPot[i] = pot[i]; pot[i] = r; }
    if (this.done && this.winner >= 0) {
      for (let i = 0; i < n; i++) pot[i] -= cfg.lose_penalty;
      pot[this.winner] += cfg.lose_penalty + cfg.win_bonus;
    }
    this.stats.steps++;
    let gs = 0;
    for (let i = 0; i < n; i++) gs += this.bodies[this._bodyOf[i]].n();
    this.stats.group += gs / n;
    if (this.done) {
      // close open alliances for the duration statistics
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j && this.pairSince[i * MAX_PLAYERS + j] >= 0) this.endPair(i, j);
    }
  }

  // n ticks (a "step" in the C++ sense clears the event list first). Returns true when the round is over.
  step(nTicks = this.decideEvery) {
    this.events.length = 0;
    for (let k = 0; k < nTicks; k++) { this.tick(); if (this.done) break; }
    return this.done;
  }

  // Events accumulated since the last step()/takeEvents(); clears the list.
  takeEvents() { const e = this.events; this.events = []; return e; }

  updateArena() {
    const cfg = this.cfg;
    const frac = this.t / cfg.time_limit;
    if (frac <= cfg.shrink_start) this.R = 1.0;
    else {
      const k = (frac - cfg.shrink_start) / (1.0 - cfg.shrink_start);
      this.R = 1.0 + (cfg.final_radius - 1.0) * Math.min(k, 1.0);
    }
    for (let m = 0; m < cfg.n_mines; m++) {
      if (this.mineAlive[m] && Math.sqrt(this.mineX[m] * this.mineX[m] + this.mineY[m] * this.mineY[m]) + cfg.mine_radius > this.R) {
        this.mineAlive[m] = 0;
        this.event({ kind: 'mine_dead', mine: m });
      }
    }
  }

  moveBodies() {
    const cfg = this.cfg;
    const a = Math.min(1.0, cfg.vel_lerp * cfg.dt);
    const bodies = this.bodies;
    for (let k = 0; k < bodies.length; k++) {
      const b = bodies[k];
      let mx = 0, my = 0;
      let pushing = 0;
      const mem = b.members;
      for (let q = 0; q < mem.length; q++) {
        const i = mem[q];
        const p = this.players[i];
        if (Math.sqrt(p.dx * p.dx + p.dy * p.dy) > 1e-6) {
          const w = (cfg.crown && mem.length > 1 && i === b.head) ? cfg.head_steer : 1.0;  // the crown steers
          mx += p.dx * w; my += p.dy * w; pushing += w;
        }
      }
      if (pushing > 0) { const inv = 1.0 / pushing; mx *= inv; my *= inv; }
      let tx = 0, ty = 0;
      if (b.stun <= 0) { const s = cfg.base_speed / Math.sqrt(mem.length); tx = mx * s; ty = my * s; }
      b.vx = b.vx + (tx - b.vx) * a;
      b.vy = b.vy + (ty - b.vy) * a;
      b.x = b.x + b.vx * cfg.dt;
      b.y = b.y + b.vy * cfg.dt;
      const r = this.radius(mem.length);
      const d = Math.sqrt(b.x * b.x + b.y * b.y);
      const limit = this.R - r;
      if (d > limit && d > 0) {
        const nx = b.x * (1.0 / d), ny = b.y * (1.0 / d);
        b.x = nx * limit; b.y = ny * limit;
        const vr = b.vx * nx + b.vy * ny;
        if (vr > 0) { b.vx -= nx * vr; b.vy -= ny * vr; }
      }
    }
  }

  updateTimers() {
    const cfg = this.cfg;
    const bodies = this.bodies;
    for (let k = 0; k < bodies.length; k++) { const b = bodies[k]; if (b.stun > 0) b.stun = Math.max(0.0, b.stun - cfg.dt); }
    for (let i = 0; i < cfg.n_players; i++) { const p = this.players[i]; if (p.joinCooldown > 0) p.joinCooldown = Math.max(0.0, p.joinCooldown - cfg.dt); }
    for (let i = 0; i < cfg.n_players; i++) if (this.brand[i] > 0) this.brand[i] = Math.max(0.0, this.brand[i] - cfg.dt);
    // co-membership time
    for (let k = 0; k < bodies.length; k++) {
      const mem = bodies[k].members;
      if (mem.length > 1) {
        for (let a = 0; a < mem.length; a++) for (let c = 0; c < mem.length; c++) if (a !== c) this.comemberTime[mem[a] * MAX_PLAYERS + mem[c]] += cfg.dt;
      }
    }
  }

  updateLeaving() {
    const cfg = this.cfg;
    for (let i = 0; i < cfg.n_players; i++) {
      const p = this.players[i];
      if (p.leaveTimer < 0) continue;
      const bi = this._bodyOf[i];
      if (this.bodies[bi].n() === 1) { p.leaveTimer = -1.0; continue; }
      p.leaveTimer -= cfg.dt;
      if (p.leaveTimer <= 0) { p.leaveTimer = -1.0; this.detach(i, bi); }
    }
  }

  endPair(i, j) {
    const ij = i * MAX_PLAYERS + j;
    if (this.pairSince[ij] >= 0) {
      const dur = this.t - this.pairSince[ij];
      this.stats.alliances++;
      this.stats.alliance_dur += dur;
      if (dur >= 10.0) this.stats.alliances_long++;
      this.pairEnded[ij] = this.t;
      this.pairSince[ij] = -1;
    }
  }

  detach(i, bi) {
    const cfg = this.cfg;
    const b = this.bodies[bi];
    const p = this.players[i];
    const weights = this._weights; weights.fill(0);
    for (const j of b.members) for (let t = 0; t < TYPES; t++) weights[t] += (this.players[j].intent === t) ? cfg.intent_weight : 1.0;
    const share = this._share;
    let taken = 0;
    for (let t = 0; t < TYPES; t++) {
      const w = (p.intent === t) ? cfg.intent_weight : 1.0;
      share[t] = b.pool[t] * (w / weights[t]) * (1.0 - cfg.leave_forfeit);
      b.pool[t] -= share[t];
      taken += share[t];
    }
    const fromSize = b.n();
    const former = [];
    for (const j of b.members) if (j !== i) former.push(j);
    let allLeavers = true;
    for (const j of former) if (!this.publicLeaver(j)) allLeavers = false;
    if (!allLeavers) this.lastUnjustLeave[i] = this.t;
    if (cfg.brand && taken >= cfg.brand_min) this.brand[i] = Math.min(cfg.brand_max, Math.max(this.brand[i], cfg.brand_base + cfg.brand_per_unit * taken));
    b.members.splice(b.members.indexOf(i), 1);
    for (const j of former) {
      this.tookFrom[j * MAX_PLAYERS + i] += taken;
      this.lastLeftMe[j * MAX_PLAYERS + i] = this.t;
      this.partnerCd[i * MAX_PLAYERS + j] = this.t + cfg.partner_cooldown;
      this.partnerCd[j * MAX_PLAYERS + i] = this.t + cfg.partner_cooldown;
      this.endPair(i, j); this.endPair(j, i);
    }
    // Eject opposite to the group's motion (or a random direction when it is still).
    let ux = -b.vx, uy = -b.vy;
    if (Math.sqrt(ux * ux + uy * uy) < 1e-3) {
      let mx = 0, my = 0;
      for (const j of b.members) { mx += this.players[j].dx; my += this.players[j].dy; }
      ux = -mx; uy = -my;
    }
    const un = Math.sqrt(ux * ux + uy * uy);
    if (un < 1e-6) { const ang = this.uniform(0, 2 * PI); ux = Math.cos(ang); uy = Math.sin(ang); }
    else { ux = ux * (1.0 / un); uy = uy * (1.0 / un); }
    const nb = new Body([i]);
    nb.head = i;
    const off = this.radius(b.n()) + this.radius(1) + 0.02;
    nb.x = b.x + ux * off; nb.y = b.y + uy * off;
    nb.vx = ux * 0.2; nb.vy = uy * 0.2;
    for (let t = 0; t < TYPES; t++) nb.pool[t] = share[t];
    p.joinable = false;
    p.joinCooldown = cfg.join_cooldown;
    p.groupSince = this.t;
    this.bodies.push(nb);
    this._bodyOf[i] = this.bodies.length - 1;
    this.crown(this.bodies[bi]);
    this.stats.leaves++;
    this.stats.units_taken += taken;
    this.event({ kind: 'leave', player: i, share: Game.nums(share, TYPES, 2), from_size: fromSize });
  }

  // ---- crown, brand and the public record
  owed(j) {  // units j watched partners receive minus units j received while grouped
    let s = 0;
    for (let i = 0; i < this.cfg.n_players; i++) if (i !== j) s += this.bankedWhile[j * MAX_PLAYERS + i] - this.bankedWhile[i * MAX_PLAYERS + j];
    return s;
  }
  vested(b, j) {
    for (const k of b.members) {
      if (k === j) continue;
      const ps = this.pairSince[j * MAX_PLAYERS + k];
      if (ps < 0 || this.t - ps < this.cfg.head_vest) return false;
    }
    return true;
  }
  branded(j) { return this.cfg.brand !== 0 && this.brand[j] > 0; }
  bodyBranded(b) { for (const j of b.members) if (this.branded(j)) return true; return false; }
  publicLeaver(j) { return this.lastUnjustLeave[j] > NEVER / 2 && this.t - this.lastUnjustLeave[j] < this.cfg.grudge_window; }
  bodyHasLeaver(b) { for (const j of b.members) if (this.publicLeaver(j)) return true; return false; }
  // Head = lowest progress among vested, unbranded members (ties: most owed, then lowest index). If nobody
  // qualifies keep the current head while it is still a member, else relax the brand condition, then vesting.
  crown(b) {
    if (b.n() <= 1) { b.head = b.n() === 1 ? b.members[0] : -1; return; }
    if (!this.cfg.crown) { b.head = b.members[0]; return; }
    const member = (h) => h >= 0 && b.members.indexOf(h) >= 0;
    const pick = (needVest, needClean) => {
      let best = -1;
      for (const j of b.members) {
        if (needVest && !this.vested(b, j)) continue;
        if (needClean && this.branded(j)) continue;
        if (best < 0) { best = j; continue; }
        const pj = this.progress(j), pb = this.progress(best);
        if (pj < pb - 1e-9 || (Math.abs(pj - pb) <= 1e-9 && (this.owed(j) > this.owed(best) + 1e-9 || (Math.abs(this.owed(j) - this.owed(best)) <= 1e-9 && j < best)))) best = j;
      }
      return best;
    };
    let h = pick(true, true);
    if (h < 0 && member(b.head) && !this.branded(b.head)) h = b.head;
    if (h < 0) h = pick(true, false);
    if (h < 0 && member(b.head)) h = b.head;
    if (h < 0) h = pick(false, true);
    if (h < 0) h = pick(false, false);
    if (h !== b.head) { this.stats.crowns++; this.event({ kind: 'crown', player: h, group: b.members.slice() }); }
    b.head = h;
  }

  allJoinable(b) {
    const mem = b.members;
    for (let q = 0; q < mem.length; q++) { const p = this.players[mem[q]]; if (!p.joinable || p.joinCooldown > 0) return false; }
    return true;
  }

  canMerge(a, b) {
    if (a.n() + b.n() > this.cfg.max_group) return false;
    if (!this.allJoinable(a) || !this.allJoinable(b)) return false;
    for (const i of a.members) for (const j of b.members) if (this.partnerCd[i * MAX_PLAYERS + j] > this.t) return false;
    return true;
  }

  collideBodies() {
    const cfg = this.cfg;
    let restart = true;
    let guard = 0;
    while (restart && guard < 20) {
      restart = false;
      guard++;
      const bodies = this.bodies;
      const nb = bodies.length;
      for (let ai = 0; ai < nb && !restart; ai++) {
        for (let bi = ai + 1; bi < nb; bi++) {
          const a = bodies[ai];
          const b = bodies[bi];
          const ra = this.radius(a.n()), rb = this.radius(b.n());
          const dxx = b.x - a.x, dyy = b.y - a.y;
          const dist = Math.sqrt(dxx * dxx + dyy * dyy);
          if (dist >= ra + rb || dist < 1e-9) continue;
          const nx = dxx * (1.0 / dist), ny = dyy * (1.0 / dist);
          if (this.canMerge(a, b)) { this.merge(ai, bi); restart = true; break; }
          const ma = a.n(), mb = b.n();
          const overlap = ra + rb - dist;
          a.x = a.x - nx * (overlap * (mb / (ma + mb))); a.y = a.y - ny * (overlap * (mb / (ma + mb)));
          b.x = b.x + nx * (overlap * (ma / (ma + mb))); b.y = b.y + ny * (overlap * (ma / (ma + mb)));
          const rel = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny;
          if (rel > 0) {
            const j = (1 + cfg.restitution) * rel / (1 / ma + 1 / mb);
            a.vx = a.vx - nx * (j / ma); a.vy = a.vy - ny * (j / ma);
            b.vx = b.vx + nx * (j / mb); b.vy = b.vy + ny * (j / mb);
            if (rel > cfg.spill_min_speed) {
              const cx = a.x + nx * ra, cy = a.y + ny * ra;
              const units = Math.trunc(Math.min(cfg.spill_max, roundHalfEven(cfg.spill_k * (rel - cfg.spill_min_speed) + 1)));
              const sa = this.spill(a, units, cx, cy);
              const sb = this.spill(b, units, cx, cy);
              // A hard bump between empty-handed bodies is just a bounce: no stun, no event.
              if (sa + sb > 0) {
                // The faster body along the normal is the aggressor and is dazed longer.
                const aFaster = (a.vx * nx + a.vy * ny) > -(b.vx * nx + b.vy * ny);
                a.stun = Math.max(a.stun, cfg.stun_time * (aFaster ? cfg.rammer_stun_mult : 1.0));
                b.stun = Math.max(b.stun, cfg.stun_time * (aFaster ? 1.0 : cfg.rammer_stun_mult));
                this.stats.spills++;
                this.event({ kind: 'spill', a: a.members.slice(), b: b.members.slice(), units: sa + sb, speed: r2(rel), x: r3(cx), y: r3(cy) });
              }
            }
          }
        }
      }
    }
  }

  merge(ai, bi) {
    const cfg = this.cfg;
    const a = this.bodies[ai], b = this.bodies[bi];
    const ma = a.n(), mb = b.n();
    const m = new Body(a.members.concat(b.members));
    m.x = (a.x * ma + b.x * mb) * (1.0 / (ma + mb)); m.y = (a.y * ma + b.y * mb) * (1.0 / (ma + mb));
    m.vx = (a.vx * ma + b.vx * mb) * (1.0 / (ma + mb)); m.vy = (a.vy * ma + b.vy * mb) * (1.0 / (ma + mb));
    m.stun = Math.max(a.stun, b.stun);
    for (let t = 0; t < TYPES; t++) m.pool[t] = a.pool[t] + b.pool[t];
    let fast = false;
    for (const i of a.members) for (const j of b.members) {
      const ij = i * MAX_PLAYERS + j, ji = j * MAX_PLAYERS + i;
      if (this.pairEnded[ij] >= 0 && this.t - this.pairEnded[ij] < 3.0) fast = true;
      this.pairSince[ij] = this.t; this.pairSince[ji] = this.t;
    }
    for (const i of m.members) if (this._bodyOf[i] >= 0 && this.bodies[this._bodyOf[i]].n() === 1) this.players[i].groupSince = this.t;
    if (cfg.brand) {  // one-hop contagion: each side inherits half of the other side's worst mark
      let maB = 0, mbB = 0;
      for (const i of a.members) maB = Math.max(maB, this.brand[i]);
      for (const j of b.members) mbB = Math.max(mbB, this.brand[j]);
      for (const i of a.members) this.brand[i] = Math.max(this.brand[i], 0.5 * mbB);
      for (const j of b.members) this.brand[j] = Math.max(this.brand[j], 0.5 * maB);
    }
    m.head = (a.n() >= b.n()) ? a.head : b.head;
    const nb = [];
    for (let k = 0; k < this.bodies.length; k++) if (k !== ai && k !== bi) nb.push(this.bodies[k]);
    nb.push(m);
    this.bodies = nb;
    this._rebuildIndex();
    this.crown(this.bodies[this.bodies.length - 1]);
    this.stats.merges++;
    if (fast) this.stats.remerge_fast++;
    this.event({ kind: 'merge', a: a.members.slice(), b: b.members.slice(), size: m.n() });
  }

  spill(b, units, cx, cy) {
    const cfg = this.cfg;
    const total = b.poolTotal();
    if (total <= 0 || units <= 0) return 0;
    units = Math.min(units, Math.floor(total));
    if (units <= 0) return 0;
    const counts = this._counts; counts.fill(0);
    const probs = this._probs;
    for (let t = 0; t < TYPES; t++) probs[t] = b.pool[t] / total;
    for (let k = 0; k < units; k++) counts[this.rng.discrete(probs)]++;
    let sum = 0;
    for (let t = 0; t < TYPES; t++) {
      counts[t] = Math.min(counts[t], Math.floor(b.pool[t]));
      b.pool[t] -= counts[t];
      sum += counts[t];
    }
    for (let t = 0; t < TYPES; t++) {
      for (let k = 0; k < counts[t]; k++) {
        const ang = this.uniform(0, 2 * PI);
        const rad = this.uniform(0.04, 0.12);
        let px = cx + rad * Math.cos(ang), py = cy + rad * Math.sin(ang);
        const d = Math.sqrt(px * px + py * py);
        if (d > this.R - 0.02) { const s = (this.R - 0.02) / d; px = px * s; py = py * s; }
        this.picks.push({ x: px, y: py, type: t, ttl: cfg.pickup_ttl });
        if (this.picks.length > cfg.max_pickups) this.picks.shift();
      }
    }
    return sum;
  }

  collideMines() {
    const cfg = this.cfg;
    const bodies = this.bodies;
    for (let k = 0; k < bodies.length; k++) {
      const b = bodies[k];
      const r = this.radius(b.n());
      for (let m = 0; m < cfg.n_mines; m++) {
        let dxx = b.x - this.mineX[m], dyy = b.y - this.mineY[m];
        let dist = Math.sqrt(dxx * dxx + dyy * dyy);
        if (dist >= r + cfg.mine_radius + 0.01) continue;
        if (dist < 1e-9) { dxx = 1.0; dyy = 0.0; dist = 1e-9; }
        const nx = dxx * (1.0 / dist), ny = dyy * (1.0 / dist);
        if (this.mineAlive[m] && this.mineStock[m] > 0 && b.stun <= 0) {
          const rate = cfg.mine_rate * Math.pow(b.n(), cfg.mine_exp);
          const amount = Math.min(rate * cfg.dt, this.mineStock[m]);
          this.mineStock[m] -= amount;
          b.pool[this.mineType[m]] += amount;
        }
        if (dist < r + cfg.mine_radius) {
          b.x = this.mineX[m] + nx * (r + cfg.mine_radius); b.y = this.mineY[m] + ny * (r + cfg.mine_radius);
          const vr = b.vx * nx + b.vy * ny;
          if (vr < 0) { b.vx -= nx * vr; b.vy -= ny * vr; }
        }
      }
    }
  }

  bank() {
    const cfg = this.cfg;
    const bodies = this.bodies;
    for (let k = 0; k < bodies.length; k++) {
      const b = bodies[k];
      if (b.poolTotal() < 0.5) continue;
      const r = this.radius(b.n());
      const mem = b.members;
      for (let q = 0; q < mem.length; q++) {
        const i = mem[q];
        if (cfg.crown && b.n() > 1 && i !== b.head) continue;  // the contract: only the head's pad pays
        const px = this.padX(i), py = this.padY(i);
        if (Math.sqrt((b.x - px) * (b.x - px) + (b.y - py) * (b.y - py)) < r + cfg.pad_radius) {
          const amount = this._amount;
          const total = b.poolTotal();
          const mult = 1.0 + cfg.group_bank_bonus * (b.n() - 1);
          let fair = true;
          const pi = this.progress(i);
          for (const j of mem) if (this.progress(j) < pi - 1e-9) fair = false;
          for (let t = 0; t < TYPES; t++) {
            amount[t] = b.pool[t] * mult;
            if (cfg.bank_round) {  // whole units bank; the fraction stays in the pool so "6/6" means six
              const whole = Math.floor(amount[t] + 0.5);
              b.pool[t] = Math.max(0.0, (amount[t] - whole) / mult);
              amount[t] = whole;
            } else b.pool[t] = 0;
            this.players[i].banked[t] += amount[t];
          }
          this.players[i].lastBankT = this.t;
          for (const j of mem) if (j !== i) {
            this.bankedWhile[j * MAX_PLAYERS + i] += total;
            if (cfg.brand && this.brand[j] > 0) this.brand[j] = Math.max(0.0, this.brand[j] - total);  // redemption: a second per unit banked for others
          }
          this.stats.banks++;
          if (b.n() > 1) { this.stats.group_banks++; if (fair) this.stats.fair_banks++; }
          this.event({ kind: 'bank', player: i, amount: Game.nums(amount, TYPES, 2), group: mem.slice() });
          this.crown(b);
          break;
        }
      }
    }
  }

  collectPickups() {
    const cfg = this.cfg;
    const picks = this.picks;
    if (picks.length === 0) return;
    if (this._keep.length < picks.length) this._keep = new Uint8Array(picks.length + 64);
    const keep = this._keep;
    for (let k = 0; k < picks.length; k++) { picks[k].ttl -= cfg.dt; keep[k] = picks[k].ttl > 0 ? 1 : 0; }
    const bodies = this.bodies;
    for (let q = 0; q < bodies.length; q++) {
      const b = bodies[q];
      const r = this.radius(b.n());
      for (let k = 0; k < picks.length; k++) {
        if (!keep[k]) continue;
        const pk = picks[k];
        if (Math.sqrt((pk.x - b.x) * (pk.x - b.x) + (pk.y - b.y) * (pk.y - b.y)) < r + 0.015) { b.pool[pk.type] += 1.0; keep[k] = 0; }
      }
    }
    let w = 0;
    for (let k = 0; k < picks.length; k++) if (keep[k]) picks[w++] = picks[k];
    picks.length = w;
  }

  regen() {
    const cfg = this.cfg;
    const M = cfg.n_mines;
    if (cfg.bloom_rate <= 0) {
      for (let m = 0; m < M; m++)
        if (this.mineAlive[m]) this.mineStock[m] = Math.min(cfg.mine_cap, this.mineStock[m] + cfg.mine_regen * cfg.dt);
      return;
    }
    // Bloom: logistic growth toward K plus seeding from live ring neighbours; a dead ring recovers from the floor term.
    const K = this.mineK();
    const ns = this._mineNext;
    ns.set(this.mineStock);
    for (let m = 0; m < M; m++) {
      if (!this.mineAlive[m]) continue;
      const x = this.mineStock[m] / K;
      const growth = cfg.bloom_rate * this.mineStock[m] * (1.0 - x);
      const free = Math.max(0.0, 1.0 - x);
      let seed = cfg.seed_floor * free;
      const nbr = this.mineNbr[m];
      for (let q = 0; q < nbr.length; q++) { const j = nbr[q]; if (this.mineAlive[j]) seed += cfg.seed_rate * Math.min(this.mineStock[j] / K, 1.0) * free; }
      ns[m] = Math.min(Math.max(0.0, this.mineStock[m] + (growth + seed) * cfg.dt), 1.5 * K);
    }
    const tmp = this.mineStock; this.mineStock = ns; this._mineNext = tmp;
  }

  // -------------------------------------------------------------- reward

  // Writes the per-player potentials into `out` (Float64Array(n)) and returns it.
  potentials(out) {
    const cfg = this.cfg, n = cfg.n_players;
    out = out || new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const b = this.bodies[this._bodyOf[i]];
      const p = this.players[i];
      let bpi = 0, cp = 0;
      for (let t = 0; t < TYPES; t++) {
        bpi += Math.min(p.banked[t] / p.need[t], 1.0);
        cp += Math.min((p.banked[t] + b.pool[t] / b.n()) / p.need[t], 1.0);
      }
      bpi /= TYPES; cp /= TYPES;
      out[i] = 10.0 * bpi + cfg.carried_shaping * (cp - bpi);
      sum += bpi;
      this._ordDist[i] = bpi;  // scratch: banked progress per player
    }
    if (n > 1 && cfg.relative_reward > 0) {
      for (let i = 0; i < n; i++) out[i] -= cfg.relative_reward * 10.0 * (sum - this._ordDist[i]) / (n - 1);
    }
    return out;
  }

  // --------------------------------------------------------- observation

  // v3 appends two features to the own block (is_head, brand) and two per other slot (is_head, brand);
  // obs_legacy=1 emits the v2 layout so checkpoints trained before the package still run.
  ownDim() { return 33 + N_DR + (this.cfg.obs_legacy ? 0 : 2); }
  otherDim() { return 22 + (this.cfg.obs_legacy ? 0 : 2); }
  static get MINE_DIM() { return 8; }
  static get PICK_DIM() { return 7; }
  obsDim() { return this.ownDim() + MAX_SLOTS * this.otherDim() + this.cfg.n_mines * 8 + N_PICKUPS_OBS * 7; }
  static get PRIV_SLOT() { return 17; }
  privDim() { return MAX_SLOTS * 17 + MAX_SLOTS; }

  drFeatures(f, c) {
    const cfg = this.cfg;
    f[c] = cfg.mine_rate / 0.3;
    f[c + 1] = cfg.mine_regen / 1.0;
    f[c + 2] = cfg.leave_time / 4.0;
    f[c + 3] = cfg.need_primary / 24.0;
  }

  // Observation of player i written at f[off..off+obsDim()).
  _observeInto(i, f, off) {
    const cfg = this.cfg, k = cfg.n_players;
    const tfrac = this.t / cfg.time_limit;
    let c = off;
    const p = this.players[i];
    const b = this.bodies[this._bodyOf[i]];
    const px = b.x, py = b.y;
    f[c++] = px; f[c++] = py; f[c++] = b.vx; f[c++] = b.vy;
    for (let t = 0; t < TYPES; t++) f[c++] = p.need[t] / cfg.need_primary;
    for (let t = 0; t < TYPES; t++) f[c++] = Math.min(p.banked[t] / p.need[t], 1.5);
    for (let t = 0; t < TYPES; t++) f[c++] = (p.intent === t) ? 1 : 0;
    f[c++] = p.joinable ? 1 : 0;
    f[c++] = b.n() / cfg.max_group;
    for (let t = 0; t < TYPES; t++) f[c++] = b.pool[t] / 20.0;
    f[c++] = p.leaveTimer >= 0 ? p.leaveTimer / cfg.leave_time : 0;
    f[c++] = this.R;
    f[c++] = tfrac;
    f[c++] = this.padX(i) - px; f[c++] = this.padY(i) - py;
    f[c++] = b.poolTotal() / 40.0;
    f[c++] = this.progress(i);
    f[c++] = Math.min((this.t - p.groupSince) / 60.0, 1.0);        // time in current membership state
    f[c++] = Math.min((this.t - p.lastBankT) / 60.0, 1.0);         // time since last bank
    f[c++] = p.joinCooldown / Math.max(cfg.join_cooldown, 1e-6);
    f[c++] = b.stun / Math.max(cfg.stun_time, 1e-6);
    this.drFeatures(f, c); c += N_DR;
    if (!cfg.obs_legacy) { f[c++] = (b.n() > 1 && b.head === i) ? 1 : 0; f[c++] = Math.min(this.brand[i] / 60.0, 1.0); }
    // Others sorted by distance, stable on index; MAX_SLOTS slots with presence flag.
    const cnt = this.nearestOthers(i);
    let slot = 0;
    for (let q = 0; q < cnt && slot < MAX_SLOTS; q++) {
      const j = this._ordIdx[q];
      const qp = this.players[j];
      const bj = this.bodies[this._bodyOf[j]];
      const ij = i * MAX_PLAYERS + j;
      f[c++] = bj.x - px; f[c++] = bj.y - py;
      f[c++] = bj.vx; f[c++] = bj.vy;
      f[c++] = bj.n() / cfg.max_group;
      f[c++] = qp.joinable ? 1 : 0;
      for (let t = 0; t < TYPES; t++) f[c++] = (qp.intent === t) ? 1 : 0;
      f[c++] = (this._bodyOf[j] === this._bodyOf[i]) ? 1 : 0;
      f[c++] = qp.leaveTimer >= 0 ? qp.leaveTimer / cfg.leave_time : 0;
      f[c++] = this.progress(j);
      f[c++] = this.padX(j) - px; f[c++] = this.padY(j) - py;
      f[c++] = bj.poolTotal() / 40.0;
      f[c++] = 1;  // present
      f[c++] = Math.min(this.comemberTime[ij] / 60.0, 1.0);
      f[c++] = Math.min(this.tookFrom[ij] / 10.0, 1.0);
      f[c++] = Math.min(this.bankedWhile[ij] / 10.0, 1.0);
      f[c++] = this.lastLeftMe[ij] < NEVER / 2 ? 1 : Math.min(Math.max(this.t - this.lastLeftMe[ij], 0.0) / 60.0, 1.0);
      f[c++] = (this.partnerCd[ij] > this.t || qp.joinCooldown > 0) ? 1 : 0;
      if (!cfg.obs_legacy) { f[c++] = (bj.n() > 1 && bj.head === j) ? 1 : 0; f[c++] = Math.min(this.brand[j] / 60.0, 1.0); }
      slot++;
    }
    const od = this.otherDim();
    for (; slot < MAX_SLOTS; slot++) for (let z = 0; z < od; z++) f[c++] = 0;
    const K = this.mineK();
    for (let m = 0; m < cfg.n_mines; m++) {
      f[c++] = this.mineX[m] - px; f[c++] = this.mineY[m] - py;
      for (let t = 0; t < TYPES; t++) f[c++] = (this.mineType[m] === t) ? 1 : 0;
      f[c++] = this.mineStock[m] / K;
      f[c++] = this.mineAlive[m] ? 1 : 0;
    }
    // nearest pickups (stable on insertion order)
    const picks = this.picks;
    const po = [];
    for (let q = 0; q < picks.length; q++) po.push([Math.hypot(picks[q].x - px, picks[q].y - py), q]);
    po.sort((a, b2) => a[0] - b2[0] || a[1] - b2[1]);
    const cntp = Math.min(po.length, N_PICKUPS_OBS);
    for (let q = 0; q < cntp; q++) {
      const pk = picks[po[q][1]];
      f[c++] = pk.x - px; f[c++] = pk.y - py;
      for (let t = 0; t < TYPES; t++) f[c++] = (pk.type === t) ? 1 : 0;
      f[c++] = 1;
    }
    for (let q = cntp; q < N_PICKUPS_OBS; q++) for (let z = 0; z < 7; z++) f[c++] = 0;
    return c - off;
  }

  observe(i) { const f = new Float32Array(this.obsDim()); this._observeInto(i, f, 0); return f; }
  observeAll() {
    const D = this.obsDim(), n = this.cfg.n_players;
    const f = new Float32Array(n * D);
    for (let i = 0; i < n; i++) this._observeInto(i, f, i * D);
    return f;
  }

  // Privileged per-seat block for the critic: every seat in fixed index order (present, need,
  // banked/need, pool share, seat-type one-hot [external, snapshot, bot, unused]) + own index.
  observePriv(i) {
    const cfg = this.cfg, k = cfg.n_players;
    const f = new Float32Array(this.privDim());
    let c = 0;
    for (let s = 0; s < MAX_SLOTS; s++) {
      if (s < k) {
        const q = this.players[s];
        const b = this.bodies[this._bodyOf[s]];
        f[c++] = 1;
        for (let t = 0; t < TYPES; t++) f[c++] = q.need[t] / cfg.need_primary;
        for (let t = 0; t < TYPES; t++) f[c++] = Math.min(q.banked[t] / q.need[t], 1.5);
        for (let t = 0; t < TYPES; t++) f[c++] = b.pool[t] / b.n() / 20.0;
        const st = this.seats[s] === SEAT_EXTERNAL ? 0 : this.seats[s] === SEAT_EXTERNAL2 ? 1 : 2;
        for (let z = 0; z < 4; z++) f[c++] = (z === st) ? 1 : 0;
      } else {
        for (let z = 0; z < 17; z++) f[c++] = 0;
      }
    }
    for (let s = 0; s < MAX_SLOTS; s++) f[c++] = (s === i) ? 1 : 0;
    return f;
  }

  // ---------------------------------------------------------------- bots

  nearestMine(x, y, types) {
    let best = -1;
    let bd = 1e9;
    for (let m = 0; m < this.cfg.n_mines; m++) {
      if (!this.mineAlive[m] || this.mineStock[m] < 1.0) continue;
      if (types && !types[this.mineType[m]]) continue;
      const d = Math.hypot(this.mineX[m] - x, this.mineY[m] - y);
      if (d < bd) { best = m; bd = d; }
    }
    return best;
  }

  // Fills out[t] (Uint8Array) and returns whether any type is still needed.
  neededTypes(i, out) {
    const b = this.bodies[this._bodyOf[i]];
    const p = this.players[i];
    let any = false;
    for (let t = 0; t < TYPES; t++) {
      out[t] = p.banked[t] + b.pool[t] / b.n() < p.need[t] ? 1 : 0;
      if (out[t]) any = true;
    }
    return any;
  }

  primaryOf(i) {
    const p = this.players[i];
    let pr = 0;
    for (let t = 1; t < TYPES; t++) if (p.need[t] > p.need[pr]) pr = t;
    return pr;
  }

  intentFor(i) {
    const needed = this._needed;
    if (this.neededTypes(i, needed)) { let k = 0; while (!needed[k]) k++; return k; }
    return this.primaryOf(i);
  }

  // Writes the target into out and returns whether one was found.
  mineTarget(i, x, y, out) {
    const needed = this._needed;
    const any = this.neededTypes(i, needed);
    let m = this.nearestMine(x, y, any ? needed : null);
    if (m < 0) m = this.nearestMine(x, y, null);
    if (m >= 0) { out.x = this.mineX[m]; out.y = this.mineY[m]; } else { out.x = 0; out.y = 0; }
    return m >= 0;
  }

  // Group consensus target: the nearest live mine of a type the member furthest behind still needs
  // (falls back to any needed type, then any mine). Every member computes the same answer.
  groupMineTarget(b, out) {
    let behind = b.members[0];
    for (const j of b.members) if (this.progress(j) < this.progress(behind)) behind = j;
    const needed = this._needed;
    const any = this.neededTypes(behind, needed);
    let m = this.nearestMine(b.x, b.y, any ? needed : null);
    if (m < 0) {
      const anyneed = this._needed2; anyneed.fill(0);
      let anyany = false;
      for (const j of b.members) { const nj = this._needed; this.neededTypes(j, nj); for (let t = 0; t < TYPES; t++) if (nj[t]) { anyneed[t] = 1; anyany = true; } }
      m = this.nearestMine(b.x, b.y, anyany ? anyneed : null);
    }
    if (m < 0) m = this.nearestMine(b.x, b.y, null);
    if (m >= 0) { out.x = this.mineX[m]; out.y = this.mineY[m]; } else { out.x = 0; out.y = 0; }
    return m >= 0;
  }

  nearestJoinableBody(i, x, y, maxd) {
    let best = -1;
    let bd = maxd;
    for (let k = 0; k < this.bodies.length; k++) {
      const b = this.bodies[k];
      if (b.members.indexOf(i) >= 0) continue;
      if (!this.allJoinable(b)) continue;
      const d = Math.hypot(b.x - x, b.y - y);
      if (d < bd) { best = k; bd = d; }
    }
    return best;
  }

  botSolo(i, act, o) {
    const b = this.bodies[this._bodyOf[i]];
    const bankAt = 0.22 * this.needTotal(i);
    let has = false;
    const target = this._botTgt;
    if (b.poolTotal() >= bankAt) { this.padPos(i, target); has = true; }
    else {
      has = this.mineTarget(i, b.x, b.y, target);
      if (!has && b.poolTotal() > 0) { this.padPos(i, target); has = true; }
    }
    this._botHas = has;
    act[o] = has ? directionToMove(target.x - b.x, target.y - b.y) : 0;
    act[o + 1] = 0;
    act[o + 2] = b.n() > 1 ? 1 : 0;
    act[o + 3] = this.primaryOf(i) + 1;
  }

  botBail(i, act, o) {
    const maxGroupBot = 3;
    const padDanger = 0.3;
    const cfg = this.cfg;
    const b = this.bodies[this._bodyOf[i]];
    const bankAt = 0.28 * this.needTotal(i), bailAt = 0.11 * this.needTotal(i);
    const joinable = b.n() < maxGroupBot ? 1 : 0;
    let leave = 0;
    let has = false;
    const target = this._botTgt;
    const mpx = this.padX(i), mpy = this.padY(i);
    if (b.n() > 1) {
      const myShare = b.poolTotal() / b.n();
      for (const j of b.members) {
        if (j === i) continue;
        if (Math.hypot(this.padX(j) - b.x, this.padY(j) - b.y) < padDanger && b.poolTotal() >= bailAt) leave = 1;
      }
      if (b.poolTotal() >= bankAt) {
        if (cfg.crown && b.head >= 0) this.padPos(b.head, target); else { target.x = mpx; target.y = mpy; }
        has = true;
      }
      if (myShare >= bailAt && Math.hypot(mpx - b.x, mpy - b.y) > 0.6 && leave === 0) leave = 1;
    } else {
      if (b.poolTotal() >= bailAt) { target.x = mpx; target.y = mpy; has = true; }
    }
    if (!has) {
      if (b.n() > 1) has = this.groupMineTarget(b, target); else has = this.mineTarget(i, b.x, b.y, target);
      if (!has) { target.x = mpx; target.y = mpy; }
    }
    this._botHas = true;
    act[o] = directionToMove(target.x - b.x, target.y - b.y);
    act[o + 1] = joinable;
    act[o + 2] = leave;
    act[o + 3] = this.intentFor(i) + 1;
  }

  // Loyal: always joinable, never leaves, banks the group at whichever member pad is nearest.
  botLoyal(i, act, o) {
    const cfg = this.cfg;
    const b = this.bodies[this._bodyOf[i]];
    const bankAt = 0.28 * this.needTotal(i);
    let has = false;
    const target = this._botTgt;
    if (b.n() > 1 && b.poolTotal() >= bankAt) {
      // Fair rotation: bank at the pad of the member who is furthest behind (the head, under the crown).
      let best = 2.0;
      for (const j of b.members) { const pj = this.progress(j); if (pj < best) { best = pj; this.padPos(j, target); has = true; } }
      if (cfg.crown && b.head >= 0) this.padPos(b.head, target);
    } else if (b.n() === 1) {
      if (b.poolTotal() >= bankAt) { this.padPos(i, target); has = true; }
      else {
        let k = this.nearestJoinableBody(i, b.x, b.y, 0.6);
        if (k >= 0 && this.bodyBranded(this.bodies[k])) k = -1;
        if (k >= 0 && this.players[i].joinCooldown <= 0) { target.x = this.bodies[k].x; target.y = this.bodies[k].y; has = true; }
      }
    }
    if (!has) {
      if (b.n() > 1) has = this.groupMineTarget(b, target); else has = this.mineTarget(i, b.x, b.y, target);
      if (!has) this.padPos(i, target);
    }
    let joinable = 1;
    if (cfg.brand) {  // shun: close the door while a branded body is within reach
      for (const ob of this.bodies) if (ob !== b && this.bodyBranded(ob) && Math.hypot(ob.x - b.x, ob.y - b.y) < 0.25 + this.radius(b.n()) + this.radius(ob.n())) joinable = 0;
    }
    this._botHas = true;
    act[o] = directionToMove(target.x - b.x, target.y - b.y);
    act[o + 1] = joinable;
    act[o + 2] = 0;
    act[o + 3] = this.intentFor(i) + 1;
  }

  // Grudge: loyal, but never seeks and never admits anyone who publicly left a partner within grudge_window s
  // (with the persistent ledger a leave in one round is remembered in the next). Leaving a leaver is not held against you.
  botGrudge(i, act, o) {
    const cfg = this.cfg;
    const b = this.bodies[this._bodyOf[i]];
    const bankAt = 0.28 * this.needTotal(i);
    let has = false;
    const target = this._botTgt;
    let joinable = 1;
    if (b.n() > 1 && b.poolTotal() >= bankAt) {
      let best = 2.0;
      for (const j of b.members) { const pj = this.progress(j); if (pj < best) { best = pj; this.padPos(j, target); has = true; } }
      if (cfg.crown && b.head >= 0) this.padPos(b.head, target);
    } else if (b.n() === 1) {
      if (b.poolTotal() >= bankAt) { this.padPos(i, target); has = true; }
      else if (this.players[i].joinCooldown <= 0) {
        let bd = 0.6;
        for (const ob of this.bodies) {
          if (ob === b || !this.allJoinable(ob) || this.bodyHasLeaver(ob) || this.bodyBranded(ob)) continue;
          const d = Math.hypot(ob.x - b.x, ob.y - b.y);
          if (d < bd) { bd = d; target.x = ob.x; target.y = ob.y; has = true; }
        }
      }
    }
    for (const ob of this.bodies)
      if (ob !== b && Math.hypot(ob.x - b.x, ob.y - b.y) < 0.25 + this.radius(b.n()) + this.radius(ob.n()) && (this.bodyHasLeaver(ob) || this.bodyBranded(ob))) joinable = 0;
    if (!has) {
      if (b.n() > 1) has = this.groupMineTarget(b, target); else has = this.mineTarget(i, b.x, b.y, target);
      if (!has) this.padPos(i, target);
    }
    this._botHas = true;
    act[o] = directionToMove(target.x - b.x, target.y - b.y);
    act[o + 1] = joinable;
    act[o + 2] = 0;
    act[o + 3] = this.intentFor(i) + 1;
  }

  // Kidnapper: always joinable, never leaves, drags any laden group to its own pad.
  botKidnap(i, act, o) {
    const b = this.bodies[this._bodyOf[i]];
    let has = false;
    const target = this._botTgt;
    if (b.n() > 1 && b.poolTotal() >= 6.0) { this.padPos(i, target); has = true; }
    else if (b.n() === 1) {
      if (b.poolTotal() >= 6.0) { this.padPos(i, target); has = true; }
      else {
        const k = this.nearestJoinableBody(i, b.x, b.y, 0.8);
        if (k >= 0 && this.players[i].joinCooldown <= 0) { target.x = this.bodies[k].x; target.y = this.bodies[k].y; has = true; }
      }
    }
    if (!has) {
      if (b.n() > 1) has = this.groupMineTarget(b, target); else has = this.mineTarget(i, b.x, b.y, target);
      if (!has) this.padPos(i, target);
    }
    this._botHas = true;
    act[o] = directionToMove(target.x - b.x, target.y - b.y);
    act[o + 1] = 1;
    act[o + 2] = 0;
    act[o + 3] = this.intentFor(i) + 1;
  }

  // Rammer: never joins, hunts the fullest body, collects the spill, banks small loads.
  botRammer(i, act, o) {
    const b = this.bodies[this._bodyOf[i]];
    let has = false;
    const target = this._botTgt;
    if (b.poolTotal() >= 4.0) { this.padPos(i, target); has = true; }
    if (!has) {
      let bd = 0.3;
      for (const pk of this.picks) { const d = Math.hypot(pk.x - b.x, pk.y - b.y); if (d < bd) { bd = d; target.x = pk.x; target.y = pk.y; has = true; } }
    }
    if (!has) {
      let best = 2.0;
      for (const ob of this.bodies) {
        if (ob === b) continue;
        if (ob.poolTotal() > best) { best = ob.poolTotal(); target.x = ob.x + ob.vx * 0.3; target.y = ob.y + ob.vy * 0.3; has = true; }
      }
    }
    if (!has) { has = this.mineTarget(i, b.x, b.y, target); if (!has) this.padPos(i, target); }
    this._botHas = true;
    act[o] = directionToMove(target.x - b.x, target.y - b.y);
    act[o + 1] = 0;
    act[o + 2] = b.n() > 1 ? 1 : 0;
    act[o + 3] = this.primaryOf(i) + 1;
  }

  _botAction(seatType_, i, act, o) {
    switch (seatType_) {
      case SEAT_SOLO: this.botSolo(i, act, o); break;
      case SEAT_BAIL: this.botBail(i, act, o); break;
      case SEAT_LOYAL: this.botLoyal(i, act, o); break;
      case SEAT_KIDNAP: this.botKidnap(i, act, o); break;
      case SEAT_RAMMER: this.botRammer(i, act, o); break;
      case SEAT_GRUDGE: this.botGrudge(i, act, o); break;
      default: act[o] = 0; act[o + 1] = 0; act[o + 2] = 0; act[o + 3] = 0;
    }
  }

  // What a scripted bot of the given type (name or index) would do for seat i: [move, join, leave, intent].
  // With macro=true the move column is the macro target class (ugb_bot_actions_macro).
  botAction(seat, i, macro = false) {
    const st = seatType(seat);
    const act = new Int32Array(4);
    this._botAction(st, i, act, 0);
    if (macro) act[0] = this.macroLabel(i);
    return Array.from(act);
  }

  // ------------------------------------------------------------ recording

  // Same content and rounding as frame_json() in the C++.
  frame() {
    const cfg = this.cfg;
    const bodies = [];
    for (const b of this.bodies) {
      bodies.push({ m: b.members.slice(), x: r3(b.x), y: r3(b.y), vx: r3(b.vx), vy: r3(b.vy), stun: r1(b.stun), pool: Game.nums(b.pool, TYPES, 1), head: b.head });
    }
    const mines = Game.nums(this.mineStock, cfg.n_mines, 1);
    const alive = [];
    for (let m = 0; m < cfg.n_mines; m++) alive.push(this.mineAlive[m] !== 0);
    const picks = this.picks.map((pk) => [r3(pk.x), r3(pk.y), pk.type]);
    const players = [];
    for (let i = 0; i < cfg.n_players; i++) {
      const p = this.players[i];
      players.push({
        banked: Game.nums(p.banked, TYPES, 1), intent: p.intent, join: p.joinable,
        leaving: r1(p.leaveTimer >= 0 ? p.leaveTimer : -1.0), cd: r1(p.joinCooldown), dir: [r2(p.dx), r2(p.dy)],
        brand: r1(this.brand[i]), leaver: this.publicLeaver(i),
      });
    }
    return { t: r3(this.t), R: r3(this.R), bodies, mines, alive, picks, players, events: this.events.slice() };
  }

  meta() {
    const cfg = this.cfg;
    const needs = [], pads = [], minePos = [], mineType = [];
    for (let i = 0; i < cfg.n_players; i++) { needs.push(Game.nums(this.players[i].need, TYPES, 0)); pads.push(r6(this.players[i].padAngle)); }
    for (let m = 0; m < cfg.n_mines; m++) { minePos.push([r6(this.mineX[m]), r6(this.mineY[m])]); mineType.push(this.mineType[m]); }
    return { seed: this.seed, needs, pads, mine_pos: minePos, mine_type: mineType, winner: this.winner, timeout_win: this.timeoutWin, cfg: configToArray(cfg) };
  }

  // Round statistics in the ep_stats convention of ugb_step (alliances counted per pair once).
  get statsSummary() {
    const s = this.stats;
    return {
      steps: s.steps, avgGroup: s.group / Math.max(1, s.steps), merges: s.merges, leaves: s.leaves, spills: s.spills,
      banks: s.banks, groupBanks: s.group_banks, remergeFast: s.remerge_fast, cancels: s.cancels,
      alliances: s.alliances / 2.0, allianceDur: s.alliances ? s.alliance_dur / s.alliances : 0,
      alliancesLong: s.alliances_long / 2.0, fairBanks: s.fair_banks, crowns: s.crowns, unitsTaken: s.units_taken,
    };
  }
}
