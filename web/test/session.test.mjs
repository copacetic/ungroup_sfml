// session.test.mjs - Host/Client protocol tests over an in-process FakeTransport with latency and jitter,
// driven by a fake clock. Run: node web/test/session.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TransportBase } from '../src/net.js';
import { Host, Client, interpolateFrame, extrapolateFrame, SNAP_EVERY, RESTART_DELAY_MS } from '../src/session.js';

// ------------------------------------------------------------------------------------ fake network
// A hub connects N endpoints; every message is JSON round-tripped (like the wire) and delivered after
// latency +- jitter, preserving order per (from, to) link like a data channel or a BroadcastChannel.
class FakeHub {
  constructor({ latency = 30, jitter = 10, seed = 1 } = {}) {
    this.latency = latency; this.jitter = jitter;
    this.now = 0;
    this.queue = [];
    this.endpoints = new Map();
    this.lastDeliver = new Map();
    this.rngState = seed >>> 0 || 1;
    this.sent = 0;
  }
  rand() { let x = this.rngState; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.rngState = x >>> 0; return (this.rngState % 10000) / 10000; }
  delay(from, to) {
    const d = this.latency + (this.rand() * 2 - 1) * this.jitter;
    const key = from + '>' + to;
    const at = Math.max(this.now + d, this.lastDeliver.get(key) || 0);
    this.lastDeliver.set(key, at);
    return at;
  }
  endpoint(id) { const e = new FakeTransport(this, id); return e; }
  enqueue(from, to, kind, payload) {
    this.sent++;
    this.queue.push({ at: this.delay(from, to), from, to, kind, payload: payload === undefined ? undefined : JSON.parse(JSON.stringify(payload)) });
  }
  // deliver everything due at or before `now`
  deliver(now) {
    this.now = now;
    if (!this.queue.length) return;
    this.queue.sort((a, b) => a.at - b.at);
    while (this.queue.length && this.queue[0].at <= now) {
      const m = this.queue.shift();
      const e = this.endpoints.get(m.to);
      if (!e || e.closed) continue;
      if (m.kind === 'join') { if (e._addPeer(m.from) && this.endpoints.get(m.from) && !this.endpoints.get(m.from).closed) this.enqueue(m.to, m.from, 'ack'); }
      else if (m.kind === 'ack') e._addPeer(m.from);
      else if (m.kind === 'leave') e._removePeer(m.from);
      else e._dispatch(m.payload.ch, m.payload.data, m.from);
    }
  }
}
class FakeTransport extends TransportBase {
  constructor(hub, id) {
    super(id);
    this.hub = hub;
    hub.endpoints.set(id, this);
    for (const [pid, p] of hub.endpoints) if (pid !== id && !p.closed) hub.enqueue(id, pid, 'join');
  }
  send(channel, obj, toId = null) {
    if (this.closed) return;
    const targets = toId === null ? this.peers() : (this._peers.has(toId) ? [toId] : []);
    for (const t of targets) this.hub.enqueue(this.id, t, 'msg', { ch: channel, data: obj });
  }
  close() {
    if (this.closed) return;
    super.close();
    for (const p of this.peers()) this.hub.enqueue(this.id, p, 'leave');
    this.hub.endpoints.delete(this.id);
  }
}

// ------------------------------------------------------------------------------------ world driver
function makeWorld(opts = {}) {
  const hub = new FakeHub(opts);
  const clock = { now: 0 };
  const now = () => clock.now;
  const actors = [];
  const world = {
    hub, clock, now, actors,
    run(untilMs, step = 5, each = null) {
      while (clock.now < untilMs) {
        clock.now += step;
        hub.deliver(clock.now);
        for (const a of actors) a.pump(clock.now);
        if (each) each(clock.now);
      }
    },
    host(settings, id = 'H') {
      const h = new Host(hub.endpoint(id), settings, { now, manual: true });
      actors.push(h);
      return h;
    },
    client(id, name) {
      const c = new Client(hub.endpoint(id), { now, manual: true, name });
      c.log = [];
      c.onLobby(() => c.log.push('lobby'));
      c.onStart(() => c.log.push('start'));
      c.onEnd(() => c.log.push('end'));
      c.onHostLeft(() => c.log.push('hostleft'));
      actors.push(c);
      return c;
    },
  };
  return world;
}

const SETTINGS = { humans: 3, bots: ['solo', 'bail'], agents: 0, preset: 'legacy', overrides: {}, seed: 7, rounds: 0, name: 'Hosty' };

test('lobby, start, snapshots at 15 Hz, interpolated view lags the host by 90-200 ms', () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  const c2 = w.client('C2', 'Bob');
  w.run(300);
  assert.equal(c1.log[0], 'lobby');
  assert.equal(c2.log[0], 'lobby');
  assert.equal(c1.seat, 1); assert.equal(c2.seat, 2);
  assert.equal(c1.spectator, false);
  assert.deepEqual(host.lobby().seats.map((s) => s.type), ['human', 'human', 'human', 'bot', 'bot']);
  assert.equal(host.lobby().seats[1].name, 'Alice');

  host.start();
  const t0 = w.clock.now;
  const samples = [];
  const snapRecv = { C1: [], C2: [] };
  c1.onSnapshot((s) => snapRecv.C1.push(w.clock.now));
  c2.onSnapshot((s) => snapRecv.C2.push(w.clock.now));
  let lastSample = 0;
  w.run(t0 + 10000, 5, (now) => {
    if (now - lastSample >= 33) {
      lastSample = now;
      for (const c of [c1, c2]) {
        const f = c.view(now);
        if (f) samples.push({ c: c.id, now, viewT: f.t, hostT: host.game.t, since: now - t0 });
      }
    }
  });
  for (const c of [c1, c2]) {
    assert.ok(c.log.indexOf('lobby') < c.log.indexOf('start'), 'lobby before start: ' + c.log.join(','));
    assert.ok(c.startMsg && c.startMsg.meta && c.startMsg.meta.needs.length === 5, 'start carries meta');
    assert.equal(c.startMsg.names[1], 'Alice');
  }
  // host ticked 30 Hz for 10 s
  assert.ok(host.tick >= 295 && host.tick <= 301, 'host ticks ' + host.tick);
  assert.equal(host.snapCount, Math.floor(host.tick / SNAP_EVERY));
  // 15 Hz snapshots at each client (minus the ones still in flight at the end)
  for (const id of ['C1', 'C2']) {
    const n = snapRecv[id].length;
    assert.ok(n >= 140 && n <= 155, id + ' received ' + n + ' snapshots in 10 s');
    const gaps = snapRecv[id].slice(1).map((v, i) => v - snapRecv[id][i]);
    const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    assert.ok(Math.abs(mean - 1000 / 15) < 4, 'mean snapshot gap ' + mean.toFixed(1) + ' ms');
  }
  // view t is monotonic and lags the host by 90..200 ms once warmed up
  for (const id of ['C1', 'C2']) {
    const mine = samples.filter((s) => s.c === id);
    assert.ok(mine.length > 250, 'samples ' + mine.length);
    for (let i = 1; i < mine.length; i++) assert.ok(mine[i].viewT >= mine[i - 1].viewT, 'monotonic t at ' + mine[i].now);
    const warm = mine.filter((s) => s.since > 1000);
    const lags = warm.map((s) => (s.hostT - s.viewT) * 1000);
    const minLag = Math.min(...lags), maxLag = Math.max(...lags);
    assert.ok(minLag >= 90 && maxLag <= 200, id + ' view lag range ' + minLag.toFixed(0) + '..' + maxLag.toFixed(0) + ' ms');
    // the view actually advances (~ real time)
    const span = warm[warm.length - 1].viewT - warm[0].viewT;
    assert.ok(Math.abs(span - (warm[warm.length - 1].now - warm[0].now) / 1000) < 0.1, 'view advances in real time: ' + span);
  }
  assert.equal(host.running, true);
});

test("a client's input reaches its seat within 3 ticks", () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  const c2 = w.client('C2', 'Bob');
  w.run(300);
  host.start();
  w.run(w.clock.now + 1000);
  const seat = c1.seat;
  assert.equal(seat, 1);
  const sentAtTick = host.tick;
  c1.sendInput({ dir: [0.6, -0.8], join: 1, leave: 0, intent: 3 });
  let arrivedTick = -1;
  w.run(w.clock.now + 500, 1, () => {
    if (arrivedTick < 0 && host.game.players[seat].joinable) arrivedTick = host.tick;
  });
  assert.ok(arrivedTick >= 0, 'input arrived');
  assert.ok(arrivedTick - sentAtTick <= 3, 'ticks to arrive: ' + (arrivedTick - sentAtTick));
  const p = host.game.players[seat];
  assert.ok(Math.abs(p.dx - 0.6) < 1e-6 && Math.abs(p.dy + 0.8) < 1e-6, 'direction applied ' + p.dx + ',' + p.dy);
  assert.equal(p.intent, 2, 'intent 3 -> type index 2');
  // 20 Hz rate limit + 500 ms keepalive: count input messages over 2 s of unchanged input
  const before = w.hub.sent;
  const inputs = [];
  host.transport.on('input', (o, from) => { if (from === 'C1') inputs.push(w.clock.now); });
  w.run(w.clock.now + 2000);
  assert.ok(inputs.length >= 3 && inputs.length <= 5, 'keepalive inputs in 2 s: ' + inputs.length);
  // rapid changes are coalesced to <= 20 Hz
  inputs.length = 0;
  for (let k = 0; k < 100; k++) { c1.sendInput({ dir: [Math.cos(k), Math.sin(k)], join: 1 }); w.run(w.clock.now + 10); }
  w.run(w.clock.now + 200);
  assert.ok(inputs.length <= 21 && inputs.length >= 15, 'rate-limited inputs during 1 s of changes: ' + inputs.length);
  // the host's own seat takes input directly
  host.setInput({ dir: [-1, 0], join: 1 });
  w.run(w.clock.now + 100);
  assert.equal(host.game.players[0].joinable, true);
  assert.ok(host.game.players[0].dx < -0.99);
  void before;
});

test('late joiner gets start with the current meta and becomes a spectator', () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  const c2 = w.client('C2', 'Bob');
  w.run(300);
  host.start();
  w.run(w.clock.now + 5000);
  const c3 = w.client('C3', 'Carol');
  w.run(w.clock.now + 300);
  assert.equal(c3.log[0], 'lobby');
  assert.ok(c3.log.includes('start'), 'late joiner got start: ' + c3.log.join(','));
  assert.equal(c3.spectator, true);
  assert.equal(c3.seat, -1);
  assert.deepEqual(c3.startMsg.meta.needs, host.game.meta().needs);
  assert.deepEqual(c3.startMsg.meta.pads, host.game.meta().pads);
  assert.deepEqual(c3.startMsg.meta.mine_pos, host.game.meta().mine_pos);
  assert.ok(host.lobby().spectators.includes('C3'));
  // spectators receive snapshots and can view, but never send input
  w.run(w.clock.now + 500);
  assert.ok(c3.snapCount > 5);
  assert.ok(c3.view(w.clock.now));
  const before = w.hub.sent;
  c3.sendInput({ dir: [1, 0] });
  w.run(w.clock.now + 600);
  const hostInputsFromC3 = [];
  host.transport.on('input', (o, from) => { if (from === 'C3') hostInputsFromC3.push(o); });
  w.run(w.clock.now + 600);
  assert.equal(hostInputsFromC3.length, 0);
  void before;
  // a seat freed by a leaving player is handed to a bot, and a newcomer can take it back
  c2.close();
  w.run(w.clock.now + 300);
  const lob = host.lobby();
  assert.equal(lob.seats[2].type, 'bot');
  assert.equal(lob.seats[2].bot, 'solo');
  assert.ok(/Bob left; seat 3 is now a solo bot/.test(lob.notice), lob.notice);
  assert.equal(host.game.seatNames[2], 'solo');
  const c4 = w.client('C4', 'Dave');
  w.run(w.clock.now + 300);
  assert.equal(c4.seat, 2);
  assert.equal(c4.spectator, false);
  assert.equal(host.game.seatNames[2], 'human');
  void c1;
});

test('host leaving is reported to every client', () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  const c2 = w.client('C2', 'Bob');
  w.run(300);
  host.start();
  w.run(w.clock.now + 1000);
  host.close();
  w.run(w.clock.now + 300);
  for (const c of [c1, c2]) {
    assert.equal(c.hostLeft, true);
    assert.ok(c.log.includes('hostleft'), c.log.join(','));
  }
  // the view freezes gracefully (extrapolation capped at 150 ms, then still)
  const f1 = c1.view(w.clock.now), f2 = c1.view(w.clock.now + 5000);
  assert.ok(f1 && f2);
  assert.ok(f2.t - f1.t <= 0.151 && f2.t >= f1.t);
});

test('bots-only host runs without any client, ends and restarts after 8 s', () => {
  const clock = { now: 0 };
  const host = new Host(null, { humans: 0, bots: ['solo', 'bail', 'loyal', 'rammer'], preset: 'legacy', seed: 3, overrides: { time_limit: 4 }, rounds: 2 }, { now: () => clock.now, manual: true });
  const frames = [], ends = [], starts = [];
  host.onFrame((f) => frames.push(f));
  host.onEnd((e) => ends.push({ e, at: clock.now }));
  host.onStart((s) => starts.push({ s, at: clock.now }));
  assert.equal(host.lobby().seats.length, 4);
  assert.equal(host.lobby().seats.every((s) => s.type === 'bot'), true);
  host.start();
  for (; clock.now < 20000; clock.now += 7) host.pump(clock.now);
  assert.equal(starts.length, 2, 'two rounds started');
  assert.equal(ends.length, 2, 'two rounds ended');
  assert.ok(ends[0].e.timeoutWin === true && ends[0].e.winner >= 0);
  assert.equal(ends[0].e.progress.length, 4);
  assert.ok(Math.abs(ends[0].at - 4000) < 50, 'round 1 ended at ' + ends[0].at);
  assert.ok(Math.abs(starts[1].at - (ends[0].at + RESTART_DELAY_MS)) < 20, 'restart after 8 s: ' + starts[1].at);
  assert.equal(starts[1].s.round, 2);
  assert.notEqual(starts[1].s.meta.seed, starts[0].s.meta.seed);
  assert.equal(host.finished, true, 'series of 2 rounds is over');
  assert.equal(host.running, false);
  assert.ok(frames.length >= 238 && frames.length <= 242, 'frames ' + frames.length);
  // frames follow the schema and carry per-tick events exactly once
  const f = frames[10];
  for (const k of ['t', 'R', 'bodies', 'mines', 'alive', 'picks', 'players', 'events']) assert.ok(k in f, k);
  assert.equal(f.bodies[0].m.length >= 1, true);
  assert.equal(f.players.length, 4);
  const evs = frames.flatMap((fr) => fr.events);
  assert.equal(evs.filter((e) => e.kind === 'timeout').length, 2);
});

test('catch-up is capped at 10 ticks per pump and a throttled host keeps time later', () => {
  const clock = { now: 0 };
  const host = new Host(null, { humans: 0, bots: ['solo', 'solo'], preset: 'legacy', seed: 5 }, { now: () => clock.now, manual: true });
  host.start();
  clock.now = 2000;   // the tab slept for 2 s
  host.pump(clock.now);
  assert.equal(host.tick, 10);
  clock.now = 2033.4;
  host.pump(clock.now);
  assert.equal(host.tick, 11);
  for (; clock.now < 5000; clock.now += 33.3333) host.pump(clock.now);
  assert.ok(host.tick >= 98 && host.tick <= 101, 'tick ' + host.tick);
});

test('interpolation and extrapolation helpers', () => {
  const mk = (t, x, vx, m = [0]) => ({ t, R: 1, bodies: [{ m, x, y: 0, vx, vy: 0, stun: 0, pool: [1, 0, 0, 0], head: 0 }], mines: [10], alive: [true], picks: [], players: [{ banked: [0, 0, 0, 0], intent: 0, join: false, leaving: -1, cd: 0, dir: [1, 0], brand: 0, leaver: false }], events: [{ kind: 'x' }] });
  const a = mk(1.0, 0.0, 0.3), b = mk(1.1, 0.03, 0.3);
  const m = interpolateFrame(a, b, 0.5);
  assert.ok(Math.abs(m.t - 1.05) < 1e-9 && Math.abs(m.bodies[0].x - 0.015) < 1e-9);
  assert.deepEqual(m.events, []);
  // an unmatched body (after a merge) takes the newer frame's position
  const c = mk(1.1, 0.5, 0, [0, 1]);
  const m2 = interpolateFrame(a, c, 0.1);
  assert.equal(m2.bodies[0].x, 0.5);
  const e = extrapolateFrame(b, 0.1);
  assert.ok(Math.abs(e.bodies[0].x - 0.06) < 1e-9 && Math.abs(e.t - 1.2) < 1e-9);
});

test('client view emits each snapshot event exactly once, in order', () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host({ ...SETTINGS, humans: 2, bots: ['rammer', 'kidnap', 'loyal', 'bail'], overrides: { time_limit: 6 } });
  const c1 = w.client('C1', 'Alice');
  w.run(300);
  host.start();
  const hostEvents = [];
  host.onFrame((f) => hostEvents.push(...f.events));
  const seen = [];
  w.run(w.clock.now + 7000, 5, (now) => { const f = c1.view(now); if (f) seen.push(...f.events); });
  assert.ok(c1.ended);
  assert.ok(hostEvents.some((e) => e.kind === 'timeout'));
  assert.equal(seen.length, hostEvents.length, 'client saw every event once: ' + seen.length + ' vs ' + hostEvents.length);
  assert.deepEqual(seen.map((e) => e.kind + '@' + e.t), hostEvents.map((e) => e.kind + '@' + e.t));
});
