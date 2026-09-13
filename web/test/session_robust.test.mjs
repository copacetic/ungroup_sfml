// session_robust.test.mjs - adversarial robustness tests for web/src/session.js and web/src/net.js over
// an in-process FakeTransport (latency, jitter, per-channel packet loss, silent crashes, throttled tabs),
// driven by a fake clock. Run: node web/test/session_robust.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { TransportBase } from '../src/net.js';
import { Host, Client, SNAP_EVERY, TICK_HZ, MAX_CATCHUP, INPUT_TIMEOUT_MS, HOST_TIMEOUT_MS, SNAP_MAX_BYTES } from '../src/session.js';

// ------------------------------------------------------------------------------------ fake network
// Every message is JSON round-tripped (like the wire) and delivered after latency +- jitter; order is
// preserved per (from, to) link like a data channel or a BroadcastChannel. loss = { channel: prob }.
class FakeHub {
  constructor({ latency = 30, jitter = 10, seed = 1, loss = {}, lossIfEvents = 0 } = {}) {
    this.latency = latency; this.jitter = jitter; this.loss = loss; this.lossIfEvents = lossIfEvents;
    this.now = 0;
    this.queue = [];
    this.endpoints = new Map();
    this.lastDeliver = new Map();
    this.rngState = seed >>> 0 || 1;
    this.sent = 0;
    this.dropped = 0;
    this.sentBy = {};      // channel -> count
    this.maxBytes = {};    // channel -> largest JSON payload
  }
  rand() { let x = this.rngState; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; this.rngState = x >>> 0; return (this.rngState % 10000) / 10000; }
  delay(from, to) {
    const d = this.latency + (this.rand() * 2 - 1) * this.jitter;
    const key = from + '>' + to;
    const at = Math.max(this.now + d, this.lastDeliver.get(key) || 0);
    this.lastDeliver.set(key, at);
    return at;
  }
  endpoint(id) { return new FakeTransport(this, id); }
  enqueue(from, to, kind, payload) {
    this.sent++;
    let json;
    if (payload !== undefined) {
      json = JSON.stringify(payload);
      const ch = payload.ch;
      this.sentBy[ch] = (this.sentBy[ch] || 0) + 1;
      if (json.length > (this.maxBytes[ch] || 0)) this.maxBytes[ch] = json.length;
      if (this.loss[ch] && this.rand() < this.loss[ch]) { this.dropped++; return; }
      // adversarial: snapshots that carry events are the ones that get lost
      if (this.lossIfEvents && ch === 'snap' && payload.data.frame.events.length && (this.evSnaps = (this.evSnaps || 0) + 1) % 2 === 1) { this.dropped++; this.droppedWithEvents = (this.droppedWithEvents || 0) + 1; return; }
    }
    this.queue.push({ at: this.delay(from, to), from, to, kind, payload: json === undefined ? undefined : JSON.parse(json) });
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
  // a tab that dies without saying goodbye (no 'leave' on the wire; peers never learn unless told later)
  vanish(id) {
    const e = this.endpoints.get(id);
    if (!e) return;
    e.closed = true;
    this.endpoints.delete(id);
  }
  // the transport layer finally notices a vanished peer (heartbeat timeout / ICE failure)
  lateLeave(id) { for (const [pid, p] of this.endpoints) if (pid !== id && !p.closed) this.enqueue(id, pid, 'leave'); }
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
    // actors with .paused are not pumped; actors with .pumpEvery only every that many ms
    run(untilMs, step = 5, each = null) {
      while (clock.now < untilMs) {
        clock.now += step;
        hub.deliver(clock.now);
        for (const a of actors) {
          if (a.paused) continue;
          if (a.pumpEvery) { if (clock.now - (a._lastPumped || -Infinity) < a.pumpEvery) continue; a._lastPumped = clock.now; }
          a.pump(clock.now);
        }
        if (each) each(clock.now);
      }
    },
    host(settings, id = 'H') {
      const h = new Host(hub.endpoint(id), settings, { now, manual: true });
      actors.push(h);
      return h;
    },
    client(id, name, opts = {}) {
      const c = new Client(hub.endpoint(id), Object.assign({ now, manual: true, name }, opts));
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

// Sample a client's view every `every` ms while running the world; returns [{now, t, hostT}].
function sampleView(w, client, host, untilMs, every = 33) {
  const out = [];
  let last = -Infinity;
  w.run(untilMs, 5, (now) => {
    if (now - last >= every) {
      last = now;
      const f = client.view(now);
      if (f) out.push({ now, t: f.t, hostT: host.game ? host.game.t : NaN });
    }
  });
  return out;
}
// Smoothness statistics of a view sample series: steps are t deltas between consecutive samples.
function smoothness(samples, { jump = 0.15 } = {}) {
  let backwards = 0, stalls = 0, jumps = 0, maxStep = 0, minStep = Infinity;
  for (let i = 1; i < samples.length; i++) {
    const d = samples[i].t - samples[i - 1].t;
    if (d < -1e-9) backwards++;
    if (Math.abs(d) < 1e-9) stalls++;
    if (d > jump) jumps++;
    maxStep = Math.max(maxStep, d); minStep = Math.min(minStep, d);
  }
  const span = samples.length ? samples[samples.length - 1].t - samples[0].t : 0;
  const real = samples.length ? (samples[samples.length - 1].now - samples[0].now) / 1000 : 0;
  return { backwards, stalls, jumps, maxStep, minStep, span, real, n: samples.length, stallFrac: samples.length > 1 ? stalls / (samples.length - 1) : 0 };
}
function seatedPeers(host) { return host.lobby().seats.map((s) => s.peer).filter((p) => p !== null); }

const SETTINGS = { humans: 3, bots: ['solo', 'bail'], agents: 0, preset: 'legacy', overrides: {}, seed: 7, rounds: 0, name: 'Hosty' };
const WAN = { latency: 300, jitter: 100, loss: { snap: 0.05 }, lossIfEvents: true, seed: 11 };

// ================================================================================ scenarios
test('WAN: 300 ms latency, 100 ms jitter, 5 % snapshot loss - the view still advances smoothly and never goes back', () => {
  const w = makeWorld(WAN);
  // a busy lineup (rammers spill, kidnappers merge, the clients chase other players) so that events flow
  const host = w.host({ ...SETTINGS, bots: ['rammer', 'rammer', 'kidnap', 'loyal', 'bail', 'solo'] });
  const c1 = w.client('C1', 'Alice');
  const c2 = w.client('C2', 'Bob');
  w.run(1500);
  assert.deepEqual([c1.seat, c2.seat].sort(), [1, 2], 'both seated (either order under jitter)');
  host.start();
  const hostEvents = [];
  host.onFrame((f) => hostEvents.push(...f.events));
  const t0 = w.clock.now;
  c1.sendInput({ move: 20, join: 1, intent: 1 });   // chase the nearest player
  c2.sendInput({ move: 21, join: 1, intent: 2 });
  host.setInput({ move: 10, join: 1 });
  // warm up 2 s, then sample 28 s at ~30 fps
  w.run(t0 + 2000);
  const seenEvents = { C1: [], C2: [] };
  const samples = { C1: [], C2: [] };
  let last = -Infinity;
  w.run(t0 + 30000, 5, (now) => {
    if (now - last >= 33) {
      last = now;
      for (const c of [c1, c2]) {
        const f = c.view(now);
        if (f) { samples[c.id].push({ now, t: f.t, hostT: host.game.t }); seenEvents[c.id].push(...f.events); }
      }
    } else for (const c of [c1, c2]) { const f = c.view(now); if (f) seenEvents[c.id].push(...f.events); }
  });
  assert.ok(w.hub.dropped > 20, 'the hub actually dropped snapshots: ' + w.hub.dropped);
  for (const id of ['C1', 'C2']) {
    const s = smoothness(samples[id]);
    assert.ok(s.n >= 700, id + ' samples ' + s.n);
    assert.equal(s.backwards, 0, id + ' view went backwards ' + s.backwards + ' times');
    assert.ok(s.maxStep <= 0.1 + 1e-9, id + ' largest step between 33 ms samples ' + (s.maxStep * 1000).toFixed(0) + ' ms');
    assert.ok(s.stallFrac < 0.03, id + ' stalled in ' + (s.stallFrac * 100).toFixed(1) + ' % of samples');
    assert.ok(Math.abs(s.span - s.real) < 0.25, id + ' view span ' + s.span.toFixed(2) + ' s over ' + s.real.toFixed(2) + ' s real');
    const lags = samples[id].map((q) => (q.hostT - q.t) * 1000);
    const maxLag = Math.max(...lags), minLag = Math.min(...lags);
    assert.ok(minLag >= 300 && maxLag <= 1200, id + ' view lag range ' + minLag.toFixed(0) + '..' + maxLag.toFixed(0) + ' ms');
  }
  // events lost with a dropped snapshot are recovered from the following snapshots (each seen exactly once)
  const hk = hostEvents.map((e) => e.kind + '@' + e.t);
  for (const id of ['C1', 'C2']) {
    const ck = seenEvents[id].map((e) => e.kind + '@' + e.t);
    // the client may still be ~1 s behind the host at the end; compare the prefix it has had time to see
    const prefix = hk.slice(0, ck.length);
    assert.ok(hk.length >= 3, 'the lineup produced events: ' + hk.length);
    assert.ok(w.hub.droppedWithEvents >= 2, 'event-carrying snapshots were dropped: ' + w.hub.droppedWithEvents);
    assert.ok(ck.length >= hk.length - 4, id + ' saw ' + ck.length + ' of ' + hk.length + ' host events (' + w.hub.dropped + ' snapshots dropped, ' + w.hub.droppedWithEvents + ' with events)');
    assert.equal(c1.eventsLost + c2.eventsLost, 0, 'no event fell outside the replay window');
    assert.deepEqual(ck, prefix, id + ' events in order without duplicates');
  }
});

test('a client that stops sending input keeps its last input for at most 1 s, then its seat stops', () => {
  const w = makeWorld({ latency: 300, jitter: 100, seed: 3 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  w.run(1500);
  host.start();
  w.run(w.clock.now + 1000);
  c1.sendInput({ dir: [1, 0], join: 1, intent: 2 });
  w.run(w.clock.now + 1500);
  const p = host.game.players[1];
  assert.ok(p.dx > 0.99, 'moving right before the freeze');
  assert.equal(p.joinable, true);
  // the client's tab freezes: no more input messages (keepalives included), transport still connected
  c1.paused = true;
  let lastInputAt = -Infinity, stoppedAt = -1;
  host.transport.on('input', (o, from) => { if (from === 'C1') lastInputAt = w.clock.now; });
  w.run(w.clock.now + 3000, 1, (now) => { if (stoppedAt < 0 && p.dx === 0 && p.dy === 0) stoppedAt = now; });
  assert.ok(stoppedAt > 0, 'the seat stopped moving after the client went silent');
  assert.ok(stoppedAt - lastInputAt <= INPUT_TIMEOUT_MS + 2 * 1000 / TICK_HZ, 'stopped ' + (stoppedAt - lastInputAt).toFixed(0) + ' ms after the last input');
  assert.ok(stoppedAt - lastInputAt >= 500, 'not stopped before the keepalive period elapsed: ' + (stoppedAt - lastInputAt).toFixed(0) + ' ms');
  assert.equal(p.joinable, true, 'joinable flag is kept, only the motion stops');
  assert.equal(host.lobby().seats[1].type, 'human', 'the seat is still the human seat (not handed to a bot)');
  // the client comes back: its seat moves again
  c1.paused = false;
  c1.sendInput({ dir: [0, -1], join: 1, intent: 2 });
  w.run(w.clock.now + 1000);
  assert.ok(p.dy < -0.99, 'moving again after the client resumed: ' + p.dx + ',' + p.dy);
  // the host's own seat has no keepalive and must never be timed out
  host.setInput({ dir: [-1, 0], join: 1 });
  w.run(w.clock.now + 3000);
  assert.ok(host.game.players[0].dx < -0.99, 'host seat still moving');
});

test('refresh: a client that joins twice gets its seat back (or a new one) without a duplicate seat', () => {
  // (a) the old tab said goodbye before the new tab said hello
  let w = makeWorld({ latency: 30, jitter: 10 });
  let host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  let c2 = w.client('C2', 'Bob', { token: 'bob-token' });
  w.run(500);
  host.start();
  w.run(w.clock.now + 1000);
  const seatA = c2.seat;
  assert.ok(seatA === 1 || seatA === 2);
  c2.close(); w.actors.splice(w.actors.indexOf(c2), 1);
  w.run(w.clock.now + 200);
  assert.equal(host.lobby().seats[seatA].type, 'bot');
  let c2b = w.client('C2b', 'Bob', { token: 'bob-token' });
  w.run(w.clock.now + 500);
  assert.equal(c2b.seat, seatA, 'the same seat back after a refresh');
  assert.equal(c2b.spectator, false);
  assert.equal(host.game.seatNames[seatA], 'human');
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C2b', 'H'], 'no duplicated seat');
  assert.ok(c2b.log.includes('start'), 'the refreshed tab got the running round');
  // (b) the new tab says hello BEFORE the transport notices the old tab is gone (WebRTC: seconds later)
  w = makeWorld({ latency: 30, jitter: 10 });
  host = w.host({ ...SETTINGS, humans: 4 });
  w.client('C1', 'Alice');
  c2 = w.client('C2', 'Bob', { token: 'bob-token' });
  const c3 = w.client('C3', 'Carol');   // takes the fourth seat; nobody is free
  w.run(500);
  host.start();
  w.run(w.clock.now + 1000);
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C2', 'C3', 'H']);
  w.hub.vanish('C2'); w.actors.splice(w.actors.indexOf(c2), 1);
  c2b = w.client('C2b', 'Bob', { token: 'bob-token' });
  w.run(w.clock.now + 500);
  assert.equal(c2b.seat, host.lobby().seats.findIndex((q) => q.peer === 'C2b'), 'client and host agree on the seat');
  assert.ok(c2b.seat >= 1 && c2b.seat <= 3 && !c2b.spectator, 'the returning token reclaims its seat even while the stale peer is still listed');
  const bobSeat = c2b.seat;
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C2b', 'C3', 'H']);
  // the stale peer's inputs are ignored, and its late 'leave' does not touch the reclaimed seat
  w.hub.lateLeave('C2');
  w.run(w.clock.now + 500);
  assert.equal(host.lobby().seats[bobSeat].type, 'human');
  assert.equal(host.lobby().seats[bobSeat].peer, 'C2b');
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C2b', 'C3', 'H']);
  assert.equal(host.game.seatNames[bobSeat], 'human');
  // (c) a refresh without a token while no seat is free: the new tab waits as a spectator, the old seat
  // goes to a bot when the stale peer is finally dropped, and the spectator's next hello takes it back
  w = makeWorld({ latency: 30, jitter: 10 });
  host = w.host({ ...SETTINGS, humans: 4 });
  w.client('C1', 'Alice');
  c2 = w.client('C2', 'Bob');
  w.client('C3', 'Carol');
  w.run(500);
  host.start();
  w.run(w.clock.now + 500);
  w.hub.vanish('C2'); w.actors.splice(w.actors.indexOf(c2), 1);
  c2b = w.client('C2b', 'Bob');
  w.run(w.clock.now + 500);
  assert.equal(c2b.spectator, true);
  const oldSeat = host.lobby().seats.findIndex((q) => q.peer === 'C2');
  w.hub.lateLeave('C2');
  w.run(w.clock.now + 500);
  assert.equal(host.lobby().seats[oldSeat].type, 'bot');
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C3', 'H']);
  c2b.setReady(true);   // any hello from a spectator takes a free seat
  w.run(w.clock.now + 300);
  assert.equal(c2b.seat, oldSeat, 'the spectator took the freed seat: ' + JSON.stringify(host.lobby().seats));
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C2b', 'C3', 'H']);
  // (d) a client saying hello repeatedly (rename, ready toggles) never gets a second seat
  c2b.setName('Robert'); c2b.setReady(true); c2b.setName('Bob');
  w.run(w.clock.now + 300);
  assert.deepEqual(seatedPeers(host).sort(), ['C1', 'C2b', 'C3', 'H']);
  assert.equal(host.lobby().seats[oldSeat].name, 'Bob');
  c3.close();
  void c1;
});

test('two peers say hello at the same instant: distinct seats, one spectator when only one seat is free', () => {
  const w = makeWorld({ latency: 50, jitter: 0 });
  const host = w.host({ ...SETTINGS, humans: 2 });   // host + 1 free seat
  const a = w.client('A', 'Ann');
  const b = w.client('B', 'Ben');
  w.run(1000);
  const seats = host.lobby().seats;
  const seated = [a, b].filter((c) => !c.spectator);
  const spect = [a, b].filter((c) => c.spectator);
  assert.equal(seated.length, 1); assert.equal(spect.length, 1);
  assert.equal(seats[1].peer, seated[0].id);
  assert.equal(seated[0].seat, 1);
  assert.ok(host.lobby().spectators.includes(spect[0].id));
  // with two free seats both get one, and each client's own idea of its seat matches the host's
  const w2 = makeWorld({ latency: 50, jitter: 0 });
  const host2 = w2.host({ ...SETTINGS, humans: 3 });
  const c = w2.client('C', 'Cid'), d = w2.client('D', 'Dee');
  w2.run(1000);
  assert.notEqual(c.seat, d.seat);
  assert.ok(c.seat > 0 && d.seat > 0);
  assert.equal(host2.lobby().seats[c.seat].peer, 'C');
  assert.equal(host2.lobby().seats[d.seat].peer, 'D');
  assert.equal(new Set(seatedPeers(host2)).size, seatedPeers(host2).length, 'no peer holds two seats');
});

test('host tab throttled to 1 Hz for 3 s then resumed: catch-up capped, clients see at most one jump', () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Alice');
  w.run(300);
  host.start();
  const t0 = w.clock.now;
  const before = sampleView(w, c1, host, t0 + 2000);
  const tickBefore = host.tick;
  host.pumpEvery = 1000;
  const during = sampleView(w, c1, host, t0 + 5000);
  const tickDuring = host.tick - tickBefore;
  assert.ok(tickDuring <= 3 * MAX_CATCHUP + 1 && tickDuring >= 2 * MAX_CATCHUP, 'ticks during the throttle ' + tickDuring);
  delete host.pumpEvery;
  const resumedAt = w.clock.now;
  const after = sampleView(w, c1, host, resumedAt + 4000);
  const all = before.concat(during, after);
  const s = smoothness(all);
  assert.equal(s.backwards, 0, 'view went backwards');
  assert.ok(s.jumps <= 1, 'jumps (> 150 ms between 33 ms samples): ' + s.jumps);
  // the host runs at 30 Hz again and the client settles back into a smooth, lagging view
  const settled = after.filter((q) => q.now > resumedAt + 2500);
  const ss = smoothness(settled);
  assert.equal(ss.stalls, 0, 'stalls after settling: ' + ss.stalls);
  assert.ok(ss.maxStep <= 0.075, 'max step after settling ' + ss.maxStep);
  assert.ok(Math.abs(ss.span - ss.real) < 0.05, 'real-time playback after settling: ' + ss.span + ' vs ' + ss.real);
  const lags = settled.map((q) => (q.hostT - q.t) * 1000);
  assert.ok(Math.min(...lags) >= 90 && Math.max(...lags) <= 250, 'lag after settling ' + Math.min(...lags).toFixed(0) + '..' + Math.max(...lags).toFixed(0));
  assert.ok(host.tick >= 30 * 4 + 30 * 2 + 20, 'host kept ticking after resume: ' + host.tick);
});

test('host leaving mid-round: every client sees host-left within 2 s (clean close and silent crash)', () => {
  // clean close (the transport reports the leave)
  let w = makeWorld({ latency: 300, jitter: 100, seed: 5 });
  let host = w.host(SETTINGS);
  let c1 = w.client('C1', 'Alice'), c2 = w.client('C2', 'Bob');
  w.run(1500);
  host.start();
  w.run(w.clock.now + 1000);
  let leftAt = w.clock.now;
  host.close(); w.actors.splice(w.actors.indexOf(host), 1);
  let seen = {};
  w.run(leftAt + 2500, 5, (now) => { for (const c of [c1, c2]) if (c.hostLeft && !seen[c.id]) seen[c.id] = now; });
  for (const c of [c1, c2]) assert.ok(seen[c.id] && seen[c.id] - leftAt <= 2000, c.id + ' host-left after ' + (seen[c.id] - leftAt) + ' ms');
  // silent crash (no leave on the wire; e.g. a killed tab on WebRTC where ICE takes ~10 s to fail)
  w = makeWorld({ latency: 300, jitter: 100, seed: 6 });
  host = w.host(SETTINGS);
  c1 = w.client('C1', 'Alice'); c2 = w.client('C2', 'Bob');
  const spec = w.client('C3', 'Carol');
  w.run(1500);
  host.start();
  w.run(w.clock.now + 1000);
  const c4 = w.client('C4', 'Dave');   // arrives mid-round: spectator
  w.run(w.clock.now + 1000);
  assert.equal(c4.spectator, true);
  leftAt = w.clock.now;
  w.hub.vanish('H'); w.actors.splice(w.actors.indexOf(host), 1);
  seen = {};
  const clients = [c1, c2, spec, c4];
  w.run(leftAt + 3000, 5, (now) => { for (const c of clients) if (c.hostLeft && !seen[c.id]) seen[c.id] = now; });
  for (const c of clients) assert.ok(seen[c.id] && seen[c.id] - leftAt <= 2000, c.id + ' host-left after ' + (seen[c.id] - leftAt) + ' ms');
  for (const c of clients) assert.ok(c.log.includes('hostleft'), c.id + ' callback fired');
  // a live host (even one throttled to 1 Hz) is never reported as gone, before or between rounds
  w = makeWorld({ latency: 300, jitter: 100, seed: 8 });
  host = w.host({ ...SETTINGS, overrides: { time_limit: 3 } });
  c1 = w.client('C1', 'Alice');
  w.run(4000);                    // 4 s in the lobby
  assert.equal(c1.hostLeft, false, 'lobby idle: host not reported gone');
  host.start();
  host.pumpEvery = 1000;
  w.run(w.clock.now + 5000);      // throttled round: ends at t=3 s game time and waits for the restart
  assert.equal(c1.hostLeft, false, 'throttled host not reported gone');
  delete host.pumpEvery;
  w.run(w.clock.now + 12000);     // the round ends, 8 s pause, next round
  assert.equal(c1.hostLeft, false, 'between rounds: host not reported gone');
  assert.ok(c1.log.filter((x) => x === 'start').length >= 2, 'rounds keep coming: ' + c1.log.join(','));
});

test('20 peers join a room with 4 human seats: 16 spectators, snapshots still 15 Hz for everyone', () => {
  const w = makeWorld({ latency: 30, jitter: 10 });
  const host = w.host({ ...SETTINGS, humans: 4, hostPlays: false, bots: ['solo', 'bail', 'loyal'] });
  const clients = [];
  for (let i = 0; i < 20; i++) { clients.push(w.client('P' + String(i).padStart(2, '0'), 'peer' + i)); w.run(w.clock.now + 25); }
  w.run(w.clock.now + 1000);
  const lob = host.lobby();
  assert.equal(lob.seats.filter((s) => s.type === 'human' && s.peer).length, 4);
  assert.equal(lob.spectators.length, 16);
  assert.equal(clients.filter((c) => !c.spectator).length, 4);
  assert.equal(clients.filter((c) => c.spectator).length, 16);
  assert.equal(new Set(seatedPeers(host)).size, 4);
  assert.ok(w.hub.sentBy.lobby < 3000, 'lobby fan-out stays bounded: ' + w.hub.sentBy.lobby);
  host.start();
  w.run(w.clock.now + 1000);
  const counts = new Map(clients.map((c) => [c.id, 0]));
  for (const c of clients) c.onSnapshot(() => counts.set(c.id, counts.get(c.id) + 1));
  const from = w.clock.now;
  w.run(from + 3000);
  for (const c of clients) {
    const n = counts.get(c.id);
    assert.ok(n >= 43 && n <= 47, c.id + ' got ' + n + ' snapshots in 3 s');
    const f = c.view(w.clock.now);
    assert.ok(f && f.players.length === 7, c.id + ' can view');
  }
  assert.ok(Math.abs(host.tick - (w.clock.now - from + 1000) / (1000 / TICK_HZ)) < 3, 'host ticks at 30 Hz with 20 peers: ' + host.tick);
  // a seated peer leaving frees its seat (a bot fills in); a spectator's next hello takes it, nobody else moves
  const seatedOne = clients.find((c) => !c.spectator);
  const spectator = clients.find((c) => c.spectator);
  seatedOne.close(); w.actors.splice(w.actors.indexOf(seatedOne), 1);
  w.run(w.clock.now + 300);
  assert.equal(clients.filter((c) => c !== seatedOne && !c.spectator).length, 3);
  spectator.setReady(true);
  w.run(w.clock.now + 300);
  assert.equal(spectator.spectator, false, 'spectator seated on its next hello');
  assert.equal(host.lobby().spectators.length, 15);
  assert.equal(new Set(seatedPeers(host)).size, 4);
});

test('a 32-player snapshot stays under 16 KB on the wire', () => {
  const w = makeWorld({ latency: 10, jitter: 0 });
  const bots = [];
  for (let i = 0; i < 32; i++) bots.push(['solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge'][i % 6]);
  const host = w.host({ humans: 0, bots, preset: 'life', seed: 11, overrides: { time_limit: 30 } });
  const c = w.client('C', 'Watcher');
  w.run(300);
  assert.equal(host.n, 32);
  host.start();
  w.run(w.clock.now + 12000);
  assert.ok(host.snapCount > 150, 'snapshots ' + host.snapCount);
  assert.ok(w.hub.maxBytes.snap < 16384, 'largest snapshot ' + w.hub.maxBytes.snap + ' bytes');
  assert.ok(w.hub.maxBytes.snap <= SNAP_MAX_BYTES, 'largest snapshot ' + w.hub.maxBytes.snap + ' bytes <= budget ' + SNAP_MAX_BYTES);
  assert.ok(c.view(w.clock.now).bodies.length >= 1);
  assert.ok(w.hub.maxBytes.start < 16384, 'start message ' + w.hub.maxBytes.start);
});

test('hostile input: non-finite directions, out-of-range intent and macro classes never poison the host state', () => {
  const w = makeWorld({ latency: 10, jitter: 0 });
  const host = w.host(SETTINGS);
  const c1 = w.client('C1', 'Mallory');
  w.run(300);
  host.start();
  w.run(w.clock.now + 500);
  const raw = (obj) => c1.transport.send('input', obj, 'H');
  raw({ dir: [1e400, 0], join: 1 });                      // JSON 1e400 -> Infinity
  w.run(w.clock.now + 100);
  raw({ dir: ['x', null], intent: 99, move: 999, join: 'yes', leave: {} });
  w.run(w.clock.now + 100);
  raw({ dir: [1e9, -1e9], intent: -3, move: 20 + 30 });
  w.run(w.clock.now + 100);
  raw({ dir: [0, 0], intent: 4, move: 12 });              // a legal macro: go to mine 2
  w.run(w.clock.now + 300);
  const p = host.game.players[1];
  assert.equal(p.macro, 12);
  raw({ dir: [0, 0], intent: 0, move: 0 });               // stop must cancel the macro
  w.run(w.clock.now + 300);
  assert.equal(p.macro, -1, 'move 0 cancels a macro target');
  assert.ok(p.dx === 0 && p.dy === 0);
  raw(null); raw(42); raw({ dir: 'nope' });
  w.run(w.clock.now + 2000);
  const f = host.frame();
  const nums = [];
  for (const b of f.bodies) nums.push(b.x, b.y, b.vx, b.vy, ...b.pool);
  for (const q of f.players) nums.push(...q.banked, ...q.dir, q.intent);
  assert.ok(nums.every((v) => Number.isFinite(v)), 'no NaN/Infinity in the host frame');
  assert.ok(f.players.every((q) => q.intent >= 0 && q.intent <= 3), 'intents in range');
  assert.equal(host.running, true);
});
