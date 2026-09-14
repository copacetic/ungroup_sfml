// net.test.mjs - unit tests for the transport wrappers without a network: LocalTransport over Node's
// global BroadcastChannel, RtcTransport over a mocked Trystero room (both the 0.25 object-style action
// API and the older [send, receive] tuple API). Run: node web/test/net.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LocalTransport, RtcTransport, MultiTransport, DirectTransport, TRYSTERO_IMPORT, TRYSTERO_STRATEGIES, DEFAULT_RTC_CONFIG, createTransport, encodeSignal, decodeSignal } from '../src/net.js';

const tick = (ms = 20) => new Promise((r) => setTimeout(r, ms));

test('LocalTransport: peers discover each other, unicast/broadcast and leave over BroadcastChannel', async () => {
  const room = 'nettest-' + Math.random().toString(36).slice(2, 8);
  const a = new LocalTransport(room, { id: 'A' });
  const b = new LocalTransport(room, { id: 'B' });
  await tick();
  assert.deepEqual(a.peers(), ['B']);
  assert.deepEqual(b.peers(), ['A']);
  const c = new LocalTransport(room, { id: 'C' });
  const seenByA = [];
  a.onPeer((id) => seenByA.push(id));   // replays existing peers, then new ones
  await tick();
  assert.deepEqual(seenByA.sort(), ['B', 'C']);
  assert.deepEqual(c.peers().sort(), ['A', 'B']);
  const got = { A: [], B: [], C: [] };
  for (const t of [a, b, c]) t.on('hello', (obj, from) => got[t.id].push([from, obj.name]));
  a.send('hello', { name: 'to-everyone' });
  b.send('hello', { name: 'only-c' }, 'C');
  b.send('hello', { name: 'dropped' }, 'nobody');
  await tick();
  assert.deepEqual(got.A, []);
  assert.deepEqual(got.B, [['A', 'to-everyone']]);
  assert.deepEqual(got.C.sort(), [['A', 'to-everyone'], ['B', 'only-c']]);
  const left = [];
  a.onLeave((id) => left.push(id));
  c.close();
  await tick();
  assert.deepEqual(left, ['C']);
  assert.deepEqual(a.peers(), ['B']);
  a.close(); b.close();
  await tick();
  assert.equal(a.closed, true);
});

// A fake Trystero room: 0.25 style (onPeerJoin property, makeAction -> {send, onMessage}) or legacy
// (onPeerJoin(fn), makeAction -> [send, receive]). Two rooms are wired together directly.
function fakeRooms(legacy) {
  const rooms = {};
  const make = (self, otherId) => {
    const actions = {};
    const room = {
      _join: null, _leave: null, left: false,
      getPeers: () => ({ [otherId]: {} }),
      makeAction: (name) => {
        const a = { name, cb: null };
        actions[name] = a;
        const send = (data, target) => {
          const other = rooms[otherId];
          const tgt = legacy ? target : (target && target.target);
          if (tgt !== undefined && tgt !== null && tgt !== otherId) return Promise.resolve();
          const oa = other._actions[name];
          if (oa && oa.cb) setTimeout(() => oa.cb(JSON.parse(JSON.stringify(data)), legacy ? self : { peerId: self }), 0);
          return Promise.resolve();
        };
        if (legacy) return [send, (cb) => { a.cb = cb; }];
        const obj = { send };
        Object.defineProperty(obj, 'onMessage', { set(cb) { a.cb = cb; }, get() { return a.cb; } });
        return obj;
      },
      leave: () => { room.left = true; const other = rooms[otherId]; if (other._leave) other._leave(self); },
      _actions: actions,
    };
    if (legacy) { room.onPeerJoin = (fn) => { room._join = fn; }; room.onPeerLeave = (fn) => { room._leave = fn; }; }
    else {
      Object.defineProperty(room, 'onPeerJoin', { set(fn) { room._join = fn; }, get() { return undefined; } });
      Object.defineProperty(room, 'onPeerLeave', { set(fn) { room._leave = fn; }, get() { return undefined; } });
    }
    return room;
  };
  rooms.p1 = make('p1', 'p2');
  rooms.p2 = make('p2', 'p1');
  return rooms;
}

for (const legacy of [false, true]) {
  test('RtcTransport over a mocked Trystero room (' + (legacy ? 'legacy tuple API' : '0.25 object API') + ')', async () => {
    const rooms = fakeRooms(legacy);
    const t1 = new RtcTransport(rooms.p1, 'p1', 'mock');
    const t2 = new RtcTransport(rooms.p2, 'p2', 'mock');
    assert.deepEqual(t1.peers(), ['p2']);   // seeded from getPeers()
    assert.deepEqual(t2.peers(), ['p1']);
    const joined = [];
    t1.onPeer((id) => joined.push(id));
    rooms.p1._join('p3');
    assert.deepEqual(joined, ['p2', 'p3']);
    const got1 = [], got2 = [];
    t1.on('snap', (o, from) => got1.push([from, o.tick]));
    t2.on('snap', (o, from) => got2.push([from, o.tick]));
    t2.on('input', (o, from) => got2.push([from, 'input', o.dir]));
    t1.send('snap', { tick: 4 });
    t2.send('input', { dir: [1, 0] }, 'p1');
    t1.send('input', { dir: [0, 1] }, 'p2');
    t1.send('input', { dir: [9, 9] }, 'p3');   // p3 is not wired in the mock: must not reach p2
    await tick();
    assert.deepEqual(got1, []);
    assert.deepEqual(got2, [['p1', 4], ['p1', 'input', [0, 1]]]);
    const gone = [];
    t2.onLeave((id) => gone.push(id));
    t1.close();
    await tick();
    assert.equal(rooms.p1.left, true);
    assert.deepEqual(gone, ['p1']);
    assert.deepEqual(t2.peers(), []);
  });
}

test('createTransport validates its arguments and pins Trystero', async () => {
  await assert.rejects(() => createTransport({ kind: 'rtc' }), /room is required/);
  await assert.rejects(() => createTransport({ kind: 'bogus', room: 'x' }), /unknown transport kind/);
  assert.match(TRYSTERO_IMPORT, /^https:\/\/cdn\.jsdelivr\.net\/npm\/@trystero-p2p\/nostr@0\.25\.4\/\+esm$/);
  assert.deepEqual(TRYSTERO_STRATEGIES.map((s) => s.name), ['nostr', 'mqtt', 'torrent']);
  for (const st of TRYSTERO_STRATEGIES) assert.ok(st.relayUrls.every((u) => u.startsWith('wss://')), st.name + ' relays are wss');
  // a strategy list whose modules cannot load: strict mode rejects with a clear error (no network in
  // tests); the default falls back to the tab channel alone and records the error
  const bad = [{ name: 'none', url: 'data:text/javascript,export const nothing = 1' }];
  await assert.rejects(() => createTransport({ kind: 'rtc', room: 'x', strategies: bad, strict: true }), /could not load Trystero/);
  await assert.rejects(() => createTransport({ kind: 'rtc-only', room: 'x', strategies: bad }), /could not load Trystero/);
  const multi = await createTransport({ kind: 'rtc', room: 'nettest-fallback', strategies: bad });
  assert.equal(multi.kind, 'multi'); assert.equal(multi.rtc, null);
  await multi.ready;
  assert.match(multi.rtcError.message, /could not load Trystero/);
  assert.equal(multi.status().localPeers, 0); assert.equal(multi.status().errors.none.length > 0, true);
  multi.close();
  assert.ok(DEFAULT_RTC_CONFIG.iceServers.some((s) => String(s.urls).includes('turn:')), 'a TURN server is configured');
  const local = await createTransport({ kind: 'local', room: 'nettest-create' });
  assert.equal(typeof local.id, 'string');
  local.close();
});

// A fake RTCDataChannel pair (open at once, messages delivered on a timer).
function fakeChannelPair() {
  const mk = () => ({ readyState: 'open', onopen: null, onmessage: null, onclose: null, onerror: null, other: null,
    send(d) { const o = this.other; setTimeout(() => { if (o.onmessage && o.readyState === 'open') o.onmessage({ data: d }); }, 0); },
    close() { this.readyState = 'closed'; if (this.onclose) this.onclose(); const o = this.other; if (o.readyState === 'open') { o.readyState = 'closed'; if (o.onclose) o.onclose(); } } });
  const x = mk(), y = mk(); x.other = y; y.other = x; return [x, y];
}

test('MultiTransport: one peer over the tab channel and two mocked signalling paths, nothing twice, best path wins', async () => {
  const room = 'nettest-multi-' + Math.random().toString(36).slice(2, 8);
  const r1 = fakeRooms(false), r2 = fakeRooms(false);
  const a = new MultiTransport('A', new LocalTransport(room, { id: 'A' }));
  const b = new MultiTransport('B', new LocalTransport(room, { id: 'B' }));
  a.add('rtc:x', 'rtc', new RtcTransport(r1.p1, 'p1', 'x', { getRelaySockets: () => ({ 'wss://x': { readyState: 1 } }) }));
  b.add('rtc:x', 'rtc', new RtcTransport(r1.p2, 'p2', 'x'));
  a.add('rtc:y', 'rtc', new RtcTransport(r2.p1, 'q1', 'y', { getRelaySockets: () => ({ 'wss://y': { readyState: 0 } }) }));
  b.add('rtc:y', 'rtc', new RtcTransport(r2.p2, 'q2', 'y'));
  await tick(60);
  assert.deepEqual(a.peers(), ['B']); assert.deepEqual(b.peers(), ['A']);
  assert.equal(a.routeOf('B'), 'local'); assert.equal(a._routes.get('B').length, 3, 'three paths to the same peer');
  const got = [];
  b.on('hello', (obj, from) => got.push(from + ':' + obj.n));
  a.send('hello', { n: 'bcast' }); a.send('hello', { n: 'uni' }, 'B');
  await tick(40);
  assert.deepEqual(got.sort(), ['A:bcast', 'A:uni']);
  const st = a.status();
  assert.equal(st.strategy, 'x+y'); assert.equal(st.relays, 2); assert.equal(st.relaysOpen, 1); assert.equal(st.peers, 1); assert.equal(st.localPeers, 1);
  // another device: no tab channel in common, two signalling paths; messages arrive once over the first
  const r3 = fakeRooms(false), r4 = fakeRooms(false);
  const c = new MultiTransport('C', new LocalTransport('br1-' + room, { id: 'C' }));
  const d = new MultiTransport('D', new LocalTransport('br2-' + room, { id: 'D' }));
  c.add('rtc:x', 'rtc', new RtcTransport(r3.p1, 'p1', 'x')); d.add('rtc:x', 'rtc', new RtcTransport(r3.p2, 'p2', 'x'));
  c.add('rtc:y', 'rtc', new RtcTransport(r4.p1, 'q1', 'y')); d.add('rtc:y', 'rtc', new RtcTransport(r4.p2, 'q2', 'y'));
  await tick(60);
  assert.deepEqual(c.peers(), ['D']); assert.equal(c.routeOf('D'), 'rtc:x');
  const got2 = [];
  d.on('hello', (obj, from) => got2.push(from + ':' + obj.n));
  c.send('hello', { n: 'bcast' }); c.send('hello', { n: 'uni' }, 'D');
  await tick(40);
  assert.deepEqual(got2.sort(), ['C:bcast', 'C:uni']);
  // the local tab closing keeps the peer over the signalling path; losing every path drops it
  const left = [];
  a.onLeave((id) => left.push(id));
  b.local.close();
  await tick(40);
  assert.deepEqual(left, []); assert.equal(a.routeOf('B'), 'rtc:x');
  r1.p2.leave(); r2.p2.leave();
  await tick(40);
  assert.deepEqual(left, ['B']); assert.deepEqual(a.peers(), []);
  for (const t of [a, b, c, d]) t.close();
  await tick();
});

test('DirectTransport: a hand-made data channel is one more path; signal codes round-trip', async () => {
  const [x, y] = fakeChannelPair();
  const a = new MultiTransport('A', new LocalTransport('nettest-direct-a', { id: 'A' }));
  const b = new MultiTransport('B', new LocalTransport('nettest-direct-b', { id: 'B' }));
  a.add('direct-1', 'direct', new DirectTransport(x, { id: 'direct-1' }));
  b.add('direct-1', 'direct', new DirectTransport(y, { id: 'direct-1' }));
  await tick(40);
  assert.deepEqual(a.peers(), ['B']); assert.deepEqual(b.peers(), ['A']); assert.equal(a.routeOf('B'), 'direct-1');
  const got = [];
  b.on('snap', (obj, from) => got.push(from + ':' + obj.tick));
  a.send('snap', { tick: 7 }); a.send('snap', { tick: 8 }, 'B');
  await tick(30);
  assert.deepEqual(got, ['A:7', 'A:8']);
  assert.equal(a.status().direct, 1);
  const left = [];
  a.onLeave((id) => left.push(id));
  y.close();
  await tick(30);
  assert.deepEqual(left, ['B']);
  a.close(); b.close();
  // the offer/answer text: compressed when the runtime has CompressionStream, plain otherwise
  const sdp = 'v=0\r\no=- 1 2 IN IP4 127.0.0.1\r\n' + 'a=candidate:1 1 udp 2 192.168.0.1 5000 typ host\r\n'.repeat(6);
  const code = await encodeSignal({ v: 1, sdp });
  assert.ok(/^[zr][A-Za-z0-9_-]+$/.test(code), 'url-safe: ' + code.slice(0, 20));
  if (typeof CompressionStream !== 'undefined') assert.ok(code.length < sdp.length, 'compressed ' + code.length + ' < ' + sdp.length);
  assert.deepEqual(await decodeSignal(code), { v: 1, sdp });
  assert.deepEqual(await decodeSignal(' ' + code + '\n'), { v: 1, sdp });
});
