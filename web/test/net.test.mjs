// net.test.mjs - unit tests for the transport wrappers without a network: LocalTransport over Node's
// global BroadcastChannel, RtcTransport over a mocked Trystero room (both the 0.25 object-style action
// API and the older [send, receive] tuple API). Run: node web/test/net.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LocalTransport, RtcTransport, DualTransport, TRYSTERO_IMPORT, TRYSTERO_STRATEGIES, DEFAULT_RTC_CONFIG, createTransport } from '../src/net.js';

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
  assert.match(TRYSTERO_IMPORT, /^https:\/\/cdn\.jsdelivr\.net\/npm\/@trystero-p2p\/torrent@0\.25\.4\/\+esm$/);
  assert.deepEqual(TRYSTERO_STRATEGIES.map((s) => s.name), ['torrent', 'nostr', 'mqtt']);
  // a strategy list whose modules cannot load: strict mode rejects with a clear error (no network in
  // tests); the default falls back to the tab channel alone and records the error
  const bad = [{ name: 'none', url: 'data:text/javascript,export const nothing = 1' }];
  await assert.rejects(() => createTransport({ kind: 'rtc', room: 'x', strategies: bad, strict: true }), /could not load Trystero/);
  await assert.rejects(() => createTransport({ kind: 'rtc-only', room: 'x', strategies: bad }), /could not load Trystero/);
  const dual = await createTransport({ kind: 'rtc', room: 'nettest-fallback', strategies: bad });
  assert.equal(dual.kind, 'dual'); assert.equal(dual.rtc, null); assert.match(dual.rtcError.message, /could not load Trystero/);
  assert.equal(dual.status().localPeers, 0);
  dual.close();
  assert.ok(DEFAULT_RTC_CONFIG.iceServers.some((s) => String(s.urls).includes('turn:')), 'a TURN server is configured');
  const local = await createTransport({ kind: 'local', room: 'nettest-create' });
  assert.equal(typeof local.id, 'string');
  local.close();
});

test('DualTransport: tabs of one browser meet on the local channel, others over rtc, nothing twice', async () => {
  const room = 'nettest-dual-' + Math.random().toString(36).slice(2, 8);
  const rooms = fakeRooms(false);
  // p1 and p2 share a BroadcastChannel room (same browser) AND the mocked rtc room: every message must
  // arrive exactly once, over the local route
  const a = new DualTransport(new LocalTransport(room, { id: 'p1' }), new RtcTransport(rooms.p1, 'p1', 'fake', { getRelaySockets: () => ({ 'wss://x': { readyState: 1 }, 'wss://y': { readyState: 0 } }) }));
  const b = new DualTransport(new LocalTransport(room, { id: 'p2' }), new RtcTransport(rooms.p2, 'p2', 'fake'));
  await tick();
  assert.deepEqual(a.peers(), ['p2']); assert.deepEqual(b.peers(), ['p1']);
  assert.equal(a._route.get('p2'), 'local');
  const got = [];
  b.on('hello', (obj, from) => got.push(from + ':' + obj.n));
  a.send('hello', { n: 'bcast' });
  a.send('hello', { n: 'uni' }, 'p2');
  await tick(40);
  assert.deepEqual(got.sort(), ['p1:bcast', 'p1:uni']);
  const st = a.status();
  assert.equal(st.strategy, 'fake'); assert.equal(st.relays, 2); assert.equal(st.relaysOpen, 1); assert.equal(st.peers, 1); assert.equal(st.localPeers, 1);
  // a peer only on rtc (another device): c shares the mocked rtc room with nobody here, so wire a third
  // fake pair and check the rtc route is used
  const rooms2 = fakeRooms(false);
  const c = new DualTransport(new LocalTransport('other-browser-' + room, { id: 'p1' }), new RtcTransport(rooms2.p1, 'p1', 'fake'));
  const d = new DualTransport(new LocalTransport('another-browser-' + room, { id: 'p2' }), new RtcTransport(rooms2.p2, 'p2', 'fake'));
  await tick();
  assert.equal(c._route.get('p2'), 'rtc'); assert.equal(d._route.get('p1'), 'rtc');
  const got2 = [];
  d.on('hello', (obj, from) => got2.push(from + ':' + obj.n));
  c.send('hello', { n: 'bcast' }); c.send('hello', { n: 'uni' }, 'p2');
  await tick(40);
  assert.deepEqual(got2.sort(), ['p1:bcast', 'p1:uni']);
  // the local tab closing while rtc still knows the peer keeps the peer, now over rtc
  const left = [];
  a.onLeave((id) => left.push(id));
  b.local.close();
  await tick(40);
  assert.deepEqual(left, []); assert.equal(a._route.get('p2'), 'rtc');
  for (const t of [a, b, c, d]) t.close();
  await tick();
});
