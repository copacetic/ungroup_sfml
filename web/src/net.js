// net.js - the serverless transport layer for Ungroup multiplayer.
//
//   createTransport({ kind: 'local' | 'rtc', room, appId })
//     -> { id, peers(), onPeer(cb(id)), onLeave(cb(id)), send(channel, obj, toId = null), on(channel, cb(obj, fromId)), close() }
//
// 'local' - BroadcastChannel('ungroup-' + room) with a per-tab id. Every tab of the same browser profile
//           on the same origin sees the channel, so this works for same-browser tests and single-machine
//           play (one host tab, N client tabs). Peer discovery is a tiny join/ack/leave protocol on the
//           channel plus a heartbeat so a killed tab is noticed within a few seconds.
// 'rtc'   - Trystero (https://github.com/dmotz/trystero), loaded as an ES module from jsdelivr. Signalling
//           goes through Trystero's public strategies (BitTorrent trackers first, Nostr relays and public
//           MQTT brokers as fallbacks); the game data then flows peer-to-peer over WebRTC data channels.
//           There is no server of ours anywhere.
//
// Trystero notes (pinned to 0.25.4; the strategy entrypoints moved to scoped packages in 0.25):
//   import { joinRoom, selfId } from 'https://cdn.jsdelivr.net/npm/@trystero-p2p/torrent@0.25.4/+esm'
//   const room = joinRoom({ appId }, roomId)          // same call for every strategy
//   room.onPeerJoin = (peerId) => ...                  // property assignment (0.25); older builds: room.onPeerJoin(fn)
//   room.onPeerLeave = (peerId) => ...
//   room.getPeers()                                    // { peerId: RTCPeerConnection }
//   const act = room.makeAction('name')                // 0.25: { send(data, {target}), onMessage = (data, {peerId}) => ... }
//                                                      // <= 0.21: [send(data, target), receive((data, peerId) => ...)]
//   room.leave()
//   Action names are limited to 12 bytes; ours are 'hello', 'lobby', 'start', 'input', 'snap', 'end'.
//   Both action shapes are handled below so the pinned version can be bumped without touching callers.
//
// Network / CSP limits: the CDN import needs the page's CSP to allow script-src https://cdn.jsdelivr.net
// (the +esm bundle imports @trystero-p2p/core and the strategy from the same host), connect-src wss: for
// the trackers/relays and the WebRTC STUN servers. A page hosted from file:// or a sandbox without
// outbound network cannot load Trystero at all; createTransport then rejects and the caller should fall
// back to 'local' (same machine) or the bots-only mode. This module is browser-only for the 'rtc' and
// 'local' kinds; tests use an in-process FakeTransport with the same interface (web/test/session.test.mjs).

const TRYSTERO_VERSION = '0.25.4';
// Signalling strategies, all joined at once (see MultiTransport): nostr relays, MQTT brokers and BitTorrent
// trackers, each with its own list of public relays (Trystero's `relayUrls`; the built-in tracker list has
// several dead entries). Each entry is a full pinned URL so the import specifier is static text.
export const TRYSTERO_STRATEGIES = Object.freeze([
  { name: 'nostr', url: `https://cdn.jsdelivr.net/npm/@trystero-p2p/nostr@${TRYSTERO_VERSION}/+esm`,
    relayUrls: ['wss://relay.damus.io', 'wss://nos.lol', 'wss://relay.nostr.band', 'wss://relay.snort.social', 'wss://relay.primal.net', 'wss://nostr.mom', 'wss://relay.nostr.bg'], relayRedundancy: 4 },
  { name: 'mqtt', url: `https://cdn.jsdelivr.net/npm/@trystero-p2p/mqtt@${TRYSTERO_VERSION}/+esm`,
    relayUrls: ['wss://broker.emqx.io:8084/mqtt', 'wss://broker.hivemq.com:8884/mqtt', 'wss://test.mosquitto.org:8081'], relayRedundancy: 2 },
  { name: 'torrent', url: `https://cdn.jsdelivr.net/npm/@trystero-p2p/torrent@${TRYSTERO_VERSION}/+esm`,
    relayUrls: ['wss://tracker.openwebtorrent.com', 'wss://tracker.webtorrent.dev', 'wss://tracker.files.fm:7073/announce', 'wss://tracker.btorrent.xyz'], relayRedundancy: 3 },
]);
export const TRYSTERO_IMPORT = TRYSTERO_STRATEGIES[0].url;
export const DEFAULT_APP_ID = 'ungroup-web-v1';
// ICE servers: Google's STUN plus the public Open Relay TURN service (openrelay.metered.ca, free, static
// credentials published by its operator), so peers behind symmetric NATs and phone networks still connect.
export const DEFAULT_RTC_CONFIG = Object.freeze({
  iceServers: [
    { urls: ['stun:stun.l.google.com:19302', 'stun:stun1.l.google.com:19302'] },
    { urls: ['turn:openrelay.metered.ca:80', 'turn:openrelay.metered.ca:443', 'turn:openrelay.metered.ca:443?transport=tcp', 'turns:openrelay.metered.ca:443'], username: 'openrelayproject', credential: 'openrelayproject' },
  ],
});

const LOCAL_HEARTBEAT_MS = 2000;
const LOCAL_TIMEOUT_MS = 6500;

export function randomId(len = 10) {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let s = '';
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const b = new Uint8Array(len);
    crypto.getRandomValues(b);
    for (let i = 0; i < len; i++) s += alphabet[b[i] % alphabet.length];
  } else {
    for (let i = 0; i < len; i++) s += alphabet[Math.floor(Math.random() * alphabet.length)];
  }
  return s;
}

// Shared bookkeeping for every transport kind: peer set, listener lists, per-channel handlers.
export class TransportBase {
  constructor(id) {
    this.id = id;
    this._peers = new Set();
    this._onPeer = [];
    this._onLeave = [];
    this._handlers = new Map();
    this.closed = false;
  }
  peers() { return Array.from(this._peers); }
  onPeer(cb) { this._onPeer.push(cb); for (const p of this._peers) cb(p); return () => { this._onPeer = this._onPeer.filter((f) => f !== cb); }; }
  onLeave(cb) { this._onLeave.push(cb); return () => { this._onLeave = this._onLeave.filter((f) => f !== cb); }; }
  on(channel, cb) {
    if (!this._handlers.has(channel)) this._handlers.set(channel, []);
    this._handlers.get(channel).push(cb);
    this._channelAdded(channel);
    return () => { const l = this._handlers.get(channel); if (l) this._handlers.set(channel, l.filter((f) => f !== cb)); };
  }
  // subclass hooks
  _channelAdded(_channel) {}
  send(_channel, _obj, _toId = null) { throw new Error('send not implemented'); }
  close() { this.closed = true; }
  // helpers for subclasses
  _addPeer(id) {
    if (id === this.id || this._peers.has(id)) return false;
    this._peers.add(id);
    for (const cb of this._onPeer.slice()) { try { cb(id); } catch (e) { console.error(e); } }
    return true;
  }
  _removePeer(id) {
    if (!this._peers.delete(id)) return false;
    for (const cb of this._onLeave.slice()) { try { cb(id); } catch (e) { console.error(e); } }
    return true;
  }
  _dispatch(channel, obj, fromId) {
    const l = this._handlers.get(channel);
    if (!l) return;
    for (const cb of l.slice()) { try { cb(obj, fromId); } catch (e) { console.error(e); } }
  }
}

// ---------------------------------------------------------------------------------------------- local
// Wire format on the BroadcastChannel: { k: 'join' | 'ack' | 'leave' | 'beat' | 'msg', from, to?, ch?, data? }
// join  - a new tab announces itself; every existing tab answers with ack (to: the new tab).
// beat  - heartbeat every LOCAL_HEARTBEAT_MS; a peer silent for LOCAL_TIMEOUT_MS is dropped.
// msg   - a channel message; to = null means broadcast.
export class LocalTransport extends TransportBase {
  constructor(room, opts = {}) {
    super(opts.id || randomId());
    this.room = room;
    const BC = opts.BroadcastChannel || (typeof BroadcastChannel !== 'undefined' ? BroadcastChannel : null);
    if (!BC) throw new Error('BroadcastChannel is not available in this environment');
    this.bc = new BC('ungroup-' + room);
    this._lastSeen = new Map();
    this._now = opts.now || (() => Date.now());
    this.bc.onmessage = (ev) => this._recv(ev.data);
    this._post({ k: 'join', from: this.id });
    this._beat = setInterval(() => {
      this._post({ k: 'beat', from: this.id });
      const now = this._now();
      for (const [p, t] of this._lastSeen) if (now - t > LOCAL_TIMEOUT_MS) { this._lastSeen.delete(p); this._removePeer(p); }
    }, LOCAL_HEARTBEAT_MS);
    this._unload = () => this.close();
    if (typeof addEventListener === 'function') addEventListener('pagehide', this._unload);
  }
  _post(m) { if (!this.closed) { try { this.bc.postMessage(m); } catch (e) { console.error('local transport post failed', e); } } }
  _recv(m) {
    if (!m || m.from === this.id || this.closed) return;
    if (m.to && m.to !== this.id) return;
    this._lastSeen.set(m.from, this._now());
    switch (m.k) {
      case 'join': this._addPeer(m.from); this._post({ k: 'ack', from: this.id, to: m.from }); break;
      case 'ack': case 'beat': this._addPeer(m.from); break;
      case 'leave': this._lastSeen.delete(m.from); this._removePeer(m.from); break;
      case 'msg': this._addPeer(m.from); this._dispatch(m.ch, m.data, m.from); break;
      default: break;
    }
  }
  send(channel, obj, toId = null) {
    if (this.closed) return;
    if (toId !== null && !this._peers.has(toId)) return;
    this._post({ k: 'msg', from: this.id, to: toId, ch: channel, data: obj });
  }
  close() {
    if (this.closed) return;
    this._post({ k: 'leave', from: this.id });
    super.close();
    clearInterval(this._beat);
    if (typeof removeEventListener === 'function') removeEventListener('pagehide', this._unload);
    try { this.bc.close(); } catch (e) { /* already closed */ }
  }
}

// ------------------------------------------------------------------------------------------------ rtc
export class RtcTransport extends TransportBase {
  // room: the Trystero room object; selfId: this peer's id; mod: the strategy module (for getRelaySockets)
  constructor(room, selfId, strategyName, mod = null) {
    super(selfId);
    this.room = room;
    this.strategy = strategyName;
    this.mod = mod;
    this.kind = 'rtc';
    this._actions = new Map();
    const onJoin = (peerId) => this._addPeer(peerId);
    const onLeave = (peerId) => this._removePeer(peerId);
    // 0.25+: callback properties; <= 0.21: registration functions.
    if (typeof room.onPeerJoin === 'function') room.onPeerJoin(onJoin); else room.onPeerJoin = onJoin;
    if (typeof room.onPeerLeave === 'function') room.onPeerLeave(onLeave); else room.onPeerLeave = onLeave;
    if (typeof room.getPeers === 'function') { try { for (const p of Object.keys(room.getPeers() || {})) this._addPeer(p); } catch (e) { /* ignore */ } }
  }
  _action(channel) {
    let a = this._actions.get(channel);
    if (a) return a;
    const made = this.room.makeAction(channel);
    if (Array.isArray(made)) {
      // legacy tuple API: [send(data, target), receive(cb(data, peerId))]
      const [send, receive] = made;
      a = { send: (data, target) => send(data, target), listen: (cb) => receive((data, peerId) => cb(data, peerId)) };
    } else {
      a = { send: (data, target) => made.send(data, target === null ? undefined : { target }), listen: (cb) => { made.onMessage = (data, meta) => cb(data, meta && meta.peerId); } };
    }
    this._actions.set(channel, a);
    return a;
  }
  _channelAdded(channel) {
    const a = this._action(channel);
    if (!a.listening) { a.listening = true; a.listen((data, peerId) => this._dispatch(channel, data, peerId)); }
  }
  // Signalling health: how many tracker/relay sockets the strategy keeps open (null when the module does
  // not expose them) and the peer count.
  status() {
    const out = { strategy: this.strategy, peers: this._peers.size, relays: null, relaysOpen: null };
    try {
      const fn = this.mod && this.mod.getRelaySockets;
      if (typeof fn === 'function') {
        const list = Object.values(fn() || {});
        out.relays = list.length;
        out.relaysOpen = list.filter((w) => w && w.readyState === 1).length;
      }
    } catch (e) { /* not fatal */ }
    return out;
  }
  send(channel, obj, toId = null) {
    if (this.closed) return;
    if (toId !== null && !this._peers.has(toId)) return;
    try {
      const r = this._action(channel).send(obj, toId);
      if (r && typeof r.catch === 'function') r.catch((e) => console.warn('rtc send failed', channel, e));
    } catch (e) { console.warn('rtc send failed', channel, e); }
  }
  close() {
    if (this.closed) return;
    super.close();
    try { this.room.leave(); } catch (e) { /* ignore */ }
  }
}

// ---------------------------------------------------------------------------------------------- multi
// One peer id over many paths. Sub-transports (the BroadcastChannel for this browser's tabs, one Trystero
// room per signalling strategy, hand-made direct data channels) each know a peer by their own id; the
// MultiTransport exchanges its canonical id over every path ('ugid' handshake) and announces a peer once,
// under that id, routing messages to it over its best path (local, then direct, then whichever
// signalling path came up first). So a friend found through nostr and through the trackers is one peer,
// and a tab of the same browser is found even when no relay answers at all.
const ROUTE_PRIORITY = { local: 0, direct: 1, rtc: 2 };
const UGID = 'ugid';
export class MultiTransport extends TransportBase {
  constructor(id, local = null) {
    super(id);
    this.kind = 'multi';
    this.subs = [];                 // { name, kind, t }
    this._routes = new Map();       // canonical peer id -> [{ sub, subPeer }] best first
    this._canon = new Map();        // sub.name + ':' + subPeer -> canonical id
    this._errors = {};              // strategy name -> error message
    this.rtcError = null;
    if (local) this.add('local', 'local', local);
  }
  add(name, kind, t) {
    if (this.closed) { try { t.close(); } catch (e) { /* ignore */ } return; }
    const sub = { name, kind, t };
    this.subs.push(sub);
    t.on(UGID, (obj, subPeer) => {
      if (!obj || typeof obj.id !== 'string' || obj.id === this.id) return;
      const key = name + ':' + subPeer;
      if (this._canon.get(key) === obj.id) return;
      this._canon.set(key, obj.id);
      const routes = this._routes.get(obj.id) || [];
      if (!routes.some((r) => r.sub === sub && r.subPeer === subPeer)) {
        routes.push({ sub, subPeer });
        routes.sort((x, y) => ROUTE_PRIORITY[x.sub.kind] - ROUTE_PRIORITY[y.sub.kind] || this.subs.indexOf(x.sub) - this.subs.indexOf(y.sub));
      }
      this._routes.set(obj.id, routes);
      if (!obj.ack) t.send(UGID, { id: this.id, ack: 1 }, subPeer);   // answer so the other side maps us too
      this._addPeer(obj.id);
    });
    for (const ch of this._handlers.keys()) if (ch !== UGID) this._forward(sub, ch);
    t.onPeer((subPeer) => t.send(UGID, { id: this.id }, subPeer));
    t.onLeave((subPeer) => {
      const key = name + ':' + subPeer;
      const canonical = this._canon.get(key);
      if (!canonical) return;
      this._canon.delete(key);
      const routes = (this._routes.get(canonical) || []).filter((r) => !(r.sub === sub && r.subPeer === subPeer));
      if (routes.length) this._routes.set(canonical, routes);
      else { this._routes.delete(canonical); this._removePeer(canonical); }
    });
    return sub;
  }
  _forward(sub, channel) {
    if (sub._fwd && sub._fwd.has(channel)) return;
    if (!sub._fwd) sub._fwd = new Set();
    sub._fwd.add(channel);
    sub.t.on(channel, (obj, subPeer) => {
      const canonical = this._canon.get(sub.name + ':' + subPeer);
      if (!canonical) return;                       // before the handshake: the session repeats what matters
      const routes = this._routes.get(canonical);
      if (routes && routes[0].sub !== sub) return;  // only the best path delivers (no duplicates)
      this._dispatch(channel, obj, canonical);
    });
  }
  _channelAdded(channel) { if (channel !== UGID) for (const sub of this.subs) this._forward(sub, channel); }
  send(channel, obj, toId = null) {
    if (this.closed) return;
    if (toId !== null) {
      const routes = this._routes.get(toId);
      if (routes && routes.length) routes[0].sub.t.send(channel, obj, routes[0].subPeer);
      return;
    }
    for (const [id, routes] of this._routes) if (routes.length) routes[0].sub.t.send(channel, obj, routes[0].subPeer);
  }
  routeOf(id) { const r = this._routes.get(id); return r && r.length ? r[0].sub.name : null; }
  get local() { const s = this.subs.find((x) => x.kind === 'local'); return s ? s.t : null; }
  get rtc() { const s = this.subs.find((x) => x.kind === 'rtc'); return s ? s.t : null; }   // the first signalling path
  // Signalling health per path plus the peer count.
  status() {
    const paths = [];
    for (const sub of this.subs) {
      const st = typeof sub.t.status === 'function' ? sub.t.status() : { peers: sub.t.peers().length };
      paths.push(Object.assign({ name: sub.name, kind: sub.kind }, st));
    }
    const rtcs = paths.filter((p) => p.kind === 'rtc');
    const relays = rtcs.reduce((n, p) => n + (p.relays || 0), 0), relaysOpen = rtcs.reduce((n, p) => n + (p.relaysOpen || 0), 0);
    return { strategy: rtcs.map((p) => p.strategy || p.name).join('+') || null, paths, relays: rtcs.length ? relays : null, relaysOpen: rtcs.length ? relaysOpen : null,
      peers: this._peers.size, localPeers: this.local ? this.local.peers().length : 0, direct: paths.filter((p) => p.kind === 'direct').length,
      errors: Object.assign({}, this._errors), rtcError: this.rtcError ? String(this.rtcError.message || this.rtcError) : null };
  }
  close() {
    if (this.closed) return;
    super.close();
    for (const sub of this.subs) { try { sub.t.close(); } catch (e) { /* ignore */ } }
  }
}
export { MultiTransport as DualTransport };   // the earlier name

// --------------------------------------------------------------------------------------------- direct
// One WebRTC data channel set up by hand (see makeInvite / acceptInvite), as a sub-transport with a
// single peer. Wire format: JSON [channel, data].
export class DirectTransport extends TransportBase {
  constructor(channel, opts = {}) {
    super(opts.id || ('direct-' + randomId(6)));
    this.kind = 'direct';
    this.channel = channel;
    this.peerId = opts.peerId || (this.id + '-peer');
    this.pc = opts.pc || null;
    const open = () => this._addPeer(this.peerId);
    channel.onopen = open;
    channel.onmessage = (ev) => {
      let m = null;
      try { m = JSON.parse(typeof ev.data === 'string' ? ev.data : String(ev.data)); } catch (e) { return; }
      if (Array.isArray(m) && typeof m[0] === 'string') this._dispatch(m[0], m[1], this.peerId);
    };
    channel.onclose = () => { this._removePeer(this.peerId); };
    channel.onerror = () => { this._removePeer(this.peerId); };
    if (channel.readyState === 'open') open();
  }
  status() { return { peers: this._peers.size, state: this.channel.readyState }; }
  send(channel, obj, toId = null) {
    if (this.closed || this.channel.readyState !== 'open') return;
    if (toId !== null && toId !== this.peerId) return;
    try { this.channel.send(JSON.stringify([channel, obj])); } catch (e) { console.warn('direct send failed', e); }
  }
  close() {
    if (this.closed) return;
    super.close();
    try { this.channel.close(); } catch (e) { /* ignore */ }
    if (this.pc) { try { this.pc.close(); } catch (e) { /* ignore */ } }
  }
}

// ---- hand-made signalling: the offer travels in a link, the answer comes back as a code (any chat)
const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
function b64url(bytes) { let out = ''; for (let i = 0; i < bytes.length; i += 3) { const n = (bytes[i] << 16) | ((bytes[i + 1] || 0) << 8) | (bytes[i + 2] || 0); out += B64[n >> 18] + B64[(n >> 12) & 63] + (i + 1 < bytes.length ? B64[(n >> 6) & 63] : '') + (i + 2 < bytes.length ? B64[n & 63] : ''); } return out; }
function unb64url(str) { const idx = {}; for (let i = 0; i < 64; i++) idx[B64[i]] = i; const out = []; let buf = 0, bits = 0; for (const ch of str) { if (idx[ch] === undefined) continue; buf = (buf << 6) | idx[ch]; bits += 6; if (bits >= 8) { bits -= 8; out.push((buf >> bits) & 255); } } return new Uint8Array(out); }
async function pipeBytes(bytes, stream) {
  const w = stream.writable.getWriter(); w.write(bytes); w.close();
  const r = stream.readable.getReader(); const chunks = []; let n = 0;
  for (;;) { const { value, done } = await r.read(); if (done) break; chunks.push(value); n += value.length; }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out;
}
// Compact text for an SDP: deflate-raw + base64url when the browser has CompressionStream, else plain.
export async function encodeSignal(obj) {
  const raw = new TextEncoder().encode(JSON.stringify(obj));
  if (typeof CompressionStream !== 'undefined') { try { return 'z' + b64url(await pipeBytes(raw, new CompressionStream('deflate-raw'))); } catch (e) { /* fall through */ } }
  return 'r' + b64url(raw);
}
export async function decodeSignal(text) {
  text = String(text || '').trim();
  const body = unb64url(text.slice(1));
  const raw = text[0] === 'z' ? await pipeBytes(body, new DecompressionStream('deflate-raw')) : body;
  return JSON.parse(new TextDecoder().decode(raw));
}
function gathered(pc, ms = 4000) {
  return new Promise((resolve) => {
    if (pc.iceGatheringState === 'complete') { resolve(); return; }
    const done = () => { pc.removeEventListener('icegatheringstatechange', check); resolve(); };
    const check = () => { if (pc.iceGatheringState === 'complete') done(); };
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(done, ms);
  });
}
// Host side: a connection waiting for one friend. Returns the sub-transport to add to the MultiTransport
// and the offer text for the link; complete(answerText) finishes the handshake.
export async function makeInvite({ rtcConfig = DEFAULT_RTC_CONFIG, id } = {}) {
  const pc = new RTCPeerConnection(rtcConfig);
  const channel = pc.createDataChannel('ungroup', { ordered: true });
  const t = new DirectTransport(channel, { pc, id });
  await pc.setLocalDescription(await pc.createOffer());
  await gathered(pc);
  const offer = await encodeSignal({ v: 1, sdp: pc.localDescription.sdp });
  return { transport: t, pc, offer, complete: async (answerText) => { const a = await decodeSignal(answerText); if (!a || a.v !== 1 || !a.sdp) throw new Error('that is not a reply code'); await pc.setRemoteDescription({ type: 'answer', sdp: a.sdp }); } };
}
// Friend side: take the offer from the link, produce the reply code.
export async function acceptInvite(offerText, { rtcConfig = DEFAULT_RTC_CONFIG, id } = {}) {
  const o = await decodeSignal(offerText);
  if (!o || o.v !== 1 || !o.sdp) throw new Error('that is not an invite');
  const pc = new RTCPeerConnection(rtcConfig);
  let resolveChannel;
  const channelReady = new Promise((r) => { resolveChannel = r; });
  pc.ondatachannel = (ev) => resolveChannel(ev.channel);
  await pc.setRemoteDescription({ type: 'offer', sdp: o.sdp });
  await pc.setLocalDescription(await pc.createAnswer());
  await gathered(pc);
  const answer = await encodeSignal({ v: 1, sdp: pc.localDescription.sdp });
  return { pc, answer, channelReady, attach: async (multi) => { const ch = await channelReady; const t = new DirectTransport(ch, { pc, id }); multi.add(t.id, 'direct', t); return t; } };
}

async function loadTrystero(strategies) {
  let lastErr = null;
  for (const s of strategies) {
    try {
      const mod = await import(/* @vite-ignore */ s.url);
      if (typeof mod.joinRoom !== 'function') throw new Error('no joinRoom export in ' + s.url);
      return { mod, strategy: s.name };
    } catch (e) { lastErr = e; console.warn('trystero strategy unavailable:', s.name, e && e.message); }
  }
  throw new Error('could not load Trystero from the CDN (' + (lastErr && lastErr.message) + ')');
}

// Factory. 'local' is the BroadcastChannel alone. 'rtc' is a MultiTransport: the BroadcastChannel at
// once, then every signalling strategy as its module loads (each joining the same room; peers are
// deduplicated by the handshake); `ready` resolves when every strategy has loaded or failed, `rtcError`
// is set when none loaded (with `strict`, that rejects). 'rtc-only' is the bare RtcTransport of the
// first strategy that loads.
export async function createTransport({ kind = 'local', room, appId = DEFAULT_APP_ID, strategies = TRYSTERO_STRATEGIES, rtcConfig = DEFAULT_RTC_CONFIG, password, strict = false, localOpts = {} } = {}) {
  if (!room) throw new Error('createTransport: room is required');
  if (kind === 'local') return new LocalTransport(room, localOpts);
  if (kind === 'rtc-only') {
    const { mod, strategy } = await loadTrystero(strategies);
    const cfg = { appId };
    if (rtcConfig) cfg.rtcConfig = rtcConfig;
    if (password) cfg.password = password;
    return new RtcTransport(mod.joinRoom(cfg, room), mod.selfId || randomId(), strategy, mod);
  }
  if (kind === 'rtc') {
    const id = (localOpts && localOpts.id) || randomId();
    const multi = new MultiTransport(id, new LocalTransport(room, Object.assign({}, localOpts, { id })));
    const loads = strategies.map(async (st) => {
      try {
        const mod = await import(/* @vite-ignore */ st.url);
        if (typeof mod.joinRoom !== 'function') throw new Error('no joinRoom export in ' + st.url);
        const cfg = { appId };
        if (rtcConfig) cfg.rtcConfig = rtcConfig;
        if (password) cfg.password = password;
        if (st.relayUrls) cfg.relayUrls = st.relayUrls.slice();
        if (st.relayRedundancy) cfg.relayRedundancy = st.relayRedundancy;
        if (multi.closed) return false;
        multi.add('rtc:' + st.name, 'rtc', new RtcTransport(mod.joinRoom(cfg, room), mod.selfId || randomId(), st.name, mod));
        return true;
      } catch (e) { console.warn('trystero strategy unavailable:', st.name, e && e.message); multi._errors[st.name] = String(e && e.message || e); return false; }
    });
    multi.ready = Promise.all(loads).then((oks) => {
      if (!oks.some(Boolean)) multi.rtcError = new Error('could not load Trystero from the CDN (' + Object.values(multi._errors).join('; ') + ')');
      return multi;
    });
    if (strict) { await multi.ready; if (multi.rtcError) { multi.close(); throw multi.rtcError; } }
    return multi;
  }
  throw new Error('unknown transport kind ' + kind);
}
