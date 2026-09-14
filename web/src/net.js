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
// Strategy order: torrent trackers first, then nostr, then mqtt. Each entry is a full pinned URL so the
// import specifier is static text (CSP-friendly, no string building at runtime).
export const TRYSTERO_STRATEGIES = Object.freeze([
  { name: 'torrent', url: `https://cdn.jsdelivr.net/npm/@trystero-p2p/torrent@${TRYSTERO_VERSION}/+esm` },
  { name: 'nostr', url: `https://cdn.jsdelivr.net/npm/@trystero-p2p/nostr@${TRYSTERO_VERSION}/+esm` },
  { name: 'mqtt', url: `https://cdn.jsdelivr.net/npm/@trystero-p2p/mqtt@${TRYSTERO_VERSION}/+esm` },
]);
export const TRYSTERO_IMPORT = TRYSTERO_STRATEGIES[0].url;
export const DEFAULT_APP_ID = 'ungroup-web-v1';

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
  // room: the Trystero room object; selfId: this peer's id
  constructor(room, selfId, strategyName) {
    super(selfId);
    this.room = room;
    this.strategy = strategyName;
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

// Factory. 'rtc' resolves once the signalling module is loaded (peers then appear asynchronously through
// onPeer); 'local' resolves immediately.
export async function createTransport({ kind = 'local', room, appId = DEFAULT_APP_ID, strategies = TRYSTERO_STRATEGIES, rtcConfig, password } = {}) {
  if (!room) throw new Error('createTransport: room is required');
  if (kind === 'local') return new LocalTransport(room);
  if (kind === 'rtc') {
    const { mod, strategy } = await loadTrystero(strategies);
    const cfg = { appId };
    if (rtcConfig) cfg.rtcConfig = rtcConfig;
    if (password) cfg.password = password;
    const trysteroRoom = mod.joinRoom(cfg, room);
    return new RtcTransport(trysteroRoom, mod.selfId || randomId(), strategy);
  }
  throw new Error('unknown transport kind ' + kind);
}
