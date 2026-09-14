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

// ----------------------------------------------------------------------------------------------- dual
// WebRTC for other devices plus the BroadcastChannel for the other tabs of this browser, under one peer
// id. Tabs of the same browser find each other at once even when no signalling tracker answers; other
// devices arrive through Trystero. A peer reachable on the local channel is spoken to there only, so
// nothing is delivered twice.
export class DualTransport extends TransportBase {
  constructor(local, rtc = null) {
    super(local.id);
    this.local = local;
    this.rtc = null;
    this.kind = 'dual';
    this.rtcError = null;
    this._route = new Map();   // peer id -> 'local' | 'rtc'
    this._subs = new Set();    // 'local:channel' / 'rtc:channel' already forwarded
    this._wire(local, 'local');
    if (rtc) this.attachRtc(rtc);
  }
  attachRtc(rtc) {
    if (this.closed) { try { rtc.close(); } catch (e) { /* ignore */ } return; }
    this.rtc = rtc;
    this._wire(rtc, 'rtc');
    for (const ch of this._handlers.keys()) this._forward(rtc, 'rtc', ch);
  }
  _wire(t, kind) {
    t.onPeer((id) => {
      const cur = this._route.get(id);
      if (kind === 'local' || !cur) this._route.set(id, kind);
      this._addPeer(id);
    });
    t.onLeave((id) => {
      if (this._route.get(id) !== kind) return;
      const other = kind === 'local' ? this.rtc : this.local;
      if (other && other.peers().includes(id)) this._route.set(id, kind === 'local' ? 'rtc' : 'local');
      else { this._route.delete(id); this._removePeer(id); }
    });
  }
  _forward(t, kind, channel) {
    const key = kind + ':' + channel;
    if (this._subs.has(key)) return;
    this._subs.add(key);
    t.on(channel, (obj, from) => { if (this._route.get(from) === kind || !this._route.has(from)) this._dispatch(channel, obj, from); });
  }
  _channelAdded(channel) {
    this._forward(this.local, 'local', channel);
    if (this.rtc) this._forward(this.rtc, 'rtc', channel);
  }
  send(channel, obj, toId = null) {
    if (this.closed) return;
    if (toId !== null) {
      const route = this._route.get(toId);
      if (route === 'local') this.local.send(channel, obj, toId);
      else if (route === 'rtc' && this.rtc) this.rtc.send(channel, obj, toId);
      return;
    }
    this.local.send(channel, obj, null);
    if (this.rtc) for (const p of this.rtc.peers()) if (this._route.get(p) === 'rtc') this.rtc.send(channel, obj, p);
  }
  status() {
    const st = this.rtc ? this.rtc.status() : { strategy: null, peers: 0, relays: null, relaysOpen: null };
    st.localPeers = this.local.peers().length;
    st.peers = this._peers.size;
    st.rtcError = this.rtcError ? String(this.rtcError.message || this.rtcError) : null;
    return st;
  }
  close() {
    if (this.closed) return;
    super.close();
    try { this.local.close(); } catch (e) { /* ignore */ }
    if (this.rtc) { try { this.rtc.close(); } catch (e) { /* ignore */ } }
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

// Factory. 'local' is the BroadcastChannel alone. 'rtc' is a DualTransport: the BroadcastChannel plus
// Trystero under Trystero's peer id (peers appear asynchronously through onPeer); when no strategy module
// can be loaded it falls back to the BroadcastChannel alone with `rtcError` set, unless `strict` is on.
// 'rtc-only' is the bare RtcTransport.
export async function createTransport({ kind = 'local', room, appId = DEFAULT_APP_ID, strategies = TRYSTERO_STRATEGIES, rtcConfig = DEFAULT_RTC_CONFIG, password, strict = false, localOpts = {} } = {}) {
  if (!room) throw new Error('createTransport: room is required');
  if (kind === 'local') return new LocalTransport(room, localOpts);
  if (kind === 'rtc' || kind === 'rtc-only') {
    let loaded = null, err = null;
    try { loaded = await loadTrystero(strategies); } catch (e) { err = e; if (strict || kind === 'rtc-only') throw e; }
    if (!loaded) { const t = new DualTransport(new LocalTransport(room, localOpts)); t.rtcError = err; return t; }
    const { mod, strategy } = loaded;
    const cfg = { appId };
    if (rtcConfig) cfg.rtcConfig = rtcConfig;
    if (password) cfg.password = password;
    const id = mod.selfId || randomId();
    const rtc = new RtcTransport(mod.joinRoom(cfg, room), id, strategy, mod);
    if (kind === 'rtc-only') return rtc;
    return new DualTransport(new LocalTransport(room, Object.assign({}, localOpts, { id })), rtc);
  }
  throw new Error('unknown transport kind ' + kind);
}
