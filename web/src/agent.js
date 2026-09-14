// web/src/agent.js — run a trained Ungroup policy (ONNX, see rl/export_onnx.py) in the browser.
//
//   import { loadAgent } from './agent.js';
//   const agent = await loadAgent('models/v9_150.onnx');          // nvec/decideEvery from the sidecar .json
//   game.decideEvery = agent.decideEvery;
//   const [move, join, leave, intent] = await agent.act(game.observe(i));
//   game.setInput(i, { move, join, leave, intent });
//
// onnxruntime-web is loaded on demand from jsDelivr as the classic UMD script (defines `globalThis.ort`):
//   https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.js
// The WebAssembly backend fetches its .wasm/.mjs binaries relative to `ort.env.wasm.wasmPaths`, which
// defaults to the page's own origin, so the loader points it back at the same jsDelivr dist folder:
//   ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/'
// It also sets ort.env.wasm.numThreads = 1 (multi-threading needs a cross-origin isolated page and a
// same-origin worker, neither of which a static page served next to index.html has) and
// ort.env.wasm.proxy = false. Pass { ort } to loadAgent to use an already loaded runtime instead.
//
// The exported graph takes obs (N, obsDim) float32 and returns four masked logit rows per observation
// (move, join, leave, intent). The action masks of the Python policy (no leave when solo, intent locked
// while grouped, macro heads: no direct steering, dead mines and absent bodies) are applied inside the
// graph as -inf; act() treats any logit < -1e8 as masked and samples the rest from a softmax.

export const ORT_VERSION = '1.29.0';
export const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
export const ORT_URL = ORT_DIST + 'ort.min.js';
export const MASKED = -1e8;
export const HEADS = ['move', 'join', 'leave', 'intent'];

let ortLoading = null;

// Loads onnxruntime-web once (classic script tag) and configures the wasm backend for CDN hosting.
export function loadOrt({ url = ORT_URL, wasmPaths = ORT_DIST, numThreads = 1 } = {}) {
  if (!ortLoading) {
    ortLoading = (globalThis.ort ? Promise.resolve(globalThis.ort) : new Promise((resolve, reject) => {
      if (typeof document === 'undefined') { reject(new Error('agent.js: no document to load onnxruntime-web into; pass { ort }')); return; }
      const s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onload = () => (globalThis.ort ? resolve(globalThis.ort) : reject(new Error('agent.js: ort global missing after ' + url)));
      s.onerror = () => reject(new Error('agent.js: failed to load ' + url));
      document.head.appendChild(s);
    })).then((ort) => configureOrt(ort, wasmPaths, numThreads));
  }
  return ortLoading;
}

export function configureOrt(ort, wasmPaths = ORT_DIST, numThreads = 1) {
  try {
    ort.env.wasm.wasmPaths = wasmPaths;
    ort.env.wasm.numThreads = numThreads;
    ort.env.wasm.proxy = false;
  } catch (e) { /* a runtime without a wasm backend (e.g. onnxruntime-node) */ }
  return ort;
}

// Numerically safe categorical sample over one logit row [off, off + n). Masked classes (< MASKED) get 0.
export function sampleRow(logits, off, n, random, deterministic = false) {
  let best = -1, max = -Infinity;
  for (let k = 0; k < n; k++) { const v = logits[off + k]; if (v >= MASKED && v > max) { max = v; best = k; } }
  if (best < 0) return 0;                      // everything masked: never happens with the exported graphs
  if (deterministic) return best;
  let z = 0;
  for (let k = 0; k < n; k++) { const v = logits[off + k]; if (v >= MASKED) z += Math.exp(v - max); }
  let u = random() * z;
  for (let k = 0; k < n; k++) {
    const v = logits[off + k];
    if (v < MASKED) continue;
    u -= Math.exp(v - max);
    if (u <= 0) return k;
  }
  return best;
}

export function softmaxRow(logits, off, n) {
  const p = new Float32Array(n);
  let max = -Infinity;
  for (let k = 0; k < n; k++) if (logits[off + k] >= MASKED && logits[off + k] > max) max = logits[off + k];
  let z = 0;
  for (let k = 0; k < n; k++) if (logits[off + k] >= MASKED) { p[k] = Math.exp(logits[off + k] - max); z += p[k]; }
  if (z > 0) for (let k = 0; k < n; k++) p[k] /= z;
  return p;
}

async function fetchSidecar(url) {
  const side = url.replace(/\.onnx(\?.*)?$/, '.json');
  if (side === url) return null;
  try {
    const r = await fetch(side);
    if (!r.ok) return null;
    return await r.json();
  } catch (e) { return null; }
}

// loadAgent(url, options) -> agent
//   options: { nvec, decideEvery, obsDim, ort, random, deterministic, wasmPaths, numThreads, executionProviders }
//   The sidecar `<url without .onnx>.json` written by rl/export_onnx.py supplies nvec, decideEvery, obsDim and the
//   training config when present; explicit options win.
export async function loadAgent(url, options = {}) {
  const side = await fetchSidecar(url);
  const ort = options.ort ? configureOrt(options.ort, options.wasmPaths ?? ORT_DIST, options.numThreads ?? 1)
    : await loadOrt({ wasmPaths: options.wasmPaths, numThreads: options.numThreads });
  const session = await ort.InferenceSession.create(url, {
    executionProviders: options.executionProviders ?? ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const nvec = options.nvec ?? side?.nvec ?? [9, 2, 2, 5];
  const decideEvery = options.decideEvery ?? side?.decide_every ?? 6;
  const obsDim = options.obsDim ?? side?.obs_dim ?? 323;
  const random = options.random ?? Math.random;
  const inputName = session.inputNames[0];
  const outNames = HEADS.every((h) => session.outputNames.includes(h)) ? HEADS : session.outputNames.slice(0, 4);
  let deterministic = !!options.deterministic;

  // Runs the graph on `count` stacked observations (Float32Array of count * obsDim) -> array of [move, join, leave, intent].
  async function actBatch(obs, count = Math.round(obs.length / obsDim), maskInfo = null) {
    if (obs.length < count * obsDim) throw new Error(`agent.js: observation has ${obs.length} floats, expected ${count * obsDim}`);
    const feed = {};
    feed[inputName] = new ort.Tensor('float32', obs.subarray ? obs.subarray(0, count * obsDim) : Float32Array.from(obs).subarray(0, count * obsDim), [count, obsDim]);
    const out = await session.run(feed);
    const det = maskInfo?.deterministic ?? deterministic;
    const actions = new Array(count);
    for (let b = 0; b < count; b++) {
      const a = new Array(4);
      for (let h = 0; h < 4; h++) {
        const data = out[outNames[h]].data;
        const n = nvec[h];
        const extra = maskInfo?.masks?.[h];       // optional extra allow-mask per head (0 = forbid)
        if (extra) {
          const row = Float32Array.from(data.subarray(b * n, b * n + n));
          for (let k = 0; k < n; k++) if (!extra[k]) row[k] = -Infinity;
          a[h] = sampleRow(row, 0, n, random, det);
        } else {
          a[h] = sampleRow(data, b * n, n, random, det);
        }
      }
      actions[b] = a;
    }
    return actions;
  }

  // Masked logits (Float32Array per head) for one observation; useful for debugging and for UIs.
  async function logits(obs) {
    const feed = {};
    feed[inputName] = new ort.Tensor('float32', obs, [1, obsDim]);
    const out = await session.run(feed);
    return outNames.map((nm) => out[nm].data);
  }

  return {
    url, ort, session, nvec, decideEvery, obsDim,
    macro: nvec[0] > 10,
    meta: side,
    get deterministic() { return deterministic; },
    set deterministic(v) { deterministic = !!v; },
    // act(obs, maskInfo) -> [move, join, leave, intent] for one observation (Float32Array from Game.observe(i)).
    async act(obs, maskInfo = null) { return (await actBatch(obs, 1, maskInfo))[0]; },
    actBatch,
    logits,
    // Converts an action to a Game.setInput() payload.
    toInput([move, join, leave, intent]) { return { move, join, leave, intent }; },
    async close() { try { await session.release(); } catch (e) { /* older runtimes */ } },
  };
}
