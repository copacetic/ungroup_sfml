#!/usr/bin/env node
// Conformance of web/src/engine.js against the canonical C++ core (rl/native/ungroup.cpp).
//
//   OMP_NUM_THREADS=1 python3 web/test/reference.py        # once: writes web/test/reference.json
//   node web/test/conformance.mjs [--games 24] [--ref web/test/reference.json] [--skip-ladder]
//
// Sections: reset exactness (seeds 1..20, life), ladder conformance (6 lineups x 2 presets, 24 rounds),
// determinism, macro steering, observation layout, legacy mine cap, performance. Exit code 1 on failure.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Game, preset, CFG_FIELDS, MACRO_BASE, crSin, crCos } from '../src/engine.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (name, dflt) => { const k = args.indexOf(name); return k >= 0 ? args[k + 1] : dflt; };
const REF = opt('--ref', path.join(HERE, 'reference.json'));
const SKIP_LADDER = args.includes('--skip-ladder');
const ref = JSON.parse(fs.readFileSync(REF, 'utf8'));
const GAMES = Number(opt('--games', ref.games || 24));

let failures = 0;
function check(ok, msg) { console.log((ok ? '  PASS ' : '  FAIL ') + msg); if (!ok) failures++; return ok; }
const fmt = (x, d = 3) => (typeof x === 'number' ? x.toFixed(d) : String(x));

// ------------------------------------------------------------------ 1. reset exactness
console.log('\n== reset exactness vs C++ (life preset, seeds 1..20)');
{
  check(JSON.stringify(CFG_FIELDS) === JSON.stringify(ref.cfg_fields), `config field order matches the Python Config (${CFG_FIELDS.length} fields)`);
  const lifeCfg = preset('life');
  const jsArr = CFG_FIELDS.map((k) => lifeCfg[k]);
  const cfgOk = jsArr.every((v, i) => Math.abs(v - ref.presets.life[i]) <= 1e-9 * Math.max(1, Math.abs(v)));
  check(cfgOk, 'preset("life") equals the Python preset("life")');
  let exact = 0, diffs = [];
  for (const e of ref.exact) {
    const g = new Game(preset('life'), e.seed);
    const m = g.meta();
    const fr = g.frame();
    const intents = fr.players.map((p) => p.intent);
    const same = JSON.stringify(m.needs) === JSON.stringify(e.needs) && JSON.stringify(m.pads) === JSON.stringify(e.pads)
      && JSON.stringify(m.mine_pos) === JSON.stringify(e.mine_pos) && JSON.stringify(m.mine_type) === JSON.stringify(e.mine_type)
      && JSON.stringify(intents) === JSON.stringify(e.intents);
    if (same) exact++; else diffs.push(e.seed);
  }
  check(exact === ref.exact.length, `needs, pads, intents, mine positions and types identical for ${exact}/${ref.exact.length} seeds` + (diffs.length ? ` (differ: ${diffs.join(',')})` : ''));
}

// ------------------------------------------------------------------ 1b. numerics: correctly rounded trig and printf rounding
console.log('\n== numerics (crSin/crCos vs glibc, printf tie rounding)');
{
  // glibc (correctly rounded) values on which V8's Math.sin/cos are one ulp off; the core must use crSin/crCos
  const GLIBC = [[0.2761655804417309, 0.2726685362302976, 0.9621080341365137], [0.27757788641275516, 0.2740270547665719, 0.9617219833485965],
    [0.6621258331119158, 0.6147948569643763, 0.7886870633211579], [-0.4096257016104661, -0.3982660232230096, 0.9172699573986544],
    [5.689250144660401, -0.5596265933296659, 0.8287448799469187], [5.453784155032886, -0.7375270909159466, 0.6753175476507781],
    [8.975872694337882, 0.43397952399815165, -0.9009227340623266], [2.1846599297966764, 0.8174285733027299, -0.576029971050347]];
  const trigOk = GLIBC.every(([x, s, c]) => crSin(x) === s && crCos(x) === c);
  check(trigOk, 'crSin/crCos reproduce glibc on 8 inputs where Math.sin/cos are one ulp off');
  // the eight compass directions and the exact points
  const compassOk = [1, 2, 3, 4, 5, 6, 7, 8].every((m) => { const a = 2 * Math.PI * (m - 1) / 8; return crCos(a) === Math.cos(a) && crSin(a) === Math.sin(a); })
    && crSin(0) === 0 && crCos(0) === 1 && Object.is(crSin(-0), -0);
  check(compassOk, 'compass directions, sin(0) and cos(0) are exact');
  // printf %.nf rounds an exact binary tie to even (0.125 -> 0.12), toFixed rounds it up; frame()/meta() use the printf rule
  const ties = [[0.125, 2, 0.12], [-0.125, 2, -0.12], [0.375, 2, 0.38], [2.5, 0, 2], [0.25, 1, 0.2], [-0.75, 1, -0.8], [0.0625, 3, 0.062], [1.5, 0, 2],
    [0.15, 1, 0.1], [0.35, 1, 0.3], [0.45, 1, 0.5], [2.675, 2, 2.67], [1e-7, 3, 0], [0.0000005, 6, 0], [1.0000005, 6, 1.000001], [0.3333335, 6, 0.333334]];
  const tieOk = ties.every(([x, p, want]) => Game.nums([x], 1, p)[0] === want);
  check(tieOk, 'frame number formatting matches printf %.nf on exact ties and near-ties (16 cases from glibc)');
}

// ------------------------------------------------------------------ 2. ladder conformance
function runLineup(lineup, cfgName, games, seed) {
  const cfg = preset(cfgName, { n_players: lineup.length });
  const n = lineup.length;
  const eps = [];
  const g = new Game(cfg, 1);
  g.setSeats(lineup);
  for (let e = 0; e < games; e++) {
    g.reset(seed + e, true);
    while (!g.done) g.step(g.decideEvery);
    const s = g.statsSummary;
    eps.push({ winner: g.winner, timeoutWin: g.timeoutWin, length: g.t, progress: [...Array(n).keys()].map((i) => g.progress(i)),
      merges: s.merges, leaves: s.leaves, spills: s.spills, banks: s.banks, groupBanks: s.groupBanks, alliances: s.alliances, alliancesLong: s.alliancesLong, allianceDur: s.allianceDur, avgGroup: s.avgGroup });
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const out = { games, eps, finished_early: mean(eps.map((e) => (e.timeoutWin ? 0 : 1))), length: mean(eps.map((e) => e.length)), avg_group: mean(eps.map((e) => e.avgGroup)),
    merges: mean(eps.map((e) => e.merges)), leaves: mean(eps.map((e) => e.leaves)), spills: mean(eps.map((e) => e.spills)), banks: mean(eps.map((e) => e.banks)),
    group_banks: mean(eps.map((e) => e.groupBanks)), alliances: mean(eps.map((e) => e.alliances)), alliances_long: mean(eps.map((e) => e.alliancesLong)), by_type: {} };
  for (const s of [...new Set(lineup)]) {
    const idx = lineup.map((x, i) => (x === s ? i : -1)).filter((i) => i >= 0);
    const perGame = eps.map((e) => mean(idx.map((i) => e.progress[i])));
    const m = mean(perGame);
    const sd = Math.sqrt(perGame.reduce((a, x) => a + (x - m) * (x - m), 0) / Math.max(1, perGame.length - 1));
    out.by_type[s] = { progress: m, se: sd / Math.sqrt(perGame.length), win_per_seat: eps.filter((e) => idx.includes(e.winner)).length / (idx.length * eps.length) };
  }
  return out;
}

if (!SKIP_LADDER) {
  const nofma = ref.ladder_nofma || null;
  console.log(`\n== ladder conformance (${GAMES} rounds per lineup, seeds ${ref.seed}..${ref.seed + GAMES - 1}; progress within 2 se of the C++, counts within 15%)`);
  if (nofma) console.log('   "noFMA" = the same C++ built with -ffp-contract=off (the shipped .so uses FMA through -march=native); it calibrates the floating-point noise floor');
  const rows = [];
  const pad = (s, w) => String(s).padEnd(w);
  const padL = (s, w) => String(s).padStart(w);
  console.log('  ' + pad('preset', 7) + pad('lineup', 30) + pad('stat', 16) + padL('C++', 9) + padL('JS', 9) + padL('tol', 9) + '  result' + (nofma ? '      noFMA  (noFMA vs C++)' : ''));
  const t0 = performance.now();
  let identical = 0, total = 0;
  for (const pname of Object.keys(ref.ladder)) {
    for (const key of Object.keys(ref.ladder[pname])) {
      const lineup = key.split(',');
      const cpp = ref.ladder[pname][key];
      const nf = nofma ? nofma[pname][key] : null;
      const js = runLineup(lineup, pname, GAMES, ref.seed);
      if (nf) {
        for (let e = 0; e < Math.min(GAMES, nf.eps.length); e++) {
          const a = js.eps[e], b = nf.eps[e];
          total++;
          if (a.merges === b.merges && a.leaves === b.leaves && a.spills === b.spills && a.banks === b.banks && a.winner === b.winner
            && a.progress.every((x, i) => Math.abs(x - b.progress[i]) < 1e-9)) identical++;
        }
      }
      const short = lineup.reduce((acc, s) => { acc[s] = (acc[s] || 0) + 1; return acc; }, {});
      const lname = Object.entries(short).map(([s, c]) => `${c} ${s}`).join(' + ');
      for (const s of Object.keys(cpp.by_type)) {
        const c = cpp.by_type[s], j = js.by_type[s];
        const tol = 2 * c.se;
        const ok = Math.abs(j.progress - c.progress) <= tol + 1e-12;
        rows.push({ ok, kind: 'progress', nfOk: nf ? Math.abs(nf.by_type[s].progress - c.progress) <= tol + 1e-12 : true });
        console.log('  ' + pad(pname, 7) + pad(lname, 30) + pad('progress ' + s, 16) + padL(fmt(c.progress), 9) + padL(fmt(j.progress), 9) + padL('±' + fmt(tol), 9)
          + (ok ? '  ok ' : '  OUT') + (nf ? padL(fmt(nf.by_type[s].progress), 11) + (rows[rows.length - 1].nfOk ? '  ok' : '  OUT') : ''));
      }
      for (const st of ['merges', 'leaves', 'banks', 'spills']) {
        const c = cpp[st], j = js[st];
        const tol = 0.15 * Math.abs(c);
        const rare = c < 1.0;  // a rate below one per round is noise-dominated at 24 rounds; report only
        const ok = rare ? true : Math.abs(j - c) <= tol + 1e-12;
        const nfOk = rare || !nf ? true : Math.abs(nf[st] - c) <= tol + 1e-12;
        rows.push({ ok, kind: st, rare, nfOk });
        console.log('  ' + pad(pname, 7) + pad(lname, 30) + pad(st, 16) + padL(fmt(c, 2), 9) + padL(fmt(j, 2), 9) + padL('±' + fmt(tol, 2), 9)
          + (rare ? (Math.abs(j - c) <= tol + 1e-12 ? '  ok (rare)' : '  n/a (rare)') : ok ? '  ok ' : '  OUT') + (nf ? padL(fmt(nf[st], 2), 11) + (rare ? '  (rare)' : nfOk ? '  ok' : '  OUT') : ''));
      }
    }
  }
  const prog = rows.filter((r) => r.kind === 'progress'), cnt = rows.filter((r) => r.kind !== 'progress' && !r.rare);
  const pOk = prog.filter((r) => r.ok).length, cOk = cnt.filter((r) => r.ok).length;
  const pNf = prog.filter((r) => r.nfOk).length, cNf = cnt.filter((r) => r.nfOk).length;
  // With two independent samples the difference of means has sd ~ 1.41 se, so "within 2 se" is a 1.4-sigma band that a
  // perfect port fails ~16% of the time per row (the 15% band on counts is not a statistical band at all). The pass
  // criterion is therefore: no more out-of-band rows than the C++'s own no-FMA build, or at most 20% of the rows.
  const allow = (rows_, nfOk) => Math.max(rows_.length - nfOk, Math.ceil(0.2 * rows_.length));
  check(prog.length - pOk <= allow(prog, pNf), `${pOk}/${prog.length} progress means within 2 se of the C++` + (nofma ? ` (no-FMA C++ build: ${pNf}/${prog.length})` : '') + `; up to ${allow(prog, pNf)} out-of-band rows allowed`);
  check(cnt.length - cOk <= allow(cnt, cNf), `${cOk}/${cnt.length} count statistics within 15% of the C++ (rates >= 1/round)` + (nofma ? ` (no-FMA C++ build: ${cNf}/${cnt.length})` : '') + `; up to ${allow(cnt, cNf)} allowed`);
  if (nofma) check(identical >= 0.5 * total, `${identical}/${total} rounds identical to the no-FMA C++ build in every outcome (winner, progress, merges, leaves, spills, banks); the rest diverge through one-ulp libm differences`);
  console.log(`  (${((performance.now() - t0) / 1000).toFixed(1)} s for ${(rows.length / 7 * GAMES) | 0} JS rounds)`);
}

// ------------------------------------------------------------------ 3. determinism
console.log('\n== determinism');
{
  const play = () => {
    const g = new Game(preset('life'), 7);
    g.setSeats(['policy', 'bail', 'bail', 'loyal', 'loyal', 'rammer']);
    const frames = [];
    let k = 0;
    while (!g.done) {
      // a scripted external seat: cycle through macro targets, hold leave for a while, toggle joinable
      g.setInput(0, { move: MACRO_BASE + (k % 12), join: (k >> 2) & 1, leave: k % 40 < 8 ? 1 : 0, intent: (k % 4) + 1 });
      g.step(6);
      frames.push(JSON.stringify(g.frame()));
      k++;
    }
    return { frames, obs: Array.from(g.observeAll()), stats: JSON.stringify(g.statsSummary) };
  };
  const a = play(), b = play();
  check(a.frames.length === b.frames.length && a.frames.every((f, i) => f === b.frames[i]), `same seed and inputs give identical frames (${a.frames.length} steps)`);
  check(a.stats === b.stats && a.obs.every((x, i) => x === b.obs[i]), 'identical statistics and observations');
  // reset() on a reused Game reproduces a fresh Game
  const g1 = new Game(preset('legacy'), 3); g1.setSeats(['bail', 'bail', 'bail', 'loyal', 'loyal', 'loyal']);
  while (!g1.done) g1.step(6);
  g1.reset(11, true);
  const g2 = new Game(preset('legacy'), 11); g2.setSeats(['bail', 'bail', 'bail', 'loyal', 'loyal', 'loyal']);
  for (let s = 0; s < 200; s++) { g1.step(6); g2.step(6); }
  check(JSON.stringify(g1.frame()) === JSON.stringify(g2.frame()), 'reset(seed) on a reused Game reproduces a fresh Game');
}

// ------------------------------------------------------------------ 4. macro steering
console.log('\n== macro steering (policy seat driven by the bail bot\'s macro labels, life preset)');
{
  const lineup = ['policy', 'bail', 'bail', 'bail', 'bail', 'bail'];
  const g = new Game(preset('life'), 1);
  g.setSeats(lineup);
  const N = 24;
  let sum = 0, sumBail = 0, macroUsed = 0, decisions = 0;
  for (let e = 0; e < N; e++) {
    g.reset(500 + e, true);
    while (!g.done) {
      const a = g.botAction('bail', 0, true);
      if (a[0] >= MACRO_BASE) macroUsed++;
      decisions++;
      g.setInput(0, { move: a[0], join: a[1], leave: a[2], intent: a[3] });
      g.step(6);
    }
    sum += g.progress(0);
    for (let i = 1; i < 6; i++) sumBail += g.progress(i) / 5;
  }
  const mean = sum / N;
  check(mean >= 0.60, `macro-driven seat mean progress ${mean.toFixed(3)} >= 0.60 over ${N} rounds (native bail seats ${(sumBail / N).toFixed(3)}; ${(100 * macroUsed / decisions).toFixed(0)}% of decisions were macro classes)`);
  // every macro class steers along the vector to its target (re-steered every tick by steerMacros())
  const h = new Game(preset('life'), 5);
  h.setSeats(['policy', 'loyal', 'loyal', 'loyal', 'loyal', 'loyal']);
  for (let k = 0; k < 60; k++) h.tick();
  let steerOk = true;
  for (let cls = MACRO_BASE; cls < MACRO_BASE + 14; cls++) {
    h.setInput(0, { move: cls, join: 0, leave: 0, intent: 0 });
    h.applyAction(0, cls, 0, 0, 0);
    const p = h.players[0], b = h.bodies[h.bodyIndex(0)];
    const tgt = { x: 0, y: 0 };
    if (h.macroTarget(0, cls, tgt)) {
      const dx = tgt.x - b.x, dy = tgt.y - b.y, n = Math.hypot(dx, dy);
      if (n > 1e-3 && (Math.abs(p.dx * dy - p.dy * dx) > 1e-9 || p.dx * dx + p.dy * dy <= 0)) steerOk = false;
    } else if (p.dx !== 0 || p.dy !== 0) steerOk = false;
  }
  check(steerOk, 'classes 10..23 (mines, own pad, head pad, 4 nearest bodies) steer along the vector to their target');
}

// ------------------------------------------------------------------ 5. observation layout
console.log('\n== observation layout (v3)');
{
  const g = new Game(preset('life'), 4);
  g.setSeats(['loyal', 'loyal', 'loyal', 'loyal', 'loyal', 'loyal']);
  check(g.obsDim() === 323 && g.observe(0).length === 323 && g.observeAll().length === 6 * 323, 'observe() has 323 floats for 8 mines (own 39 + 8 x 24 + 8 x 8 + 4 x 7)');
  let grp = null;
  for (let s = 0; s < 600 && !grp; s++) { g.step(6); grp = g.bodies.find((b) => b.n() > 1) || null; }
  check(grp !== null, 'a group formed to test is_head');
  if (grp) {
    const h = grp.head;
    const other = grp.members.find((j) => j !== h);
    g.brand[h] = 30;
    const fh = g.observe(h), fo = g.observe(other);
    check(fh[37] === 1 && fo[37] === 0, 'own block: is_head at index 37 (1 for the head, 0 for another member)');
    check(Math.abs(fh[38] - 0.5) < 1e-6 && fo[38] === 0, 'own block: brand/60 at index 38');
    const cnt = g.nearestOthers(other);
    let slot = -1;
    for (let q = 0; q < cnt; q++) if (g._ordIdx[q] === h) slot = q;
    const base = 39 + slot * 24;
    check(slot >= 0 && fo[base + 16] === 1 && fo[base + 10] === 1, `other slot ${slot}: present flag at +16 and same-body flag at +10`);
    check(fo[base + 22] === 1 && Math.abs(fo[base + 23] - 0.5) < 1e-6, 'other slot: is_head at +22 and brand/60 at +23');
    const f = g.observe(0);
    const mineBase = 39 + 8 * 24;
    let minesOk = true;
    for (let m = 0; m < 8; m++) { const t = g.mineType[m]; if (f[mineBase + m * 8 + 2 + t] !== 1 || f[mineBase + m * 8 + 7] !== (g.mineAlive[m] ? 1 : 0)) minesOk = false; }
    check(minesOk, 'mine block at 231: type one-hot at +2..+5, alive flag at +7; pickups block at 295');
    const l = new Game(preset('life', { obs_legacy: 1 }), 4);
    check(l.obsDim() === 305, 'obs_legacy=1 emits the v2 layout (37 + 8 x 22 + 64 + 28 = 305 floats)');
  }
}

// ------------------------------------------------------------------ 6. legacy mine cap
console.log('\n== legacy preset: bloom off keeps mine stock at the cap');
{
  const g = new Game(preset('legacy'), 9);
  g.setSeats(['policy', 'policy', 'policy', 'policy', 'policy', 'policy']);  // nobody moves
  for (let k = 0; k < 300; k++) g.tick();
  check(Array.from(g.mineStock).every((s) => s === 30), 'untouched mines stay exactly at mine_cap (30) for 300 ticks');
  const h = new Game(preset('legacy'), 9);
  h.setSeats(['loyal', 'loyal', 'loyal', 'loyal', 'loyal', 'loyal']);
  let over = false, minStock = 30, backAtCap = false;
  while (!h.done) {
    h.tick();
    for (const s of h.mineStock) { if (s > 30 + 1e-9) over = true; if (s < minStock) minStock = s; }
    if (h.t > 120 && Array.from(h.mineStock).some((s, m) => h.mineAlive[m] && s === 30)) backAtCap = true;
  }
  check(!over && minStock < 25 && backAtCap, `with six loyal miners no stock ever exceeds the cap, groups draw it down (min ${minStock.toFixed(1)}) and it regenerates back to exactly 30`);
  const b = new Game(preset('life'), 9);
  check(Array.from(b.mineStock).every((s) => s === 8), 'life preset starts mines at bloom_cap K = n_players + 2 = 8');
}

// ------------------------------------------------------------------ 7. performance
console.log('\n== performance (node, one 6-player round = up to 7200 ticks)');
{
  const g = new Game(preset('life'), 1);
  g.setSeats(['bail', 'bail', 'bail', 'loyal', 'loyal', 'loyal']);
  g.reset(100, true); while (!g.done) g.step(6);  // warm-up
  const N = 10;
  let ticks = 0;
  const t0 = performance.now();
  for (let e = 0; e < N; e++) { g.reset(200 + e, true); while (!g.done) g.step(6); ticks += g.tickCount; }
  const ms = (performance.now() - t0) / N;
  check(ms < 60, `${ms.toFixed(1)} ms per round (${(ticks / N) | 0} ticks; ${(1000 * ms / (ticks / N)).toFixed(1)} us per tick), target < 60 ms`);
}

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall checks passed');
process.exit(failures ? 1 : 0);
