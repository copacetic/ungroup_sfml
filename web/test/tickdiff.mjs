#!/usr/bin/env node
// Tick-level diff of web/src/engine.js against the canonical C++ core at full precision (17 significant
// digits of every body, player, ledger, mine and pickup field after every tick; bot actions, events,
// statistics; optionally observations, frame_json/meta_json and macro labels). The C++ side is
// web/test/probe.cpp, which #includes rl/native/ungroup.cpp and is built with -ffp-contract=off (the
// shipped .so uses FMA through -march=native, which no other build reproduces bit for bit):
//
//   g++ -O2 -std=c++17 -ffp-contract=off -o /tmp/ungroup_probe web/test/probe.cpp
//   node web/test/tickdiff.mjs <preset> <seats,comma> <seed> <maxTicks> <decideEvery> <rounds> <flags> [overridesJSON] [--quiet]
//   e.g. node web/test/tickdiff.mjs life policy,bail,policy,loyal,loyal,rammer 1 7200 6 1 15
//
// flags: 1 = script the external (policy/snapshot) seats through setInput (macro classes 10..23, move 9 with a
// direction, compass moves, stop, joinable toggles, held leaves, intent changes; the probe applies the same
// script through apply_actions/set_direction), 2 = compare observe() at decision ticks, 4 = compare frame()
// and meta() with frame_json/meta_json, 8 = compare botAction(..., macro=true) labels. rounds > 1 resets the same
// game with seed+1, seed+2, ... (exercises persist=1). Prints the first field differing by more than 1e-9 and
// the first one-ulp difference; the probe binary is looked up at $UNGROUP_PROBE or /tmp/ungroup_probe.
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { Game, preset, configToArray, seatType, MACRO_BASE } from '../src/engine.js';
const argv = process.argv.slice(2).filter((a) => a !== '--quiet');
const quiet = process.argv.includes('--quiet');
const [pname, seatsArg, seedArg, maxArg, deArg, roundsArg, flagsArg, ovArg] = argv;
const seats = seatsArg.split(',');
const seed = Number(seedArg), maxTicks = Number(maxArg), de = Number(deArg), rounds = Number(roundsArg), flags = Number(flagsArg);
const overrides = ovArg ? JSON.parse(ovArg) : {};
const cfg = preset(pname, Object.assign({ n_players: seats.length }, overrides));
import os from 'node:os';
import path from 'node:path';
const PROBE = process.env.UNGROUP_PROBE || '/tmp/ungroup_probe';
if (!fs.existsSync(PROBE)) { console.error(`probe binary ${PROBE} not found: g++ -O2 -std=c++17 -ffp-contract=off -o ${PROBE} web/test/probe.cpp`); process.exit(2); }
const cfgPath = path.join(os.tmpdir(), `ungroup_cfg_${process.pid}.txt`);
fs.writeFileSync(cfgPath, configToArray(cfg).map((v) => v.toPrecision(17)).join(','));
const out = execFileSync(PROBE, [cfgPath, seats.map((s) => seatType(s)).join(','), String(seed), String(maxTicks), String(de), String(rounds), String(flags)], { maxBuffer: 4 * 1024 * 1024 * 1024 });
fs.unlinkSync(cfgPath);
const ref = JSON.parse(out.toString());
const g = new Game(cfg, seed);
g.setSeats(seats);
g.decideEvery = de;
const n = seats.length;
const TOL = 1e-9;
let firstUlp = -1, firstUlpMsg = '', firstDiff = -1, firstDiffMsg = '', ulpCount = 0;
let cur = '';
function cmp(path, a, b) {
  if (typeof a === 'number' && typeof b === 'number') {
    if (a !== b) { ulpCount++; if (firstUlp < 0) { firstUlp = cur; firstUlpMsg = `${path}: js ${a} cpp ${b} (diff ${a - b})`; } }
    if (!(Math.abs(a - b) <= TOL) && firstDiff < 0) { firstDiff = cur; firstDiffMsg = `${path}: js ${a} cpp ${b} (diff ${a - b})`; }
  } else if (a !== b && firstDiff < 0) { firstDiff = cur; firstDiffMsg = `${path}: js ${String(a).slice(0, 300)} cpp ${String(b).slice(0, 300)}`; }
}
const canon = (e) => JSON.stringify(Object.fromEntries(Object.keys(e).sort().map((k2) => [k2, e[k2]])));
function deepEq(a, b, path) {
  if (typeof a === 'number' && typeof b === 'number') { if (a !== b) return `${path}: js ${a} cpp ${b}`; return null; }
  if (Array.isArray(a) && Array.isArray(b)) { if (a.length !== b.length) return `${path}: length js ${a.length} cpp ${b.length}`; for (let i = 0; i < a.length; i++) { const r = deepEq(a[i], b[i], `${path}[${i}]`); if (r) return r; } return null; }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
    if (ka.join() !== kb.join()) return `${path}: keys js ${ka} cpp ${kb}`;
    for (const k of ka) { const r = deepEq(a[k], b[k], `${path}.${k}`); if (r) return r; }
    return null;
  }
  if (a !== b) return `${path}: js ${JSON.stringify(a)} cpp ${JSON.stringify(b)}`;
  return null;
}
let idx = 0;
let events = [];
for (let round = 0; round < rounds && firstDiff < 0; round++) {
  if (round === 0) g.reset(seed, true); else g.reset(seed + round);
  g.events.length = 0;
  for (let k = 0; k <= maxTicks && firstDiff < 0; k++) {
    if (k > 0) {
      if (g.done) break;
      if ((k - 1) % de === 0) {
        const step = (k - 1) / de;
        g.events.length = 0;
        if (flags & 1) for (let i = 0; i < n; i++) if (g.seats[i] < 2) {
          const phase = (step + i) % 20;
          let move, dir;
          if (phase < 14) move = MACRO_BASE + phase;
          else if (phase < 18) { move = 9; dir = [(step * 7 + i * 3) % 11 - 5, (step * 5 + i) % 7 - 3]; }
          else if (phase === 18) move = 1 + (step % 8);
          else move = 0;
          const inp = { move, join: ((step + i) >> 2) & 1, leave: (step + 7 * i) % 40 < 8 ? 1 : 0, intent: ((step + i) % 5 === 0) ? (step % 4) + 1 : 0 };
          if (dir) inp.dir = dir;
          g.setInput(i, inp);
        }
      }
      g.tick();
    }
    const r = ref[idx++];
    if (!r) { firstDiff = cur; firstDiffMsg = 'C++ trace ended early'; break; }
    cur = `round ${round} tick ${k}`;
    if (r.round !== round || r.k !== k) { firstDiff = cur; firstDiffMsg = `trace misaligned: cpp round ${r.round} k ${r.k}`; break; }
    cmp('t', g.t, r.t); cmp('R', g.R, r.R);
    cmp('nbodies', g.bodies.length, r.bodies.length);
    for (let b = 0; b < Math.min(g.bodies.length, r.bodies.length); b++) {
      const B = g.bodies[b], C = r.bodies[b];
      cmp(`b${b}.m`, JSON.stringify(B.members), JSON.stringify(C.m));
      cmp(`b${b}.x`, B.x, C.x); cmp(`b${b}.y`, B.y, C.y); cmp(`b${b}.vx`, B.vx, C.vx); cmp(`b${b}.vy`, B.vy, C.vy);
      cmp(`b${b}.stun`, B.stun, C.stun); cmp(`b${b}.head`, B.head, C.head);
      for (let t = 0; t < 4; t++) cmp(`b${b}.pool${t}`, B.pool[t], C.pool[t]);
    }
    for (let i = 0; i < n; i++) {
      const P = g.players[i], Q = r.players[i];
      for (let t = 0; t < 4; t++) cmp(`p${i}.banked${t}`, P.banked[t], Q.banked[t]);
      cmp(`p${i}.intent`, P.intent, Q.intent); cmp(`p${i}.join`, P.joinable ? 1 : 0, Q.join);
      cmp(`p${i}.lt`, P.leaveTimer, Q.lt); cmp(`p${i}.cd`, P.joinCooldown, Q.cd);
      cmp(`p${i}.dx`, P.dx, Q.dx); cmp(`p${i}.dy`, P.dy, Q.dy); cmp(`p${i}.macro`, P.macro, Q.macro);
      cmp(`p${i}.brand`, g.brand[i], Q.brand); cmp(`p${i}.gs`, P.groupSince, Q.gs); cmp(`p${i}.lb`, P.lastBankT, Q.lb);
      cmp(`p${i}.ul`, g.lastUnjustLeave[i], Q.ul); cmp(`p${i}.pad`, P.padAngle, Q.pad); cmp(`p${i}.padx`, g.padX(i), Q.padx); cmp(`p${i}.pady`, g.padY(i), Q.pady);
    }
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const ij = i * 32 + j, q = r.pair[i * n + j];
      cmp(`pair${i},${j}.comember`, g.comemberTime[ij], q[0]); cmp(`pair${i},${j}.took`, g.tookFrom[ij], q[1]); cmp(`pair${i},${j}.bankedWhile`, g.bankedWhile[ij], q[2]);
      cmp(`pair${i},${j}.lastLeft`, g.lastLeftMe[ij], q[3]); cmp(`pair${i},${j}.partnerCd`, g.partnerCd[ij], q[4]); cmp(`pair${i},${j}.since`, g.pairSince[ij], q[5]); cmp(`pair${i},${j}.ended`, g.pairEnded[ij], q[6]);
    }
    for (let m = 0; m < cfg.n_mines; m++) { cmp(`mine${m}`, g.mineStock[m], r.mines[m]); cmp(`alive${m}`, g.mineAlive[m], r.alive[m]); cmp(`minex${m}`, g.mineX[m], r.minepos[m][0]); cmp(`miney${m}`, g.mineY[m], r.minepos[m][1]); }
    cmp('npicks', g.picks.length, r.picks.length);
    for (let q = 0; q < Math.min(g.picks.length, r.picks.length); q++) {
      cmp(`pick${q}.x`, g.picks[q].x, r.picks[q][0]); cmp(`pick${q}.y`, g.picks[q].y, r.picks[q][1]); cmp(`pick${q}.type`, g.picks[q].type, r.picks[q][2]); cmp(`pick${q}.ttl`, g.picks[q].ttl, r.picks[q][3]);
    }
    if (k > 0 && (k - 1) % de === 0) {
      // bot columns of the action array (external columns are the script, applied through setInput)
      for (let i = 0; i < n; i++) if (g.seats[i] >= 2) cmp(`act${i}`, JSON.stringify(Array.from(g._act.subarray(i * 4, i * 4 + 4))), JSON.stringify(r.act.slice(i * 4, i * 4 + 4)));
    }
    cmp('events', g.events.map(canon).join(';'), r.events.map(canon).join(';'));
    const s = g._stats;
    cmp('stats', JSON.stringify([s.merges, s.leaves, s.spills, s.banks, s.group_banks, s.remerge_fast, s.cancels, s.alliances, s.alliances_long, s.fair_banks, s.crowns]), JSON.stringify(r.stats.slice(0, 11)));
    cmp('stats.alliance_dur', s.alliance_dur, r.stats[11]); cmp('stats.units_taken', s.units_taken, r.stats[12]);
    cmp('done', g.done ? 1 : 0, r.done); cmp('winner', g.winner, r.winner); cmp('tw', g.timeoutWin ? 1 : 0, r.tw);
    if ((flags & 2) && r.obs && k % de === 0) {
      const o = g.observeAll();
      cmp('obs.len', o.length, r.obs.length);
      for (let q = 0; q < Math.min(o.length, r.obs.length); q++) { const c = Math.fround(r.obs[q]); if (o[q] !== c) { cmp(`obs[${q}] (player ${Math.floor(q / g.obsDim())}, feature ${q % g.obsDim()})`, o[q], c); if (Math.abs(o[q] - c) > 1e-6) break; } }
    }
    if (flags & 4) {
      const fr = g.frame(); const rf = r.frame;
      const d = deepEq(fr, rf, 'frame'); if (d) cmp('frameJSON', d, null);
      const m = g.meta(); const rm = r.meta;
      const md = deepEq({ needs: m.needs, pads: m.pads, mine_pos: m.mine_pos, mine_type: m.mine_type, winner: m.winner, timeout_win: m.timeout_win }, { needs: rm.needs, pads: rm.pads, mine_pos: rm.mine_pos, mine_type: rm.mine_type, winner: rm.winner, timeout_win: rm.timeout_win }, 'meta');
      if (md) cmp('metaJSON', md, null);
      for (let q = 0; q < rm.cfg.length; q++) if (Math.abs(m.cfg[q] - rm.cfg[q]) > 1e-5 * Math.max(1, Math.abs(rm.cfg[q]))) cmp('meta.cfg' + q, m.cfg[q], rm.cfg[q]);
    }
    if ((flags & 8) && r.labels && k % de === 0) {
      for (let i = 0; i < n; i++) if (g.seats[i] >= 2) cmp(`label${i}`, g.botAction(g.seats[i], i, true)[0], r.labels[i]);
    }
  }
}
const res = { preset: pname, seats: seatsArg, seed, de, rounds, flags, overrides, entries: ref.length, firstUlp, firstUlpMsg, ulpCount, firstDiff, firstDiffMsg };
console.log(quiet ? JSON.stringify(res) : JSON.stringify(res, null, 1));
process.exit(firstDiff === -1 ? 0 : 1);
