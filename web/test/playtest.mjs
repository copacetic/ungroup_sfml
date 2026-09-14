// Scripted playtest of the web app through the real UI: a "human" seat driven by a simple plan (keys only),
// screenshots every 20 s, the event feed and the end table written to a log. Usage:
//   PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/playtest.mjs [outdir] [agents] [bots] [seconds] [viewport]
// e.g. node web/test/playtest.mjs /tmp/pt 2 loyal,bail,grudge 240 1280x800
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
let playwright; try { playwright = require('playwright'); } catch (e) { playwright = require('/opt/node22/lib/node_modules/playwright'); }
const { chromium } = playwright;
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const webDir = path.resolve(here, '..');
const out = process.argv[2] || '/tmp/ungroup-playtest';
const agents = Number(process.argv[3] ?? 2);
const bots = (process.argv[4] || 'loyal,bail,grudge').split(',').filter(Boolean);
const seconds = Number(process.argv[5] ?? 240);
const [vw, vh] = (process.argv[6] || '1280x800').split('x').map(Number);
fs.mkdirSync(out, { recursive: true });
const log = fs.createWriteStream(path.join(out, 'playtest.log'));
const say = (s) => { log.write(s + '\n'); console.log(s); };

const port = 8100 + Math.floor(Math.random() * 500);
const server = spawn('python3', ['-m', 'http.server', String(port), '--directory', webDir], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));

const browser = await chromium.launch({ executablePath: process.env.CHROME || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: vw, height: vh } });
page.on('pageerror', (e) => say('PAGEERROR ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') say('CONSOLE ' + m.text()); });
await page.goto(`http://127.0.0.1:${port}/index.html#local`);
await page.waitForSelector('#humans');
await page.fill('#humans', '1');
await page.fill('#agents', String(agents));
for (const b of ['solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge']) await page.fill(`input.bot[data-bot=${b}]`, String(bots.filter((x) => x === b).length));
await page.fill('#name', 'Tester');
await page.selectOption('#kind', 'local').catch(() => {});
await page.click('#create');
await page.waitForSelector('#lobby.active');
await page.screenshot({ path: path.join(out, 'lobby.png') });
await page.click('#start');
await page.waitForFunction(() => window.__app && window.__app.frame && window.__app.me >= 0, null, { timeout: 30000 });
say(`round started: seats ${JSON.stringify(await page.evaluate(() => window.__app.seats.map((s) => s.type === 'bot' ? s.bot || s.name : s.type)))}`);

// --- key mapping calibration: press D and W briefly, see which way the body moves in world space
async function myBody() {
  return page.evaluate(() => { const S = window.__app; const f = S.frame; const me = S.me; const b = f.bodies.find((b) => b.m.includes(me)); return { x: b.x, y: b.y, n: b.m.length, pool: b.pool, head: b.head, stun: b.stun, t: f.t, R: f.R, banked: f.players[me].banked, join: f.players[me].join, leaving: f.players[me].leaving, brand: f.players[me].brand }; });
}
const keysDown = new Set();
async function setKeys(want) {
  for (const k of [...keysDown]) if (!want.has(k)) { await page.keyboard.up(k); keysDown.delete(k); }
  for (const k of want) if (!keysDown.has(k)) { await page.keyboard.down(k); keysDown.add(k); }
}
const b0 = await myBody();
await setKeys(new Set(['KeyD'])); await page.waitForTimeout(1200); const b1 = await myBody();
await setKeys(new Set(['KeyW'])); await page.waitForTimeout(1200); const b2 = await myBody();
await setKeys(new Set());
const sx = Math.sign(b1.x - b0.x) || 1, sy = Math.sign(b2.y - b1.y) || -1;   // world dx per D, world dy per W
say(`calibration: D moves x by ${(b1.x - b0.x).toFixed(3)}, W moves y by ${(b2.y - b1.y).toFixed(3)} -> sx ${sx} sy ${sy}`);
function keysToward(dx, dy) {
  const want = new Set();
  const n = Math.hypot(dx, dy); if (n < 0.03) return want;
  const ux = dx / n, uy = dy / n;
  if (ux * sx > 0.38) want.add('KeyD'); else if (ux * sx < -0.38) want.add('KeyA');
  if (uy * sy > 0.38) want.add('KeyW'); else if (uy * sy < -0.38) want.add('KeyS');
  return want;
}

const meta = await page.evaluate(() => ({ needs: window.__app.meta.needs, pads: window.__app.meta.pads, mines: window.__app.meta.mine_pos, types: window.__app.meta.mine_type, me: window.__app.me, cfg: window.__app.cfg }));
const me = meta.me;
const need = meta.needs[me];
say(`my needs ${JSON.stringify(need)} (types A B C D)`);
function padPos(i, R) { const r = Math.max(R - meta.cfg.pad_radius - 0.02, 0.1); return [r * Math.cos(meta.pads[i]), r * Math.sin(meta.pads[i])]; }

// --- the plan: fill cheap types first, group when open bodies are near, bank when the pool is worth it,
//     leave when the crown feeds others and I am close to done
let joinOn = false, leaveHold = 0, lastShot = -20, seen = 0, lastFeed = 0;
const t0 = Date.now();
while ((Date.now() - t0) / 1000 < seconds + 5) {
  const st = await page.evaluate(() => { const S = window.__app; return { end: !!S.endMsg, screen: S.screen, feed: S.feed ? S.feed.length : 0, alive: S.frame ? S.frame.alive : [], mines: S.frame ? S.frame.mines : [], bodies: S.frame ? S.frame.bodies : [], players: S.frame ? S.frame.players : [] }; });
  if (st.end) break;
  const b = await myBody();
  // progress
  const prog = need.map((n, t) => Math.min(1, b.banked[t] / n));
  const P = prog.reduce((a, x) => a + x, 0) / 4;
  const share = b.pool.map((p) => p / b.n);
  const remaining = need.map((n, t) => Math.max(0, n - b.banked[t]));
  // needed types ordered cheap first (secondaries 6 before primary 18)
  const order = [0, 1, 2, 3].filter((t) => remaining[t] - share[t] > 0.3).sort((a, c) => need[a] - need[c]);
  let target = null, why = '';
  const poolTotal = b.pool.reduce((a, x) => a + x, 0);
  const iAmHead = b.n > 1 && b.head === me;
  const closeToDone = remaining.reduce((a, x) => a + x, 0) - share.reduce((a, x) => a + x, 0) < 5;
  if (b.n > 1 && !iAmHead && closeToDone) {
    leaveHold = 5; why = 'leave: close to done and not head';
  }
  if (leaveHold > 0) { leaveHold -= 0.5; }
  if (b.n === 1 && poolTotal >= 7 || (iAmHead && poolTotal >= 10)) { target = padPos(me, b.R); why = 'bank'; }
  else if (b.n > 1 && !iAmHead && poolTotal >= 10) { target = padPos(b.head, b.R); why = 'follow to head pad'; }
  else {
    let best = null, bd = 9;
    for (const t of order) for (let m = 0; m < meta.mines.length; m++) {
      if (meta.types[m] !== t || !st.alive[m] || st.mines[m] < 1.5) continue;
      const d = Math.hypot(meta.mines[m][0] - b.x, meta.mines[m][1] - b.y) + 0.15 * order.indexOf(t);
      if (d < bd) { bd = d; best = m; }
    }
    if (best == null) { let bd2 = 9; for (let m = 0; m < meta.mines.length; m++) { if (!st.alive[m]) continue; const d = Math.hypot(meta.mines[m][0] - b.x, meta.mines[m][1] - b.y); if (d < bd2) { bd2 = d; best = m; } } }
    if (best != null) { target = meta.mines[best]; why = 'mine ' + best + ' type ' + 'ABCD'[meta.types[best]]; }
  }
  // joinable: open while I still have a lot to do and I am not about to bank solo with a big pool
  const wantJoin = P < 0.75 && !(b.n === 1 && poolTotal >= 6);
  if (wantJoin !== joinOn) { await page.keyboard.press('KeyJ'); joinOn = wantJoin; }
  if (leaveHold > 0) { if (!keysDown.has('KeyL')) { await page.keyboard.down('KeyL'); keysDown.add('KeyL'); } }
  else if (keysDown.has('KeyL')) { await page.keyboard.up('KeyL'); keysDown.delete('KeyL'); }
  const want = target ? keysToward(target[0] - b.x, target[1] - b.y) : new Set();
  if (keysDown.has('KeyL')) want.add('KeyL');
  await setKeys(want);
  if (b.t - lastShot >= 20) {
    lastShot = b.t;
    await page.screenshot({ path: path.join(out, `t${String(Math.round(b.t)).padStart(3, '0')}.png`) });
    say(`t=${b.t.toFixed(0)} P=${P.toFixed(2)} pos (${b.x.toFixed(2)},${b.y.toFixed(2)}) group ${b.n} head ${b.head} pool ${poolTotal.toFixed(1)} banked ${b.banked.map((x) => x.toFixed(0)).join('/')} join ${b.join} ${b.leaving >= 0 ? 'LEAVING' : ''} ${b.brand > 0 ? 'BRANDED' : ''} -> ${why}`);
  }
  if (st.feed > lastFeed) {
    const items = await page.evaluate((n) => window.__app.feed.slice(n).map((e) => (e.text || JSON.stringify(e))), lastFeed);
    for (const it of items) say('  feed: ' + it);
    lastFeed = st.feed;
  }
  await page.waitForTimeout(500);
}
await setKeys(new Set());
const end = await page.evaluate(() => window.__app.endMsg);
say('END ' + JSON.stringify(end));
await page.screenshot({ path: path.join(out, 'end.png') });
await browser.close();
server.kill();
log.end();
