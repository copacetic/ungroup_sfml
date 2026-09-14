// e2e.mjs - Playwright end-to-end test of the playable app (web/index.html) over the 'local' transport.
//
//   PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/e2e.mjs [--shots DIR] [--headed]
//
// Serves web/ on port 8090 with python3 -m http.server (reused if something already listens there).
// Page A creates a room (#local; 3 human seats, 3 bots, seed 12) and copies the link; page B opens the
// link and is seated; A starts; both pages receive frames for 12 s; B holds D for 2 s and its body's x
// increases in B's own view; screenshots of both pages at 8 s; then page C runs the bots-only watch
// mode for 10 s. Exit code 1 on any failed assertion or page error.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); } catch (e) { playwright = require('/opt/node22/lib/node_modules/playwright'); }
const { chromium } = playwright;

const args = process.argv.slice(2);
const argVal = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SHOTS = argVal('--shots', '/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/web/app');
const PORT = +argVal('--port', '8090');
fs.mkdirSync(SHOTS, { recursive: true });

function portOpen(port) {
  return new Promise((res) => { const s = net.connect(port, '127.0.0.1'); s.once('connect', () => { s.destroy(); res(true); }); s.once('error', () => res(false)); });
}
async function waitPort(port, ms = 8000) {
  const t0 = Date.now();
  while (!(await portOpen(port))) { if (Date.now() - t0 > ms) throw new Error('server did not start'); await sleep(100); }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log('[e2e]', ...a);

let server = null;
if (!(await portOpen(PORT))) {
  server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1', '--directory', webDir], { stdio: 'ignore' });
  await waitPort(PORT);
  log('serving', webDir, 'on', PORT);
} else log('reusing server on', PORT);

const base = `http://127.0.0.1:${PORT}/index.html`;
let browser = null;
const failures = [];
const check = (cond, msg) => { if (cond) log('ok  ', msg); else { log('FAIL', msg); failures.push(msg); } };

try {
  const exe = process.env.CHROME_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
  const launch = { headless: !args.includes('--headed'), args: ['--no-sandbox'] };
  try { browser = await chromium.launch({ ...launch, executablePath: exe }); } catch (e) { browser = await chromium.launch(launch); }
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const errors = [];
  const consoleErrors = [];
  const newPage = async (label) => {
    const p = await ctx.newPage();
    p.on('pageerror', (e) => errors.push(label + ': ' + String(e)));
    p.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(label + ': ' + m.text()); });
    return p;
  };

  // ---------------------------------------------------------------- A creates a room
  const A = await newPage('A');
  await A.goto(base + '#local&n=Alice');
  await A.waitForSelector('#home.active');
  await A.fill('#humans', '3');
  for (const b of ['solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge']) await A.fill(`.bot[data-bot=${b}]`, '0');
  await A.fill('.bot[data-bot=bail]', '2');
  await A.fill('.bot[data-bot=loyal]', '1');
  await A.selectOption('#preset', 'life');
  await A.fill('#seed', '12');
  check((await A.inputValue('#kind')) === 'local', 'A: #local selects the local transport');
  await A.click('#create');
  await A.waitForSelector('#lobby.active', { timeout: 10000 });
  await A.click('#copy');
  const link = await A.inputValue('#link');
  check(/#r=[A-Z0-9]{6}&local$/.test(link), 'A: share link looks right: ' + link);
  const seatsA = await A.$$eval('#seats li', (els) => els.map((e) => e.textContent));
  check(seatsA.length === 6, 'A: lobby lists 6 seats (' + seatsA.length + ')');
  check(/Alice/.test(seatsA[0]) && /you/.test(seatsA[0]), 'A: host sits in seat 1: ' + seatsA[0].trim());
  const connA = await A.textContent('#conn');
  check(/local/.test(connA), 'A: connection state shown: ' + connA);

  // ---------------------------------------------------------------- B joins from the link
  const B = await newPage('B');
  await B.goto(link + '&n=Bob');
  await B.waitForFunction(() => window.__app && window.__app.client && window.__app.client.seat >= 0, null, { timeout: 15000 });
  await B.waitForSelector('#lobby.active');
  const seatB = await B.evaluate(() => window.__app.client.seat);
  check(seatB === 1, 'B: seated in seat 2 (index ' + seatB + ')');
  const seatsB = await B.$$eval('#seats li', (els) => els.map((e) => e.textContent));
  check(seatsB.some((s) => /Bob/.test(s) && /you/.test(s)), 'B: sees itself in the lobby');
  await A.waitForFunction(() => /Bob/.test(document.querySelector('#seats').textContent), null, { timeout: 5000 });
  check(true, 'A: sees Bob in the lobby');
  check(/2 peers|1 peer/.test(await A.textContent('#conn')), 'A: peer count in the status bar: ' + (await A.textContent('#conn')));

  // ---------------------------------------------------------------- A starts
  await A.click('#start');
  await A.waitForSelector('#game.active', { timeout: 5000 });
  await B.waitForSelector('#game.active', { timeout: 5000 });
  await A.waitForFunction(() => window.__app.frames > 5, null, { timeout: 5000 });
  await B.waitForFunction(() => window.__app.frames > 5, null, { timeout: 5000 });
  const t0 = Date.now();
  const framesA0 = await A.evaluate(() => window.__app.frames);
  const framesB0 = await B.evaluate(() => window.__app.frames);
  check((await B.evaluate(() => window.__app.me)) === 1, 'B: plays seat index 1 in the game');
  check((await A.evaluate(() => window.__app.me)) === 0, 'A: plays seat index 0 in the game');

  // ---------------------------------------------------------------- B holds D for 2 s
  const bodyX = (page) => page.evaluate(() => { const S = window.__app; const b = S.frame && S.frame.bodies.find((q) => q.m.includes(S.me)); return b ? b.x : null; });
  await sleep(600);
  await B.bringToFront();
  await B.focus('body');
  const x0 = await bodyX(B);
  await B.keyboard.down('KeyD');
  await sleep(2000);
  await B.keyboard.up('KeyD');
  const x1 = await bodyX(B);
  check(x0 !== null && x1 !== null && x1 > x0 + 0.05, `B: x increased while holding D (${x0} -> ${x1})`);
  const dirSent = await B.evaluate(() => window.__app.lastSent && window.__app.lastSent.dir.join(','));
  check(dirSent === '0,0', 'B: releasing D sends a stop (' + dirSent + ')');
  // A's view agrees on where B is (host-authoritative). B's view is deliberately 100-200 ms behind the
  // host, so compare once the stop has propagated and the body has settled (vel_lerp 6 -> ~1 s).
  await sleep(1200);
  const xHost = await A.evaluate(() => { const S = window.__app; const b = S.frame.bodies.find((q) => q.m.includes(1)); return b ? b.x : null; });
  const xB = await bodyX(B);
  check(xHost !== null && xB !== null && Math.abs(xHost - xB) < 0.03, `A: host view of B's body agrees after the stop (${xHost} vs ${xB})`);
  // toggle joinable and declare an intent on B, check they reach the host frame
  await B.keyboard.press('KeyJ');
  await B.keyboard.press('Digit3');
  await sleep(400);
  const pB = await A.evaluate(() => window.__app.frame.players[1]);
  check(pB.join === true, 'B: joinable toggle reached the host');
  check(pB.intent === 2, 'B: intent 3 reached the host (intent index ' + pB.intent + ')');

  // ---------------------------------------------------------------- screenshots at 8 s
  await sleep(Math.max(0, 8000 - (Date.now() - t0)));
  await A.screenshot({ path: path.join(SHOTS, 'pageA_8s.png') });
  await B.screenshot({ path: path.join(SHOTS, 'pageB_8s.png') });
  log('screenshots', path.join(SHOTS, 'pageA_8s.png'), path.join(SHOTS, 'pageB_8s.png'));

  // ---------------------------------------------------------------- 12 s of frames
  await sleep(Math.max(0, 12000 - (Date.now() - t0)));
  const framesA1 = await A.evaluate(() => window.__app.frames);
  const framesB1 = await B.evaluate(() => window.__app.frames);
  const snapsB = await B.evaluate(() => window.__app.client.snapCount);
  const tA = await A.evaluate(() => window.__app.frame.t);
  const tB = await B.evaluate(() => window.__app.frame.t);
  check(framesA1 - framesA0 >= 250, `A: received ${framesA1 - framesA0} host frames in 12 s`);
  check(framesB1 - framesB0 >= 250, `B: rendered ${framesB1 - framesB0} view frames in 12 s (${snapsB} snapshots)`);
  check(snapsB >= 120, `B: >= 120 snapshots in 12 s (${snapsB})`);
  check(tA > 11 && tA - tB < 0.6 && tA - tB > 0, `B lags the host by ${(tA - tB).toFixed(3)} s (host t ${tA})`);
  const panelB = await B.textContent('#panel');
  check(/Bob/.test(panelB) && /Alice/.test(panelB) && /bail bot/.test(panelB), 'B: panel lists the players and seat types');
  check(/room [A-Z0-9]{6}/.test(panelB) && /seat 2/.test(panelB), 'B: panel shows the room and own seat');
  const modeA = await A.evaluate(() => window.__app.renderer.mode);
  log('renderer mode', modeA);
  const feedA = await A.$$eval('#feed li', (els) => els.length);
  log('event feed entries on A:', feedA);

  // B leaves: A's lobby notice reports the handover to a bot
  await B.close();
  await A.waitForFunction(() => window.__app.lobby && /Bob left/.test(window.__app.lobby.notice || ''), null, { timeout: 10000 });
  check(true, 'A: Bob leaving is noticed (' + (await A.evaluate(() => window.__app.lobby.notice)) + ')');
  await A.close();

  // ---------------------------------------------------------------- C: watch mode
  const C = await newPage('C');
  await C.goto(base + '#watch&seed=5&bots=bail,bail,loyal,loyal,rammer,solo');
  await C.waitForSelector('#game.active', { timeout: 5000 });
  await C.waitForFunction(() => window.__app.frames > 5, null, { timeout: 5000 });
  const fc0 = await C.evaluate(() => window.__app.frames);
  await sleep(10000);
  const fc1 = await C.evaluate(() => window.__app.frames);
  const tC = await C.evaluate(() => window.__app.frame.t);
  const meC = await C.evaluate(() => window.__app.me);
  check(fc1 - fc0 >= 200, `C: watch mode produced ${fc1 - fc0} frames in 10 s (t=${tC})`);
  check(meC === -1, 'C: spectator (me = -1)');
  check(/no network/.test(await C.textContent('#conn')), 'C: no network needed');
  await C.screenshot({ path: path.join(SHOTS, 'pageC_watch_10s.png') });
  log('screenshot', path.join(SHOTS, 'pageC_watch_10s.png'));
  await C.close();

  const realErrors = consoleErrors.filter((m) => !/favicon/.test(m));
  check(errors.length === 0, 'no page errors' + (errors.length ? ': ' + errors.join(' | ') : ''));
  check(realErrors.length === 0, 'no console errors' + (realErrors.length ? ': ' + realErrors.join(' | ') : ''));
} catch (e) {
  failures.push('exception: ' + (e && e.stack || e));
  console.error(e);
} finally {
  if (browser) await browser.close();
  if (server) server.kill();
}
if (failures.length) { console.log('e2e: FAILED\n - ' + failures.join('\n - ')); process.exit(1); }
console.log('e2e: OK');
