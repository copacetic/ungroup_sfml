// net_local.pw.mjs - Playwright smoke test for the 'local' (BroadcastChannel) transport: two pages of
// web/dev/net_demo.html in one Chromium context exchange 'hello'.
// Run: PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/net_local.pw.mjs
// Serves web/ itself on a free port with python3 -m http.server.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import net from 'node:net';
import assert from 'node:assert/strict';

const require = createRequire(import.meta.url);
let playwright;
try { playwright = require('playwright'); } catch (e) { playwright = require('/opt/node22/lib/node_modules/playwright'); }
const { chromium } = playwright;

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function freePort() {
  return new Promise((res) => { const s = net.createServer(); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });
}
async function waitPort(port, ms = 8000) {
  const t0 = Date.now();
  for (;;) {
    const ok = await new Promise((res) => { const s = net.connect(port, '127.0.0.1'); s.once('connect', () => { s.destroy(); res(true); }); s.once('error', () => res(false)); });
    if (ok) return;
    if (Date.now() - t0 > ms) throw new Error('server did not start');
    await new Promise((r) => setTimeout(r, 100));
  }
}

const port = await freePort();
const server = spawn('python3', ['-m', 'http.server', String(port), '--bind', '127.0.0.1', '--directory', webDir], { stdio: 'ignore' });
let browser = null;
try {
  await waitPort(port);
  const exe = process.env.CHROME_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
  const launch = { headless: true, args: ['--no-sandbox'] };
  try { browser = await chromium.launch({ ...launch, executablePath: exe }); } catch (e) { browser = await chromium.launch(launch); }
  const ctx = await browser.newContext();
  const room = 'pw' + Math.random().toString(36).slice(2, 8);
  const url = (n) => `http://127.0.0.1:${port}/dev/net_demo.html#r=${room}&n=${n}`;
  const p1 = await ctx.newPage();
  const p2 = await ctx.newPage();
  const errors = [];
  for (const p of [p1, p2]) p.on('pageerror', (e) => errors.push(String(e)));
  await p1.goto(url('one'));
  await p2.goto(url('two'));
  await p1.waitForFunction(() => window.__net && window.__net.ready && window.__net.hellos.length >= 1, null, { timeout: 10000 });
  await p2.waitForFunction(() => window.__net && window.__net.ready && window.__net.hellos.length >= 1, null, { timeout: 10000 });
  const s1 = await p1.evaluate(() => ({ id: window.__net.id, hellos: window.__net.hellos, peers: window.__net.peers }));
  const s2 = await p2.evaluate(() => ({ id: window.__net.id, hellos: window.__net.hellos, peers: window.__net.peers }));
  assert.equal(errors.length, 0, 'page errors: ' + errors.join('; '));
  assert.notEqual(s1.id, s2.id);
  assert.deepEqual(s1.hellos, [{ from: s2.id, name: 'two' }]);
  assert.deepEqual(s2.hellos, [{ from: s1.id, name: 'one' }]);
  assert.deepEqual(s1.peers, [s2.id]);
  assert.deepEqual(s2.peers, [s1.id]);
  // closing a tab is reported to the other one
  await p2.close();
  await p1.waitForFunction(() => window.__net.peers.length === 0, null, { timeout: 5000 });
  const left = await p1.evaluate(() => Array.from(document.querySelectorAll('li[data-kind=leave]')).map((li) => li.textContent));
  assert.equal(left.length, 1);
  console.log('net_local.pw: OK', JSON.stringify({ ids: [s1.id, s2.id], hellos: [s1.hellos, s2.hellos], left }));
} finally {
  if (browser) await browser.close();
  server.kill();
}
