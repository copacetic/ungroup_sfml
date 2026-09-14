// Screenshots of the game screen for visual comparison with the original SFML client (which renders at
// 1080x810). Starts a local round with one human seat and bots, drives the human a little so the arrows
// show, and saves chase-camera and arena-view frames.
//   PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/shots.mjs [outdir] [bots] [viewport] [seed]
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
const out = process.argv[2] || '/tmp/ungroup-shots';
const bots = (process.argv[3] || 'loyal,bail,grudge,solo,kidnap').split(',').filter(Boolean);
const [vw, vh] = (process.argv[4] || '1080x810').split('x').map(Number);
const seed = Number(process.argv[5] ?? 7);
fs.mkdirSync(out, { recursive: true });
const port = 8600 + Math.floor(Math.random() * 300);
const server = spawn('python3', ['-m', 'http.server', String(port), '--directory', webDir], { stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 800));
const browser = await chromium.launch({ executablePath: process.env.CHROME || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome', args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: vw, height: vh } });
page.on('pageerror', (e) => console.log('PAGEERROR ' + e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('CONSOLE ' + m.text()); });
await page.goto(`http://127.0.0.1:${port}/index.html#${process.env.HASH || 'local'}`);
await page.waitForSelector('#humans');
await page.fill('#humans', '1');
await page.fill('#agents', '0');
for (const b of ['solo', 'bail', 'loyal', 'kidnap', 'rammer', 'grudge']) await page.fill(`input.bot[data-bot=${b}]`, String(bots.filter((x) => x === b).length));
await page.fill('#name', 'Tester');
await page.fill('#seed', String(seed)).catch(() => {});
await page.selectOption('#kind', 'local').catch(() => {});
await page.click('#create');
await page.waitForSelector('#lobby.active');
await page.click('#start');
await page.waitForFunction(() => window.__app && window.__app.frame && window.__app.me >= 0, null, { timeout: 30000 });
console.log('mode', await page.evaluate(() => window.__app.renderer.mode));
const shot = async (name) => { await page.screenshot({ path: path.join(out, name + '.png') }); console.log('shot', name, 't=', await page.evaluate(() => window.__app.frame.t.toFixed(1))); };
await page.waitForTimeout(1500);
await shot('chase_start');
await page.keyboard.down('KeyD'); await page.waitForTimeout(2500); await shot('chase_moving_right');
await page.keyboard.up('KeyD'); await page.keyboard.down('KeyW'); await page.keyboard.press('KeyJ'); await page.waitForTimeout(2500); await shot('chase_joinable_up');
await page.keyboard.up('KeyW');
await page.waitForTimeout(6000); await shot('chase_later');
await page.keyboard.press('KeyC'); await page.waitForTimeout(1200); await shot('arena');
await page.keyboard.press('Tab'); await page.waitForTimeout(400); await shot('arena_panel');
await page.keyboard.press('Tab'); await page.keyboard.press('KeyC'); await page.waitForTimeout(8000); await shot('chase_end');
await browser.close(); server.kill();
