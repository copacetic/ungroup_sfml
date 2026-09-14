# Ungroup in the browser

A static, serverless port of Ungroup: `index.html` plus ES modules under `src/`, no bundler, no
framework, no backend. The rules core is a line-by-line JavaScript port of the canonical C++ engine
(`rl/native/ungroup.cpp`, the same core the reinforcement-learning agents are trained on), the renderer
reproduces the look of the SFML client (a 1x world buffer upscaled 3x with nearest sampling, a camera
that chases you, Voronoi cells, the dotted background, the letter HUD), and
multiplayer is host-authoritative over WebRTC with signalling through Trystero's public strategies, so
there is no server of ours anywhere. Trained policies (`models/*.onnx`) can fill seats through
onnxruntime-web.

```
python3 -m http.server 8080 --directory web     # then open http://localhost:8080/
```

Any static file server works. Chromium and Edge refuse ES modules from `file://` (origin `null`), so a
local http server is needed there; Firefox opens `index.html` directly, and the page says so if it is
opened the wrong way.

## How to play

**The loop.** You are a circle in a round arena. Touch a mine to draw its stock (the four resource types
are the letters in the top-right HUD), then touch your home pad on the arena edge to bank everything you
carry. Each player needs a private mix of the four types (`count/goal` per letter); the first to complete
their needs wins the round, otherwise the highest progress wins when the clock runs out.

**Groups.** Bodies that touch while both are *joinable* merge into one group body with a shared pool.
A group of `n` mines `n²` times faster but moves `1/√n` as fast, in the mean direction its members push.
Touching any member's pad banks the whole pool to that member (under the crown rule only the group
head's pad pays out, so the head decides where the pool goes). Leaving is a held action with a timer:
the leaver takes a share of the pool weighted toward their declared intent, minus a forfeit, is ejected
and gets cooldowns; under the brand rule a betrayal marks the leaver publicly for a while. Hard
collisions spill units onto the floor as pickups and stun both bodies. Late in the round the arena
shrinks and the outer mines die; mines regenerate constantly (legacy) or bloom logistically and seed
their neighbours (bloom / life).

**Controls**

| key | action |
|---|---|
| `W A S D` / arrows, or hold the mouse or a finger | steer (continuous direction) |
| `J` or space | toggle *joinable* (touching bodies merge with you) |
| hold `L` | leave the group (release to cancel) |
| `1` `2` `3` `4` | declare an intent: the resource type you want (also colours your arrow) |
| `C` | toggle the whole-arena camera (the default is the original's close chase camera) |
| `Tab` or the `menu` button | open the panel: every player's progress, banked counts, group, crown, brand, leaving/stunned/cooldown state, the alliances and an event feed |

The game screen is the plain view of the original client: the buttons and the panel are hidden until
you open them (touch devices always get the buttons). The HUD is the original's: your four counts as
`banked/needed` with the tinted letters, off-screen mines of your declared type as a letter at the edge
of the view, your pad as a ring icon when it is off screen, and the round clock under the counts.
Toasts on the canvas explain what just happened to you.

**Presets** (the rule packages from `rl/ungroup/native.py`)

| preset | rules |
|---|---|
| `legacy` | the original rules: any member's pad banks the pool, constant mine regeneration |
| `crown` | + crown: only the group head's pad banks |
| `bloom` | + logistic mine bloom with neighbour seeding (capacity `n_players + 2`) |
| `life` | crown + brand (public betrayal mark) + bloom + whole-unit banking; the default and the preset the agents were trained on |
| `series` | life + persistence across rounds (ledgers and grudges carry over) |

**Modes**

* **Create a room** (host): choose human seats, bots by type (`solo`, `bail`, `loyal`, `kidnap`,
  `rammer`, `grudge`), agent seats (when a model is found next to the page), preset, seed and the number
  of rounds (0 = endless). Share the link, `index.html#r=CODE`; everyone who opens it takes a free human
  seat, the rest spectate. Empty human seats sit idle until someone joins; a player who leaves mid-round
  is replaced by a bot, and a page refresh reclaims the seat. Rounds restart 8 s after they end.
* **Join a room**: open the host's link, or paste the code on the home screen.
* **Watch bots**: a bots-only game with no network at all, eight trained agents by default. `index.html#watch&agents=8&seed=5&preset=life&time=240`, or `bots=bail,loyal,rammer` for scripted bots
  starts one directly.

## Hosting with no server

The whole game is the `web/` folder. Put it on any static host and open `index.html`:

* **GitHub Pages**: `.github/workflows/pages.yml` uploads `web/` with `actions/upload-pages-artifact`
  and publishes it with `actions/deploy-pages` on every push to `main` (or `master`) that touches
  `web/`, after running the Node tests. Enable Pages once with *Settings → Pages → Source: GitHub
  Actions*; the page then lives at `https://<owner>.github.io/<repo>/`. `web/.nojekyll` keeps Pages from
  running Jekyll over the folder.
* **Anything else**: `python3 -m http.server`, nginx, Netlify, S3, a USB stick with Firefox. Nothing is
  built and nothing is fetched from us at runtime; the only external resources, pinned to exact
  versions on jsDelivr, are Trystero (WebRTC signalling, `src/net.js`) and onnxruntime-web (agent seats,
  `src/agent.js`). Without them the game still runs in the `local` transport and with scripted bots.

**Rooms.** The host's browser runs the engine at 30 Hz and is the single authority: it applies every
client's latest input each tick and broadcasts a snapshot every 2 ticks (15 Hz) with the events since the
previous one; clients render an interpolated view about 100 ms behind the newest snapshot (more when
the connection jitters). Bots and agents run on the host too. The room code is the only thing peers
share: `src/net.js` joins the Trystero room `ungroup-web-v1/CODE`, and peers find each other through
Trystero's public signalling strategies (BitTorrent trackers first, nostr and MQTT relays as fallbacks)
and then talk directly over WebRTC data channels (Google STUN plus the public Open Relay TURN service
for peers behind strict NATs). No server of ours exists, nothing is stored anywhere, and a room dies
with its host. The `rtc` transport is a `MultiTransport`: one peer id over many paths. It opens a
`BroadcastChannel` for the other tabs of this browser at once, then joins the room through every
signalling strategy as its module loads (nostr relays, MQTT brokers and BitTorrent trackers, each with
its own list of public relays), and can take hand-made direct data channels. Peers are recognised across
paths by a handshake and spoken to over their best path only, so a friend found through two relays is
one peer and a tab of the same browser is found even when no relay answers. The lobby shows the
signalling state per path (relays answering, peers, direct links) and what to do when nothing answers.

**Invite links (no relay at all).** When no relay works for someone, the host presses *Make an invite
link* in the lobby: the link carries the host's WebRTC offer (compressed, about 600 characters), the
friend's browser answers it and shows a reply code of the same size, the friend sends that code back
over any chat, the host pastes it, and the two are connected directly (one invite per friend; a refresh
needs a new one). The `local` transport alone (`#r=CODE&local`, or automatically on `file://`) is the
`BroadcastChannel` by itself; it is what the tests use.

**The host's tab runs the game.** Its tick timer lives in a Web Worker, because browsers slow the
main-thread timers of a background tab to once a second; with the worker the simulation keeps its 30 Hz
even while the host looks at another tab. Should the timer still be slowed (some browsers throttle
workers too), the host catches up in one-second bursts, both the host and the clients get a status
line saying so, and clients widen their playback lag (up to 1.2 s) so movement stays smooth, only late.

**When the host leaves** (closes the tab, loses the connection or crashes) every client shows a
*host left* overlay within about two seconds and can go back home; the room is gone because the game
state lived only in the host's page. If the host merely stalled and speaks again the overlay is hidden.
A client that stops sending input for a second has its body stopped by the host; a client that comes
back (same tab, refresh) reclaims its seat.

**Browsers.** Any current Chromium, Firefox or Safari with WebGL2 (the renderer falls back to a 2D
canvas without it, with blobbier cells). WebRTC data channels and `BroadcastChannel` are needed for the
two transports. Touch devices get on-screen buttons. The chase camera shows the same slice of the
world as the original's 1080x810 window did (3 CSS px per world pixel, 2 on screens narrower than
700 px), and the panel slides over the canvas on narrow screens.

## Tests

All tests are plain Node 22 scripts (no packages) except the two browser tests, which need Playwright's
Chromium. `python3 -m http.server` is spawned by the browser tests themselves.

```
node web/test/session.test.mjs            # host/client protocol over an in-process fake transport (8)
node web/test/session_robust.test.mjs     # adversarial: loss/jitter, silent clients, refresh, throttled host, host crash, 20 peers, 32-player snapshots (9)
node web/test/net.test.mjs                # transports: BroadcastChannel, mocked Trystero in both API shapes, the multi-path transport, direct channels and signal codes (6)
node web/test/conformance.mjs             # engine vs the C++ core: bit-exact resets, ladder statistics, determinism, macro steering, observation layout, mine cap, perf
node web/test/conformance.mjs --skip-ladder                              # the fast checks only
OMP_NUM_THREADS=1 python3 web/test/reference.py                          # regenerate test/reference.json from the C++ core (needs rl/)
PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/net_local.pw.mjs # two tabs exchange hello over BroadcastChannel
PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/e2e.mjs          # the whole app: create, join, play, watch (screenshots in --shots DIR)
PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/shots.mjs DIR     # game-screen screenshots at 1080x810 for comparison with the original client
PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node web/test/playtest.mjs DIR  # a scripted human plays a round through the UI, screenshots every 20 s
```

`web/test/README.md` describes the conformance methodology (why the ladder is compared statistically:
the shipped C++ core uses FMA, the JS engine reproduces a `-ffp-contract=off` build frame for frame).
`web/test/probe.cpp` + `web/test/tickdiff.mjs` diff the two engines tick by tick at full precision.

## Module map

| file | what |
|---|---|
| `index.html`, `src/app.js`, `src/ui.css` | the app: home (create / join / watch), lobby with share link, game screen (canvas, HUD, panel, toasts), end screen with play-again countdown, spectator and host-left overlays, keyboard / pointer / touch input, hash routing (`#r=CODE[&local][&n=NAME]`, `#watch[&bots=..&agents=..&seed=..&preset=..&time=..]`) |
| `src/engine.js` | the rules core, ported function by function from `rl/native/ungroup.cpp`: `CONFIG_DEFAULTS`, `PRESETS`, `preset()`, `Game` (`reset`, `setSeats`, `setInput`, `tick`/`step`, `frame()`/`meta()` in the replay schema, `observe()` in the v3 layout, `stats`, the six scripted bots, macro targets) |
| `src/rng.js` | `std::mt19937_64` plus the libstdc++ `uniform_real`, `uniform_int`, `discrete_distribution` algorithms, so a seed produces the same needs, pads and mines as the C++ core |
| `src/render.js` | `createRenderer(canvas, assets)`: the look of the SFML client (the 1x world buffer blitted up with nearest sampling, chase camera, dark disc with the grey out-of-bounds, the two dotted layers, `voronoi_counts` cells for bodies and mines, direction arrows in intent colours, joinable / ungroup rings, sparks on collisions and on every mine hit while a body gathers, the letter HUD in the monogram font from a crisp glyph atlas) plus the pixel-style extras for the new rules (pads, crown notch, brand rim, pickups, clock) in WebGL2 with a 2D-canvas fallback (`#...&2d` forces it) |
| `src/net.js` | `createTransport({kind, room})`: `local` (BroadcastChannel with heartbeats), `rtc` (a `MultiTransport`: the BroadcastChannel plus Trystero over nostr / mqtt / torrent signalling with curated relay lists, pinned version, STUN + TURN, plus `DirectTransport` channels from `makeInvite` / `acceptInvite`) and `rtc-only`, behind one `{id, peers, onPeer, onLeave, send, on, status, close}` interface |
| `src/session.js` | `Host` and `Client`: lobby, `hello`/`lobby`/`start`/`input`/`snap`/`end` protocol, 30 Hz fixed-step engine on the host, 15 Hz snapshots with an event window, input validation and stale-input timeout, seat reclaim by token, host heartbeat and host-left detection, rate-controlled interpolated client view, automatic round restart |
| `src/agent.js` | `loadAgent(url)`: a trained policy from `models/*.onnx` through onnxruntime-web, returning `act(obs) -> [move, join, leave, intent]` |
| `models/` | exported policies (`v9_150`: legacy 9-way moves, `v10_100`: macro targets, the peak snapshot; `v10_latest` is the drifted end of the same run) with their sidecar json; see `models/README.md` for regeneration with `rl/export_onnx.py` |
| `assets/` | `monogram.ttf`, `dotted_background.png`, `mine_pattern.png`, `spark.png`, the four letter sprites (copied from `resources/`) |
| `dev/` | `render_demo.html` (plays a recorded replay through the renderer), `net_demo.html` (transport smoke page), `agent_demo.html` (six agent seats through the engine, statistics vs the Python evaluation) |
| `test/` | the tests above, `reference.py`/`reference.json` (C++ reference numbers), `probe.cpp`/`tickdiff.mjs` (tick-level diffing), `README.md` |

## Known gaps

* Relay signalling could not be exercised end to end in the sandbox this was built in (no WebSocket
  upgrades through its proxy); the wrappers are covered by mocked-room unit tests, and the invite-link
  handshake was driven for real between two isolated browser contexts. Public relays are best-effort:
  joining another device can take several seconds, and a network that blocks every `wss://` relay
  leaves the invite link (and the same-browser channel) as the way in; the lobby says so.
* Room settings are fixed when the room is created; there is no in-lobby editing (`Host.updateSettings`
  exists but has no UI).
* The host's screen is drawn from `LocalView`, an interpolated playback 50 ms behind its own 30 Hz
  simulation (clients are interpolated the same way), so what the host sees is that little bit late.
* Agent seats need onnxruntime-web from jsDelivr and a model next to the page; without them they play
  as `solo` bots (noted in the console only). Inference calls are serialised, so many agent seats with a
  short `decideEvery` cost host frame time.
* Engine numerics: resets are bit-exact with the C++ core, and whole games match a `-ffp-contract=off`
  build frame for frame, but the shipped `libungroup.so` uses FMA, so long games diverge chaotically
  from it at contact thresholds; only statistics are compared, not trajectories.
* The 2D-canvas fallback approximates the Voronoi cells with blobs and anti-aliases the circle edges.
* `dev/render_demo.html` needs `dev/sample_replay.json`, which is git-ignored; generate one with
  `record_game` from `rl/ungroup/native.py`.
* The chase camera shows about a third of the arena, as the original did; the whole-arena view (`C`)
  is the way to see everyone at once.
