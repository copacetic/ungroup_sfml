# Ungroup: Product Requirements and State of the Game

*Written September 2026 against commit `d870d29` (last upstream change January 2021). Everything below was checked by reading the full source (about 8,000 lines of C++), building it on a current toolchain, running the unit tests, running bot-vs-bot games over the real network stack, and running bot-vs-bot games in a new offline simulator (`ug-sim`, added alongside this document).*

> **Addendum, 7 September 2026.** The roadmap in section 7 has been superseded. After the first RL run, three adversarial reviews found that the reported alliance behavior was a rule exploit, that the SFML engine should be frozen rather than extended, and that the C++ core under `rl/native` should be the only rules implementation. The rules were revised (held leave with cooldowns, spill stun, intent lock, timeout tiebreak, 30 Hz physics), the trainer was rebuilt, and the roadmap now starts with humans playing the new rules. See [PLAN_REVIEW_2026-09.md](PLAN_REVIEW_2026-09.md) for the findings, their verification status, the ladder results under the new rules, and the revised plan.

## 1. What the game is trying to be

Ungroup is a real-time, top-down multiplayer game about **temporary alliances**. Each player is a circle with a private shopping list of four resource types. Resources are mined by bumping into mines. Players can merge circles into a group that moves and mines as one body, and can leave a group at any moment. The intended tension is:

- You need partners to move fast and mine fast.
- Your partners want different resources than you and will leave when it suits them.
- The first player to fill their list wins, so every alliance has a built-in expiry date.

The intended experience is a five-minute round with friends where you form, exploit, and abandon alliances, and where watching who groups with whom is as interesting as the racing.

### Target outcomes (the user's stated goals)

1. **Playable with friends** with very little setup, ideally in a browser via a shared link.
2. **Smooth** networked movement. The current interpolation "never worked quite right".
3. **Strong AI opponents** for playtesting alone or filling out a lobby.
4. **A bot-only mode** that is worth watching, and an environment suitable for training agents with reinforcement learning (RL).
5. A game design where the alliance mechanic actually matters, so that the optimal strategy is not trivially greedy.

## 2. Where it stands today

### 2.1 Feature inventory

| Area | Status | Notes |
| --- | --- | --- |
| Core loop: move, bump mines, collect, win | Done | Works end to end, offline and networked. |
| Grouping: join on contact when both sides are "joinable" | Done | Any one member can make the whole group joinable. |
| Ungrouping | Done | The leaver spawns a new circle at the group's position. |
| Group movement: velocity is the sum of member directions | Done | Members pulling in different directions cancel out. |
| Group size, mass scale with member count | Done | Radius 10 px per member, mass n/10. |
| Private win condition per player | Done | 10 of one random resource plus 3 of each other (19 total). |
| Mines with finite capacity (20 each), 8 mines in a ring, 2 per type | Done | Capacity is never reached in practice, see 4.1. |
| "Intent" signal (E key cycles the resource you claim to want) | Done, unused | Rendered as arrow color; nothing in the rules reads it. |
| Authoritative server, TCP for reliable input, UDP for state and movement | Done | Full-state snapshot broadcast at 60 Hz. |
| NAT punch-through for symmetric NAT | Done | Works per the authors; not re-verified here. |
| Client "play in the past" snapshot buffer with Hermite interpolation | Built, defective | See 3.2 for the specific bugs. |
| Rendering: shaders, animations, parallax, camera, HUD | Done | SFML 2.5, GLSL Voronoi shader for group composition. |
| Server terminal dashboard (termbox) | Done | Requires a TTY; cannot run headless as-is. |
| Scripted bots: Random, NearestGreedy, Groupie | Done | Run as headless network clients. |
| Rounds, lobby, restart, player names, ready-up | Missing | Game starts at server boot; game over is terminal. |
| Windows server build, static linking | Broken | Upstream issues #194 and #192. |
| CI | Dead | Travis CI config only; travis-ci.com no longer runs open-source builds. |
| Unit tests | 12 tests, all passing | Physics, IDs, events, metrics. No tests for grouping, mining, or winning. |
| Offline simulator | New in this branch | `ug-sim` runs bot games with no network and no window. |

### 2.2 Build health

The code did not compile on GCC 13 (missing `<cstdint>`, `<limits>` includes; old Catch2 hitting the glibc `SIGSTKSZ` change). This branch fixes those, fixes an `ONLY_CLEINT` typo in CMake that silently disabled the client-only build, and logs game over to stderr so headless servers can be observed. With those changes: build succeeds against the distro's SFML 2.6.1, all 12 unit tests pass, and a 4-bot networked game reaches a winner.

### 2.3 Measured facts

| Measurement | Value |
| --- | --- |
| Time for one greedy bot to win, offline sim | 14.5 s |
| Time for 4 greedy bots to reach a winner, offline sim | 14.6 s |
| Same, over the real network stack on one machine | 15 to 17 s |
| Game-state datagram size, 0 / 4 / 50 players | 3.3 / 3.5 / 4.9 KB |
| Server upload per connected client at 60 Hz | 195 to 288 KB/s |
| Groups formed in 20 games of 4 Groupie bots | 0 |
| Offline simulator speed, 4 players, one core | about 150,000 ticks/s (a full game in 12 ms, roughly 1,200x real time) |

## 3. Structural issues

These are ordered by how much they block the stated goals.

### 3.1 The rules make alliances irrelevant (blocks goals 4 and 5)

Detailed in section 4. Summary: a lone greedy bot wins in the same time as any mix of grouping bots, games last 15 seconds, and no scripted or learned policy has a reason to group.

### 3.2 Interpolation is wrong in four specific places (blocks goal 2)

The "play in the past" design (buffer server snapshots, render a few ticks behind, interpolate toward the oldest buffered snapshot) is a sound approach. The implementation has these defects:

1. **Snapshot velocity is never sent.** `Group::getUpdate()` builds the update struct without setting `velocity`, so every snapshot arrives with velocity (0, 0). The Hermite end-tangent is always zero, which turns the curve into an ease-out that decelerates to a stop at every snapshot and then lurches toward the next one. This alone explains most of the visible jitter.
2. **Unit mismatch in the Hermite formula.** Velocities are in pixels per second but are multiplied by a delta in milliseconds, so any non-zero tangent is 1000x too large. Today this is masked by bug 1.
3. **The velocity update uses the already-overwritten position.** `hermiteInterpolatePosition` assigns `m_position` and then computes the new `m_velocity` from that new position instead of the old one.
4. **Re-anchoring every tick.** Interpolation starts from the current interpolated position each tick with a recomputed blend factor, rather than blending between two fixed snapshots. That produces asymptotic rubber-banding, not a curve through the samples.

Two surrounding problems make it worse: snapshots are captured once per server *step* (a 16 ms sleep plus compute), not per tick, while the client assumes tick-uniform spacing; and the server's tick rate is set by a sleep, so it drifts as clients connect (noted in the original commit message).

**Recommended fix:** standard snapshot interpolation. Stamp snapshots with server time, render at server-time minus a fixed delay (about 100 ms), find the two bracketing snapshots, and interpolate between them with their real velocities. Drive the server tick from a clock, not a sleep. This is transport-independent and carries over to a browser client unchanged.

### 3.3 Netcode does not scale past a LAN (blocks goal 1)

- **Full-state broadcast.** Every 16 ms the server sends every client the state of all 50 player slots, all 50 group slots, and the full resource table, whether or not the slots are in use. Measured at 3.3 to 4.9 KB per datagram, which fragments into 3 or 4 IP fragments; losing any fragment drops the whole snapshot. Per-client bandwidth is 195 to 288 KB/s. Over a home connection with 4 players this is borderline; with 8 it will stutter.
- **Input timing check.** The server rejects inputs whose tick is more than 25 ticks (200 ms) from its own. The client deliberately runs at least 13 ticks in the past, so a 100 ms round trip consumes most of the margin. After game over the client tick freezes and every input is rejected, which floods the server log.
- **Lock discipline.** The server's main loop holds four mutexes for the entire game step and releases them only during a 16 ms sleep window. It also unlocks mutexes it does not own on the first call and in the destructor, which is undefined behavior.
- **Fixed UDP ports.** The `-p` flag changes only the TCP port; the two UDP ports come from the config file, so two servers cannot share a host.

### 3.4 No session lifecycle (blocks goal 1)

The server starts in the `playing` state at boot with zero players. Whoever connects first has a head start. When someone wins, the server sits in `game_over` forever and must be restarted. Player slots are never recycled: after 50 total connections across the server's lifetime it throws and exits. Disconnected sockets are never removed from the client list. There are no names, no lobby, no ready check, no rounds, no scoreboard.

### 3.5 Engine coupling (blocks goals 1 and 4)

- The rules layer (`common-lib`) uses SFML types throughout (`sf::Vector2f`, `sf::Packet`, `sf::Clock`) and reads a global config at static-initialization time from a path relative to the working directory. The binaries only run from the repo root, and the rules cannot be compiled for the browser or wrapped for Python without first removing SFML from them.
- `IdFactory` and `EventController` are process-wide singletons with static state, so only one game can exist per process. The simulator works around this by resetting them between games; a vectorized RL environment could not.
- The server's terminal dashboard is constructed unconditionally and needs a TTY.

### 3.6 Physics details

- The collision loop visits every ordered pair, so each contact is examined twice per tick and, if the first resolution did not fully separate the bodies, resolved twice.
- Resolution for a movable body against an immovable body moves the movable body by the full overlap in one branch and half the overlap in the other.
- The out-of-bounds correction formula is not the reflection it appears to be intended as.

None of these break play at 4 to 8 circles, but they will need cleaning before the rules become the foundation of an RL environment, where every determinism quirk becomes a training artifact.

### 3.7 Code health

Deterministic wall-clock seeding (`srand(time(NULL) + player_id)`) means every game started in the same second has the same win conditions. Non-standard C++ (GNU designated initializers under `-std=c++11`). No CI. The most important rules (grouping, mining, win condition) have no tests.

## 4. Game design analysis: is the optimal strategy greedy?

Yes, under the current rules. This was tested empirically, not just argued.

### 4.1 Evidence from the simulator

All runs use the shipped `game_settings.json`. Strategies: 1 = NearestGreedy (go to the closest mine you still need), 2 = Groupie (chase the nearest joinable group, else greedy), 3 = Greedy with "joinable" switched on at spawn.

| Lineup | Time to a winner | Groups formed | Winner |
| --- | --- | --- | --- |
| 1 greedy alone | 14.5 s | n/a | the only player |
| 4 greedy | 14.6 s | 0 | always the same player |
| 4 groupie | 14.6 s | 0 | always the same player |
| 2 greedy + 2 groupie | 14.7 s | 0 | a groupie |
| 4 greedy, joinable on | 14.3 s | 2 joins, avg group size 1.2 | a joined player |
| 2 greedy + 2 greedy-joinable | 14.8 s | 0 | a lone greedy |
| 8 mixed | 15.7 s | 3 joins | a lone greedy |

Reading this: a player alone wins in 14.5 seconds; with three opponents doing anything at all, someone wins in 14.6 seconds. The other players are not competition, they are scenery. Grouping is worth at most a few tenths of a second. Every game with the same lineup produced identical results across 20 repetitions, because nothing in the rules or the bots is random.

The Groupie bots never grouped because they only turn on "joinable" when standing next to a group that is already joinable, and no one starts joinable. That is a bot bug, but strategy 3 shows that fixing it would not change the outcome.

### 4.2 Why the rules produce this

1. **No per-capita mining bonus.** A group of n bumping a mine gives each member 1 unit. Solo you also get 1 unit per bump. The group drains the mine n times faster, but a mine holds 20 units and a winner only needs 19 units total across all types, so mines never run dry in a 4-player game (160 units of supply vs. about 76 units of demand). Scarcity only starts to bite at 8 or more players.
2. **Group movement rewards agreement, and greedy players never agree.** Velocity is the sum of member directions, capped at 4x. Two members heading for different mines pull against each other and move slower than either would alone.
3. **Nothing forces contact.** Eight mines on a ring, two per resource type, mean there is always an uncontested mine nearby. There is no blocking, stealing, trading, or gating.
4. **Games are too short for alliances to have a life cycle.** Fifteen seconds is not long enough to form, exploit, and betray anything, and the physics of bump-mining (a bounce and re-approach every 0.75 s) sets the pace.
5. **Private information is not used.** The "intent" signal exists but nothing in the rules or the bots reads it, so there is no negotiation surface.

### 4.3 What this means for RL and for spectating

As it stands, an RL agent trained on this environment would converge on nearest-greedy within minutes, would never learn to group because grouping is never rewarded, and would produce games of four circles bouncing off mines for 15 seconds. That is neither an interesting learning problem nor watchable.

The good news is that the *shape* of the game (mixed-motive, private goals, cheap-talk signal, continuous space, discrete actions) is exactly the shape that produces interesting multi-agent behavior once the payoff for cooperating is real and time-limited. The fix is in the rules, not the engine.

### 4.4 Design levers to make alliances matter

These are proposals to test in the simulator, cheapest first. The target is a measurable "alliance advantage": a well-played grouping policy should beat the best solo policy by a wide margin, and a policy that groups and never leaves should lose to one that leaves at the right moment.

1. **Superlinear mining for groups, divided among members.** A group of n extracts f(n) units per bump with f(n) > n (for example n^1.5), split among members. Solo mining becomes slow; a pair is much better than solo; a trio better still.
2. **Real scarcity.** Four mines, one per type, with capacity that runs out and regenerates slowly. Large groups now drain a mine that everyone needs, which is the trigger for the alliance to break.
3. **Gated mines.** A mine only yields when the touching group's mass exceeds a threshold. Nobody can mine alone. The interesting decision becomes who to bring, since partners consume the same mine.
4. **Leaving costs something.** Splitting a group splits its pooled resources by some rule (equal shares, or the leaver forfeits a fraction). Joining requires both sides to opt in. This gives betrayal a price and timing a meaning.
5. **Heavier groups can take from lighter ones.** A collision between a group and a solo circle transfers a unit to the heavier body. Grouping becomes protection as well as speed.
6. **Longer games.** Requirements of 60 to 80 units, a short mining cooldown per mine, and a regeneration rate tuned so that a 4-player game lasts 3 to 5 minutes.
7. **Make intent public and consequential.** Show every player's declared intent and give a small bonus when a group's declared intents match the mine being tapped. This turns cheap talk into a coordination device that can also be lied with.

Evaluation loop: change one rule, run the scripted ladder (greedy, greedy-joinable, groupie, a new cooperative-mover bot that steers toward the group's shared target), and report alliance advantage, average group size over time, join and leave counts, and game length. Only move rules to the networked game once the ladder shows grouping winning and stable grouping losing.

## 5. Path to playing in the browser

Three options were considered.

| Option | Effort | Outcome |
| --- | --- | --- |
| A. Compile the C++ client to WebAssembly | High | SFML has no Emscripten support; graphics and networking would both have to be replaced. Not recommended. |
| B. Keep the C++ server, write a TypeScript client, add a WebSocket transport | Medium | Two languages, two copies of any client-side rules, but the server work overlaps with fixes needed anyway. |
| C. Port the rules to TypeScript, run the server on Node, client in the browser | Medium | One language, one deploy, easiest for friends. Loses the C++ engine. |

**Recommendation:** first extract a pure, deterministic rules core from `common-lib` with no SFML types and a single `step(state, inputs) -> state` entry point, with golden-trace tests. That core is small (the rules are a few hundred lines once separated from rendering and transport) and it is the thing both the browser and the RL environment need. Then choose B or C based on appetite; C is the faster route to "send a link to friends", and the C++ simulator remains the tool for rule iteration and RL.

Browser constraints that shape the design either way: browsers cannot open raw UDP or TCP sockets, so state goes over WebSocket (reliable, ordered) or WebRTC data channels (unreliable). WebSocket at 20 Hz with delta-compressed snapshots is enough for this game at 4 to 8 players. Hosting is a single small VM or a free-tier container; players join with a room code in the URL.

## 6. Bots and reinforcement learning

### 6.1 Scripted bots (near term)

The bot interface is the right one: a function from game state to an input pair. Two improvements make bots useful for playtesting immediately: fix Groupie's chicken-and-egg joinable logic, and add a "cooperative mover" that, when in a group, steers toward the group's consensus target instead of its own. Both can be validated in the simulator without a server.

### 6.2 RL environment (after the rules change)

- **Interface.** Multi-agent, simultaneous move, discrete action space (9 movement actions x joinable toggle x ungroup toggle x intent cycle), partial observation (own needs, everyone's position, size, joinable flag, declared intent, mine positions and remaining capacity). A PettingZoo-style parallel environment is the natural fit.
- **Simulation speed.** The offline simulator runs about 150,000 ticks per second on one core: a full 4-player game (about 1,800 ticks at 125 Hz) in 12 ms, and 8-player games at about 125,000 ticks per second. That is enough for millions of games per day on a laptop before any optimization. Frame-skipping to one decision per 6 ticks (the current bot cadence) is already built in.
- **Blockers to remove first.** Process-wide singletons (one game per process), SFML types in the core, wall-clock seeding, and the physics double-resolution quirk. All are addressed by the rules-core extraction in section 5.
- **Training approach.** Self-play with a population (not a single opponent), because alliance behavior only emerges against partners who can also cooperate and defect. Sparse win reward plus a small shaped term for resource progress. Expect greedy to emerge first; the rules ladder in section 4.4 is what makes the next stage (grouping, then timed defection) learnable.
- **Watchability.** Bot-only games should be replayable in the browser client from a recorded input log, which the deterministic core makes trivial.

## 7. Roadmap

| Phase | Scope | Exit criteria |
| --- | --- | --- |
| 0. Foundations (this branch) | Build fixed on current toolchains, offline simulator, this document. | `ug-sim` runs; tests pass; PRD merged. |
| 1. Rules that reward alliances | Implement levers from 4.4 behind config flags; add rule tests; add cooperative bot; iterate in the simulator. | Grouping policy beats solo by a clear margin; stable grouping loses to timed leaving; 4-player game lasts 3 to 5 minutes. |
| 2. Session and netcode | Lobby, names, ready-up, rounds and restart, slot recycling, configurable ports, headless server, clock-driven tick, snapshot interpolation fix, delta-compressed 20 Hz snapshots, GitHub Actions CI. | Four people on different home networks play three rounds in a row without a restart and without visible jitter. |
| 3. Browser | Rules core extracted; TypeScript client; WebSocket transport; room codes; hosted server. | A friend with only a link is in a game within 30 seconds. |
| 4. Agents | Python environment over the rules core; scripted ladder as evaluation; population self-play; replay viewer. | Trained agents beat every scripted bot and exhibit join-then-leave behavior; bot-only matches are watchable in the browser. |

Phase 1 comes before netcode on purpose: there is no point polishing the delivery of a game whose interesting mechanic does not yet work, and every rule change is far cheaper to test in the simulator than over the network.

## 8. Open questions for the authors

1. Is the private win condition core to the vision, or would public, differing goals (everyone can see what everyone needs) make for better alliances? Public goals make negotiation legible to spectators and to learning agents.
2. Should ungrouping be instant and free, or should it be a visible, delayed action that others can react to? The latter gives betrayal a tell.
3. Is a 3 to 5 minute round the right length, or is Ungroup meant to be a longer, more strategic game?
4. How much of the current C++ engine is worth preserving for its own sake versus as a reference implementation for a TypeScript port?
