# Ungroup: design, architecture and plan review (September 2026)

This document consolidates three adversarial reviews run on 7 September 2026 against the state of the
repository after the first RL results: one on the policy network architecture, one on the overall plan,
and one on how to make agents generalize so rule tweaks stay cheap. Each review used six or three
independent reviewers with different lenses, then three skeptics per finding who tried to refute it by
reading the code and re-running experiments. About a third of findings were refuted or downgraded.
The verification phase was cut short by an API session limit, so findings are marked as **verified**
(reproduced by at least two skeptics), **partly verified** (one skeptic, or the mechanism corrected), or
**unverified** (reviewer measurement only).

The second half of the document records what was changed in response (the v2 rules and trainer) and
the revised roadmap.

## 1. The headline finding

**What the first RL run reported as alliance behavior was a rule exploit.** (verified, reproduced
independently by four skeptics)

The trained agents' median alliance lasted 2.0 to 2.3 seconds, exactly the leave timer. About 60 to 70
percent of merges happened within a second of a leave by one of the participants. Three rule holes made
this the best strategy: a leaver respawned touching the group it left with its joinable flag still on;
any merge cancelled every member's leave timer; and leaving was a free one-shot request. On top of that,
the DAgger warm start had cloned the scripted bot's joinable and leave rules to certainty (joinable
exactly when the group had fewer than 3 members, with probability 1.000 after 11 million PPO samples),
and the entropy bonus was too small to reopen those heads. So the "120 joins per round" I reported
measured the bot's constant and collision geometry, not learned decisions.

Two further verified rule problems: **ramming was free** (an empty body could spill a laden one at no
cost; a scripted pirate halved the finish rate of a lobby while reaching parity with the mining bots),
and **intent leaked every player's private need in the first observation** and could be flipped at zero
cost on the tick before a split.

Two verified process problems: **no timeout tiebreak**, so once agents got good at denying each other,
rounds ended with nobody winning and the response (raising mine yield) was a rule change made to satisfy
a bot metric; and **nothing recorded which rules a checkpoint or log row ran under**, so a two-line
constant change silently re-baselined every number.

## 2. Architecture review

| Finding | Status | Action taken |
| --- | --- | --- |
| Alliance heads collapsed by the warm start; a summed entropy bonus of 0.001 cannot reopen them | verified | Per-head entropy coefficients normalized by head size (0.003 move, 0.02 joinable, 0.02 leave, 0.01 intent); alliance heads cloned with label smoothing 0.25; per-head entropy and conditional marginals logged every update |
| Leave is a memoryless one-bit latch; nothing in the observation says how long you have been grouped or what a partner did | verified | Leave is now a held action that cancels when released; own-state features for time in group, time since last bank, cooldowns and stun; per-other features for co-membership time, units they took when leaving you, units banked to them while you were a member, time since they last left you |
| Other-player slots swap identity on 20 percent of decisions and share no weights | partly verified (mechanism disputed, direction accepted) | Shared per-entity encoder with masked max and mean pooling, plus a pool over co-members |
| Observation width hard-wired to 6 players and 8 mines | verified | Eight padded other-player slots with a presence flag; the same checkpoint can play 2 to 8 seats |
| Attention not warranted at this budget; pooling is | unverified | Pooling used |
| Critic shares the trunk and is blind to partner banks and hidden needs; explained variance near zero in the last 40 percent of a round | partly verified | Separate critic fed a privileged block (every seat's needs, banked, pool share, seat type); value normalization; Huber loss; explained variance logged; critic-only warm-up before joint PPO |
| PPO barely moved the weights: 6 Adam steps per 37k-sample update, trunk changed 3 to 5 percent in 11M samples; second tanh layer 60 percent saturated after BC | unverified | Minibatch 4096, 3 epochs, lr 2.5e-4 with a KL stop, gradient clip 1.0; LayerNorm plus ReLU trunk with orthogonal init; weight decay in the warm start |
| A GRU is affordable only at 128 units; history features first | partly verified | History features; no GRU yet |
| Snapshot pool was ten near-copies of the current policy; needs a persistent league | unverified | Snapshots kept for the whole run; scripted bots and the warm-start clone are permanent members; opponents sampled per seat, not per round |
| A single unconditioned trunk cannot represent heterogeneous roles; add a style latent | unverified | Not done yet; the DR constants in the observation give a first conditioning axis |
| Intent's "keep" option is dead weight and the head is neither a signal nor a lever | verified | Intent locked while grouped (masked in the policy), random at reset, split weight raised to 3:1 |
| 8-way movement is fine; move=0 should be a free "follow" inside a group | unverified | Members with no direction no longer brake the group |
| 85 percent of wall time was spent under CPU oversubscription | verified | Training and evaluation are no longer co-scheduled; passive OpenMP waiting |

## 3. Plan review

| Finding | Status | Action taken |
| --- | --- | --- |
| Alliance metrics were an oscillation, not decisions | verified | Rules fixed (section 5); the core now reports alliance count, mean duration, alliances over 10 s, group banks and fast re-merges |
| Nobody group-banks and kidnapping loses, so the bank-to-owner rule is dead | refuted as a conclusion (measurements reproduced, inference contradicted by the skeptics' own bots) | Kept the rule; a "follow" action and consensus targeting make kidnapping feasible; kidnap is a held-out test bot |
| Ramming is free | verified | Spill threshold raised above solo speed and both bodies stunned for a second on a spill |
| Intent is a zero-cost bluff and leaks needs | verified | Random at reset, locked while grouped |
| The shrink is a mine-death clock; the last 18 s of every round are dead | partly verified (geometry right, severity overstated) | Final radius 0.5 so the inner ring survives; shrink starts at 50 percent |
| The playable game is further from "play with friends" than at PRD time; Phase 2 on the SFML engine is dead work | verified | Roadmap rewritten (section 7); SFML engine tagged `sfml-legacy` |
| No implementation is the source of truth; Python and C++ cores differ and were tuned out of step | verified | The C++ core is canonical; Python tools drive it through ctypes; checkpoints record their config |
| Physics tick 0.1 s and decisions at 0.2 s are the human input-latency budget | verified | Physics at 30 Hz; agents still decide every 0.2 s; a continuous direction entry point exists for human seats |
| Reward has no adversarial gradient and a 40-second horizon | verified | Relative term (minus half the others' banked progress), gamma 0.998, timeout tiebreak so every round has a winner |
| No human has played the new rules | verified | Highest-priority next step (section 7) |
| Evaluation cannot support a "great bots" claim | verified | Native evaluation with paired seeds, held-out bots, per-lineup reporting; cross-play and exploitability still to do |
| Spectator event feed is a firehose and leave_start was never recorded | verified | Events now captured per step; alliance summary view still to do |

## 4. Generalization review

| Finding | Status | Action taken |
| --- | --- | --- |
| Frozen-checkpoint margin over the ladder changes 2x within plus or minus 30 percent of one constant; the mine-yield commit silently removed 83 percent of a checkpoint's edge | unverified | Rule randomization per game with the sampled constants in the observation; config sweep in `play_v2.py sweep` |
| Randomization only works with the constants exposed and a per-game config in the core | partly verified | Done in the core |
| Held-out bots (kidnapper, spill hunter, pad mugger) erase most of the agent's edge | unverified | Kidnap and rammer bots exist in the core and are never used in training |
| Seven rounds per log row is noise; margins need 60 to 160 rounds, finish and win rates 4 to 10x more | unverified | Evaluation defaults to 48 to 96 paired rounds; the training log is for trend only |
| Cross-play and exploitability cannot be run from one lineage | unverified | Not yet done |
| Nothing records which rules a checkpoint ran under | verified | Checkpoints store config, observation layout, git sha and sample count; replays store config and the action log |

## 5. The v2 rules, and what the ladder says about them

Changes to the carry-bank-spill rules, all in `rl/native/ungroup.cpp`:

- Leaving is held: the timer runs only while the member keeps requesting it and cancels when released.
- The leaver is ejected opposite to the group's motion with joinable forced off, cannot join anyone for 3 s, and cannot rejoin former partners for 10 s. Merging never cancels a leave in progress.
- A group merges only when every member of both bodies is joinable, so any member can close a group.
- The leaver's share is weighted 3:1 toward their declared intent and reduced by a 15 percent forfeit that stays with the group.
- Intent is random at reset and can only be changed while solo.
- Spills need an approach speed above 0.40 (solo top speed is 0.45), and both bodies are stunned for 1 s.
- Members with no direction follow the group instead of braking it.
- At the time limit the player with the highest progress wins.
- Physics at 30 Hz; the arena shrinks from 50 percent of the round to a radius that keeps the inner mines alive.
- Mine rate 0.12 per second for a solo body, times group size squared.

The scripted ladder under these rules (32 paired rounds per lineup, six seats):

| Lineup | Progress by seat type | Reading |
| --- | --- | --- |
| six solo | 0.62 | baseline |
| six defectors (bail) | 0.63 | mutual defection barely beats solo |
| six cooperators (loyal, rotating banks) | 0.70, 28 percent finish early | mutual cooperation is best for everyone |
| three cooperators, three solo | 0.77 vs 0.60 | grouping beats solo decisively |
| three defectors, three cooperators | 0.57 vs 0.51 | defecting in a cooperative lobby pays |
| two defectors, two cooperators, two kidnappers | 0.52, 0.36, 0.20 | dumb kidnapping loses |
| two defectors, two cooperators, two rammers | 0.64, 0.64, 0.45 | freeloading by spilling is viable but never wins |

That is a spatial prisoner's dilemma: cooperation is the efficient equilibrium, defection is the tempting
deviation, and neither collapses into the other. Instant re-merges are now zero per round and scripted
alliances last 13 to 150 seconds depending on the bot. Whether this is fun is a question for humans;
the numbers only say it is no longer degenerate.

## 5b. The v3 rule changes (7 September): the rammer pays, the group is paid

The first trained v2 agents used alliances, but the strongest snapshot drifted toward five-second
alliances and 54 spills per round: crashing into a laden group cost the rammer nothing and paid it
half the floor. Two changes, both in `rl/native/ungroup.cpp`:

- In a spill the faster body along the collision normal is stunned 2.5 times longer than its victim
  (`rammer_stun_mult`), so the victim reaches the floor first.
- Banking as a group multiplies the pool by 1 + 0.15 (n - 1) (`group_bank_bonus`), so a pair that banks
  together beats a solo of the same yield even after the leaver's forfeit.

The ladder ordering survived (32 paired rounds, six seats): six loyal 0.72 with half the rounds finishing
early, six bail 0.63, six solo 0.63, three bail among three loyal 0.55 vs 0.48, two rammers among bail
and loyal 0.44 and never a win. The dilemma shape is unchanged; the rammer went from viable freeloader
to loser.

Agents retrained from scratch under the v3 rules (run `rl/checkpoints/v3`, snapshot at update 350,
8.8 M samples) against the previous strongest v2 snapshot evaluated under the same v3 rules:

| Lineup (32 paired rounds) | v3 agent | v2 agent under v3 rules |
| --- | --- | --- |
| three agents, three loyal: alliance mean duration, spills | 34 s, 10.5 | 19 s, 15.8 |
| three agents, three kidnappers: agent vs kidnapper progress | 0.48 vs 0.19 | 0.53 vs 0.16 |
| three agents, three rammers: agent vs rammer progress | 0.48 vs 0.35 | 0.49 vs 0.30 |
| six agents: alliances, mean duration, over 10 s, spills | 39, 12.9 s, 17.6, 23 | (v2 rules) 45, 5 s, 15, 54 |

Alliances roughly doubled in length and spills halved at similar strength against the held-out bots.
Against the training bots the v3 snapshot is weaker than the v2 candidate was (0.54 vs 0.62 against
bail, 0.51 vs 0.60 against solo); it is an earlier snapshot of a run that was still improving when this
was written, and the win-rate columns of the training log were rising through update 400. Twenty-agent
self-play under v3: 137 merges, 144 spills and 192 banks in a round, against 253, 220 and 114 for the v2
snapshot, so large lobbies also shifted from demolition derby toward mining and banking.

**Why the spill counts were misleading.** Classifying the spills in the recorded v3 rounds by the
bodies' loads before impact: in the six-agent round 29 of 34 spills were between two solo bodies carrying
under three units, 20 of them dropped nothing at all, and only 5 were an empty body hitting a laden one.
The twenty-agent round had the same shape (121 of 144 light-on-light, 63 with nothing dropped). The agents
were not ramming for loot; they were bumping into each other at full speed near contested mines, and the
one-second stun on every bump was the movement tax that let the scripted bots, which stop cleanly, outscore
them. The core now stuns and counts a collision only when at least one unit hits the floor. The ladder
ordering is unchanged, and the same v3 snapshot re-evaluated under the fix went from 0.54 to 0.59 against
the bail bot with real spills in self-play at 14 per round. Run `rl/checkpoints/v4` retrains from scratch
under the fixed rules.

**v4 (retrained under the fixed rules) and what the health checker showed.** `rl/health.py` evaluated every
v4 snapshot as it appeared (`rl/models/v4_health.csv`). Update 200 is the best snapshot: 32 paired rounds give
self-play progress 0.61 (v3: 0.53) with alliances of 9.9 s and 17 over ten seconds per round, 0.61 vs 0.64
against bail, 0.58 vs 0.62 against solo, 0.64 vs 0.71 against loyal, 0.56 vs 0.25 against the kidnapper,
0.50 vs 0.37 against the rammer, and the top progress in the mixed lobby. From update 250 the checker flagged
the same drift as v3: self-play progress fell to 0.54, alliances to 6 s, spills rose, and the bot margins
went from within noise to minus 0.08. The drift is therefore not the collision tax; it is something in the
self-play objective. The v5 run tests the two candidates together, the relative-reward term (which pays for
other agents' failure and so rewards mutual harassment) and the carried-unit shaping (which pays for
picking spilled units off the floor before they are banked).

The published replays use `rl/models/v4_200.pt`.

## 6. The v2 training stack

`rl/train_v2.py` implements the architecture-review recommendations: entity encoder with masked pooling,
separate privileged critic with value normalization, per-head entropy, action masks, a persistent league
with per-seat opponent sampling, DAgger warm start with softened alliance heads, a critic-only warm-up,
domain randomization of mine rate, regen, leave time and need size with the sampled values in the
observation, and checkpoints that carry their config. `rl/play_v2.py` evaluates a checkpoint against
training and held-out lineups with paired seeds, sweeps rule constants to measure memorization, and
records replays with the action log.

Acceptance tests for a run, per the reviews: P(joinable | solo) between 0.3 and 0.9 rather than 1.000;
median merge-to-leave latency well above 2 s; alliances over 10 s per round above zero; margin over the
bail bot retained under plus or minus 25 percent constant sweeps; and progress against the held-out
kidnapper and rammer within 0.1 of progress against the training bots.

## 7. Revised roadmap

The old roadmap routed delivery through the SFML engine. That engine is now tagged `sfml-legacy` and is
a reference for art and shaders only. Nothing in `src/` will be extended.

1. **Humans play the v2 rules (next, one to two evenings).** A small WebSocket server over the C++ core
   through the existing ctypes wrapper, keyboard input mapped to the four actions with the continuous
   direction entry point, bail or policy seats filling the rest, and the replay viewer's draw loop as the
   client with two-snapshot interpolation at about 100 ms. No lobby, no room codes: a shared tunnel link is
   enough for friends. Every human round is saved as a replay. Keep a "what confused me, what felt bad" log
   and treat it as a required input to any rule change.
2. **Rule constants freeze until two or three human sessions have happened.** The scripted ladder is the
   pacing gate; the RL agent is a regression check for exploits, not the design oracle.
3. **Agents.** Continue the v2 run; add the cross-play matrix across two more seeds and a best-response
   exploitability run; add a style latent if the league converges to one personality; consider a small
   GRU only if the history features prove insufficient.
4. **Browser for real.** Either compile the C++ core to WebAssembly for a Node server and browser client
   (one rules implementation everywhere), or keep the Python server and ship the browser client alone.
   Decide after the first human sessions, based on whether latency over a real network is acceptable
   with interpolation only. Rooms, names, rounds and a scoreboard live in that server.
5. **Spectator features.** Alliance summaries derived from events (formed at, members, ended by bank to X
   or leave with N units), highlights instead of the raw feed, replay library.

Explicitly not on the roadmap: the four old interpolation bugs (they do not exist in the new snapshot
format), delta compression and lock discipline on the old server, the terminal dashboard, and a
TypeScript port of the rules.

## 8. How to know it is working

- Ladder gates after any rule change: the seven lineups in section 5 keep their ordering.
- Alliance duration distribution, group-bank fraction and fast re-merges reported per lineup, never pooled.
- Agent margin over the bail bot with standard error at 96 paired rounds; retention of that margin under
  the constant sweep; progress against held-out bots.
- Per-head entropy and conditional marginals every update; a checkpoint whose alliance heads sit outside
  0.02 to 0.98 is flagged and its alliance statistics are not used for design conclusions.
- Human sessions: can a first-time player explain why an agent left a group, and does a round with a
  human in it end with a winner they can name.
