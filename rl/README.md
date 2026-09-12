# Ungroup v2: rules core, RL environment, replay viewer, and play server

This directory is a self-contained prototype of the redesigned "carry, bank, spill" rules
described in `docs/PRD.md` and the design discussion that followed it. It does not depend on
the SFML engine. It exists to answer two questions cheaply: do the new rules make alliances
matter, and what do learned agents do with them.

## Layout

| Path | What it is |
| --- | --- |
| `ungroup/core.py` | The rules and physics. Deterministic, no rendering, no networking. Records replays. |
| `ungroup/env.py` | Multi-agent environment: observations, MultiDiscrete actions, shaped rewards. |
| `ungroup/bots.py` | Scripted baselines: `solo` (never groups), `bail` (groups, mines, leaves at the right time), `random`. |
| `ladder.py` | Runs scripted bots against each other and reports wins, game length, and grouping stats. |
| `train_ppo.py` | PPO self-play with one shared policy, past-snapshot opponents, and scripted-bot seats. |
| `record_replay.py` | Plays a game with a checkpoint and writes a replay JSON. |
| `viewer/template.html`, `viewer/build_viewer.py` | Standalone HTML replay viewer; the build script embeds a replay. |

## The rules in one paragraph

Players are circles. Two circles that touch while both are "joinable" merge into one group
body with a shared pool. Group speed is `base / sqrt(n)` times the mean of member directions, so
mass costs speed and disagreement slows the group. Mines yield while a body touches them at
`mine_rate * n ** mine_exp` units per second, so groups mine far faster per head than solos.
Nothing counts until it is banked: each player has a home pad on the boundary, and when a group
touches a member's pad the entire pool goes to that member. Leaving takes two seconds, is
visible, and gives the leaver a per-type share weighted toward members whose declared intent
matches the type. Hard collisions spill carried units onto the floor for anyone to collect.
The arena shrinks in the second part of the round and kills the mines it leaves behind. First
player to bank their private need vector wins; otherwise the round ends at the time limit.

## Actions and observations

Actions per agent are MultiDiscrete `[9, 2, 2, 5]`: move (none or 8 directions), joinable
(off/on), leave (request), intent (keep or set one of 4 types). A decision is taken every
two physics ticks (0.2 s).

Observations are a flat float vector: own state (position, velocity, needs, banked, intent,
group size, pool, leave timer, arena radius, time, own pad), every other player sorted by
distance (relative position, velocity, group size, joinable, intent, same-group flag, leaving,
public progress, their pad, their carried total), every mine (relative position, type, stock,
alive), and the four nearest floor pickups. Other players' needs are private and not observed.

Rewards are potential-based: 10 x banked progress plus 2 x (carried-share progress minus banked
progress), so mining is rewarded a little and banking a lot. The winner gets +10 and everyone
else -2 when a round ends with a winner.

## Quick start

```
pip install numpy torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python3 rl/ladder.py --games 10 solo solo solo bail bail bail      # scripted ladder
python3 rl/train_ppo.py --updates 600 --workers 4 --envs 4          # train, logs to rl/checkpoints/log.csv
python3 rl/record_replay.py --checkpoint rl/checkpoints/policy_latest.pt --players 6 --tries 5 --out replay.json
python3 rl/viewer/build_viewer.py replay.json replay.html           # open replay.html in a browser
```

`record_replay.py --seats policy,policy,policy,bail,solo,policy` mixes trained agents with
scripted bots. `--tries N` records several seeds and keeps the most eventful finished game.

## Reading the training log

`log.csv` has one row per update: `policy_winrate`, `bot_winrate`, and `snapshot_winrate` are
wins per seat relative to chance (1.0 means a seat of that type wins exactly as often as a
uniformly random seat would); `finish_rate` is the fraction of rounds that ended with a winner
rather than at the time limit; `avg_group`, `merges`, `leaves`, `spills`, `banks` are per-round
averages that describe how much alliance behavior is happening.


## v2 (September 2026): canonical C++ core, new rules, new trainer

After the design review in `docs/PLAN_REVIEW_2026-09.md` the rules moved to `native/ungroup.cpp` as the
only implementation (`ungroup/core.py` is the original Python prototype and is no longer the source of
truth). The v2 tools:

| Command | What it does |
| --- | --- |
| `python3 rl/ladder_native.py --gates` | Rules pacing gates: scripted lineups with alliance statistics and paired seeds |
| `python3 rl/ladder_native.py --set mine_rate=0.1 --games 48 bail bail bail loyal loyal loyal` | Any lineup under any constants |
| `python3 rl/train_v2.py --dr --out rl/checkpoints/v2` | DAgger warm start, critic warm-up, then PPO with the entity encoder, privileged critic, league, and rule randomisation |
| `python3 rl/play_v2.py eval --checkpoint rl/checkpoints/v2/latest.pt` | Paired-seed evaluation against training and held-out bots |
| `python3 rl/train_v2.py --resume rl/checkpoints/v4/warmup.pt --anchor-kl 0.3 --anchor-heads 0 --p-snapshot 0 --p-policy 0.34 ...` | Anchored run: keeps the movement head close to the imitation prior, which stopped the drift every earlier run showed |
| `python3 rl/health.py watch --run rl/checkpoints/v4` | Run alongside training: checks every new snapshot for losing to a bot, self-play below the bot baseline, alliance collapse, bump-dominated spills, collapsed heads; writes `health.csv` and `alerts.log` |
| `python3 rl/play_v2.py sweep --checkpoint ...` | Margin over the ladder under rule-constant perturbations (memorisation check) |
| `python3 rl/play_v2.py record --checkpoint ... --tries 6 --out replay.json` | Record a replay with the action log; build the page with `viewer/build_viewer.py` |
| `python3 rl/play_v2.py record --checkpoint ... --seats $(python3 -c "print(','.join(['policy']*20))") --tries 3 --out replay20.json` | Record a 20-agent round (about 8 MB of JSON; the viewer handles up to 32 seats) |
| `python3 rl/play_server.py --humans 2 --agents 2 --checkpoint ... --bots bail,loyal` | Play in a browser: serves `viewer/play.html`, one room, rounds restart, replays saved |

Scripted bots: `solo` (never groups), `bail` (groups, leaves when its share is worth taking or a partner's
pad is near), `loyal` (never leaves, banks at the pad of the member furthest behind), and two held-out
bots never used in training: `kidnap` (drags laden groups to its own pad) and `rammer` (spills laden
bodies and collects the floor).

Models in `rl/models/`: `v2_candidate.pt` and `v2_strong.pt` were trained under the v2 rules; `v3_*.pt`
were trained after the 7 September changes (rammer stunned longer than its victim, group bank bonus).
Checkpoints carry their own config, so evaluating an old model runs it under the current core's rules
with the old model's constants.
