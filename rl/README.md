# Ungroup v2: rules core, RL environment, and replay viewer

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
