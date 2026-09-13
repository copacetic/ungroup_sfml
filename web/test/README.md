# Engine conformance tests

`web/src/engine.js` is a line-by-line port of the canonical rules core `rl/native/ungroup.cpp`
(same function names and order, so the two files can be diffed side by side). These tests prove
that it behaves like the C++ core.

## Run

```sh
# 1. reference numbers from the C++ core (needs python3 + numpy and the rl/ package; ~1 min on one core)
OMP_NUM_THREADS=1 python3 web/test/reference.py            # writes web/test/reference.json

# 2. the JS side (Node 22, no dependencies; ~15 s)
node web/test/conformance.mjs                              # exit code 1 on failure
node web/test/conformance.mjs --skip-ladder                # only the fast unit checks
node web/test/conformance.mjs --ref /tmp/reference96.json  # a reference made with --games 96 --seed 2000
```

Numerics checks (no reference needed): `crSin/crCos` against glibc values on which `Math.sin/cos` are
one ulp off, and the `printf %.nf` tie rule on 16 cases.

`reference.py` options: `--games N` (default 24), `--seed S` (default 1000; round e uses seed S+e),
`--no-nofma` (skip the calibration build, see below), `--trace out.json --preset life --lineup
bail,bail,bail,loyal,loyal,loyal --trace-seed 1` (dump one game's frames after every decision step,
for tick-level diffing; `--lib path.so` picks another build of the core).

## What is checked

1. **Reset exactness** – for seeds 1..20 under the life preset, `Game.meta()` (needs, pad angles,
   mine positions and types) and the initial intents are identical to the C++ `meta_json`/`frame_json`.
   `web/src/rng.js` implements `std::mt19937_64` and the libstdc++ (GCC 13) algorithms used by the
   core: `generate_canonical<double,53>` (one 64-bit draw / 2^64), `uniform_real_distribution`,
   `uniform_int_distribution` (Lemire's nearly divisionless method with a 128-bit product) and
   `discrete_distribution` (normalised cumulative table, lower_bound on one canonical draw). It was
   verified draw-for-draw against a C++ probe for three seeds, including across a twist boundary.
2. **Ladder conformance** – the lineups [6 solo], [6 bail], [6 loyal], [3 bail + 3 loyal],
   [2 bail + 2 loyal + 2 rammer], [5 bail + 1 loyal] under the legacy and life presets, 24 rounds
   each with paired seeds, run through `ladder_native.run` (C++) and through `Game` (JS). The table
   prints mean progress per seat type (tolerance 2 C++ standard errors), and merges, leaves, banks,
   spills (tolerance 15 %; rates below one per round are printed but not judged).
3. **Determinism** – the same seed and the same external inputs (a scripted policy seat cycling
   through macro targets, joinable and held leaves) give identical frames, statistics and
   observations; `reset(seed)` on a reused `Game` reproduces a fresh one.
4. **Macro steering** – a policy seat driven by the bail bot's macro labels (`botAction('bail', i,
   true)`, classes 10..23) in a six-seat life lobby reaches at least 0.60 mean progress (it reaches
   the same progress as the native bail seats), and every class steers along the vector to its target.
5. **Observation layout** – 323 floats for 8 mines (own block 39 with `is_head` at 37 and `brand/60`
   at 38; 8 other slots of 24 with present at +16, same-body at +10, `is_head` at +22, `brand/60` at
   +23; mines at 231, pickups at 295); `obs_legacy=1` gives the 305-float v2 layout.
6. **Legacy mine cap** – with bloom off, untouched mines stay exactly at `mine_cap`, groups draw the
   stock down and it regenerates back to the cap and never above it; the life preset starts mines at
   `bloom_cap = n_players + 2`.
7. **Performance** – a six-player round (up to 7200 ticks) in about 20 ms in Node (3.0 µs per tick; a 32-player round with macro-steered policy seats and `frame()` every 2 ticks costs about 80 µs per tick).

## Floating point: why the ladder is compared statistically

The shipped `rl/native/libungroup.so` is built with `-O3 -march=native`; GCC keeps floating-point
contraction on for C++ even under `-std=c++17`, so the core uses FMA instructions, which JavaScript
cannot reproduce. Both engines are chaotic at contact thresholds (a body sliding on a mine, a merge
test, a pad boundary), so one-ulp differences flip branches and full trajectories diverge, even though
every rule is identical.

`reference.py` therefore also builds a copy of the core with `-ffp-contract=off` and runs the same
ladder through it ("noFMA" column in the table). Against that build the JS engine reproduces whole
games bit for bit: at 24 rounds x 12 lineups all 288 rounds end with the identical winner, progress
vector and merge/leave/spill/bank counts (it was 190 of 288 while the engine used `Math.sin/cos`,
which V8 does not round correctly on about 3% of inputs, and `Math.hypot`, which is not
`sqrt(x*x + y*y)`; `engine.js` now evaluates sin/cos in double-double arithmetic and rounds once, and
`tickdiff.mjs` shows whole rounds with zero one-ulp differences in any state variable). The remaining
noise floor is glibc itself, which is not correctly rounded on roughly 0.1% of inputs (a pad angle hit
one such value in one 12-player round out of 40; the round still matched to 1e-9 for 2400 ticks). The
pass criterion for the ladder is that the JS agrees with the shipped C++ at least as often as the
C++'s own no-FMA build does (with two independent 24-round samples "within 2 se" is a 1.4-sigma band
that even a perfect port fails in about one row in six).

## Tick-level diff against the C++ (probe.cpp + tickdiff.mjs)

`probe.cpp` `#include`s `rl/native/ungroup.cpp` and prints every state variable (bodies, players,
the pairwise ledger, mines, pickups, bot actions, events, statistics; optionally observations,
`frame_json`/`meta_json` and macro labels) with 17 significant digits after every tick;
`tickdiff.mjs` runs the same lineup through `Game` and reports the first field that differs by more
than 1e-9 and the first one-ulp difference. External seats can be scripted identically on both sides
(macro classes 10..23, move 9 with `set_direction`, compass moves, stop, joinable, held leaves,
intent), and `rounds > 1` resets the same game repeatedly (persist=1).

```sh
g++ -O2 -std=c++17 -ffp-contract=off -o /tmp/ungroup_probe web/test/probe.cpp
node web/test/tickdiff.mjs life policy,bail,policy,loyal,loyal,rammer 1 7200 6 1 15   # flags: 1 script, 2 obs, 4 frame/meta, 8 labels
node web/test/tickdiff.mjs series policy,bail,bail,loyal,grudge,rammer 2 7200 6 3 9  # three persisted rounds
node web/test/tickdiff.mjs life policy,bail,bail,loyal,loyal,rammer 1 7200 6 1 9 '{"head_steer":2}'
```

Reviewed this way (40 whole rounds: legacy/crown/bloom/life/series, 6/8/12/32 players, decide_every
1/3/6, scripted external seats, head_steer=2, bank_round=0, obs_legacy=1): every field identical for
the whole round in 39 of 40, the 40th within 1e-9 (a glibc rounding case, see above).

## Interpretations made while porting

- `apply_actions`: the C++ computes every bot's action, then applies all seats in index order. The JS
  does the same at decision ticks (every `decideEvery` = 6 ticks); an external seat's `setInput`
  additionally applies at the next tick so human input is not delayed. The C++ step accounting
  (rewards, mean group size, closing open alliances at the end) runs at the end of every decision
  step and when the round ends, as in `ugb_step`.
- Events are objects with the same fields as the C++ JSON; `step()` clears the list first like
  `ugb_step`, `tick()` never does, `takeEvents()` drains it (for the multiplayer host).
- Numbers in `frame()`/`meta()` are rounded like the C++ `printf` formats (3, 2, 1 and 6 decimals),
  including glibc's round-half-to-even on exact binary ties (0.125 -> 0.12, where `toFixed` gives
  0.13); `meta().cfg` keeps full precision instead of `%.6g` (a replay's `cfg` from the C++ has
  `dt = 0.0333333`, not 1/30, so rebuild a config from the preset rather than from a C++ meta);
  `meta().seed` is a Number (the exact BigInt is `game.seedBig`).
- `Math.hypot` is never used (V8 scales before the square root, so it is not `sqrt(x*x + y*y)` to the
  ulp); every norm is written out like the C++ `Vec::norm()`. `Math.atan2` only selects a compass sector
  in `directionToMove` and `Math.pow(n, 2.0)` is exact, so those stay.
- Domain randomisation of the config (`ugb_set_cfg_range`) is not ported; `sampleCfg()` is a no-op.
- The working-tree core added `head_steer` (weight of the head's push direction under the crown) and
  `bank_round` (whole-unit banking, on in the life/series presets); both are ported and covered by the
  ladder.
