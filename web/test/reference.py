#!/usr/bin/env python3
"""
Reference numbers from the canonical C++ core for web/test/conformance.mjs.

Writes a JSON file with
  exact:  for seeds 1..20 under the life preset, the C++ meta (needs, pads, mine_pos, mine_type) and
          the players' initial intents (from the first frame), to check that Game.reset() draws the
          same values;
  ladder: for every lineup x preset, ladder_native.run(...) with 24 paired seeds (progress per seat
          type with standard errors, merges, leaves, spills, banks, ...).
  trace:  (optional, --trace) per-step frames of one bots-only game so the JS engine can be diffed
          tick by tick until the first divergence.

Usage:
  OMP_NUM_THREADS=1 python3 web/test/reference.py [--out web/test/reference.json] [--games 24] [--seed 1000]
  OMP_NUM_THREADS=1 python3 web/test/reference.py --trace /tmp/trace.json --preset life --seed 1
"""
import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "rl"))

import numpy as np  # noqa: E402
from ladder_native import run  # noqa: E402
from ungroup.native import Config, NativeBatch, preset, CFG_FIELDS  # noqa: E402

LINEUPS = [
    ["solo"] * 6,
    ["bail"] * 6,
    ["loyal"] * 6,
    ["bail"] * 3 + ["loyal"] * 3,
    ["bail"] * 2 + ["loyal"] * 2 + ["rammer"] * 2,
    ["bail"] * 5 + ["loyal"],
]
PRESET_NAMES = ["legacy", "life"]


def exact_block(seeds):
    cfg = preset("life")
    out = []
    for s in seeds:
        b = NativeBatch(1, cfg, seed=s)
        b.reset(0, s)
        m = b.meta(0)
        fr = b.frame(0)
        out.append({"seed": s, "needs": m["needs"], "pads": m["pads"], "mine_pos": m["mine_pos"], "mine_type": m["mine_type"],
                    "intents": [p["intent"] for p in fr["players"]], "cfg": m["cfg"]})
    return out


def ladder_block(games, seed):
    out = {}
    for pname in PRESET_NAMES:
        cfg = preset(pname)
        out[pname] = {}
        for lineup in LINEUPS:
            key = ",".join(lineup)
            r = run(lineup, games, seed, cfg, quiet=True)
            r.pop("sps", None)
            out[pname][key] = r
            print(f"[{pname}] {key}: merges={r['merges']:.2f} leaves={r['leaves']:.2f} spills={r['spills']:.2f} banks={r['banks']:.2f} "
                  + " ".join(f"{k}={v['progress']:.3f}±{v['se']:.3f}" for k, v in r["by_type"].items()), file=sys.stderr)
    return out


def trace_block(lineup, pname, seed, decide_every=6):
    """Frames after every decision step (bots decide inside the C++ at every ugb_step call)."""
    cfg = preset(pname).replace(n_players=len(lineup))
    b = NativeBatch(1, cfg, seed=seed, decide_every=decide_every)
    b.set_seats(0, lineup)
    b.reset(0, seed)
    acts = np.zeros((1, len(lineup), 4), dtype=np.int32)
    frames = [b.frame(0)]
    obs0 = b.observe()[0].tolist()
    stats = None
    while not b.done(0):
        _, _, _, ep = b.step(acts, auto_reset=False)
        frames.append(b.frame(0))
        if ep:
            stats = ep[0]
    return {"lineup": lineup, "preset": pname, "seed": seed, "decide_every": decide_every, "meta": b.meta(0), "frames": frames,
            "obs0": obs0, "stats": stats, "progress": [b.progress(0, i) for i in range(len(lineup))]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "reference.json"))
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--trace", default=None, help="write a per-step trace of one game to this path instead of the reference")
    ap.add_argument("--preset", default="life")
    ap.add_argument("--lineup", default="bail,bail,bail,loyal,loyal,loyal")
    ap.add_argument("--trace-seed", type=int, default=1)
    a = ap.parse_args()
    if a.trace:
        tr = trace_block(a.lineup.split(","), a.preset, a.trace_seed)
        with open(a.trace, "w") as f:
            json.dump(tr, f)
        print(f"wrote {a.trace}: {len(tr['frames'])} frames, winner {tr['meta']['winner']}")
        sys.exit(0)
    ref = {"cfg_fields": CFG_FIELDS, "defaults": Config().to_array(), "games": a.games, "seed": a.seed,
           "presets": {p: preset(p).to_array() for p in PRESET_NAMES},
           "exact": exact_block(range(1, 21)), "ladder": ladder_block(a.games, a.seed)}
    with open(a.out, "w") as f:
        json.dump(ref, f)
    print(f"wrote {a.out}")
