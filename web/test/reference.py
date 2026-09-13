#!/usr/bin/env python3
"""
Reference numbers from the canonical C++ core for web/test/conformance.mjs.

Writes a JSON file with
  exact:  for seeds 1..20 under the life preset, the C++ meta (needs, pads, mine_pos, mine_type) and
          the players' initial intents (from the first frame), to check that Game.reset() draws the
          same values;
  ladder: for every lineup x preset, ladder_native.run(...) with 24 paired seeds (progress per seat
          type with standard errors, merges, leaves, spills, banks, ...).
  ladder_nofma: (--nofma, default on when g++ is available) the same ladder from a copy of the core
          built with -ffp-contract=off. The shipped libungroup.so is built with -march=native and GCC
          keeps FMA contraction on for C++, which JavaScript cannot reproduce; the no-FMA build is what
          the JS engine matches tick for tick, so this block calibrates how much of the JS-vs-C++ gap
          is floating-point chaos rather than rules.
  trace:  (optional, --trace) per-step frames of one bots-only game so the JS engine can be diffed
          tick by tick until the first divergence (--lib picks the shared library).

Usage:
  OMP_NUM_THREADS=1 python3 web/test/reference.py [--out web/test/reference.json] [--games 24] [--seed 1000] [--no-nofma]
  OMP_NUM_THREADS=1 python3 web/test/reference.py --trace /tmp/trace.json --preset life --trace-seed 1
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "rl")
sys.path.insert(0, RL)

import numpy as np  # noqa: E402
import ungroup.native as native  # noqa: E402
if "--lib" in sys.argv:  # use an alternative build of the core (see --nofma)
    native.LIB = sys.argv[sys.argv.index("--lib") + 1]
    native.SRC = native.LIB  # never rebuild it
from ladder_native import summarize  # noqa: E402
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


def run_eps(seats, games, seed, cfg):
    """ladder_native.run with the per-game records kept (paired seeds seed..seed+games-1)."""
    n = len(seats)
    cfg = cfg.replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed)
    for e in range(games):
        b.set_seats(e, seats)
        b.reset(e, seed + e)
    acts = np.zeros((games, n, 4), dtype=np.int32)
    eps = []
    while len(eps) < games:
        _, _, _, ep = b.step(acts, auto_reset=False)
        for d in ep:
            if all(d["env"] != x["env"] for x in eps):
                eps.append(d)
    eps.sort(key=lambda d: d["env"])
    res = summarize(seats, eps)
    keep = ["env", "winner", "length", "merges", "leaves", "spills", "banks", "group_banks", "progress", "timeout_win"]
    res["eps"] = [{k: (bool(d[k]) if k == "timeout_win" else d[k]) for k in keep} for d in eps]
    return res


def ladder_block(games, seed):
    out = {}
    for pname in PRESET_NAMES:
        cfg = preset(pname)
        out[pname] = {}
        for lineup in LINEUPS:
            key = ",".join(lineup)
            r = run_eps(lineup, games, seed, cfg)
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
            stats = {k: (v.tolist() if hasattr(v, "tolist") else (bool(v) if isinstance(v, (np.bool_, bool)) else v)) for k, v in ep[0].items()}
    return {"lineup": lineup, "preset": pname, "seed": seed, "decide_every": decide_every, "meta": b.meta(0), "frames": frames,
            "obs0": obs0, "stats": stats, "progress": [b.progress(0, i) for i in range(len(lineup))]}


def nofma_ladder(games, seed):
    """Build the core with FMA contraction off in a temp dir and run the ladder there (a subprocess, since
    ctypes cannot load two builds of the same library into one process cleanly)."""
    if shutil.which("g++") is None:
        print("g++ not found: skipping the no-FMA calibration", file=sys.stderr)
        return None
    tmp = tempfile.mkdtemp(prefix="ungroup_nofma_")
    lib = os.path.join(tmp, "libungroup_nofma.so")
    src = os.path.join(RL, "native", "ungroup.cpp")
    subprocess.check_call(["g++", "-O2", "-std=c++17", "-ffp-contract=off", "-fopenmp", "-shared", "-fPIC", "-o", lib, src])
    out = os.path.join(tmp, "ladder_nofma.json")
    subprocess.check_call([sys.executable, os.path.abspath(__file__), "--lib", lib, "--ladder-only", "--out", out,
                           "--games", str(games), "--seed", str(seed)], env=dict(os.environ, OMP_NUM_THREADS="1"))
    with open(out) as f:
        return json.load(f)["ladder"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "reference.json"))
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--lib", default=None, help="alternative build of libungroup.so")
    ap.add_argument("--ladder-only", action="store_true")
    ap.add_argument("--no-nofma", action="store_true", help="skip the -ffp-contract=off calibration build")
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
    if a.ladder_only:
        with open(a.out, "w") as f:
            json.dump({"ladder": ladder_block(a.games, a.seed)}, f)
        sys.exit(0)
    ref = {"cfg_fields": CFG_FIELDS, "defaults": Config().to_array(), "games": a.games, "seed": a.seed,
           "presets": {p: preset(p).to_array() for p in PRESET_NAMES},
           "exact": exact_block(range(1, 21)), "ladder": ladder_block(a.games, a.seed)}
    if not a.no_nofma:
        print("== no-FMA calibration build", file=sys.stderr)
        ref["ladder_nofma"] = nofma_ladder(a.games, a.seed)
    with open(a.out, "w") as f:
        json.dump(ref, f)
    print(f"wrote {a.out}")
