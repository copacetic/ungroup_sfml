"""
Scripted-bot ladder on the canonical C++ core, with paired seeds and alliance statistics.

Usage:
  python3 rl/ladder_native.py --games 48 bail bail bail solo solo solo
  python3 rl/ladder_native.py --gates            # the rules pacing gates (no policy involved)
  python3 rl/ladder_native.py --set mine_rate=0.2 --set leave_time=3 --games 48 bail bail bail loyal loyal loyal
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ungroup.native import Config, NativeBatch, SEAT_NAMES  # noqa: E402


def run(seats, games=48, seed=1000, cfg=None, quiet=False):
    n = len(seats)
    cfg = (cfg or Config()).replace(n_players=n)
    E = games
    b = NativeBatch(E, cfg, seed=seed)
    for e in range(E):
        b.set_seats(e, seats)
        b.reset(e, seed + e)
    acts = np.zeros((E, n, 4), dtype=np.int32)
    eps = []
    t0 = time.time()
    steps = 0
    while len(eps) < E:
        _, _, _, ep = b.step(acts, auto_reset=False)
        steps += E
        for d in ep:
            if all(d["env"] != x["env"] for x in eps):
                eps.append(d)
    dt = time.time() - t0
    res = summarize(seats, eps)
    res["sps"] = steps / dt
    if not quiet:
        print(fmt(res))
    return res


def summarize(seats, eps):
    n = len(seats)
    prog = np.array([e["progress"] for e in eps])
    wins = np.zeros(n)
    for e in eps:
        if e["winner"] >= 0:
            wins[e["winner"]] += 1
    out = dict(
        seats=list(seats), games=len(eps),
        finished_early=float(np.mean([not e["timeout_win"] for e in eps])),
        length=float(np.mean([e["length"] for e in eps])),
        avg_group=float(np.mean([e["avg_group"] for e in eps])),
        merges=float(np.mean([e["merges"] for e in eps])),
        leaves=float(np.mean([e["leaves"] for e in eps])),
        remerge_fast=float(np.mean([e["remerge_fast"] for e in eps])),
        spills=float(np.mean([e["spills"] for e in eps])),
        banks=float(np.mean([e["banks"] for e in eps])),
        group_banks=float(np.mean([e["group_banks"] for e in eps])),
        alliances=float(np.mean([e["alliances"] for e in eps])),
        alliance_dur=float(np.mean([e["alliance_dur"] for e in eps if e["alliances"] > 0] or [0])),
        alliances_long=float(np.mean([e["alliances_long"] for e in eps])),
        by_type={},
    )
    for s in sorted(set(seats), key=seats.index):
        idx = [i for i, x in enumerate(seats) if x == s]
        out["by_type"][s] = dict(progress=float(prog[:, idx].mean()), se=float(prog[:, idx].mean(axis=1).std(ddof=1) / np.sqrt(len(eps))) if len(eps) > 1 else 0.0,
                                 win_per_seat=float(wins[idx].sum() / (len(idx) * len(eps))))
    return out


def fmt(r):
    s = (f"{','.join(r['seats'])}: n={r['games']} early_finish={r['finished_early']:.2f} len={r['length']:.0f}s group={r['avg_group']:.2f} "
         f"merges={r['merges']:.1f} leaves={r['leaves']:.1f} remerge<3s={r['remerge_fast']:.1f} spills={r['spills']:.1f} banks={r['banks']:.1f} "
         f"group_banks={r['group_banks']:.1f} alliances={r['alliances']:.1f} mean_dur={r['alliance_dur']:.1f}s long(>=10s)={r['alliances_long']:.1f}")
    if "sps" in r:
        s += f" [{r['sps']:.0f} steps/s]"
    for k, v in r["by_type"].items():
        s += f"\n    {k:7s} progress {v['progress']:.3f} (se {v['se']:.3f}) win/seat {v['win_per_seat']:.2f} (chance {1 / len(r['seats']):.2f})"
    return s


def gates(cfg=None, games=48, seed=1000):
    """Rules pacing gates. Prints each lineup with the alliance columns."""
    cfg = cfg or Config()
    print("== solo alone (1 player)"); run(["solo"], games, seed, cfg)
    print("== pair of loyal (2 players)"); run(["loyal", "loyal"], games, seed, cfg)
    print("== six solo"); run(["solo"] * 6, games, seed, cfg)
    print("== six bail"); run(["bail"] * 6, games, seed, cfg)
    print("== six loyal"); run(["loyal"] * 6, games, seed, cfg)
    print("== three bail + three solo"); run(["bail"] * 3 + ["solo"] * 3, games, seed, cfg)
    print("== three bail + three loyal"); run(["bail"] * 3 + ["loyal"] * 3, games, seed, cfg)
    print("== two bail + two loyal + two kidnap"); run(["bail"] * 2 + ["loyal"] * 2 + ["kidnap"] * 2, games, seed, cfg)
    print("== two bail + two loyal + two rammer"); run(["bail"] * 2 + ["loyal"] * 2 + ["rammer"] * 2, games, seed, cfg)


def parse_sets(sets):
    cfg = Config()
    for kv in sets or []:
        k, v = kv.split("=")
        f = Config.__dataclass_fields__[k]
        cfg = cfg.replace(**{k: int(v) if f.type is int else float(v)})
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--set", action="append", help="override a Config field, e.g. --set mine_rate=0.2")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("bots", nargs="*")
    a = ap.parse_args()
    cfg = parse_sets(a.set)
    if a.gates:
        gates(cfg, a.games, a.seed)
    else:
        for b in a.bots:
            assert b in SEAT_NAMES, f"unknown bot {b}"
        run(a.bots, a.games, a.seed, cfg)
