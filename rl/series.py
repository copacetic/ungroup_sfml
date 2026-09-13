"""
Series play on the canonical core: the same seats play consecutive rounds and, with persist=1, the
pairwise ledger (who left whom, who banked for whom) carries over between rounds.

  python3 rl/series.py --preset series --rounds 5 --games 24 bail bail bail bail grudge grudge
  python3 rl/series.py --preset series --rounds 5 --set persist=0 bail bail bail bail grudge grudge   # control

Prints, per round, each seat type's mean progress and wins per seat, plus the lobby's merges and
leaves, so the round-over-round change (the ledger gate in docs/SKILL_CEILING.md section 7) is visible.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_native import parse_sets, summarize  # noqa: E402
from ungroup.native import PRESETS, NativeBatch  # noqa: E402


def play_series(seats, rounds, games, seed, cfg):
    n = len(seats)
    cfg = cfg.replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed)
    for e in range(games):
        b.set_seats(e, seats)
        b.reset(e, seed + e, fresh=True)
    acts = np.zeros((games, n, 4), dtype=np.int32)
    per_round = []
    for r in range(rounds):
        eps = {}
        while len(eps) < games:
            _, _, _, ep = b.step(acts, auto_reset=False)
            for d in ep:
                eps.setdefault(d["env"], d)
        per_round.append(summarize(seats, [eps[e] for e in range(games)]))
        if r + 1 < rounds:
            for e in range(games):
                b.reset(e, seed + 1000 * (r + 1) + e)   # persist decides whether the ledger survives
    return per_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--preset", default="series", choices=sorted(PRESETS))
    ap.add_argument("--set", action="append")
    ap.add_argument("seats", nargs="+")
    a = ap.parse_args()
    cfg = parse_sets(a.set, a.preset)
    rows = play_series(a.seats, a.rounds, a.games, a.seed, cfg)
    types = list(dict.fromkeys(a.seats))
    print(f"persist={cfg.persist} crown={cfg.crown} brand={cfg.brand} bloom_rate={cfg.bloom_rate}  seats={','.join(a.seats)}  {a.games} lobbies")
    head = "round  " + "  ".join(f"{t:>7s}" for t in types) + "   wins/seat " + " ".join(f"{t:>6s}" for t in types) + "   merges leaves early"
    print(head)
    for r, s in enumerate(rows):
        prog = "  ".join(f"{s['by_type'][t]['progress']:7.3f}" for t in types)
        wins = " ".join(f"{s['by_type'][t]['win_per_seat']:6.2f}" for t in types)
        print(f"{r:>5}  {prog}             {wins}   {s['merges']:6.1f} {s['leaves']:6.1f} {s['finished_early']:5.2f}")


if __name__ == "__main__":
    main()
