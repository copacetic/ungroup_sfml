"""
Scripted-bot ladder: run games between scripted policies and report who wins, how long games
last, and how much grouping happens. Used to check that the rules reward alliances before
spending compute on RL.

Usage: python3 rl/ladder.py --games 20 solo solo solo bail bail bail
"""

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ungroup import Config, UngroupEnv  # noqa: E402
from ungroup.bots import BOTS  # noqa: E402


def run(bot_names, games, seed, verbose=False, record=False, decide_every=2):
    cfg = Config(n_players=len(bot_names))
    env = UngroupEnv(cfg, seed=seed, decide_every=decide_every, record=record)
    wins = np.zeros(len(bot_names))
    finished = 0
    lengths, group_sizes, merges, leaves, spills, banks = [], [], [], [], [], []
    scores = np.zeros(len(bot_names))
    t0 = time.time()
    steps = 0
    last_game = None
    for gi in range(games):
        env.reset()
        bots = [BOTS[n](**({"seed": seed + k} if n == "random" else {})) for k, n in enumerate(bot_names)]
        done = False
        gs, nm, nl, ns, nb = [], 0, 0, 0, 0
        while not done:
            acts = [bots[i].act(env.game, i) for i in range(len(bots))]
            _, _, done, info = env.step(acts)
            steps += 1
            gs.append(np.mean([env.game.body_of(i).n for i in range(len(bots))]))
            for e in info["events"]:
                nm += e["kind"] == "merge"
                nl += e["kind"] == "leave"
                ns += e["kind"] == "spill"
                nb += e["kind"] == "bank"
        g = env.game
        if g.winner >= 0:
            wins[g.winner] += 1
            finished += 1
        for i in range(len(bots)):
            scores[i] += g.progress(i)
        lengths.append(g.t)
        group_sizes.append(np.mean(gs))
        merges.append(nm)
        leaves.append(nl)
        spills.append(ns)
        banks.append(nb)
        last_game = g
        if verbose:
            print(f"game {gi}: winner={g.winner} t={g.t:.0f}s avg_group={np.mean(gs):.2f} merges={nm} "
                  f"leaves={nl} spills={ns} banks={nb} progress={[round(g.progress(i), 2) for i in range(len(bots))]}")
    dt = time.time() - t0
    print(f"summary: finished {finished}/{games}, avg length {np.mean(lengths):.0f}s, avg group size "
          f"{np.mean(group_sizes):.2f}, merges {np.mean(merges):.1f}, leaves {np.mean(leaves):.1f}, "
          f"spills {np.mean(spills):.1f}, banks {np.mean(banks):.1f}, {steps / dt:.0f} steps/s")
    for i, n in enumerate(bot_names):
        print(f"  seat {i} ({n}): wins {int(wins[i])}, avg progress {scores[i] / games:.2f}")
    return last_game


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("bots", nargs="+")
    a = ap.parse_args()
    run(a.bots, a.games, a.seed, a.verbose)
