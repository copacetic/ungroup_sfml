"""
Evaluate a checkpoint against scripted bots and against itself.

Usage: python3 rl/evaluate.py --checkpoint rl/checkpoints/policy_latest.pt --games 20
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ppo import Policy  # noqa: E402
from ungroup import Config, UngroupEnv  # noqa: E402
from ungroup.bots import BOTS  # noqa: E402


def run(policy, seats, games, seed, deterministic=False):
    n = len(seats)
    env = UngroupEnv(Config(n_players=n), seed=seed, decide_every=2)
    wins = np.zeros(n)
    prog = np.zeros(n)
    finished = 0
    group, merges, leaves, spills, banks, lengths = [], [], [], [], [], []
    for g in range(games):
        obs = env.reset(seed=seed + g)
        bots = [None if s == "policy" else BOTS[s]() for s in seats]
        done = False
        gs, nm, nl, ns, nb = [], 0, 0, 0, 0
        while not done:
            act, _, _ = policy.act(torch.from_numpy(obs), deterministic=deterministic)
            act = act.numpy()
            for i, b in enumerate(bots):
                if b is not None:
                    act[i] = b.act(env.game, i)
            obs, _, done, info = env.step(act)
            gs.append(np.mean([env.game.body_of(i).n for i in range(n)]))
            for e in info["events"]:
                nm += e["kind"] == "merge"
                nl += e["kind"] == "leave"
                ns += e["kind"] == "spill"
                nb += e["kind"] == "bank"
        gm = env.game
        if gm.winner >= 0:
            wins[gm.winner] += 1
            finished += 1
        for i in range(n):
            prog[i] += gm.progress(i)
        group.append(np.mean(gs)); merges.append(nm); leaves.append(nl); spills.append(ns); banks.append(nb)
        lengths.append(gm.t)
    by_type = {}
    for i, s in enumerate(seats):
        d = by_type.setdefault(s, dict(seats=0, wins=0, prog=0.0))
        d["seats"] += 1
        d["wins"] += wins[i]
        d["prog"] += prog[i]
    print(f"seats={','.join(seats)} games={games} finished={finished}/{games} avg_len={np.mean(lengths):.0f}s "
          f"avg_group={np.mean(group):.2f} merges={np.mean(merges):.1f} leaves={np.mean(leaves):.1f} "
          f"spills={np.mean(spills):.1f} banks={np.mean(banks):.1f}")
    for s, d in by_type.items():
        print(f"  {s:7s}: win rate per seat {d['wins'] / (d['seats'] * games):.2f} (chance {1 / n:.2f}), "
              f"avg progress {d['prog'] / (d['seats'] * games):.2f}")
    return by_type


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--players", type=int, default=6)
    a = ap.parse_args()
    probe = UngroupEnv(Config(n_players=a.players))
    policy = Policy(probe.obs_dim)
    policy.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
    policy.eval()
    torch.set_num_threads(1)
    n = a.players
    half = n // 2
    run(policy, ["policy"] * n, a.games, a.seed, a.deterministic)
    run(policy, ["policy"] * half + ["bail"] * (n - half), a.games, a.seed, a.deterministic)
    run(policy, ["policy"] * half + ["solo"] * (n - half), a.games, a.seed, a.deterministic)
    run(policy, ["bail"] * half + ["solo"] * (n - half), a.games, a.seed, a.deterministic)
