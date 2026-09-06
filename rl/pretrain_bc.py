"""
Behavior-cloning warm start: imitate the scripted bots so PPO starts from a policy that already
mines, groups, leaves, and banks, instead of discovering those from random toggles.

Usage: python3 rl/pretrain_bc.py --games 40 --out rl/checkpoints/policy_bc.pt
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ppo import Policy  # noqa: E402
from ungroup import ACTION_NVEC, Config, UngroupEnv  # noqa: E402
from ungroup.bots import BOTS  # noqa: E402


def generate(args):
    seed, games, n, bail_frac = args
    rng = np.random.default_rng(seed)
    env = UngroupEnv(Config(n_players=n), seed=seed, decide_every=2)
    X, Y = [], []
    for g in range(games):
        obs = env.reset(seed=seed * 10000 + g)
        bots = [BOTS["bail"]() if rng.random() < bail_frac else BOTS["solo"]() for _ in range(n)]
        done = False
        while not done:
            acts = np.array([bots[i].act(env.game, i) for i in range(n)], dtype=np.int64)
            X.append(obs.copy())
            Y.append(acts)
            obs, _, done, _ = env.step(acts)
    return np.concatenate(X), np.concatenate(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--bail-frac", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "policy_bc.pt"))
    a = ap.parse_args()

    t0 = time.time()
    per = max(1, a.games // a.workers)
    with mp.get_context("fork").Pool(a.workers) as pool:
        parts = pool.map(generate, [(w + 1, per, a.players, a.bail_frac) for w in range(a.workers)])
    X = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    print(f"dataset: {X.shape[0]} samples in {time.time() - t0:.0f}s")

    policy = Policy(X.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    N = Xt.shape[0]
    bs = 2048
    for ep in range(a.epochs):
        perm = torch.randperm(N)
        tot, correct = 0.0, np.zeros(len(ACTION_NVEC))
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            logits, _ = policy(Xt[idx])
            loss = 0.0
            for k, lg in enumerate(logits):
                loss = loss + F.cross_entropy(lg, Yt[idx, k])
                correct[k] += (lg.argmax(-1) == Yt[idx, k]).float().sum().item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        print(f"epoch {ep}: loss {tot / N:.3f} accuracy per head {np.round(correct / N, 3).tolist()}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(policy.state_dict(), a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
