"""
Imitation warm start with DAgger: clone the scripted group-and-bail bot, then repeatedly roll
out the current policy, label the states it visits with the bot's choices, aggregate, and
retrain. PPO then starts from a policy that already mines, groups, leaves, and banks.

Usage: python3 rl/pretrain_bc.py --iters 4 --games 32 --out rl/checkpoints/policy_bc.pt
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
from train_ppo import Policy, load_policy  # noqa: E402
from ungroup import ACTION_NVEC, Config, UngroupEnv  # noqa: E402
from ungroup.bots import BOTS  # noqa: E402


def rollout(args):
    """Play games; return (obs, expert action) pairs for every seat.

    When a checkpoint is given, seats act with the policy (sampled) except a few bail-bot seats
    kept as partners; the expert label is always what the bail bot would do in that state.
    """
    seed, games, n, ckpt, bail_frac = args
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    env = UngroupEnv(Config(n_players=n), seed=seed, decide_every=2)
    policy = load_policy(ckpt, env.obs_dim) if ckpt else None
    expert = BOTS["bail"]()
    X, Y = [], []
    for g in range(games):
        obs = env.reset(seed=seed * 10000 + g)
        if policy is None:
            actors = [BOTS["bail"]() if rng.random() < bail_frac else BOTS["solo"]() for _ in range(n)]
        else:
            actors = [BOTS["bail"]() if rng.random() < 0.3 else None for _ in range(n)]
        done = False
        while not done:
            labels = np.array([expert.act(env.game, i) for i in range(n)], dtype=np.int64)
            if policy is None:
                acts = np.array([actors[i].act(env.game, i) for i in range(n)], dtype=np.int64)
            else:
                acts, _, _ = policy.act(torch.from_numpy(obs))
                acts = acts.numpy()
                for i in range(n):
                    if actors[i] is not None:
                        acts[i] = actors[i].act(env.game, i)
            X.append(obs.copy())
            Y.append(labels)
            obs, _, done, _ = env.step(acts)
    return np.concatenate(X), np.concatenate(Y)


def evaluate(ckpt, n, games, seed):
    torch.set_num_threads(1)
    env = UngroupEnv(Config(n_players=n), seed=seed, decide_every=2)
    policy = load_policy(ckpt, env.obs_dim)
    prog = []
    for g in range(games):
        obs = env.reset(seed=seed + g)
        done = False
        while not done:
            acts, _, _ = policy.act(torch.from_numpy(obs))
            obs, _, done, _ = env.step(acts.numpy())
        prog.extend(env.game.progress(i) for i in range(n))
    return float(np.mean(prog))


def train(policy, X, Y, epochs, lr):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    N = Xt.shape[0]
    bs = 2048
    policy.train()
    for ep in range(epochs):
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
        if ep == epochs - 1 or ep % 5 == 0:
            print(f"    epoch {ep}: loss {tot / N:.3f} accuracy per head {np.round(correct / N, 3).tolist()}", flush=True)
    policy.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4, help="DAgger iterations after the initial clone")
    ap.add_argument("--games", type=int, default=32, help="games per data collection round")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--bail-frac", type=float, default=0.75)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "policy_bc.pt"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    ctx = mp.get_context("fork")
    per = max(1, a.games // a.workers)

    X_all, Y_all = None, None
    policy = None
    for it in range(a.iters + 1):
        t0 = time.time()
        ckpt = None if it == 0 else a.out
        with ctx.Pool(a.workers) as pool:
            parts = pool.map(rollout, [(it * 100 + w + 1, per, a.players, ckpt, a.bail_frac) for w in range(a.workers)])
        X = np.concatenate([p[0] for p in parts])
        Y = np.concatenate([p[1] for p in parts])
        X_all = X if X_all is None else np.concatenate([X_all, X])
        Y_all = Y if Y_all is None else np.concatenate([Y_all, Y])
        print(f"iter {it}: +{X.shape[0]} samples ({X_all.shape[0]} total) collected in {time.time() - t0:.0f}s", flush=True)
        if policy is None:
            policy = Policy(X.shape[1], hidden=a.hidden)
        train(policy, X_all, Y_all, a.epochs if it == 0 else max(6, a.epochs // 2), 1e-3 if it == 0 else 5e-4)
        torch.save(policy.state_dict(), a.out)
        with ctx.Pool(a.workers) as pool:
            progs = pool.starmap(evaluate, [(a.out, a.players, 1, 5000 + it * 10 + w) for w in range(a.workers)])
        print(f"iter {it}: self-play mean progress {np.mean(progs):.2f} (bail bot ~0.8)", flush=True)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
