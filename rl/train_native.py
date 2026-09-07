"""
PPO self-play on the C++ rules core: one process, one NativeBatch of many games stepped in
parallel with OpenMP, one shared policy. Seats per episode are policy, frozen snapshot, or a
scripted bot run inside C++; only policy seats produce training samples.

Usage: python3 rl/train_native.py --updates 2000 --envs 64 --resume rl/checkpoints/policy_latest.pt
"""

import argparse
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ppo import Policy, load_policy  # noqa: E402
from ungroup import ACTION_NVEC, Config  # noqa: E402
from ungroup.native import SEAT_BAIL, SEAT_EXTERNAL, SEAT_EXTERNAL2, SEAT_SOLO, NativeBatch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.001)
    ap.add_argument("--bot-prob", type=float, default=0.25)
    ap.add_argument("--snapshot-prob", type=float, default=0.35)
    ap.add_argument("--snapshot-every", type=int, default=25)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg = Config(n_players=args.players)
    batch = NativeBatch(args.envs, cfg, seed=args.seed * 100000 + 1, decide_every=2)
    E, n, D = batch.E, batch.n, batch.obs_dim

    if args.resume:
        policy = load_policy(args.resume, D)
        policy.train()
    else:
        policy = Policy(D, hidden=args.hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    snapshots = [copy.deepcopy(policy).eval()]

    seat_types = np.zeros((E, n), dtype=int)
    snap_idx = np.zeros(E, dtype=int)

    def assign(e):
        st = np.zeros(n, dtype=int)
        r = rng.random()
        if r < args.bot_prob:
            k = int(rng.integers(1, n))
            idx = rng.choice(n, k, replace=False)
            st[idx] = np.where(rng.random(k) < 0.5, SEAT_SOLO, SEAT_BAIL)
        elif r < args.bot_prob + args.snapshot_prob:
            k = int(rng.integers(1, n))
            idx = rng.choice(n, k, replace=False)
            st[idx] = SEAT_EXTERNAL2
            snap_idx[e] = int(rng.integers(len(snapshots)))
        seat_types[e] = st
        batch.set_seats(e, st.tolist())

    for e in range(E):
        assign(e)
    obs = batch.observe()

    log_path = os.path.join(args.out, "log_native.csv")
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if os.path.getsize(log_path) == 0:
        logw.writerow(["update", "samples", "time", "policy_winrate", "bot_winrate", "snapshot_winrate", "finish_rate",
                       "length", "policy_progress", "bot_progress", "avg_group", "merges", "leaves", "spills", "banks",
                       "entropy", "value_loss", "policy_loss", "kl", "sps"])
    total_samples = 0
    t_start = time.time()
    ep_buffer = []

    for update in range(1, args.updates + 1):
        T = args.rollout
        t_roll = time.time()
        b_obs = np.zeros((T, E, n, D), dtype=np.float32)
        b_act = np.zeros((T, E, n, len(ACTION_NVEC)), dtype=np.int64)
        b_logp = np.zeros((T, E, n), dtype=np.float32)
        b_val = np.zeros((T + 1, E, n), dtype=np.float32)
        b_rew = np.zeros((T, E, n), dtype=np.float32)
        b_done = np.zeros((T + 1, E), dtype=np.float32)
        b_mask = np.zeros((T, E, n), dtype=np.float32)
        policy.eval()
        for t in range(T):
            obs_t = torch.from_numpy(obs.reshape(E * n, D))
            act, logp, val = policy.act(obs_t)
            act = act.numpy().reshape(E, n, -1)
            logp = logp.numpy().reshape(E, n)
            val = val.numpy().reshape(E, n)
            snap_seats = seat_types == SEAT_EXTERNAL2
            if snap_seats.any():
                for k in np.unique(snap_idx[snap_seats.any(1)]):
                    sel = snap_seats & (snap_idx[:, None] == k)
                    if sel.any():
                        a, _, _ = snapshots[k % len(snapshots)].act(torch.from_numpy(obs[sel]))
                        act[sel] = a.numpy()
            b_obs[t] = obs
            b_act[t] = act
            b_logp[t] = logp
            b_val[t] = val
            b_mask[t] = (seat_types == SEAT_EXTERNAL)
            obs, rew, done, eps = batch.step(act)
            b_rew[t] = rew
            b_done[t + 1] = done
            for ep in eps:
                ep["types"] = seat_types[ep["env"]].tolist()
                ep["winner_type"] = int(seat_types[ep["env"]][ep["winner"]]) if ep["winner"] >= 0 else -1
                ep_buffer.append(ep)
                assign(ep["env"])
        with torch.no_grad():
            _, last_val = policy.forward(torch.from_numpy(obs.reshape(E * n, D)))
        b_val[T] = last_val.numpy().reshape(E, n)
        roll_time = time.time() - t_roll

        adv = np.zeros((T, E, n), dtype=np.float32)
        last = np.zeros((E, n), dtype=np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t + 1][:, None]
            delta = b_rew[t] + args.gamma * b_val[t + 1] * nonterm - b_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + b_val[:T]

        mask = b_mask.reshape(-1) > 0
        f_obs = torch.from_numpy(b_obs.reshape(-1, D)[mask])
        f_act = torch.from_numpy(b_act.reshape(-1, len(ACTION_NVEC))[mask])
        f_logp = torch.from_numpy(b_logp.reshape(-1)[mask])
        f_adv = torch.from_numpy(adv.reshape(-1)[mask])
        f_ret = torch.from_numpy(ret.reshape(-1)[mask])
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        N = f_obs.shape[0]
        total_samples += N

        policy.train()
        ent_acc, vl_acc, pl_acc, kl_acc, count = 0.0, 0.0, 0.0, 0.0, 0
        for _ in range(args.epochs):
            perm = torch.randperm(N)
            for start in range(0, N, args.minibatch):
                idx = perm[start:start + args.minibatch]
                logp, ent, val = policy.evaluate(f_obs[idx], f_act[idx])
                ratio = torch.exp(logp - f_logp[idx])
                a = f_adv[idx]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                vl = F.mse_loss(val, f_ret[idx])
                loss = pl + 0.5 * vl - args.entropy * ent.mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                ent_acc += ent.mean().item(); vl_acc += vl.item(); pl_acc += pl.item()
                kl_acc += (f_logp[idx] - logp).mean().item(); count += 1

        if update % args.snapshot_every == 0:
            snapshots.append(copy.deepcopy(policy).eval())
            snapshots = snapshots[-10:]
            torch.save(policy.state_dict(), os.path.join(args.out, f"native_{update}.pt"))
        torch.save(policy.state_dict(), os.path.join(args.out, "policy_latest.pt"))

        sps = N / (time.time() - t_roll)
        if ep_buffer:
            eps = ep_buffer
            ep_buffer = []
            fin = [e for e in eps if e["winner"] >= 0]
            seats = {k: sum(e["types"].count(k) for e in eps) for k in (SEAT_EXTERNAL, SEAT_EXTERNAL2, SEAT_SOLO, SEAT_BAIL)}
            def rate(kinds):
                w = sum(1 for e in fin if e["winner_type"] in kinds)
                s = sum(seats[k] for k in kinds)
                return (w / s) * n if s else float("nan")
            def prog(kinds):
                vals = [p for e in eps for p, ty in zip(e["progress"], e["types"]) if ty in kinds]
                return float(np.mean(vals)) if vals else float("nan")
            row = [update, total_samples, round(time.time() - t_start),
                   round(rate((SEAT_EXTERNAL,)), 3), round(rate((SEAT_SOLO, SEAT_BAIL)), 3), round(rate((SEAT_EXTERNAL2,)), 3),
                   round(len(fin) / len(eps), 3), round(np.mean([e["length"] for e in eps]), 1),
                   round(prog((SEAT_EXTERNAL,)), 3), round(prog((SEAT_SOLO, SEAT_BAIL)), 3),
                   round(np.mean([e["avg_group"] for e in eps]), 2), round(np.mean([e["merges"] for e in eps]), 1),
                   round(np.mean([e["leaves"] for e in eps]), 1), round(np.mean([e["spills"] for e in eps]), 1),
                   round(np.mean([e["banks"] for e in eps]), 1),
                   round(ent_acc / count, 3), round(vl_acc / count, 3), round(pl_acc / count, 4), round(kl_acc / count, 4), round(sps)]
        else:
            row = [update, total_samples, round(time.time() - t_start)] + [""] * 12 + [
                round(ent_acc / count, 3), round(vl_acc / count, 3), round(pl_acc / count, 4), round(kl_acc / count, 4), round(sps)]
        logw.writerow(row)
        logf.flush()
        print(" ".join(str(x) for x in row), flush=True)
    logf.close()


if __name__ == "__main__":
    main()
