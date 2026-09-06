"""
PPO self-play trainer for Ungroup v2.

One policy network is shared by every seat. Rollouts come from W worker processes, each
stepping E environments. To keep the opponent distribution diverse, each seat in each episode is
assigned one of: the current policy, a frozen past snapshot of the policy, or a scripted bot.
Only current-policy seats contribute training samples.

Usage: python3 rl/train_ppo.py --updates 400 --workers 4 --envs 4
"""

import argparse
import copy
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ungroup import ACTION_NVEC, Config, UngroupEnv  # noqa: E402
from ungroup.bots import BOTS  # noqa: E402

SEAT_POLICY, SEAT_SNAPSHOT, SEAT_BOT = 0, 1, 2


# --------------------------------------------------------------------------- network


class Policy(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in ACTION_NVEC])
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs):
        h = self.body(obs)
        return [head(h) for head in self.heads], self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits, value = self.forward(obs)
        actions, logps = [], []
        for lg in logits:
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.probs.argmax(-1) if deterministic else dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))
        return torch.stack(actions, -1), torch.stack(logps, -1).sum(-1), value

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        logps, ents = [], []
        for k, lg in enumerate(logits):
            dist = torch.distributions.Categorical(logits=lg)
            logps.append(dist.log_prob(actions[:, k]))
            ents.append(dist.entropy())
        return torch.stack(logps, -1).sum(-1), torch.stack(ents, -1).sum(-1), value


# --------------------------------------------------------------------------- workers


def worker(conn, n_envs, seed, cfg_kwargs, bot_prob, snapshot_prob, decide_every):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    cfg = Config(**cfg_kwargs)
    envs = [UngroupEnv(cfg, seed=seed * 1000 + k, decide_every=decide_every) for k in range(n_envs)]
    n = cfg.n_players
    seat_types = np.zeros((n_envs, n), dtype=int)
    bots = [[None] * n for _ in range(n_envs)]
    stats = [dict(merges=0, leaves=0, spills=0, banks=0, group=0.0, steps=0) for _ in range(n_envs)]

    def assign(e):
        st = np.zeros(n, dtype=int)
        r = rng.random()
        if r < bot_prob:
            k = int(rng.integers(1, n))  # number of bot seats
            idx = rng.choice(n, k, replace=False)
            st[idx] = SEAT_BOT
        elif r < bot_prob + snapshot_prob:
            k = int(rng.integers(1, n))
            idx = rng.choice(n, k, replace=False)
            st[idx] = SEAT_SNAPSHOT
        seat_types[e] = st
        for i in range(n):
            if st[i] == SEAT_BOT:
                name = "solo" if rng.random() < 0.5 else "bail"
                bots[e][i] = BOTS[name]()
            else:
                bots[e][i] = None
        stats[e] = dict(merges=0, leaves=0, spills=0, banks=0, group=0.0, steps=0)

    obs = np.stack([env.reset() for env in envs])
    for e in range(n_envs):
        assign(e)
    conn.send((obs, seat_types.copy()))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        actions = msg  # (n_envs, n, 4)
        new_obs = np.zeros_like(obs)
        rewards = np.zeros((n_envs, n), dtype=np.float32)
        dones = np.zeros(n_envs, dtype=bool)
        episodes = []
        for e, env in enumerate(envs):
            acts = actions[e].copy()
            for i in range(n):
                if bots[e][i] is not None:
                    acts[i] = bots[e][i].act(env.game, i)
            o, r, d, info = env.step(acts)
            s = stats[e]
            s["steps"] += 1
            s["group"] += np.mean([env.game.body_of(i).n for i in range(n)])
            for ev in info["events"]:
                if ev["kind"] in ("merge", "leave", "spill", "bank"):
                    s[ev["kind"] + "s"] += 1
            rewards[e] = r
            dones[e] = d
            if d:
                g = env.game
                episodes.append(dict(
                    winner=g.winner, length=g.t,
                    winner_type=int(seat_types[e][g.winner]) if g.winner >= 0 else -1,
                    progress=[g.progress(i) for i in range(n)],
                    types=seat_types[e].tolist(),
                    avg_group=s["group"] / max(1, s["steps"]),
                    merges=s["merges"], leaves=s["leaves"], spills=s["spills"], banks=s["banks"],
                ))
                o = env.reset()
                assign(e)
            new_obs[e] = o
        obs = new_obs
        conn.send((obs, rewards, dones, seat_types.copy(), episodes))
    conn.close()


# --------------------------------------------------------------------------- training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--bot-prob", type=float, default=0.3)
    ap.add_argument("--snapshot-prob", type=float, default=0.3)
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--decide-every", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg_kwargs = dict(n_players=args.players)
    probe = UngroupEnv(Config(**cfg_kwargs))
    obs_dim, n = probe.obs_dim, probe.n_agents

    policy = Policy(obs_dim)
    if args.resume:
        policy.load_state_dict(torch.load(args.resume))
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    snapshots = [copy.deepcopy(policy)]

    ctx = mp.get_context("fork")
    conns, procs = [], []
    for w in range(args.workers):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(b, args.envs, args.seed * 100 + w + 1, cfg_kwargs,
                                             args.bot_prob, args.snapshot_prob, args.decide_every))
        p.daemon = True
        p.start()
        conns.append(a)
        procs.append(p)
    E = args.workers * args.envs
    obs = np.zeros((E, n, obs_dim), dtype=np.float32)
    seat_types = np.zeros((E, n), dtype=int)
    for w, c in enumerate(conns):
        o, st = c.recv()
        obs[w * args.envs:(w + 1) * args.envs] = o
        seat_types[w * args.envs:(w + 1) * args.envs] = st
    # Each snapshot seat picks a snapshot index at episode start; store per env-seat.
    snap_idx = np.zeros((E, n), dtype=int)

    log_path = os.path.join(args.out, "log.csv")
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if os.path.getsize(log_path) == 0:
        logw.writerow(["update", "samples", "time", "reward", "policy_winrate", "bot_winrate", "snapshot_winrate",
                       "finish_rate", "length", "avg_group", "merges", "leaves", "spills", "banks",
                       "entropy", "value_loss", "policy_loss", "kl"])
    total_samples = 0
    t_start = time.time()
    ep_buffer = []

    for update in range(1, args.updates + 1):
        T = args.rollout
        b_obs = np.zeros((T, E, n, obs_dim), dtype=np.float32)
        b_act = np.zeros((T, E, n, len(ACTION_NVEC)), dtype=np.int64)
        b_logp = np.zeros((T, E, n), dtype=np.float32)
        b_val = np.zeros((T + 1, E, n), dtype=np.float32)
        b_rew = np.zeros((T, E, n), dtype=np.float32)
        b_done = np.zeros((T + 1, E), dtype=np.float32)
        b_mask = np.zeros((T, E, n), dtype=np.float32)
        policy.eval()
        for t in range(T):
            obs_t = torch.from_numpy(obs.reshape(E * n, obs_dim))
            act, logp, val = policy.act(obs_t)
            act = act.numpy().reshape(E, n, -1)
            logp = logp.numpy().reshape(E, n)
            val = val.numpy().reshape(E, n)
            # Snapshot seats act with their frozen network.
            snap_seats = seat_types == SEAT_SNAPSHOT
            if snap_seats.any() and len(snapshots) > 0:
                for k in np.unique(snap_idx[snap_seats]):
                    sel = snap_seats & (snap_idx == k)
                    o = torch.from_numpy(obs[sel])
                    a, _, _ = snapshots[k % len(snapshots)].act(o)
                    act[sel] = a.numpy()
            b_obs[t] = obs
            b_act[t] = act
            b_logp[t] = logp
            b_val[t] = val
            b_mask[t] = (seat_types == SEAT_POLICY)
            for w, c in enumerate(conns):
                c.send(act[w * args.envs:(w + 1) * args.envs])
            for w, c in enumerate(conns):
                o, r, d, st, eps = c.recv()
                sl = slice(w * args.envs, (w + 1) * args.envs)
                obs[sl] = o
                b_rew[t, sl] = r
                b_done[t + 1, sl] = d
                new_ep = d
                if new_ep.any():
                    for e_local in np.where(new_ep)[0]:
                        snap_idx[w * args.envs + e_local] = np.random.randint(0, max(1, len(snapshots)))
                seat_types[sl] = st
                ep_buffer.extend(eps)
        with torch.no_grad():
            _, last_val = policy.forward(torch.from_numpy(obs.reshape(E * n, obs_dim)))
        b_val[T] = last_val.numpy().reshape(E, n)

        # GAE
        adv = np.zeros((T, E, n), dtype=np.float32)
        last = np.zeros((E, n), dtype=np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - b_done[t + 1][:, None]
            delta = b_rew[t] + args.gamma * b_val[t + 1] * nonterm - b_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + b_val[:T]

        mask = b_mask.reshape(-1) > 0
        f_obs = torch.from_numpy(b_obs.reshape(-1, obs_dim)[mask])
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
                ent_acc += ent.mean().item()
                vl_acc += vl.item()
                pl_acc += pl.item()
                kl_acc += (f_logp[idx] - logp).mean().item()
                count += 1

        if update % args.snapshot_every == 0:
            snapshots.append(copy.deepcopy(policy))
            snapshots = snapshots[-8:]
            torch.save(policy.state_dict(), os.path.join(args.out, f"policy_{update}.pt"))
        torch.save(policy.state_dict(), os.path.join(args.out, "policy_latest.pt"))

        # Episode stats
        if ep_buffer:
            eps = ep_buffer
            ep_buffer = []
            fin = [e for e in eps if e["winner"] >= 0]
            pw = sum(1 for e in fin if e["winner_type"] == SEAT_POLICY)
            bw = sum(1 for e in fin if e["winner_type"] == SEAT_BOT)
            sw = sum(1 for e in fin if e["winner_type"] == SEAT_SNAPSHOT)
            # Win rate per seat type normalised by how many seats of that type played.
            seats = {k: sum(e["types"].count(k) for e in eps) for k in (0, 1, 2)}
            def rate(w, k):
                return (w / max(1, seats[k])) * n if seats[k] else float("nan")
            row = [update, total_samples, round(time.time() - t_start), round(float(b_rew[b_mask > 0].sum() / max(1, len(eps))), 3),
                   round(rate(pw, 0), 3), round(rate(bw, 2), 3), round(rate(sw, 1), 3),
                   round(len(fin) / len(eps), 3), round(np.mean([e["length"] for e in eps]), 1),
                   round(np.mean([e["avg_group"] for e in eps]), 2), round(np.mean([e["merges"] for e in eps]), 1),
                   round(np.mean([e["leaves"] for e in eps]), 1), round(np.mean([e["spills"] for e in eps]), 1),
                   round(np.mean([e["banks"] for e in eps]), 1),
                   round(ent_acc / count, 3), round(vl_acc / count, 3), round(pl_acc / count, 4), round(kl_acc / count, 4)]
        else:
            row = [update, total_samples, round(time.time() - t_start), "", "", "", "", "", "", "", "", "", "", "",
                   round(ent_acc / count, 3), round(vl_acc / count, 3), round(pl_acc / count, 4), round(kl_acc / count, 4)]
        logw.writerow(row)
        logf.flush()
        print(" ".join(str(x) for x in row), flush=True)

    for c in conns:
        c.send(None)
    logf.close()


if __name__ == "__main__":
    main()
