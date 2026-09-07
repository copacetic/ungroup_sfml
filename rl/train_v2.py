"""
Ungroup v2 trainer: PPO self-play on the canonical C++ core with the changes recommended by the
2026-09 architecture and plan reviews.

- Entity encoder policy: shared per-other-player MLP with masked max/mean pooling (also pooled
  over co-members), mines as a flat block, pooled pickups, LayerNorm trunk, orthogonal init.
- Separate privileged critic: sees the seat's observation plus every seat's needs, banked,
  pool share and seat type (training only). Value targets are normalised; Huber loss.
- Per-head entropy bonus normalised by head cardinality, per-head entropy logging, and
  conditional marginals (P(joinable | solo), P(leave | grouped)) logged every update.
- Action masks: leave is masked when solo, intent changes are masked while grouped.
- League: permanent scripted bots (solo, bail, loyal), the warm-start clone, and snapshots kept
  for the whole run; opponents are sampled per seat. Held-out bots (kidnap, rammer) are never
  used in training.
- DAgger warm start from the bail bot with softened alliance-head targets, then a critic-only
  warm-up before joint PPO.
- Optional domain randomisation of rule constants (exposed in the observation).
- Checkpoints carry the config, observation layout, and sample count.

Usage:
  python3 rl/train_v2.py --warmup-iters 4 --updates 800 --out rl/checkpoints/v2
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ungroup.native import (HELDOUT_BOTS, MAX_SLOTS, N_DR, SEAT_BAIL, SEAT_EXTERNAL, SEAT_EXTERNAL2, SEAT_NAMES,  # noqa: E402
                            TRAINING_BOTS, TYPES, Config, NativeBatch)

ACTION_NVEC = (9, 2, 2, 5)
OWN_DIM = 33 + N_DR
OTHER_DIM = 22
MINE_DIM = 8
PICK_DIM = 7
N_PICKUPS = 4
IDX_GROUP_N = 17      # own block: n / max_group
IDX_OTHER_PRESENT = 16
IDX_OTHER_SAME = 10
IDX_PICK_PRESENT = 6
ARCH = "v2-entity"


def ortho(layer, gain=math.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Policy(nn.Module):
    def __init__(self, obs_dim, n_mines=8, hidden=256, ent=64, max_group=6):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_mines = n_mines
        self.max_group = max_group
        self.own = nn.Sequential(ortho(nn.Linear(OWN_DIM, 128)), nn.ReLU())
        self.oth = nn.Sequential(ortho(nn.Linear(OTHER_DIM, ent)), nn.ReLU(), ortho(nn.Linear(ent, ent)), nn.ReLU())
        self.mine = nn.Sequential(ortho(nn.Linear(n_mines * MINE_DIM, 64)), nn.ReLU())
        self.pick = nn.Sequential(ortho(nn.Linear(PICK_DIM, 32)), nn.ReLU())
        tin = 128 + 3 * ent + 64 + 32
        self.trunk = nn.Sequential(ortho(nn.Linear(tin, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
                                   ortho(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.ReLU())
        self.heads = nn.ModuleList([ortho(nn.Linear(hidden, n), gain=0.01) for n in ACTION_NVEC])
        assert obs_dim == OWN_DIM + MAX_SLOTS * OTHER_DIM + n_mines * MINE_DIM + N_PICKUPS * PICK_DIM, obs_dim

    def split(self, obs):
        c = 0
        own = obs[:, c:c + OWN_DIM]; c += OWN_DIM
        oth = obs[:, c:c + MAX_SLOTS * OTHER_DIM].reshape(-1, MAX_SLOTS, OTHER_DIM); c += MAX_SLOTS * OTHER_DIM
        mines = obs[:, c:c + self.n_mines * MINE_DIM]; c += self.n_mines * MINE_DIM
        picks = obs[:, c:c + N_PICKUPS * PICK_DIM].reshape(-1, N_PICKUPS, PICK_DIM)
        return own, oth, mines, picks

    def features(self, obs):
        own, oth, mines, picks = self.split(obs)
        h_own = self.own(own)
        e = self.oth(oth)                                   # (B, S, ent)
        present = oth[:, :, IDX_OTHER_PRESENT:IDX_OTHER_PRESENT + 1]
        same = oth[:, :, IDX_OTHER_SAME:IDX_OTHER_SAME + 1] * present
        neg = torch.finfo(e.dtype).min
        e_max = torch.where(present > 0, e, torch.full_like(e, neg)).max(1).values
        e_max = torch.where(present.sum(1) > 0, e_max, torch.zeros_like(e_max))
        e_mean = (e * present).sum(1) / present.sum(1).clamp(min=1)
        c_max = torch.where(same > 0, e, torch.full_like(e, neg)).max(1).values
        c_max = torch.where(same.sum(1) > 0, c_max, torch.zeros_like(c_max))
        h_mine = self.mine(mines)
        p = self.pick(picks)
        pp = picks[:, :, IDX_PICK_PRESENT:IDX_PICK_PRESENT + 1]
        p_max = torch.where(pp > 0, p, torch.full_like(p, neg)).max(1).values
        p_max = torch.where(pp.sum(1) > 0, p_max, torch.zeros_like(p_max))
        return self.trunk(torch.cat([h_own, e_max, e_mean, c_max, h_mine, p_max], 1))

    def masks(self, obs):
        """Returns per-head additive logit masks based on the observation."""
        n = obs[:, IDX_GROUP_N] * self.max_group
        solo = n < 1.5
        m_leave = torch.zeros(obs.shape[0], 2, device=obs.device)
        m_leave[solo, 1] = float("-inf")            # cannot leave when solo
        m_intent = torch.zeros(obs.shape[0], 5, device=obs.device)
        m_intent[~solo, 1:] = float("-inf")         # intent locked while grouped
        return [None, None, m_leave, m_intent]

    def forward(self, obs):
        h = self.features(obs)
        masks = self.masks(obs)
        logits = []
        for k, head in enumerate(self.heads):
            lg = head(h)
            if masks[k] is not None:
                lg = lg + masks[k]
            logits.append(lg)
        return logits

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        logits = self.forward(obs)
        actions, logps = [], []
        for lg in logits:
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.probs.argmax(-1) if deterministic else dist.sample()
            actions.append(a)
            logps.append(dist.log_prob(a))
        return torch.stack(actions, -1), torch.stack(logps, -1).sum(-1)

    def evaluate(self, obs, actions):
        logits = self.forward(obs)
        logps, ents = [], []
        for k, lg in enumerate(logits):
            dist = torch.distributions.Categorical(logits=lg)
            logps.append(dist.log_prob(actions[:, k]))
            ents.append(dist.entropy())
        return torch.stack(logps, -1), torch.stack(ents, -1)


class Critic(nn.Module):
    def __init__(self, obs_dim, priv_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(ortho(nn.Linear(obs_dim + priv_dim, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
                                 ortho(nn.Linear(hidden, hidden)), nn.LayerNorm(hidden), nn.ReLU(),
                                 ortho(nn.Linear(hidden, 1), gain=1.0))

    def forward(self, obs, priv):
        return self.net(torch.cat([obs, priv], 1)).squeeze(-1)


class RunningNorm:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        bm, bv, bc = x.mean(), x.var(), x.size
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return math.sqrt(self.var) + 1e-6


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(path, policy, critic, cfg, samples, extra=None):
    torch.save({"arch": ARCH, "policy": policy.state_dict(), "critic": critic.state_dict() if critic else None,
                "config": asdict(cfg), "obs_dim": policy.obs_dim, "n_mines": policy.n_mines, "max_group": policy.max_group,
                "samples": samples, "git": git_sha(), "extra": extra or {}}, path)


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck.get("arch") == ARCH, f"{path} is not a {ARCH} checkpoint"
    cfg = Config(**ck["config"])
    policy = Policy(ck["obs_dim"], n_mines=ck["n_mines"], max_group=ck["max_group"])
    policy.load_state_dict(ck["policy"])
    policy.eval()
    return policy, cfg, ck


# --------------------------------------------------------------------------- helpers


def head_marginals(obs, act):
    """P(joinable=1 | solo), P(leave=1 | grouped), P(intent change | solo) from a rollout buffer."""
    n = obs[..., IDX_GROUP_N] * 6.0
    solo = n < 1.5
    grouped = ~solo
    out = {}
    out["p_join_solo"] = float(act[..., 1][solo].mean()) if solo.any() else float("nan")
    out["p_join_grouped"] = float(act[..., 1][grouped].mean()) if grouped.any() else float("nan")
    out["p_leave_grouped"] = float((act[..., 2][grouped] == 1).mean()) if grouped.any() else float("nan")
    out["p_intent_change"] = float((act[..., 3][solo] > 0).mean()) if solo.any() else float("nan")
    return out


def dr_ranges(cfg):
    lo = cfg.replace(mine_rate=cfg.mine_rate * 0.8, mine_regen=cfg.mine_regen * 0.8, leave_time=1.5, need_primary=14)
    hi = cfg.replace(mine_rate=cfg.mine_rate * 1.25, mine_regen=cfg.mine_regen * 1.25, leave_time=3.0, need_primary=22)
    return lo, hi


# --------------------------------------------------------------------------- warm start


def masked_smooth_ce(logits, target, smooth):
    """Cross-entropy with label smoothing spread only over classes whose logit is finite."""
    allowed = torch.isfinite(logits)
    logp = torch.log_softmax(logits, -1)
    logp = torch.where(allowed, logp, torch.zeros_like(logp))
    n_allowed = allowed.sum(-1, keepdim=True).clamp(min=1).float()
    tgt = torch.zeros_like(logp)
    tgt.scatter_(1, target[:, None], 1.0)
    tgt = tgt * (1 - smooth) + allowed.float() * (smooth / n_allowed)
    return -(tgt * logp).sum(-1).mean()


def warmup(policy, batch, args, log):
    """DAgger from the bail bot: labels come from ugb_bot_actions for every external seat."""
    E, n, D = batch.E, batch.n, batch.obs_dim
    for e in range(E):
        seats = ["policy"] * n
        for i in range(n):
            if np.random.random() < 0.3:
                seats[i] = TRAINING_BOTS[np.random.randint(len(TRAINING_BOTS))]
        batch.set_seats(e, seats)
    obs = batch.observe()
    X, Y = [], []
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-5)
    for it in range(args.warmup_iters):
        steps = args.warmup_steps
        for t in range(steps):
            labels = np.stack([batch.bot_actions(e, "bail") for e in range(E)])  # (E, n, 4)
            if it == 0:
                act = labels.copy()
            else:
                a, _ = policy.act(torch.from_numpy(obs.reshape(E * n, D)))
                act = a.numpy().reshape(E, n, 4)
            # Labels must respect the policy's action masks: no intent change while grouped, no leave when solo.
            flat_obs = obs.reshape(E * n, D)
            lab = labels.reshape(E * n, 4).copy()
            solo = flat_obs[:, IDX_GROUP_N] * batch.cfg.max_group < 1.5
            lab[~solo, 3] = 0
            lab[solo, 2] = 0
            X.append(flat_obs.copy())
            Y.append(lab)
            obs, _, _, _ = batch.step(act)
        Xt = torch.from_numpy(np.concatenate(X))
        Yt = torch.from_numpy(np.concatenate(Y)).long()
        N = Xt.shape[0]
        policy.train()
        epochs = args.warmup_epochs if it == 0 else max(4, args.warmup_epochs // 2)
        for ep in range(epochs):
            perm = torch.randperm(N)
            tot, acc = 0.0, np.zeros(4)
            for s in range(0, N, 4096):
                idx = perm[s:s + 4096]
                logits = policy(Xt[idx])
                loss = 0.0
                for k, lg in enumerate(logits):
                    smooth = 0.0 if k == 0 else args.warmup_smooth
                    loss = loss + masked_smooth_ce(lg, Yt[idx, k], smooth)
                    acc[k] += (lg.argmax(-1) == Yt[idx, k]).float().sum().item()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                tot += loss.item() * len(idx)
        policy.eval()
        log(f"warmup iter {it}: {N} samples, loss {tot / N:.3f}, acc {np.round(acc / N, 3).tolist()}")


# --------------------------------------------------------------------------- training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=800)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--rollout", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.998)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--target-kl", type=float, default=0.02)
    ap.add_argument("--entropy", type=str, default="0.003,0.02,0.02,0.01")
    ap.add_argument("--p-policy", type=float, default=0.5, help="per-seat probability of the current policy")
    ap.add_argument("--p-snapshot", type=float, default=0.25)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--dr", action="store_true", help="randomise rule constants per game")
    ap.add_argument("--warmup-iters", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=400)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--warmup-smooth", type=float, default=0.25)
    ap.add_argument("--critic-warmup", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "v2"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", help="override a Config field, e.g. --set mine_rate=0.12")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg = Config(n_players=args.players)
    for kv in args.set or []:
        k, v = kv.split("=")
        f = Config.__dataclass_fields__[k]
        cfg = cfg.replace(**{k: int(v) if f.type is int else float(v)})
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=1)
    logf = open(os.path.join(args.out, "train.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    batch = NativeBatch(args.envs, cfg, seed=args.seed * 100000 + 1, decide_every=6)
    if args.dr:
        lo, hi = dr_ranges(cfg)
        batch.set_cfg_range(lo, hi)
        for e in range(args.envs):
            batch.reset(e)
    E, n, D, P = batch.E, batch.n, batch.obs_dim, batch.priv_dim
    ent_coef = [float(x) for x in args.entropy.split(",")]
    ent_norm = [math.log(k) for k in ACTION_NVEC]

    critic = Critic(D, P)
    samples_done = 0
    if args.resume:
        policy, _, ck = load_checkpoint(args.resume)
        policy.train()
        if ck.get("critic"):
            critic.load_state_dict(ck["critic"])
        samples_done = ck.get("samples", 0)
        log(f"resumed {args.resume} ({samples_done} samples)")
    else:
        policy = Policy(D, n_mines=cfg.n_mines, max_group=cfg.max_group)
        if args.warmup_iters > 0:
            warmup(policy, batch, args, log)
            save_checkpoint(os.path.join(args.out, "warmup.pt"), policy, None, cfg, 0)
    league = [("warmup", copy.deepcopy(policy).eval())]
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    copt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
    vnorm = RunningNorm()

    # Seat assignment: per seat, current policy / league snapshot / training bot.
    seat_types = np.zeros((E, n), dtype=int)
    seat_snap = np.full((E, n), -1, dtype=int)
    rollout_league = [0]

    def assign(e):
        st = np.zeros(n, dtype=int)
        sn = np.full(n, -1, dtype=int)
        seats = []
        for i in range(n):
            r = rng.random()
            if r < args.p_policy:
                st[i] = SEAT_EXTERNAL; seats.append("policy")
            elif r < args.p_policy + args.p_snapshot:
                st[i] = SEAT_EXTERNAL2; sn[i] = rollout_league[rng.integers(len(rollout_league))]; seats.append("snapshot")
            else:
                name = TRAINING_BOTS[rng.integers(len(TRAINING_BOTS))]
                st[i] = SEAT_NAMES[name]; seats.append(name)
        if not (st == SEAT_EXTERNAL).any():
            i = rng.integers(n); st[i] = SEAT_EXTERNAL; sn[i] = -1; seats[i] = "policy"
        seat_types[e] = st
        seat_snap[e] = sn
        batch.set_seats(e, seats)

    for e in range(E):
        assign(e)
    obs = batch.observe()
    priv = batch.observe_priv()

    log_path = os.path.join(args.out, "log.csv")
    new_log = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    csvf = open(log_path, "a", newline="")
    csvw = csv.writer(csvf)
    cols = ["update", "samples", "time", "sps", "policy_prog", "bot_prog", "snap_prog", "policy_win", "bot_win", "snap_win",
            "early_finish", "length", "avg_group", "alliances", "alliance_dur", "alliances_long", "group_banks", "remerge_fast",
            "merges", "leaves", "spills", "banks", "ent_move", "ent_join", "ent_leave", "ent_intent",
            "p_join_solo", "p_join_grouped", "p_leave_grouped", "p_intent_change", "ev", "vloss", "ploss", "kl", "epochs_run"]
    if new_log:
        csvw.writerow(cols)
    t_start = time.time()
    ep_buffer = []
    total_samples = samples_done

    def run_update(update, critic_only=False):
        nonlocal obs, priv, total_samples, ep_buffer
        T = args.rollout
        t_roll = time.time()
        b_obs = np.zeros((T, E, n, D), dtype=np.float32)
        b_priv = np.zeros((T, E, n, P), dtype=np.float32)
        b_act = np.zeros((T, E, n, 4), dtype=np.int64)
        b_logp = np.zeros((T, E, n), dtype=np.float32)
        b_val = np.zeros((T + 1, E, n), dtype=np.float32)
        b_rew = np.zeros((T, E, n), dtype=np.float32)
        b_done = np.zeros((T + 1, E), dtype=np.float32)
        b_mask = np.zeros((T, E, n), dtype=np.float32)
        policy.eval(); critic.eval()
        # choose the snapshots used in this rollout (at most 3 distinct networks)
        for t in range(T):
            obs_t = torch.from_numpy(obs.reshape(E * n, D))
            with torch.no_grad():
                act, logp = policy.act(obs_t)
                val = critic(obs_t, torch.from_numpy(priv.reshape(E * n, P))).numpy() * vnorm.std + vnorm.mean
            act = act.numpy().reshape(E, n, 4)
            logp = logp.numpy().reshape(E, n)
            snap_seats = seat_types == SEAT_EXTERNAL2
            if snap_seats.any():
                for k in np.unique(seat_snap[snap_seats]):
                    sel = snap_seats & (seat_snap == k)
                    a, _ = league[k][1].act(torch.from_numpy(obs[sel]))
                    act[sel] = a.numpy()
            b_obs[t] = obs; b_priv[t] = priv; b_act[t] = act; b_logp[t] = logp
            b_val[t] = val.reshape(E, n)
            b_mask[t] = (seat_types == SEAT_EXTERNAL)
            obs, rew, done, eps, priv = batch.step(act, want_priv=True)
            b_rew[t] = rew
            b_done[t + 1] = done
            for ep in eps:
                e = ep["env"]
                ep["types"] = seat_types[e].tolist()
                ep["winner_type"] = int(seat_types[e][ep["winner"]]) if ep["winner"] >= 0 else -1
                ep_buffer.append(ep)
                assign(e)
        with torch.no_grad():
            last_val = critic(torch.from_numpy(obs.reshape(E * n, D)), torch.from_numpy(priv.reshape(E * n, P))).numpy() * vnorm.std + vnorm.mean
        b_val[T] = last_val.reshape(E, n)
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
        f_priv = torch.from_numpy(b_priv.reshape(-1, P)[mask])
        f_act = torch.from_numpy(b_act.reshape(-1, 4)[mask])
        f_logp = torch.from_numpy(b_logp.reshape(-1)[mask])
        f_adv = torch.from_numpy(adv.reshape(-1)[mask])
        f_ret = torch.from_numpy(ret.reshape(-1)[mask])
        ev = 1.0 - float(np.var(ret.reshape(-1)[mask] - b_val[:T].reshape(-1)[mask]) / (np.var(ret.reshape(-1)[mask]) + 1e-8))
        vnorm.update(f_ret.numpy())
        f_ret_n = (f_ret - vnorm.mean) / vnorm.std
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        N = f_obs.shape[0]
        total_samples += N

        policy.train(); critic.train()
        ent_acc = np.zeros(4); vl_acc = pl_acc = kl_acc = 0.0; count = 0; epochs_run = 0
        stop = False
        for ep in range(args.epochs):
            perm = torch.randperm(N)
            for s in range(0, N - args.minibatch // 2, args.minibatch):
                idx = perm[s:s + args.minibatch]
                vpred = critic(f_obs[idx], f_priv[idx])
                vl = F.smooth_l1_loss(vpred, f_ret_n[idx])
                copt.zero_grad(); vl.backward(); nn.utils.clip_grad_norm_(critic.parameters(), 1.0); copt.step()
                vl_acc += vl.item()
                if not critic_only:
                    logps, ents = policy.evaluate(f_obs[idx], f_act[idx])
                    logp = logps.sum(-1)
                    ratio = torch.exp(logp - f_logp[idx])
                    a = f_adv[idx]
                    pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    ent_term = sum(ent_coef[k] * ents[:, k].mean() / ent_norm[k] for k in range(4))
                    loss = pl - ent_term
                    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    ent_acc += ents.mean(0).detach().numpy(); pl_acc += pl.item(); kl_acc += kl
                    if kl > args.target_kl * 1.5:
                        stop = True
                count += 1
                if stop:
                    break
            epochs_run += 1
            if stop:
                break
        sps = N / (time.time() - t_roll)
        margs = head_marginals(b_obs[b_mask > 0], b_act[b_mask > 0])

        if ep_buffer:
            eps = ep_buffer; ep_buffer = []
            def prog(kinds):
                v = [p for e in eps for p, ty in zip(e["progress"], e["types"]) if ty in kinds]
                return float(np.mean(v)) if v else float("nan")
            def winr(kinds):
                w = sum(1 for e in eps if e["winner_type"] in kinds)
                s = sum(e["types"].count(k) for e in eps for k in kinds)
                return w / s * n if s else float("nan")
            bots = tuple(SEAT_NAMES[b] for b in TRAINING_BOTS)
            row = [update, total_samples, round(time.time() - t_start), round(sps),
                   round(prog((SEAT_EXTERNAL,)), 3), round(prog(bots), 3), round(prog((SEAT_EXTERNAL2,)), 3),
                   round(winr((SEAT_EXTERNAL,)), 2), round(winr(bots), 2), round(winr((SEAT_EXTERNAL2,)), 2),
                   round(float(np.mean([not e["timeout_win"] for e in eps])), 2), round(float(np.mean([e["length"] for e in eps])), 1),
                   round(float(np.mean([e["avg_group"] for e in eps])), 2), round(float(np.mean([e["alliances"] for e in eps])), 1),
                   round(float(np.mean([e["alliance_dur"] for e in eps])), 1), round(float(np.mean([e["alliances_long"] for e in eps])), 1),
                   round(float(np.mean([e["group_banks"] for e in eps])), 1), round(float(np.mean([e["remerge_fast"] for e in eps])), 1),
                   round(float(np.mean([e["merges"] for e in eps])), 1), round(float(np.mean([e["leaves"] for e in eps])), 1),
                   round(float(np.mean([e["spills"] for e in eps])), 1), round(float(np.mean([e["banks"] for e in eps])), 1)]
        else:
            row = [update, total_samples, round(time.time() - t_start), round(sps)] + [""] * 18
        c = max(1, count)
        row += [round(float(x), 3) for x in (ent_acc / c)]
        row += [round(margs["p_join_solo"], 3), round(margs["p_join_grouped"], 3), round(margs["p_leave_grouped"], 3), round(margs["p_intent_change"], 3)]
        row += [round(ev, 3), round(vl_acc / c, 4), round(pl_acc / c, 4), round(kl_acc / c, 4), epochs_run]
        csvw.writerow(row); csvf.flush()
        log(" ".join(str(x) for x in row))

    # Critic-only warm-up so PPO does not start with a random baseline.
    for u in range(args.critic_warmup):
        run_update(-args.critic_warmup + u, critic_only=True)

    for update in range(1, args.updates + 1):
        run_update(update)
        if update % args.snapshot_every == 0:
            snap = copy.deepcopy(policy).eval()
            league.append((f"u{update}", snap))
            rollout_league[:] = list(range(len(league)))[-6:] + [0]
            save_checkpoint(os.path.join(args.out, f"v2_{update}.pt"), policy, critic, cfg, total_samples)
        save_checkpoint(os.path.join(args.out, "latest.pt"), policy, critic, cfg, total_samples)
    logf.close(); csvf.close()


if __name__ == "__main__":
    main()
