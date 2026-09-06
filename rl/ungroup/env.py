"""
Multi-agent environment wrapper over the rules core.

All agents share one observation and action format so a single policy can be used for every
seat (parameter sharing / self-play). Actions are MultiDiscrete [9, 2, 2, 5]:
move (none + 8 directions), joinable (off/on), leave (no/yes), intent (keep or set type).
"""

import math

import numpy as np


def _norm(v):
    return math.hypot(float(v[0]), float(v[1]))

from .core import TYPES, Config, Game

ACTION_NVEC = (9, 2, 2, 5)
N_PICKUPS_OBS = 4


class UngroupEnv:
    def __init__(self, cfg: Config = None, seed: int = 0, decide_every: int = 2, record: bool = False,
                 carried_shaping: float = 2.0, win_bonus: float = 10.0, lose_penalty: float = 2.0):
        self.cfg = cfg or Config()
        self.decide_every = decide_every
        self.record = record
        self.carried_shaping = carried_shaping
        self.win_bonus = win_bonus
        self.lose_penalty = lose_penalty
        self.game = Game(self.cfg, seed=seed, record=record)
        self.n_agents = self.cfg.n_players
        self.obs_dim = self._obs_dim()
        self._prev_potential = np.zeros(self.n_agents)
        self._seed = seed

    # ------------------------------------------------------------------ api

    def reset(self, seed=None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed += 1
        self.game = Game(self.cfg, seed=self._seed, record=self.record)
        self._prev_potential = self._potentials()
        return self.observe()

    def step(self, actions):
        """actions: array (n_agents, 4) of ints. Returns obs, rewards, done, info."""
        g = self.game
        g.apply_actions(np.asarray(actions))
        events = []
        for _ in range(self.decide_every):
            g.tick()
            events.extend(g.events)
            if g.done:
                break
        pot = self._potentials()
        rewards = pot - self._prev_potential
        self._prev_potential = pot
        if g.done and g.winner >= 0:
            rewards = rewards - self.lose_penalty
            rewards[g.winner] += self.lose_penalty + self.win_bonus
        info = {"events": events, "winner": g.winner, "t": g.t}
        return self.observe(), rewards, g.done, info

    # ---------------------------------------------------------------- reward

    def _potentials(self):
        g = self.game
        pots = np.zeros(self.n_agents)
        for i, p in enumerate(g.players):
            b = g.body_of(i)
            banked_prog = np.mean(np.minimum(p.banked / p.need, 1.0))
            carried_prog = np.mean(np.minimum((p.banked + b.pool / b.n) / p.need, 1.0))
            pots[i] = 10.0 * banked_prog + self.carried_shaping * (carried_prog - banked_prog)
        return pots

    # ----------------------------------------------------------- observation

    def _obs_dim(self):
        k = self.cfg.n_players
        m = self.cfg.n_mines
        return 29 + (k - 1) * 16 + m * 8 + N_PICKUPS_OBS * 6

    def observe_slow(self):
        g = self.game
        cfg = self.cfg
        k = self.n_agents
        obs = np.zeros((k, self.obs_dim), dtype=np.float32)
        tfrac = g.t / cfg.time_limit
        body_of = [g.body_of(i) for i in range(k)]
        pads = [g.pad_pos(i) for i in range(k)]
        progress = [g.progress(i) for i in range(k)]
        for i in range(k):
            p = g.players[i]
            b = body_of[i]
            pos = b.pos
            f = []
            f.extend(pos)
            f.extend(b.vel)
            f.extend(p.need / cfg.need_primary)
            f.extend(np.minimum(p.banked / p.need, 1.5))
            oh = np.zeros(TYPES)
            oh[p.intent] = 1
            f.extend(oh)
            f.append(1.0 if p.joinable else 0.0)
            f.append(b.n / cfg.max_group)
            f.extend(b.pool / 20.0)
            f.append(p.leave_timer / cfg.leave_time if p.leave_timer >= 0 else 0.0)
            f.append(g.arena_radius)
            f.append(tfrac)
            f.extend(pads[i] - pos)
            f.append(b.pool.sum() / 40.0)
            f.append(progress[i])
            assert len(f) == 29
            # Others, sorted by distance.
            others = [j for j in range(k) if j != i]
            others.sort(key=lambda j: _norm(body_of[j].pos - pos))
            for j in others:
                q = g.players[j]
                bj = body_of[j]
                f.extend(bj.pos - pos)
                f.extend(bj.vel)
                f.append(bj.n / cfg.max_group)
                f.append(1.0 if q.joinable else 0.0)
                oh = np.zeros(TYPES)
                oh[q.intent] = 1
                f.extend(oh)
                f.append(1.0 if bj is b else 0.0)
                f.append(1.0 if q.leave_timer >= 0 else 0.0)
                f.append(progress[j])
                f.extend(pads[j] - pos)
                f.append(bj.pool.sum() / 40.0)
            for m in range(cfg.n_mines):
                f.extend(g.mine_pos[m] - pos)
                oh = np.zeros(TYPES)
                oh[g.mine_type[m]] = 1
                f.extend(oh)
                f.append(g.mine_stock[m] / cfg.mine_cap)
                f.append(1.0 if g.mine_alive[m] else 0.0)
            npk = len(g.pick_ttl)
            if npk:
                d = np.linalg.norm(g.pick_pos - pos, axis=1)
                order = np.argsort(d)[:N_PICKUPS_OBS]
            else:
                order = []
            for idx in order:
                f.extend(g.pick_pos[idx] - pos)
                oh = np.zeros(TYPES)
                oh[g.pick_type[idx]] = 1
                f.extend(oh)
            for _ in range(N_PICKUPS_OBS - len(order)):
                f.extend([0.0] * 6)
            obs[i] = np.asarray(f, dtype=np.float32)
        return obs

    def observe(self):
        """Vectorized observation builder (same layout as observe_slow)."""
        g = self.game
        cfg = self.cfg
        k = self.n_agents
        tfrac = g.t / cfg.time_limit
        bodies = [g.body_of(i) for i in range(k)]
        body_id = np.array([id(b) for b in bodies])
        P = np.array([b.pos for b in bodies])              # (k,2)
        V = np.array([b.vel for b in bodies])              # (k,2)
        NB = np.array([b.n for b in bodies], dtype=float)  # (k,)
        POOL = np.array([b.pool for b in bodies])          # (k,4)
        PT = POOL.sum(1)                                   # (k,)
        NEED = np.array([p.need for p in g.players])
        BANK = np.array([p.banked for p in g.players])
        INT = np.zeros((k, TYPES)); INT[np.arange(k), [p.intent for p in g.players]] = 1
        JOIN = np.array([1.0 if p.joinable else 0.0 for p in g.players])
        LT = np.array([p.leave_timer for p in g.players])
        LEAVING = (LT >= 0).astype(float)
        PADS = np.array([g.pad_pos(i) for i in range(k)])
        PROG = np.mean(np.minimum(BANK / NEED, 1.0), axis=1)

        own = np.concatenate([
            P, V, NEED / cfg.need_primary, np.minimum(BANK / NEED, 1.5), INT,
            JOIN[:, None], (NB / cfg.max_group)[:, None], POOL / 20.0,
            np.where(LT >= 0, LT / cfg.leave_time, 0.0)[:, None],
            np.full((k, 1), g.arena_radius), np.full((k, 1), tfrac),
            PADS - P, (PT / 40.0)[:, None], PROG[:, None],
        ], axis=1)

        # Others sorted by distance (excluding self).
        diff = P[None, :, :] - P[:, None, :]               # (k,k,2) diff[i,j] = P[j]-P[i]
        dist = np.linalg.norm(diff, axis=2)
        dist[np.arange(k), np.arange(k)] = np.inf
        order = np.argsort(dist, axis=1, kind='stable')[:, : k - 1]      # (k,k-1)
        rel = np.take_along_axis(diff, order[:, :, None], axis=1)  # (k,k-1,2)
        same = (body_id[order] == body_id[:, None]).astype(float)
        padrel = PADS[order] - P[:, None, :]
        others = np.concatenate([
            rel, V[order], (NB[order] / cfg.max_group)[:, :, None], JOIN[order][:, :, None], INT[order],
            same[:, :, None], LEAVING[order][:, :, None], PROG[order][:, :, None], padrel,
            (PT[order] / 40.0)[:, :, None],
        ], axis=2).reshape(k, -1)

        m = cfg.n_mines
        mrel = g.mine_pos[None, :, :] - P[:, None, :]      # (k,m,2)
        moh = np.zeros((m, TYPES)); moh[np.arange(m), g.mine_type] = 1
        mines = np.concatenate([
            mrel, np.broadcast_to(moh, (k, m, TYPES)),
            np.broadcast_to((g.mine_stock / cfg.mine_cap)[None, :, None], (k, m, 1)),
            np.broadcast_to(g.mine_alive.astype(float)[None, :, None], (k, m, 1)),
        ], axis=2).reshape(k, -1)

        picks = np.zeros((k, N_PICKUPS_OBS, 6))
        npk = len(g.pick_ttl)
        if npk:
            prel = g.pick_pos[None, :, :] - P[:, None, :]  # (k,npk,2)
            pd = np.linalg.norm(prel, axis=2)
            po = np.argsort(pd, axis=1, kind='stable')[:, :N_PICKUPS_OBS]
            cnt = min(npk, N_PICKUPS_OBS)
            poh = np.zeros((npk, TYPES)); poh[np.arange(npk), g.pick_type] = 1
            picks[:, :cnt, :2] = np.take_along_axis(prel, po[:, :, None], axis=1)
            picks[:, :cnt, 2:] = poh[po]
        picks = picks.reshape(k, -1)

        return np.concatenate([own, others, mines, picks], axis=1).astype(np.float32)


def direction_to_move(v):
    """Quantize a 2D vector into the 9-way move action."""
    n = math.hypot(v[0], v[1])
    if n < 1e-6:
        return 0
    ang = math.atan2(v[1], v[0])
    idx = int(round(ang / (2 * math.pi / 8))) % 8
    return idx + 1
