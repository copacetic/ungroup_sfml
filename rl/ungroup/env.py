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

    def observe(self):
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


def direction_to_move(v):
    """Quantize a 2D vector into the 9-way move action."""
    n = math.hypot(v[0], v[1])
    if n < 1e-6:
        return 0
    ang = math.atan2(v[1], v[0])
    idx = int(round(ang / (2 * math.pi / 8))) % 8
    return idx + 1
