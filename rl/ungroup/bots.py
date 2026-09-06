"""
Scripted baseline policies. They read the Game state directly (full information about
positions; they only use their own needs). Used for balance checks and as evaluation opponents.
"""

import math

import numpy as np

from .core import TYPES, Game
from .env import direction_to_move


def _needed_types(g: Game, i, include_pool=True):
    p = g.players[i]
    b = g.body_of(i)
    have = p.banked + (b.pool / b.n if include_pool else 0)
    return [t for t in range(TYPES) if have[t] < p.need[t]]


def _nearest_mine(g: Game, pos, types):
    best, best_d = -1, 1e9
    for m in range(g.cfg.n_mines):
        if not g.mine_alive[m] or g.mine_stock[m] < 1.0:
            continue
        if types is not None and g.mine_type[m] not in types:
            continue
        d = math.hypot(*(g.mine_pos[m] - pos))
        if d < best_d:
            best, best_d = m, d
    return best


class SoloGreedy:
    """Never groups. Mines the nearest needed mine, banks when carrying enough."""

    def __init__(self, bank_at=8.0):
        self.bank_at = bank_at

    def act(self, g: Game, i):
        p = g.players[i]
        b = g.body_of(i)
        primary = int(np.argmax(p.need))
        target = None
        if b.pool.sum() >= self.bank_at:
            target = g.pad_pos(i)
        else:
            needed = _needed_types(g, i)
            m = _nearest_mine(g, b.pos, needed if needed else None)
            if m < 0:
                m = _nearest_mine(g, b.pos, None)
            if m >= 0:
                target = g.mine_pos[m]
            elif b.pool.sum() > 0:
                target = g.pad_pos(i)
        move = 0 if target is None else direction_to_move(target - b.pos)
        return (move, 0, 1 if b.n > 1 else 0, primary + 1)


class GroupAndBail:
    """Joins groups, mines with them, and leaves when the pool is worth taking or when the
    group nears another member's pad."""

    def __init__(self, max_group=3, bank_at=10.0, bail_at=4.0, pad_danger=0.3):
        self.max_group = max_group
        self.bank_at = bank_at
        self.bail_at = bail_at
        self.pad_danger = pad_danger

    def act(self, g: Game, i):
        p = g.players[i]
        b = g.body_of(i)
        needed = _needed_types(g, i)
        intent = needed[0] if needed else int(np.argmax(p.need))
        joinable = 1 if b.n < self.max_group else 0
        leave = 0
        target = None
        my_pad = g.pad_pos(i)
        if b.n > 1:
            my_share = b.pool.sum() / b.n
            # Danger: group is close to another member's pad and carrying something.
            for j in b.members:
                if j == i:
                    continue
                if math.hypot(*(g.pad_pos(j) - b.pos)) < self.pad_danger and b.pool.sum() >= self.bail_at:
                    leave = 1
            if b.pool.sum() >= self.bank_at:
                target = my_pad
            if my_share >= self.bail_at and math.hypot(*(my_pad - b.pos)) > 0.6 and leave == 0:
                # Far from home with a decent share: take it and go bank alone.
                leave = 1
        else:
            if b.pool.sum() >= self.bail_at:
                target = my_pad
        if target is None:
            m = _nearest_mine(g, b.pos, needed if needed else None)
            if m < 0:
                m = _nearest_mine(g, b.pos, None)
            target = g.mine_pos[m] if m >= 0 else my_pad
        move = direction_to_move(target - b.pos)
        return (move, joinable, leave, intent + 1)


class RandomBot:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, g: Game, i):
        return (int(self.rng.integers(9)), int(self.rng.integers(2)), int(self.rng.integers(2) if self.rng.random() < 0.05 else 0),
                int(self.rng.integers(5)))


BOTS = {"solo": SoloGreedy, "bail": GroupAndBail, "random": RandomBot}
