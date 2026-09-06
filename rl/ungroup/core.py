"""
Ungroup v2 rules core: carry, bank, spill.

A dependency-light, deterministic implementation of the redesigned rules described in
docs/PRD.md and the design discussion. It has no rendering and no networking. It is used by
the RL environment (env.py), the scripted bots (bots.py), and the replay recorder.

Units: the arena is a circle of radius 1.0 at the start of a round. Time is in seconds.

Rules summary
- Every player is a circle. Circles that touch while both are "joinable" merge into one group
  body with a shared, typed resource pool.
- Group speed is base_speed / sqrt(n) times the mean of the members' direction vectors, so
  mass costs speed and members pulling in different directions slow the group.
- Mines yield while a body is in contact. Yield rate is mine_rate * n ** mine_exp (superlinear).
- Nothing counts until it is banked. Each player owns a pad on the arena boundary. When a
  group touches a member's pad, the whole pool is banked to that member.
- Leaving a group takes leave_time seconds and is visible. On detaching, the leaver takes a
  share of the pool per type, weighted 2:1 toward members whose declared intent matches the
  type (intent is public cheap talk that also shapes the split).
- Hard collisions spill carried units from both bodies as floor pickups anyone can collect.
- The arena shrinks in the second part of the round; mines outside it die; pads slide inward.
- A player wins when banked >= need for every type. Otherwise the round ends at time_limit.
"""

import math
from dataclasses import dataclass, field

import numpy as np

TYPES = 4


def _norm(v):
    return math.hypot(float(v[0]), float(v[1]))


@dataclass
class Config:
    n_players: int = 6
    n_mines: int = 8
    dt: float = 0.1  # physics step, seconds
    time_limit: float = 240.0
    base_speed: float = 0.45
    vel_lerp: float = 6.0
    solo_radius: float = 0.045
    mine_radius: float = 0.08
    mine_cap: float = 30.0
    mine_regen: float = 0.5  # units per second
    mine_rate: float = 0.15  # units per second for a solo body; scales n ** mine_exp
    mine_exp: float = 2.0
    pad_radius: float = 0.06
    need_primary: int = 18
    need_secondary: int = 6
    leave_time: float = 2.0
    spill_min_speed: float = 0.25
    spill_k: float = 6.0
    spill_max: int = 6
    pickup_ttl: float = 8.0
    max_pickups: int = 64
    shrink_start: float = 0.4  # fraction of time_limit
    final_radius: float = 0.35
    restitution: float = 0.5
    max_group: int = 6


@dataclass
class Body:
    members: list
    pos: np.ndarray
    vel: np.ndarray
    pool: np.ndarray  # shape (TYPES,)

    @property
    def n(self):
        return len(self.members)


@dataclass
class Player:
    need: np.ndarray
    banked: np.ndarray
    pad_angle: float
    intent: int = 0
    joinable: bool = False
    direction: np.ndarray = field(default_factory=lambda: np.zeros(2))
    leave_timer: float = -1.0  # seconds remaining, < 0 when not leaving


class Game:
    """One round of Ungroup v2."""

    def __init__(self, cfg: Config, seed: int = 0, record: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.record = record
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self):
        cfg = self.cfg
        n = cfg.n_players
        self.t = 0.0
        self.step_count = 0
        self.done = False
        self.winner = -1
        self.events = []  # events of the current tick
        self.frames = [] if self.record else None
        self.arena_radius = 1.0

        # Players and pads spread evenly, with a random rotation.
        rot = self.rng.uniform(0, 2 * math.pi)
        self.players = []
        for i in range(n):
            primary = int(self.rng.integers(TYPES))
            need = np.full(TYPES, cfg.need_secondary, dtype=float)
            need[primary] = cfg.need_primary
            angle = rot + 2 * math.pi * i / n
            self.players.append(
                Player(need=need, banked=np.zeros(TYPES), pad_angle=angle, intent=primary))

        # Mines: alternating inner and outer ring, types spread so each type has one of each.
        self.mine_pos = np.zeros((cfg.n_mines, 2))
        self.mine_type = np.zeros(cfg.n_mines, dtype=int)
        self.mine_stock = np.full(cfg.n_mines, cfg.mine_cap)
        self.mine_alive = np.ones(cfg.n_mines, dtype=bool)
        mrot = self.rng.uniform(0, 2 * math.pi)
        for m in range(cfg.n_mines):
            ring = 0.62 if m % 2 == 0 else 0.35
            angle = mrot + 2 * math.pi * m / cfg.n_mines
            self.mine_pos[m] = ring * np.array([math.cos(angle), math.sin(angle)])
            self.mine_type[m] = (m // 2) % TYPES if cfg.n_mines >= 2 * TYPES else m % TYPES

        # Bodies: every player starts solo near their pad, slightly inside.
        self.bodies = []
        for i, p in enumerate(self.players):
            pos = 0.72 * np.array([math.cos(p.pad_angle), math.sin(p.pad_angle)])
            self.bodies.append(Body(members=[i], pos=pos, vel=np.zeros(2), pool=np.zeros(TYPES)))

        # Pickups as parallel arrays.
        self.pick_pos = np.zeros((0, 2))
        self.pick_type = np.zeros(0, dtype=int)
        self.pick_ttl = np.zeros(0)

        if self.record:
            self._record_frame()

    # --------------------------------------------------------------- helpers

    def body_of(self, player_id):
        for b in self.bodies:
            if player_id in b.members:
                return b
        raise KeyError(player_id)

    def radius(self, n):
        return self.cfg.solo_radius * math.sqrt(n)

    def pad_pos(self, i):
        p = self.players[i]
        r = max(self.arena_radius - self.cfg.pad_radius - 0.02, 0.1)
        return r * np.array([math.cos(p.pad_angle), math.sin(p.pad_angle)])

    def progress(self, i):
        p = self.players[i]
        return float(np.mean(np.minimum(p.banked / p.need, 1.0)))

    def _event(self, kind, **kw):
        kw["kind"] = kind
        kw["t"] = round(self.t, 2)
        self.events.append(kw)

    # ---------------------------------------------------------------- actions

    def apply_actions(self, actions):
        """actions: dict or list indexed by player id of (move, joinable, leave, intent).

        move: 0 = none, 1..8 = compass directions starting east going counter-clockwise.
        joinable: 0 off, 1 on.
        leave: 1 requests leaving the current group (no-op when solo or already leaving).
        intent: 0 keep, 1..4 set intent to type 0..3.
        """
        for i, act in enumerate(actions):
            move, joinable, leave, intent = act
            p = self.players[i]
            if move == 0:
                p.direction = np.zeros(2)
            else:
                a = 2 * math.pi * (move - 1) / 8
                p.direction = np.array([math.cos(a), math.sin(a)])
            p.joinable = bool(joinable)
            if intent > 0:
                p.intent = int(intent) - 1
            if leave and p.leave_timer < 0 and self.body_of(i).n > 1:
                p.leave_timer = self.cfg.leave_time
                self._event("leave_start", player=i)

    # ------------------------------------------------------------------ tick

    def tick(self):
        if self.done:
            return
        cfg = self.cfg
        dt = cfg.dt
        self.events = []

        self._update_arena()
        self._move_bodies()
        self._update_leaving()
        self._collide_bodies()
        self._collide_mines()
        self._bank()
        self._collect_pickups()
        self._regen()

        self.t += dt
        self.step_count += 1

        # Win check
        for i, p in enumerate(self.players):
            if np.all(p.banked >= p.need):
                self.done = True
                self.winner = i
                self._event("win", player=i)
                break
        if not self.done and self.t >= cfg.time_limit:
            self.done = True
            self._event("timeout")

        if self.record:
            self._record_frame()

    def _update_arena(self):
        cfg = self.cfg
        frac = self.t / cfg.time_limit
        if frac <= cfg.shrink_start:
            self.arena_radius = 1.0
        else:
            k = (frac - cfg.shrink_start) / (1.0 - cfg.shrink_start)
            self.arena_radius = 1.0 + (cfg.final_radius - 1.0) * min(k, 1.0)
        # Mines outside the arena die.
        for m in range(cfg.n_mines):
            if self.mine_alive[m] and _norm(self.mine_pos[m]) + cfg.mine_radius > self.arena_radius:
                self.mine_alive[m] = False
                self._event("mine_dead", mine=m)

    def _move_bodies(self):
        cfg = self.cfg
        dt = cfg.dt
        a = min(1.0, cfg.vel_lerp * dt)
        for b in self.bodies:
            mean_dir = np.zeros(2)
            for i in b.members:
                mean_dir += self.players[i].direction
            mean_dir /= b.n
            target = (cfg.base_speed / math.sqrt(b.n)) * mean_dir
            b.vel += (target - b.vel) * a
            b.pos = b.pos + b.vel * dt
            # Keep inside the arena.
            r = self.radius(b.n)
            d = _norm(b.pos)
            limit = self.arena_radius - r
            if d > limit and d > 0:
                nrm = b.pos / d
                b.pos = nrm * limit
                vr = float(np.dot(b.vel, nrm))
                if vr > 0:
                    b.vel = b.vel - vr * nrm

    def _update_leaving(self):
        cfg = self.cfg
        for i, p in enumerate(self.players):
            if p.leave_timer < 0:
                continue
            b = self.body_of(i)
            if b.n == 1:
                p.leave_timer = -1.0
                continue
            p.leave_timer -= cfg.dt
            if p.leave_timer <= 0:
                p.leave_timer = -1.0
                self._detach(i, b)

    def _detach(self, i, b):
        cfg = self.cfg
        p = self.players[i]
        weights = np.zeros(TYPES)
        for j in b.members:
            w = np.ones(TYPES)
            w[self.players[j].intent] = 2.0
            weights += w
        my_w = np.ones(TYPES)
        my_w[p.intent] = 2.0
        share = b.pool * (my_w / weights)
        b.pool = b.pool - share
        b.members.remove(i)
        u = p.direction.copy()
        if _norm(u) < 1e-6:
            ang = self.rng.uniform(0, 2 * math.pi)
            u = np.array([math.cos(ang), math.sin(ang)])
        else:
            u = u / _norm(u)
        pos = b.pos + u * (self.radius(b.n) + self.radius(1) + 0.01)
        nb = Body(members=[i], pos=pos, vel=b.vel + u * 0.15, pool=share)
        self.bodies.append(nb)
        self._event("leave", player=i, share=[round(float(x), 2) for x in share],
                    from_size=b.n + 1)

    def _collide_bodies(self):
        cfg = self.cfg
        restart = True
        guard = 0
        while restart and guard < 20:
            restart = False
            guard += 1
            nb = len(self.bodies)
            for ai in range(nb):
                for bi in range(ai + 1, nb):
                    a, b = self.bodies[ai], self.bodies[bi]
                    ra, rb = self.radius(a.n), self.radius(b.n)
                    delta = b.pos - a.pos
                    dist = _norm(delta)
                    if dist >= ra + rb or dist < 1e-9:
                        continue
                    nrm = delta / dist
                    # Merge if both are joinable and the result is not too large.
                    if self._joinable(a) and self._joinable(b) and a.n + b.n <= cfg.max_group:
                        self._merge(ai, bi)
                        restart = True
                        break
                    # Otherwise resolve the collision and possibly spill.
                    ma, mb = float(a.n), float(b.n)
                    overlap = ra + rb - dist
                    a.pos = a.pos - nrm * overlap * (mb / (ma + mb))
                    b.pos = b.pos + nrm * overlap * (ma / (ma + mb))
                    rel = float(np.dot(a.vel - b.vel, nrm))  # > 0 when approaching
                    if rel > 0:
                        j = (1 + cfg.restitution) * rel / (1 / ma + 1 / mb)
                        a.vel = a.vel - (j / ma) * nrm
                        b.vel = b.vel + (j / mb) * nrm
                        if rel > cfg.spill_min_speed:
                            contact = a.pos + nrm * ra
                            units = int(min(cfg.spill_max, round(cfg.spill_k * (rel - cfg.spill_min_speed) + 1)))
                            sa = self._spill(a, units, contact)
                            sb = self._spill(b, units, contact)
                            if sa or sb:
                                self._event("spill", a=list(a.members), b=list(b.members),
                                            units=sa + sb, speed=round(rel, 2),
                                            x=round(float(contact[0]), 3), y=round(float(contact[1]), 3))
                if restart:
                    break

    def _joinable(self, b):
        return any(self.players[i].joinable for i in b.members)

    def _merge(self, ai, bi):
        a, b = self.bodies[ai], self.bodies[bi]
        ma, mb = float(a.n), float(b.n)
        pos = (a.pos * ma + b.pos * mb) / (ma + mb)
        vel = (a.vel * ma + b.vel * mb) / (ma + mb)
        merged = Body(members=a.members + b.members, pos=pos, vel=vel, pool=a.pool + b.pool)
        for i in merged.members:
            self.players[i].leave_timer = -1.0
        self.bodies = [x for k, x in enumerate(self.bodies) if k not in (ai, bi)] + [merged]
        self._event("merge", a=list(a.members), b=list(b.members), size=merged.n)

    def _spill(self, b, units, contact):
        total = float(b.pool.sum())
        if total <= 0 or units <= 0:
            return 0
        units = min(units, int(math.floor(total)))
        if units <= 0:
            return 0
        # Draw types proportional to the pool composition.
        probs = b.pool / total
        counts = self.rng.multinomial(units, probs)
        counts = np.minimum(counts, np.floor(b.pool)).astype(int)
        b.pool = b.pool - counts
        for tp in range(TYPES):
            for _ in range(int(counts[tp])):
                ang = self.rng.uniform(0, 2 * math.pi)
                rad = self.rng.uniform(0.04, 0.12)
                pos = contact + rad * np.array([math.cos(ang), math.sin(ang)])
                d = _norm(pos)
                if d > self.arena_radius - 0.02:
                    pos = pos / d * (self.arena_radius - 0.02)
                self._add_pickup(pos, tp)
        return int(counts.sum())

    def _add_pickup(self, pos, tp):
        cfg = self.cfg
        self.pick_pos = np.vstack([self.pick_pos, pos[None, :]])
        self.pick_type = np.append(self.pick_type, tp)
        self.pick_ttl = np.append(self.pick_ttl, cfg.pickup_ttl)
        if len(self.pick_ttl) > cfg.max_pickups:
            self.pick_pos = self.pick_pos[1:]
            self.pick_type = self.pick_type[1:]
            self.pick_ttl = self.pick_ttl[1:]

    def _collide_mines(self):
        cfg = self.cfg
        dt = cfg.dt
        for b in self.bodies:
            r = self.radius(b.n)
            for m in range(cfg.n_mines):
                delta = b.pos - self.mine_pos[m]
                dist = _norm(delta)
                if dist >= r + cfg.mine_radius + 0.01:
                    continue
                if dist < 1e-9:
                    delta = np.array([1.0, 0.0])
                    dist = 1e-9
                nrm = delta / dist
                # Mining happens on contact with a live mine.
                if self.mine_alive[m] and self.mine_stock[m] > 0:
                    rate = cfg.mine_rate * (b.n ** cfg.mine_exp)
                    amount = min(rate * dt, float(self.mine_stock[m]))
                    self.mine_stock[m] -= amount
                    b.pool[self.mine_type[m]] += amount
                # Push out, sticky (no bounce), so holding a direction keeps contact.
                if dist < r + cfg.mine_radius:
                    b.pos = self.mine_pos[m] + nrm * (r + cfg.mine_radius)
                    vr = float(np.dot(b.vel, nrm))
                    if vr < 0:
                        b.vel = b.vel - vr * nrm

    def _bank(self):
        cfg = self.cfg
        for b in self.bodies:
            if b.pool.sum() < 0.5:
                continue
            r = self.radius(b.n)
            for i in b.members:
                pad = self.pad_pos(i)
                if _norm(b.pos - pad) < r + cfg.pad_radius:
                    amount = b.pool.copy()
                    self.players[i].banked += amount
                    b.pool = np.zeros(TYPES)
                    self._event("bank", player=i, amount=[round(float(x), 2) for x in amount],
                                group=list(b.members))
                    break

    def _collect_pickups(self):
        cfg = self.cfg
        if len(self.pick_ttl) == 0:
            return
        self.pick_ttl -= cfg.dt
        keep = self.pick_ttl > 0
        for b in self.bodies:
            r = self.radius(b.n)
            d = np.linalg.norm(self.pick_pos - b.pos, axis=1)
            hit = (d < r + 0.015) & keep
            if hit.any():
                for tp in range(TYPES):
                    b.pool[tp] += float(np.sum(hit & (self.pick_type == tp)))
                keep &= ~hit
        self.pick_pos = self.pick_pos[keep]
        self.pick_type = self.pick_type[keep]
        self.pick_ttl = self.pick_ttl[keep]

    def _regen(self):
        cfg = self.cfg
        alive = self.mine_alive
        self.mine_stock[alive] = np.minimum(cfg.mine_cap, self.mine_stock[alive] + cfg.mine_regen * cfg.dt)

    # ------------------------------------------------------------- recording

    def _record_frame(self):
        frame = {
            "t": round(self.t, 2),
            "R": round(self.arena_radius, 3),
            "bodies": [
                {
                    "m": list(b.members),
                    "x": round(float(b.pos[0]), 3),
                    "y": round(float(b.pos[1]), 3),
                    "vx": round(float(b.vel[0]), 3),
                    "vy": round(float(b.vel[1]), 3),
                    "pool": [round(float(x), 1) for x in b.pool],
                }
                for b in self.bodies
            ],
            "mines": [round(float(x), 1) for x in self.mine_stock],
            "alive": [bool(x) for x in self.mine_alive],
            "picks": [[round(float(p[0]), 3), round(float(p[1]), 3), int(tp)]
                      for p, tp in zip(self.pick_pos, self.pick_type)],
            "players": [
                {
                    "banked": [round(float(x), 1) for x in p.banked],
                    "intent": int(p.intent),
                    "join": bool(p.joinable),
                    "leaving": round(float(p.leave_timer), 1) if p.leave_timer >= 0 else -1,
                    "dir": [round(float(p.direction[0]), 2), round(float(p.direction[1]), 2)],
                }
                for p in self.players
            ],
            "events": list(self.events),
        }
        self.frames.append(frame)

    def replay(self, names=None, meta=None):
        cfg = self.cfg
        return {
            "config": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in vars(cfg).items()},
            "seed": self.seed,
            "names": names or [f"P{i}" for i in range(cfg.n_players)],
            "meta": meta or {},
            "needs": [[float(x) for x in p.need] for p in self.players],
            "pads": [float(p.pad_angle) for p in self.players],
            "mine_pos": [[float(x), float(y)] for x, y in self.mine_pos],
            "mine_type": [int(x) for x in self.mine_type],
            "winner": self.winner,
            "frames": self.frames,
        }
