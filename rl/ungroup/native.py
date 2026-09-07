"""
ctypes wrapper for the canonical C++ rules core (rl/native/ungroup.cpp).

NativeBatch steps many games at once with OpenMP. Seats are either external (actions supplied by
the caller, e.g. a policy network) or scripted bots run inside C++.
"""

import ctypes
import json
import os
import subprocess
from dataclasses import dataclass, asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE_DIR = os.path.join(os.path.dirname(HERE), "native")
SRC = os.path.join(NATIVE_DIR, "ungroup.cpp")
LIB = os.path.join(NATIVE_DIR, "libungroup.so")

TYPES = 4
MAX_SLOTS = 8
N_DR = 4


@dataclass
class Config:
    """Rule constants. Field order must match cfg_fill() in ungroup.cpp."""
    n_players: int = 6
    n_mines: int = 8
    dt: float = 1.0 / 30.0
    time_limit: float = 240.0
    base_speed: float = 0.45
    vel_lerp: float = 6.0
    solo_radius: float = 0.045
    mine_radius: float = 0.08
    mine_cap: float = 30.0
    mine_regen: float = 0.5
    mine_rate: float = 0.12
    mine_exp: float = 2.0
    pad_radius: float = 0.06
    need_primary: int = 18
    need_secondary: int = 6
    leave_time: float = 2.0
    spill_min_speed: float = 0.40
    spill_k: float = 6.0
    spill_max: int = 6
    pickup_ttl: float = 8.0
    max_pickups: int = 64
    shrink_start: float = 0.5
    final_radius: float = 0.5
    restitution: float = 0.5
    max_group: int = 6
    join_cooldown: float = 3.0
    partner_cooldown: float = 10.0
    leave_forfeit: float = 0.15
    intent_weight: float = 3.0
    stun_time: float = 1.0
    leave_hold: float = 1.0
    carried_shaping: float = 2.0
    win_bonus: float = 10.0
    lose_penalty: float = 2.0
    relative_reward: float = 0.5

    def to_array(self):
        return [float(v) for v in asdict(self).values()]

    @classmethod
    def from_array(cls, arr):
        names = list(cls.__dataclass_fields__.keys())
        kw = {}
        for name, v in zip(names, arr):
            f = cls.__dataclass_fields__[name]
            kw[name] = int(round(v)) if f.type is int else float(v)
        return cls(**kw)

    def replace(self, **kw):
        d = asdict(self)
        d.update(kw)
        return Config(**d)


CFG_FIELDS = list(Config.__dataclass_fields__.keys())

SEAT_EXTERNAL, SEAT_EXTERNAL2, SEAT_SOLO, SEAT_BAIL, SEAT_LOYAL, SEAT_KIDNAP, SEAT_RAMMER = range(7)
SEAT_NAMES = {"policy": SEAT_EXTERNAL, "snapshot": SEAT_EXTERNAL2, "solo": SEAT_SOLO, "bail": SEAT_BAIL,
              "loyal": SEAT_LOYAL, "kidnap": SEAT_KIDNAP, "rammer": SEAT_RAMMER}
SEAT_LABEL = {v: k for k, v in SEAT_NAMES.items()}
TRAINING_BOTS = ["solo", "bail", "loyal"]
HELDOUT_BOTS = ["kidnap", "rammer"]

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        subprocess.check_call(["g++", "-O3", "-march=native", "-std=c++17", "-fopenmp", "-shared", "-fPIC", "-o", LIB, SRC])
    lib = ctypes.CDLL(LIB)
    P = ctypes.POINTER
    lib.ugb_cfg_len.restype = ctypes.c_int
    lib.ugb_stats_base.restype = ctypes.c_int
    lib.ugb_create.restype = ctypes.c_void_p
    lib.ugb_create.argtypes = [ctypes.c_int, P(ctypes.c_double), ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int]
    lib.ugb_destroy.argtypes = [ctypes.c_void_p]
    for name in ("ugb_obs_dim", "ugb_priv_dim", "ugb_n_players"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
        getattr(lib, name).restype = ctypes.c_int
    lib.ugb_set_cfg_range.argtypes = [ctypes.c_void_p, P(ctypes.c_double), P(ctypes.c_double), ctypes.c_int]
    lib.ugb_set_seats.argtypes = [ctypes.c_void_p, ctypes.c_int, P(ctypes.c_int)]
    lib.ugb_reset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    lib.ugb_observe.argtypes = [ctypes.c_void_p, P(ctypes.c_float)]
    lib.ugb_observe_priv.argtypes = [ctypes.c_void_p, P(ctypes.c_float)]
    lib.ugb_bot_actions.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, P(ctypes.c_int)]
    lib.ugb_set_direction.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double]
    lib.ugb_cfg.argtypes = [ctypes.c_void_p, ctypes.c_int, P(ctypes.c_double)]
    lib.ugb_step.argtypes = [ctypes.c_void_p, P(ctypes.c_int), P(ctypes.c_float), P(ctypes.c_ubyte), P(ctypes.c_float),
                             P(ctypes.c_float), ctypes.c_int, P(ctypes.c_double), ctypes.c_int]
    lib.ugb_frame_json.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.ugb_frame_json.restype = ctypes.c_int
    lib.ugb_meta_json.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.ugb_meta_json.restype = ctypes.c_int
    lib.ugb_progress.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ugb_progress.restype = ctypes.c_double
    lib.ugb_winner.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ugb_winner.restype = ctypes.c_int
    lib.ugb_done.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ugb_done.restype = ctypes.c_int
    lib.ugb_time.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ugb_time.restype = ctypes.c_double
    assert lib.ugb_cfg_len() == len(CFG_FIELDS), f"config layout mismatch: C++ {lib.ugb_cfg_len()} vs Python {len(CFG_FIELDS)}"
    _lib = lib
    return lib


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


STAT_KEYS = ["winner", "length", "avg_group", "merges", "leaves", "spills", "banks", "ended", "group_banks",
             "remerge_fast", "alliances", "alliance_dur", "alliances_long"]


class NativeBatch:
    def __init__(self, n_envs, cfg: Config = None, seed=1, decide_every=6, dr_lo: Config = None, dr_hi: Config = None):
        self.lib = _load()
        self.cfg = cfg or Config()
        vals = self.cfg.to_array()
        arr = (ctypes.c_double * len(vals))(*vals)
        self.h = self.lib.ugb_create(n_envs, arr, len(vals), seed, decide_every)
        self.E = n_envs
        self.n = self.lib.ugb_n_players(self.h)
        self.obs_dim = self.lib.ugb_obs_dim(self.h)
        self.priv_dim = self.lib.ugb_priv_dim(self.h)
        self.stats_base = self.lib.ugb_stats_base()
        self._obs = np.zeros((self.E, self.n, self.obs_dim), dtype=np.float32)
        self._priv = np.zeros((self.E, self.n, self.priv_dim), dtype=np.float32)
        self._rew = np.zeros((self.E, self.n), dtype=np.float32)
        self._done = np.zeros(self.E, dtype=np.uint8)
        self._stats = np.zeros((self.E, self.stats_base + self.n + N_DR), dtype=np.float64)
        self._buf = ctypes.create_string_buffer(1 << 20)
        self._act = np.zeros((self.n, 4), dtype=np.int32)
        if dr_lo is not None and dr_hi is not None:
            self.set_cfg_range(dr_lo, dr_hi)

    def __del__(self):
        try:
            self.lib.ugb_destroy(self.h)
        except Exception:
            pass

    def set_cfg_range(self, lo: Config, hi: Config):
        a = lo.to_array(); b = hi.to_array()
        la = (ctypes.c_double * len(a))(*a); lb = (ctypes.c_double * len(b))(*b)
        self.lib.ugb_set_cfg_range(self.h, la, lb, len(a))

    def set_seats(self, env, seats):
        vals = [SEAT_NAMES[s] if isinstance(s, str) else int(s) for s in seats]
        arr = (ctypes.c_int * self.n)(*vals)
        self.lib.ugb_set_seats(self.h, env, arr)

    def reset(self, env, seed=0):
        self.lib.ugb_reset(self.h, env, seed)

    def observe(self):
        self.lib.ugb_observe(self.h, _ptr(self._obs, ctypes.c_float))
        return self._obs.copy()

    def observe_priv(self):
        self.lib.ugb_observe_priv(self.h, _ptr(self._priv, ctypes.c_float))
        return self._priv.copy()

    def bot_actions(self, env, seat_type):
        st = SEAT_NAMES[seat_type] if isinstance(seat_type, str) else int(seat_type)
        self.lib.ugb_bot_actions(self.h, env, st, _ptr(self._act, ctypes.c_int))
        return self._act.copy()

    def set_direction(self, env, seat, dx, dy):
        self.lib.ugb_set_direction(self.h, env, seat, dx, dy)

    def game_cfg(self, env):
        out = np.zeros(len(CFG_FIELDS), dtype=np.float64)
        self.lib.ugb_cfg(self.h, env, _ptr(out, ctypes.c_double))
        return Config.from_array(out)

    def step(self, actions, auto_reset=True, decide_every=0, want_priv=False):
        """actions: (E, n, 4) ints. Returns obs, rewards, dones, episodes[, priv]."""
        act = np.ascontiguousarray(actions, dtype=np.int32).reshape(self.E, self.n, 4)
        self.lib.ugb_step(self.h, _ptr(act, ctypes.c_int), _ptr(self._rew, ctypes.c_float), _ptr(self._done, ctypes.c_ubyte),
                          _ptr(self._obs, ctypes.c_float), _ptr(self._priv, ctypes.c_float) if want_priv else None,
                          1 if auto_reset else 0, _ptr(self._stats, ctypes.c_double), decide_every)
        episodes = []
        for e in np.where(self._stats[:, 7] > 0)[0]:
            st = self._stats[e]
            d = {k: float(st[i]) for i, k in enumerate(STAT_KEYS)}
            d["env"] = int(e)
            d["winner"] = int(d["winner"])
            d["timeout_win"] = st[7] == 2
            d["progress"] = [float(x) for x in st[self.stats_base:self.stats_base + self.n]]
            d["dr"] = [float(x) for x in st[self.stats_base + self.n:self.stats_base + self.n + N_DR]]
            episodes.append(d)
        if want_priv:
            return self._obs.copy(), self._rew.copy(), self._done.astype(bool), episodes, self._priv.copy()
        return self._obs.copy(), self._rew.copy(), self._done.astype(bool), episodes

    def frame(self, env):
        r = self.lib.ugb_frame_json(self.h, env, self._buf, len(self._buf))
        if r < 0:
            self._buf = ctypes.create_string_buffer(-r + 1024)
            r = self.lib.ugb_frame_json(self.h, env, self._buf, len(self._buf))
        return json.loads(self._buf.value[:r].decode())

    def meta(self, env):
        r = self.lib.ugb_meta_json(self.h, env, self._buf, len(self._buf))
        return json.loads(self._buf.value[:r].decode())

    def progress(self, env, player):
        return self.lib.ugb_progress(self.h, env, player)

    def winner(self, env):
        return self.lib.ugb_winner(self.h, env)

    def done(self, env):
        return bool(self.lib.ugb_done(self.h, env))

    def time(self, env):
        return self.lib.ugb_time(self.h, env)


def record_game(policy_act, seats, cfg: Config = None, seed=1, names=None, meta=None, decide_every=6, frame_every=3):
    """Play one game with per-tick recording. policy_act(obs (n,D)) -> (n,4) ints for external seats.
    Records a frame every `frame_every` ticks (10 Hz at the default 30 Hz physics) and the action log."""
    cfg = cfg or Config(n_players=len(seats))
    b = NativeBatch(1, cfg, seed=seed, decide_every=1)
    b.set_seats(0, seats)
    b.reset(0, seed)
    frames = [b.frame(0)]
    obs = b.observe()
    act = None
    actions = []
    k = 0
    pending = []
    while not b.done(0):
        if k % decide_every == 0:
            act = policy_act(obs[0])
            actions.append(act.tolist())
        obs, _, _, _ = b.step(act[None], auto_reset=False, decide_every=1)
        k += 1
        fr = b.frame(0)
        pending.extend(fr["events"])
        if k % frame_every == 0 or b.done(0):
            fr["events"] = pending
            pending = []
            frames.append(fr)
    m = b.meta(0)
    game_cfg = Config.from_array(m["cfg"])
    return {
        "config": asdict(game_cfg),
        "seed": seed,
        "names": names or [f"P{i}" for i in range(len(seats))],
        "meta": dict(meta or {}, progress=[b.progress(0, i) for i in range(len(seats))], seats=list(seats),
                     timeout_win=m["timeout_win"], decide_every=decide_every, frame_every=frame_every),
        "needs": m["needs"], "pads": m["pads"], "mine_pos": m["mine_pos"], "mine_type": m["mine_type"],
        "winner": m["winner"], "frames": frames, "actions": actions,
    }
