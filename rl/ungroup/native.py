"""
ctypes wrapper for the C++ rules core (rl/native/ungroup.cpp).

NativeBatch steps many games at once with OpenMP. Seats can be external (actions supplied by the
caller, e.g. a policy network) or scripted bots run inside C++.
"""

import ctypes
import json
import os
import subprocess

import numpy as np

from .core import Config

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE_DIR = os.path.join(os.path.dirname(HERE), "native")
SRC = os.path.join(NATIVE_DIR, "ungroup.cpp")
LIB = os.path.join(NATIVE_DIR, "libungroup.so")

CFG_FIELDS = ["n_players", "n_mines", "dt", "time_limit", "base_speed", "vel_lerp", "solo_radius",
              "mine_radius", "mine_cap", "mine_regen", "mine_rate", "mine_exp", "pad_radius",
              "need_primary", "need_secondary", "leave_time", "spill_min_speed", "spill_k", "spill_max",
              "pickup_ttl", "max_pickups", "shrink_start", "final_radius", "restitution", "max_group"]

SEAT_EXTERNAL, SEAT_EXTERNAL2, SEAT_SOLO, SEAT_BAIL = 0, 1, 2, 3
SEAT_NAMES = {"policy": SEAT_EXTERNAL, "snapshot": SEAT_EXTERNAL2, "solo": SEAT_SOLO, "bail": SEAT_BAIL}

_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        cmd = ["g++", "-O3", "-march=native", "-std=c++17", "-fopenmp", "-shared", "-fPIC", "-o", LIB, SRC]
        subprocess.check_call(cmd)
    lib = ctypes.CDLL(LIB)
    lib.ugb_create.restype = ctypes.c_void_p
    lib.ugb_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_ulonglong, ctypes.c_int]
    lib.ugb_destroy.argtypes = [ctypes.c_void_p]
    lib.ugb_obs_dim.argtypes = [ctypes.c_void_p]
    lib.ugb_obs_dim.restype = ctypes.c_int
    lib.ugb_n_players.argtypes = [ctypes.c_void_p]
    lib.ugb_n_players.restype = ctypes.c_int
    lib.ugb_set_seats.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.ugb_reset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong]
    lib.ugb_observe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.ugb_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
                             ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_int]
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
    _lib = lib
    return lib


def _ptr(arr, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


class NativeBatch:
    def __init__(self, n_envs, cfg: Config = None, seed=1, decide_every=2, carried_shaping=2.0, win_bonus=10.0,
                 lose_penalty=2.0):
        self.lib = _load()
        self.cfg = cfg or Config()
        vals = [float(getattr(self.cfg, f)) for f in CFG_FIELDS] + [carried_shaping, win_bonus, lose_penalty]
        arr = (ctypes.c_double * len(vals))(*vals)
        self.h = self.lib.ugb_create(n_envs, arr, len(vals), seed, decide_every)
        self.E = n_envs
        self.n = self.lib.ugb_n_players(self.h)
        self.obs_dim = self.lib.ugb_obs_dim(self.h)
        self._obs = np.zeros((self.E, self.n, self.obs_dim), dtype=np.float32)
        self._rew = np.zeros((self.E, self.n), dtype=np.float32)
        self._done = np.zeros(self.E, dtype=np.uint8)
        self._stats = np.zeros((self.E, 8 + self.n), dtype=np.float64)
        self._buf = ctypes.create_string_buffer(1 << 20)

    def __del__(self):
        try:
            self.lib.ugb_destroy(self.h)
        except Exception:
            pass

    def set_seats(self, env, seats):
        """seats: list of ints (SEAT_*) or names (policy, snapshot, solo, bail)."""
        vals = [SEAT_NAMES[s] if isinstance(s, str) else int(s) for s in seats]
        arr = (ctypes.c_int * self.n)(*vals)
        self.lib.ugb_set_seats(self.h, env, arr)

    def reset(self, env, seed=0):
        self.lib.ugb_reset(self.h, env, seed)

    def observe(self):
        self.lib.ugb_observe(self.h, _ptr(self._obs, ctypes.c_float))
        return self._obs.copy()

    def step(self, actions, auto_reset=True, decide_every=0):
        """actions: (E, n, 4) ints. Returns obs (E,n,D), rewards (E,n), dones (E,), episodes (list of dict)."""
        act = np.ascontiguousarray(actions, dtype=np.int32).reshape(self.E, self.n, 4)
        self.lib.ugb_step(self.h, _ptr(act, ctypes.c_int), _ptr(self._rew, ctypes.c_float), _ptr(self._done, ctypes.c_ubyte),
                          _ptr(self._obs, ctypes.c_float), 1 if auto_reset else 0, _ptr(self._stats, ctypes.c_double),
                          decide_every)
        episodes = []
        for e in np.where(self._stats[:, 7] > 0)[0]:
            st = self._stats[e]
            episodes.append(dict(env=int(e), winner=int(st[0]), length=float(st[1]), avg_group=float(st[2]),
                                 merges=int(st[3]), leaves=int(st[4]), spills=int(st[5]), banks=int(st[6]),
                                 progress=[float(x) for x in st[8:8 + self.n]]))
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


def record_game(policy_act, seats, cfg: Config = None, seed=1, names=None, meta=None):
    """Play one game in the native core with per-tick recording. policy_act(obs (n,D)) -> (n,4) ints,
    used for external seats; other seats are scripted bots. Returns a replay dict."""
    cfg = cfg or Config(n_players=len(seats))
    b = NativeBatch(1, cfg, seed=seed, decide_every=1)
    b.set_seats(0, seats)
    b.reset(0, seed)
    frames = [b.frame(0)]
    obs = b.observe()
    act = None
    k = 0
    while not b.done(0):
        if k % 2 == 0:
            act = policy_act(obs[0])
        obs, _, _, _ = b.step(act[None], auto_reset=False, decide_every=1)
        frames.append(b.frame(0))
        k += 1
    m = b.meta(0)
    return {
        "config": {f: getattr(cfg, f) for f in CFG_FIELDS},
        "seed": seed,
        "names": names or [f"P{i}" for i in range(len(seats))],
        "meta": dict(meta or {}, progress=[b.progress(0, i) for i in range(len(seats))], seats=list(seats)),
        "needs": m["needs"], "pads": m["pads"], "mine_pos": m["mine_pos"], "mine_type": m["mine_type"],
        "winner": m["winner"], "frames": frames,
    }
