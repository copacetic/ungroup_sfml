import ctypes, os, sys
sys.path.insert(0, "rl")
import ungroup.native as base
D = os.path.dirname(os.path.abspath(__file__))
base.SRC = os.path.join(D, "native", "ungroup_ev.cpp")
base.LIB = os.path.join(D, "native", "libungroup_ev.so")
base.SEAT_NAMES.update({"grudge": 7, "cash": 8})
base.SEAT_LABEL.update({7: "grudge", 8: "cash"})
from ungroup.native import Config, NativeBatch, SEAT_NAMES  # noqa
def _ext(lib):
    P = ctypes.POINTER
    lib.ugb_set_inherit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, P(ctypes.c_double)]
    lib.ugb_set_evo.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.ugb_clear_history.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ugb_start_progress.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.ugb_start_progress.restype = ctypes.c_double
class EvoBatch(NativeBatch):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); _ext(self.lib)
    def set_inherit(self, env, seat, v):
        arr = (ctypes.c_double * 4)(*[float(x) for x in v]); self.lib.ugb_set_inherit(self.h, env, seat, arr)
    def set_evo(self, env, inherit_cap=0.5, persist=False, ledger_decay=0.5, grudge_window=500.0, cash_at=0.4):
        self.lib.ugb_set_evo(self.h, env, inherit_cap, 1 if persist else 0, ledger_decay, grudge_window, cash_at)
    def clear_history(self, env, seat): self.lib.ugb_clear_history(self.h, env, seat)
    def start_progress(self, env, seat): return self.lib.ugb_start_progress(self.h, env, seat)
def _ext2(lib):
    P = ctypes.POINTER
    lib.ugb_set_mines.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int]
    lib.ugb_remap_history.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, P(ctypes.c_int), ctypes.c_int, ctypes.c_double]
class EvoBatch2(EvoBatch):
    def __init__(self, *a, **k):
        super().__init__(*a, **k); _ext2(self.lib)
    def set_mines(self, env, mine_diff=0.0, persist_mines=False): self.lib.ugb_set_mines(self.h, env, mine_diff, 1 if persist_mines else 0)
    def remap_history(self, old, env_old, env_new, mapping, t_shift):
        arr = (ctypes.c_int * len(mapping))(*[int(m) for m in mapping])
        self.lib.ugb_remap_history(old.h, env_old, self.h, env_new, arr, len(mapping), t_shift)
