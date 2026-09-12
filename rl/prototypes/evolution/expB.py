"""Baseline imitation league on the existing bot types (no new mechanics)."""
import sys, json, math
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
TYPES = ["solo", "bail", "loyal", "kidnap", "rammer"]
L, N = 24, 6
def gen_round(types, seed):
    cfg = Config()
    b = NativeBatch(L, cfg, seed=seed)
    for e in range(L):
        b.set_seats(e, [TYPES[t] for t in types[e]])
        b.reset(e, seed + e)
    acts = np.zeros((L, N, 4), dtype=np.int32)
    eps = {}
    while len(eps) < L:
        _, _, _, ep = b.step(acts, auto_reset=False)
        for d in ep:
            eps.setdefault(d["env"], d)
    prog = np.array([eps[e]["progress"] for e in range(L)])
    win = np.array([eps[e]["winner"] for e in range(L)])
    return prog, win
def evolve(types, prog, win, rule, rng, mut=0.02, mig=0.1, T=0.05, pcopy=0.3):
    new = types.copy()
    for e in range(L):
        for i in range(N):
            if rule == "W":
                if i != win[e] and rng.random() < pcopy: new[e, i] = types[e, win[e]]
            else:
                j = rng.integers(N - 1); j = j + (j >= i)
                p = 1 / (1 + math.exp(-(prog[e, j] - prog[e, i]) / T))
                if rng.random() < p: new[e, i] = types[e, j]
            if rng.random() < mut: new[e, i] = rng.integers(len(TYPES))
    # migration: swap with a neighbouring lobby
    for e in range(L):
        for i in range(N):
            if rng.random() < mig:
                e2 = (e + rng.choice([-1, 1])) % L; i2 = rng.integers(N)
                new[e, i], new[e2, i2] = new[e2, i2], new[e, i]
    return new
def spatial_ac(frac):
    f = frac - frac.mean()
    if f.std() < 1e-9: return 0.0
    return float((f * np.roll(f, 1)).mean() / (f ** 2).mean())
def run_league(rule, init, gens, seed, mig=0.1, label=""):
    rng = np.random.default_rng(seed)
    if init == "uniform": types = rng.integers(len(TYPES), size=(L, N))
    elif init == "loyal70": types = np.where(rng.random((L, N)) < 0.7, 2, rng.integers(len(TYPES), size=(L, N)))
    elif init == "bail70": types = np.where(rng.random((L, N)) < 0.7, 1, rng.integers(len(TYPES), size=(L, N)))
    hist = []
    for g in range(gens):
        prog, win = gen_round(types, 20000 + 1000 * seed + 37 * g)
        freq = [float((types == k).mean()) for k in range(len(TYPES))]
        loyal_frac = (types == 2).mean(axis=1)
        pay = [float(prog[types == k].mean()) if (types == k).any() else None for k in range(len(TYPES))]
        hist.append(dict(gen=g, freq=freq, pay=pay, mean_prog=float(prog.mean()), ac=spatial_ac(loyal_frac),
                         pure_lobbies=int(sum(len(set(types[e])) == 1 for e in range(L)))))
        if g % 10 == 0 or g == gens - 1:
            print(f"{label} g={g:3d} freq " + " ".join(f"{TYPES[k][:3]}={freq[k]:.2f}" for k in range(5)) +
                  f" | mean_prog={prog.mean():.3f} ac(loyal)={hist[-1]['ac']:.2f} pure={hist[-1]['pure_lobbies']}", flush=True)
        types = evolve(types, prog, win, rule, rng, mig=mig)
    return hist
out = {}
for rule, init, mig in [("W", "uniform", 0.1), ("F", "uniform", 0.1), ("F", "loyal70", 0.1), ("F", "bail70", 0.1), ("F", "uniform", 1.0)]:
    label = f"{rule}_{init}_mig{mig}"
    out[label] = run_league(rule, init, 60, 1, mig=mig, label=label)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expB.json", "w"))
