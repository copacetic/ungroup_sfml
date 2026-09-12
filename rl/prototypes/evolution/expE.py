"""Variable-density league on the UNMODIFIED rules: birth on high progress, death on low progress, Fermi copying within lobby.
Lobbies are independent arenas of variable size n in [3, 32]."""
import sys, json, math, time
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
TYPES = ["solo", "bail", "loyal"]
L = 8
B_THR, D_THR, MUT, T = 0.75, 0.30, 0.03, 0.05
def play(lobby, seed):
    n = len(lobby)
    b = NativeBatch(1, Config().replace(n_players=n), seed=seed)
    b.set_seats(0, [TYPES[t] for t in lobby]); b.reset(0, seed)
    acts = np.zeros((1, n, 4), dtype=np.int32)
    while True:
        _, _, _, ep = b.step(acts, auto_reset=False)
        if ep: return ep[0]
def step_lobby(lobby, ep, rng):
    prog = np.array(ep["progress"]); n = len(lobby)
    new = list(lobby)
    for i in range(n):  # Fermi copy of a random lobby-mate by progress
        if n < 2: break
        j = rng.integers(n - 1); j = j + (j >= i)
        if rng.random() < 1 / (1 + math.exp(-(prog[j] - prog[i]) / T)): new[i] = lobby[j]
    births = [new[i] for i in range(n) if prog[i] >= B_THR]
    survivors = [new[i] for i in range(n) if prog[i] >= D_THR]
    out = survivors + births
    out = [rng.integers(len(TYPES)) if rng.random() < MUT else t for t in out]
    while len(out) < 3: out.append(rng.integers(len(TYPES)))
    if len(out) > 32: out = list(rng.permutation(out)[:32])
    return [int(t) for t in out], len(births), n - len(survivors)
def main(gens=45, seed=7, init="uniform6"):
    rng = np.random.default_rng(seed)
    lobbies = [[int(x) for x in rng.integers(len(TYPES), size=6)] for _ in range(L)]
    hist = []
    t0 = time.time()
    for g in range(gens):
        eps = [play(lob, 40000 + 1000 * seed + 31 * g + e) for e, lob in enumerate(lobbies)]
        allt = np.concatenate([np.array(l) for l in lobbies]); allp = np.concatenate([np.array(e["progress"]) for e in eps])
        freq = [float((allt == k).mean()) for k in range(3)]
        pay = [float(allp[allt == k].mean()) if (allt == k).any() else float("nan") for k in range(3)]
        sizes = [len(l) for l in lobbies]
        loyal_by_lobby = [float(np.mean(np.array(l) == 2)) for l in lobbies]
        nb = nd = 0
        newl = []
        for lob, ep in zip(lobbies, eps):
            nl, b_, d_ = step_lobby(lob, ep, rng); newl.append(nl); nb += b_; nd += d_
        hist.append(dict(gen=g, freq=freq, pay=pay, sizes=sizes, loyal_by_lobby=loyal_by_lobby, births=nb, deaths=nd,
                         early=float(np.mean([not e["timeout_win"] for e in eps])), mean_prog=float(allp.mean()),
                         spills=float(np.mean([e["spills"] for e in eps]))))
        print(f"g={g:3d} N={sum(sizes):3d} sizes={sizes} freq sol={freq[0]:.2f} bai={freq[1]:.2f} loy={freq[2]:.2f} pay={['%.2f'%p for p in pay]} "
              f"births={nb} deaths={nd} early={hist[-1]['early']:.2f} spills={hist[-1]['spills']:.1f} [{time.time()-t0:.0f}s]", flush=True)
        lobbies = newl
    json.dump(hist, open(f"/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expE_{seed}.json", "w"))
main(gens=int(sys.argv[1]) if len(sys.argv) > 1 else 45, seed=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
