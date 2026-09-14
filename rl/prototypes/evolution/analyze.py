"""Per-run and per-lobby dynamics statistics for the density leagues (expF_*.json)."""
import json, sys, glob, os
import numpy as np
D = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution"


def acf(x, lag):
    x = np.asarray(x, float); x = x - x.mean()
    if x.std() < 1e-9 or lag >= len(x): return 0.0
    return float((x[:-lag] * x[lag:]).mean() / (x ** 2).mean())


def dominant_period(x):
    x = np.asarray(x, float) - np.mean(x)
    if x.std() < 1e-9: return None
    f = np.abs(np.fft.rfft(x)) ** 2
    f[0] = 0
    k = int(np.argmax(f[1:]) + 1)
    return len(x) / k


def analyze(path):
    h = json.load(open(path))
    G = len(h)
    types = None
    coop = np.array([r["coop_by_lobby"] for r in h])  # (G, L)
    N = np.array([r["N"] for r in h])
    freq = np.array([r["freq"] for r in h])
    L = coop.shape[1]
    # per-lobby regime switches with hysteresis (0.35 / 0.65)
    switches = 0; ext = 0; recol = 0
    for l in range(L):
        s = None
        for g in range(G):
            c = coop[g, l]
            if c >= 0.65 and s != "c": switches += (s is not None); s = "c"
            elif c <= 0.35 and s != "b": switches += (s is not None); s = "b"
        z = coop[:, l] == 0
        for g in range(1, G):
            if z[g] and not z[g - 1]: ext += 1
            if (not z[g]) and z[g - 1] and coop[g, l] > 0: recol += 1
    spatial = []
    for g in range(G):
        f = coop[g] - coop[g].mean()
        spatial.append(0.0 if f.std() < 1e-9 else float((f * np.roll(f, 1)).mean() / (f ** 2).mean()))
    coopfrac = coop.mean(axis=1)
    last = slice(G // 2, G)
    out = dict(run=os.path.basename(path), gens=G, L=L,
               coop_mean_2nd_half=float(coopfrac[last].mean()), coop_min_2nd_half=float(coopfrac[last].min()), coop_max_2nd_half=float(coopfrac[last].max()),
               coop_sd_2nd_half=float(coopfrac[last].std()),
               bail_min_2nd_half=float(freq[last, 1].min()), bail_max_2nd_half=float(freq[last, 1].max()),
               N_mean_2nd_half=float(N[last].mean()), N_sd_2nd_half=float(N[last].std()),
               N_acf=[round(acf(N[last], k), 2) for k in (1, 2, 3, 5, 8)], coop_acf=[round(acf(coopfrac[last], k), 2) for k in (1, 2, 3, 5, 8)],
               N_period=dominant_period(N[last]), coop_period=dominant_period(coopfrac[last]),
               lobby_switches_per_lobby=switches / L, local_extinctions=ext, recolonisations=recol,
               spatial_ac_mean=float(np.mean(spatial[G // 4:])), lobby_coop_sd_mean=float(coop[last].std(axis=1).mean()),
               fixation_gen=next((g for g in range(G) if coopfrac[g] < 0.02 or coopfrac[g] > 0.98), None))
    return out, h


if __name__ == "__main__":
    for p in sorted(glob.glob(f"{D}/expF_*.json")):
        o, h = analyze(p)
        print(json.dumps(o))
        # trace of coop fraction and N every 5 gens
        cf = [round(float(np.mean(r["coop_by_lobby"])), 2) for r in h]
        print("  coop:", cf[::5])
        print("  N   :", [r["N"] for r in h][::5])
        print("  bail:", [round(r["freq"][1], 2) for r in h][::5])
