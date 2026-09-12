import json, math, itertools, csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fsolve(g, u0, full_output=True):
    u = np.array(u0, float)
    for it in range(60):
        r = np.array(g(u))
        if np.abs(r).max() < 1e-12: return u, None, 1, ""
        J = np.zeros((2, 2)); h = 1e-7
        for k in range(2):
            d = np.zeros(2); d[k] = h; J[:, k] = (np.array(g(u + d)) - r) / h
        try: du = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError: return u, None, 0, ""
        u = u + du
        if not (np.isfinite(u).all() and u.min() > -0.5 and u.max() < 1.5): return u, None, 0, ""
    return u, None, (1 if np.abs(np.array(g(u))).max() < 1e-9 else 0), ""

D = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/payoff-landscape/"
T = ["solo", "bail", "loyal"]
rowsA = json.load(open(D + "compositions.json"))
rowsB = json.load(open(D + "compositions_seed3000.json"))

def table(rows):
    P = {t: {} for t in T}; SE = {t: {} for t in T}
    for r in rows:
        if r["kind"] != "comp": continue
        key = (r["n_solo"], r["n_bail"], r["n_loyal"])
        for t in T:
            if r[f"{t}_prog"] != "":
                P[t][key] = float(r[f"{t}_prog"]); SE[t][key] = float(r[f"{t}_se"])
    return P, SE

PA, SA = table(rowsA); PB, SB = table(rowsB)
PP = {t: {k: 0.5 * (PA[t][k] + PB[t][k]) for k in PA[t]} for t in T}
SP = {t: {k: 0.5 * math.hypot(SA[t][k], SB[t][k]) for k in PA[t]} for t in T}

# ---- pooled composition table CSV
with open(D + "compositions_pooled32.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["n_solo", "n_bail", "n_loyal"] + [f"{t}_{m}" for t in T for m in ("prog", "se")] + ["seedA_" + t for t in T] + ["seedB_" + t for t in T])
    for k in sorted(PA["solo"].keys() | PA["bail"].keys() | PA["loyal"].keys(), key=lambda k: (-k[0], -k[1])):
        w.writerow(list(k) + [f"{PP[t][k]:.3f}" if k in PP[t] else "" for t in T for _ in (0,)] and
                   list(k) + sum([[f"{PP[t][k]:.3f}", f"{SP[t][k]:.3f}"] if k in PP[t] else ["", ""] for t in T], [])
                   + [f"{PA[t][k]:.3f}" if k in PA[t] else "" for t in T] + [f"{PB[t][k]:.3f}" if k in PB[t] else "" for t in T])

# ---- multinomial (random-matching) payoff: focal type i, 5 co-players ~ Multinomial(5, x)
COMPS5 = [(a, b, 5 - a - b) for a in range(6) for b in range(6 - a)]
def multinom(c, x):
    a, b, l = c
    return math.factorial(5) / (math.factorial(a) * math.factorial(b) * math.factorial(l)) * x[0] ** a * x[1] ** b * x[2] ** l
def payoff(P, x):
    out = np.zeros(3)
    for i, t in enumerate(T):
        e = [0, 0, 0]; e[i] = 1
        out[i] = sum(multinom(c, x) * P[t][(c[0] + e[0], c[1] + e[1], c[2] + e[2])] for c in COMPS5)
    return out
def step(P, x):
    f = payoff(P, x); m = float(x @ f)
    y = x * f / m
    return y / y.sum()
def run(P, x0, gens=200):
    x = np.array(x0, float); x = x / x.sum(); traj = [x.copy()]
    for _ in range(gens):
        x = step(P, x); traj.append(x.copy())
    return np.array(traj)

def bary_to_xy(x):  # solo left, bail right, loyal top
    return x[..., 1] + 0.5 * x[..., 2], x[..., 2] * math.sqrt(3) / 2

def fixed_points(P):
    fps = []
    # vertices
    for i in range(3):
        v = np.zeros(3); v[i] = 1; fps.append(v)
    # edges: find sign changes of payoff difference along each edge
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        ts = np.linspace(0.001, 0.999, 999); d = []
        for t in ts:
            x = np.zeros(3); x[i] = 1 - t; x[j] = t
            f = payoff(P, x); d.append(f[j] - f[i])
        d = np.array(d)
        for k in range(len(ts) - 1):
            if d[k] * d[k + 1] < 0:
                t = ts[k] - d[k] * (ts[k + 1] - ts[k]) / (d[k + 1] - d[k])
                x = np.zeros(3); x[i] = 1 - t; x[j] = t; fps.append(x)
    # interior: solve f_solo = f_bail = f_loyal from grid seeds
    def g(u):
        x = np.array([u[0], u[1], 1 - u[0] - u[1]]); f = payoff(P, x); return [f[0] - f[1], f[1] - f[2]]
    seen = []
    for a in np.linspace(0.05, 0.95, 19):
        for b in np.linspace(0.05, 0.95, 19):
            if a + b >= 0.98: continue
            sol, info, ier, _ = fsolve(g, [a, b], full_output=True)
            if ier == 1 and sol.min() > 1e-3 and sol.sum() < 1 - 1e-3 and np.abs(g(sol)).max() < 1e-9:
                x = np.array([sol[0], sol[1], 1 - sol.sum()])
                if all(np.abs(x - s).max() > 1e-4 for s in seen):
                    seen.append(x)
    fps += seen
    return fps

def stability(P, x):
    # eigenvalues of the map Jacobian restricted to the simplex tangent space (numerical); |lambda|<1 stable
    h = 1e-6; J = np.zeros((3, 3)); fx = step(P, x)
    for k in range(3):
        d = np.zeros(3); d[k] = h
        xp = np.clip(x + d, 0, None); xp = xp / xp.sum()
        J[:, k] = (step(P, xp) - fx) / h
    # project onto tangent directions
    B = np.array([[1, -1, 0], [0, 1, -1]], float).T
    M = np.linalg.lstsq(B, J @ B, rcond=None)[0]
    ev = np.linalg.eigvals(M)
    return ev

def analyse(P, label):
    print(f"\n===== replicator dynamics, table = {label} =====")
    fps = fixed_points(P)
    out = dict(label=label, fixed=[])
    for x in fps:
        ev = stability(P, x); f = payoff(P, x)
        kind = "stable" if np.all(np.abs(ev) < 1 - 1e-9) else ("unstable/saddle")
        # for vertices eigenvalue may be exactly 1 in a neutral direction; also report invasion fitness
        inv = ""
        if x.max() > 0.999:
            i = int(x.argmax()); inv = " invasion fitness: " + " ".join(f"{T[j]} {f[j]-f[i]:+.3f}" for j in range(3) if j != i)
        print(f"  fixed point solo={x[0]:.3f} bail={x[1]:.3f} loyal={x[2]:.3f}  payoffs {np.round(f,3)}  eig|.|={np.round(np.abs(ev),3)} -> {kind}{inv}")
        out["fixed"].append(dict(x=x.tolist(), payoff=f.tolist(), eig_abs=np.abs(ev).tolist(), kind=kind))
    # starts
    starts = [(0.98, 0.01, 0.01), (0.01, 0.98, 0.01), (0.01, 0.01, 0.98), (0.9, 0.1, 0.0), (0.9, 0.0, 0.1), (0.1, 0.9, 0.0), (0.0, 0.9, 0.1),
              (0.1, 0.0, 0.9), (0.0, 0.1, 0.9), (1/3, 1/3, 1/3), (0.6, 0.2, 0.2), (0.2, 0.6, 0.2), (0.2, 0.2, 0.6), (0.45, 0.45, 0.1), (0.1, 0.45, 0.45), (0.45, 0.1, 0.45),
              (0.0, 0.5, 0.5), (0.0, 0.4, 0.6), (0.0, 0.6, 0.4), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)]
    trajs = {}
    print("  start -> gen 200 state (moved in last gen)")
    for s in starts:
        tr = run(P, s); trajs[s] = tr
        mv = np.abs(tr[-1] - tr[-2]).max()
        print(f"   {np.round(s,3)} -> {np.round(tr[-1],3)} (|dx| last gen {mv:.1e}) gen50 {np.round(tr[50],3)}")
    out["starts"] = {str(s): trajs[s][-1].tolist() for s in starts}
    # basins on a fine simplex grid
    N = 60; ends = []; pts = []; conv = 0
    for a in range(N + 1):
        for b in range(N + 1 - a):
            x0 = np.array([a, b, N - a - b], float) / N
            if x0.min() <= 0:  # edges: keep as they are (types absent stay absent), but use tiny mass to probe interior basin instead
                pass
            tr = run(P, x0)
            ends.append(tr[-1]); pts.append(x0)
            if np.abs(tr[-1] - tr[-2]).max() < 1e-6: conv += 1
    ends = np.array(ends); pts = np.array(pts)
    lab = ends.argmax(axis=1); pure = ends.max(axis=1) > 0.99
    frac = {T[i]: float(np.mean((lab == i) & pure)) for i in range(3)}
    frac["mixed"] = float(np.mean(~pure))
    print(f"  basins over {len(pts)} grid points (step 1/{N}): {frac}; converged (|dx|<1e-6 at gen 200): {conv}/{len(pts)}")
    inter = pts[pts.min(axis=1) > 0]; labi = lab[pts.min(axis=1) > 0]; purei = pure[pts.min(axis=1) > 0]
    fraci = {T[i]: float(np.mean((labi == i) & purei)) for i in range(3)}; fraci["mixed"] = float(np.mean(~purei))
    print(f"  basins, interior grid points only ({len(inter)}): {fraci}")
    out["basins_all"] = frac; out["basins_interior"] = fraci
    # separatrix on the bail-loyal edge and where solo-heavy starts go
    return out, trajs, pts, ends, lab, pure

def plot(P, label, trajs, pts, ends, lab, pure, fname):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    ax = axs[0]
    cols = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}
    X, Y = bary_to_xy(pts)
    c = [cols[l] if p else "#999999" for l, p in zip(lab, pure)]
    ax.scatter(X, Y, c=c, s=14, marker="h", alpha=0.6, linewidths=0)
    for s, tr in trajs.items():
        x, y = bary_to_xy(tr); ax.plot(x, y, "k-", lw=0.8, alpha=0.7)
        ax.plot(x[0], y[0], "ko", ms=3); ax.plot(x[-1], y[-1], "k*", ms=7)
    for x in fixed_points(P):
        xx, yy = bary_to_xy(x); ev = stability(P, x)
        ax.plot(xx, yy, "o", ms=11, mfc=("white" if np.any(np.abs(ev) > 1) else "black"), mec="black", mew=1.5)
    ax.text(-0.03, -0.04, "SOLO", fontsize=12, weight="bold"); ax.text(0.95, -0.04, "BAIL", fontsize=12, weight="bold"); ax.text(0.44, 0.89, "LOYAL", fontsize=12, weight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Replicator basins, {label}\nblue->solo, red->bail, green->loyal, grey=mixed at gen 200; lines: 21 trajectories (dot=start, star=gen 200); circles: fixed points (filled=stable)", fontsize=8)
    ax = axs[1]
    tr = trajs[(1/3, 1/3, 1/3)]
    for i in range(3): ax.plot(tr[:, i], color=cols[i], label=T[i], lw=2)
    tr2 = trajs[(0.6, 0.2, 0.2)]
    for i in range(3): ax.plot(tr2[:, i], color=cols[i], ls="--", lw=1.2)
    tr3 = trajs[(0.2, 0.2, 0.6)]
    for i in range(3): ax.plot(tr3[:, i], color=cols[i], ls=":", lw=1.2)
    ax.set_xlabel("generation"); ax.set_ylabel("population share"); ax.set_ylim(0, 1); ax.legend()
    ax.set_title("Time series: solid start (1/3,1/3,1/3); dashed (0.6 solo,0.2,0.2); dotted (0.2,0.2,0.6 loyal)", fontsize=9)
    fig.tight_layout(); fig.savefig(fname, dpi=130); plt.close(fig)

results = {}
for P, label, fname in [(PP, "pooled 32 rounds (seeds 1000+3000)", "replicator_pooled.png"), (PA, "seed 1000, 16 rounds", "replicator_seed1000.png"), (PB, "seed 3000, 16 rounds", "replicator_seed3000.png")]:
    out, trajs, pts, ends, lab, pure = analyse(P, label)
    plot(P, label, trajs, pts, ends, lab, pure, D + fname)
    results[label] = out
    if "pooled" in label:
        with open(D + "timeseries_mixed_start.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["gen", "solo", "bail", "loyal", "pay_solo", "pay_bail", "pay_loyal"])
            for g, x in enumerate(trajs[(1/3, 1/3, 1/3)]):
                w.writerow([g] + [f"{v:.4f}" for v in x] + [f"{v:.4f}" for v in payoff(P, x)])

# ---- payoff as a function of the mix, along edges and at the centroid (pooled), for the report
print("\n===== pooled payoff along the edges (multinomial random matching) =====")
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    print(f"  edge {T[i]}->{T[j]}:")
    for t in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = np.zeros(3); x[i] = 1 - t; x[j] = t; f = payoff(PP, x)
        print(f"    share_{T[j]}={t:.1f}: {T[i]} {f[i]:.3f}  {T[j]} {f[j]:.3f}  ({T[3-i-j]} would get {f[3-i-j]:.3f})")
x = np.ones(3) / 3; print("  centroid payoffs", np.round(payoff(PP, x), 3))

# ---- exploitability of pure lobbies: single invader among 5 residents
print("\n===== exploitability of pure lobbies (1 invader + 5 residents) =====")
inv = {}
for res in T:
    marg = {}
    for r in rowsA:
        seats = r["seats"].split(",")
        if seats.count(res) == 5:
            other = [s for s in seats if s != res][0]
            marg[other] = float(r[f"{other}_prog"]) - float(r[f"{res}_prog"])
            # add seed-3000 replicate for solo/bail/loyal invaders
            for r2 in rowsB:
                if r2["seats"] == r["seats"]:
                    marg[other] = (marg[other] + float(r2[f"{other}_prog"]) - float(r2[f"{res}_prog"])) / 2
    best = max(marg, key=marg.get)
    inv[res] = dict(margins=marg, best=best, gap=marg[best])
    print(f"  pure {res}: " + " ".join(f"{k} {v:+.3f}" for k, v in marg.items()) + f"  -> best invader {best}, gap {marg[best]:+.3f}")
print("  (solo/bail/loyal invader margins are pooled over both seeds = 32 rounds; kidnap/rammer are 16 rounds, seed 1000)")
print("\n===== two invaders (kidnap + rammer) into pure lobbies =====")
for r in rowsA:
    if r["kind"] == "inv2":
        print("  " + r["seats"] + ": " + " ".join(f"{t} {float(r[f'{t}_prog']):.3f} (win/seat {float(r[f'{t}_win']):.2f})" for t in ["kidnap", "rammer", "solo", "bail", "loyal"] if r[f"{t}_prog"] != ""))
json.dump(dict(results=results, exploitability=inv), open(D + "replicator_results.json", "w"), indent=1)
