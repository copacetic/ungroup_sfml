import sys, os, json, math
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
variant = sys.argv[1]
sys.path.insert(0, "rl")
import ungroup.native as native
native.LIB = os.path.join(SC, f"lib_{variant}.so")
from ungroup.native import Config, NativeBatch
LINEUPS = {"12bh": ["bail","bail","bail","bail","solo","loyal"]*2, "20eq": (["solo","bail","loyal"]*7)[:20], "6b3l3": ["bail"]*3+["loyal"]*3, "6loyal": ["loyal"]*6}
ROUNDS = 6
for lk, seats in LINEUPS.items():
    n = len(seats); cfg = Config().replace(n_players=n)
    b = NativeBatch(ROUNDS, cfg, seed=5000)
    for e in range(ROUNDS): b.set_seats(e, seats); b.reset(e, 5000 + n + e)
    acts = np.zeros((ROUNDS, n, 4), dtype=np.int32)
    metas = [b.meta(e) for e in range(ROUNDS)]; done = [False]*ROUNDS
    R = [dict(frames=[]) for _ in range(ROUNDS)]
    while not all(done):
        _, _, _, eps = b.step(acts, auto_reset=False)
        for e in range(ROUNDS):
            if done[e]: continue
            f = b.frame(e); R[e]["frames"].append(f)
        for ep in eps: done[ep["env"]] = True
    agg = dict(chain_max=[], chain_mean=[], outbreak_n=[], outbreak_dur=[], outbreak_peak=[], pads_per_life=[], ang_step=[], fair=[], prog_sd=[], prog_min=[], bank_gap=[])
    for e in range(ROUNDS):
        F = R[e]["frames"]; T = len(F); pads = metas[e]["pads"]; needs = metas[e]["needs"]
        # transmission trees: brand acquisition at step k without own leave -> parent = branded member of the other body merged this step
        parent = {}; depth = {}
        for k in range(1, T):
            evs = F[k]["events"]; leavers = {ev["player"] for ev in evs if ev["kind"] == "leave"}
            merges = [ev for ev in evs if ev["kind"] == "merge"]
            for i in range(n):
                b0 = F[k-1]["players"][i].get("brand", 0); b1 = F[k]["players"][i].get("brand", 0)
                if b1 > b0 + 1e-6:
                    if i in leavers: depth[(k, i)] = 0; continue
                    src = None
                    for ev in merges:
                        A, B = ev["a"], ev["b"]
                        if i in A: others = B
                        elif i in B: others = A
                        else: continue
                        cand = [(F[k-1]["players"][j].get("brand", 0), j) for j in others]
                        cand = [c for c in cand if c[0] > 0]
                        if cand: src = max(cand)[1]
                    d = 0
                    if src is not None:
                        # depth of the source's latest brand
                        ds = [v for (kk, jj), v in depth.items() if jj == src and kk <= k]
                        d = (ds[-1] if ds else 0) + 1
                    depth[(k, i)] = d
        ds = [v for v in depth.values()]
        agg["chain_max"].append(max(ds) if ds else 0); agg["chain_mean"].append(float(np.mean(ds)) if ds else 0)
        # outbreaks: contiguous spans with >= 2 branded players
        cnt = np.array([sum(1 for p in f["players"] if p.get("brand", 0) > 0) for f in F]); dt = F[1]["t"] - F[0]["t"]
        on = cnt >= 2; spans = []; s = None
        for k in range(T):
            if on[k] and s is None: s = k
            if (not on[k] or k == T-1) and s is not None: spans.append((s, k)); s = None
        agg["outbreak_n"].append(len(spans)); agg["outbreak_dur"].append(float(np.mean([(b_-a_)*dt for a_, b_ in spans])) if spans else 0); agg["outbreak_peak"].append(int(cnt.max()))
        # pad migration per group life: consecutive group banks of the same member set
        life = {}
        for k in range(T):
            for ev in F[k]["events"]:
                if ev["kind"] == "bank" and len(ev["group"]) > 1:
                    key = tuple(sorted(ev["group"])); life.setdefault(key, []).append((F[k]["t"], ev["player"]))
        for key, seq in life.items():
            # collapse repeated banks to the same pad within 3 s
            col = [seq[0]]
            for tt, p in seq[1:]:
                if p != col[-1][1] or tt - col[-1][0] > 3: col.append((tt, p))
            agg["pads_per_life"].append(len(set(p for _, p in col)))
            for (t0, p0), (t1, p1) in zip(col, col[1:]):
                a = abs((pads[p0] - pads[p1] + math.pi) % (2*math.pi) - math.pi); agg["ang_step"].append(a); agg["bank_gap"].append(t1 - t0)
        # fairness: receiver has the lowest progress in the group at bank time (using previous frame banked)
        fair = []
        for k in range(1, T):
            for ev in F[k]["events"]:
                if ev["kind"] == "bank" and len(ev["group"]) > 1:
                    pr = {j: np.mean([min(F[k-1]["players"][j]["banked"][t]/needs[j][t], 1) for t in range(4)]) for j in ev["group"]}
                    fair.append(1.0 if pr[ev["player"]] <= min(pr.values()) + 1e-9 else 0.0)
        agg["fair"].append(float(np.mean(fair)) if fair else 0)
        pf = [np.mean([min(F[-1]["players"][i]["banked"][t]/needs[i][t], 1) for t in range(4)]) for i in range(n)]
        agg["prog_sd"].append(float(np.std(pf))); agg["prog_min"].append(float(np.min(pf)))
    print(variant, lk, {k: round(float(np.mean(v)), 3) if v else 0 for k, v in agg.items()}, "pads_per_life max", max(agg["pads_per_life"]) if agg["pads_per_life"] else 0, flush=True)
