import sys, gzip, pickle, json, numpy as np
S = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"
def analyze(name, lineups):
    rec = pickle.load(gzip.open(f"{S}/rec_{name}.pkl.gz"))
    for ln in lineups:
        if ln not in rec: continue
        R = rec[ln]; K = R["cfg"].get("mine_cap", 30.0); G = len(R["ts"])
        rec_full, rec_dead = [], []; dorm_pairs = []; dorm_chance = []; front_same = []; arrival = []; seat_arr = {}
        cluster_runs = []
        for e in range(G):
            t = np.array(R["ts"][e]); st = np.array(R["stocks"][e]); al = np.array(R["alive"][e], dtype=bool)
            nm = st.shape[1]; dt = t[1]-t[0]
            below = (st < 1.0) & al
            evs = []
            for m in range(nm):
                bm = below[:, m]
                idx = np.where(bm[1:] & ~bm[:-1])[0] + 1
                for i in idx:
                    nb = [(m-1) % nm, (m+1) % nm]
                    nbs = np.mean([st[i, j] / K if al[i, j] else 0.0 for j in nb])
                    j = i
                    while j < len(t) and st[j, m] < 0.5*K and al[j, m]: j += 1
                    if j < len(t) and al[j, m] and t[i] < 150:
                        (rec_full if nbs >= 0.5 else rec_dead).append(t[j] - t[i])
                    evs.append((t[i], m))
            evs.sort()
            # front persistence: consecutive neighbour steps in the same rotational direction
            steps = []
            for (t1, m1), (t2, m2) in zip(evs[:-1], evs[1:]):
                if t2 - t1 < 30 and (m2 - m1) % nm in (1, nm-1): steps.append((t2, +1 if (m2-m1) % nm == 1 else -1))
                else: steps.append((t2, 0))
            for (ta, a), (tb, b) in zip(steps[:-1], steps[1:]):
                if a != 0 and b != 0: front_same.append(1.0 if a == b else 0.0)
            # dormancy clustering on the ring (t < 190 s, live mines only)
            sel = (t < 190)
            d = below[sel]; a = al[sel]
            for fi in range(0, d.shape[0], 5):
                live = np.where(a[fi])[0]
                if len(live) < 4 or d[fi].sum() < 1: continue
                p = d[fi][live].mean(); dorm_chance.append(p)
                pairs = [(m, (m+1) % nm) for m in live if a[fi][(m+1) % nm]]
                if pairs: dorm_pairs.append(np.mean([d[fi][m1] and d[fi][m2] for m1, m2 in pairs]))
                # run lengths of dormant mines around the ring
                ring = [int(d[fi][m]) for m in range(nm)]
                runs = []; cur = 0
                for v in ring + ring[:1]:
                    if v: cur += 1
                    else:
                        if cur: runs.append(cur); cur = 0
                if runs: cluster_runs.append(max(runs))
            # arrival stock: stock at the frame a body first touches a mine (per player), vs progress
            mpos = np.array(R["meta"][e]["mine_pos"]); last = {}
            for fi, frame in enumerate(R["bodies"][e]):
                if t[fi] > 190: break
                for (mem, bx, by, pool) in frame:
                    rr = 0.045*np.sqrt(len(mem)) + 0.08 + 0.01
                    dd = np.hypot(mpos[:,0]-bx, mpos[:,1]-by); mm = int(np.argmin(dd)) if dd.min() < rr else -1
                    for p in mem:
                        if mm >= 0 and last.get(p, -1) != mm:
                            x = st[fi, mm] / K; arrival.append(x); seat_arr.setdefault((e, p), []).append(x)
                        last[p] = mm
        # correlation of a seat's mean arrival stock with its final progress
        xs, ys = [], []
        for (e, p), v in seat_arr.items():
            xs.append(np.mean(v)); ys.append(R["eps"][e]["progress"][p])
        corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 3 else None
        pc = float(np.mean(dorm_chance)) if dorm_chance else 0
        print(f"{name:8s} {ln:12s} recov_full_nbrs {np.median(rec_full) if rec_full else None} (n={len(rec_full)})  recov_dead_nbrs {np.median(rec_dead) if rec_dead else None} (n={len(rec_dead)}) | "
              f"dormant-pair frac {np.mean(dorm_pairs) if dorm_pairs else 0:.3f} vs independent {pc*pc:.3f} (p={pc:.3f}) | max dead run {np.mean(cluster_runs) if cluster_runs else 0:.2f} | "
              f"front same-direction {np.mean(front_same) if front_same else None} (n={len(front_same)}) | arrival stock mean {np.mean(arrival):.2f} sd {np.std(arrival):.2f} | corr(arrival stock, progress) {corr}")
if __name__ == "__main__":
    L = ["6loyal", "3bail3loyal", "6bail", "12equal", "12loyal"]
    for nm in sys.argv[1:]: analyze(nm, L)
