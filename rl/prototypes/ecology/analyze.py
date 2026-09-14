import sys, gzip, pickle, json, numpy as np
S = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"

def acf_period(x, dt):
    x = x - x.mean()
    if x.std() < 1e-6: return None, 0.0
    n = len(x); ac = np.correlate(x, x, "full")[n-1:] / (x.var() * n)
    # first minimum then next maximum
    i = 1
    while i < n-1 and ac[i+1] <= ac[i]: i += 1
    j = i
    while j < n-1 and ac[j+1] >= ac[j]: j += 1
    if j >= n-1 or j == i: return None, 0.0
    return j*dt, float(ac[j])

def analyze(name):
    rec = pickle.load(gzip.open(f"{S}/rec_{name}.pkl.gz"))
    res = {}
    for ln, R in rec.items():
        K = R["cfg"].get("mine_cap", 30.0)
        G = len(R["ts"])
        dorm = []; dep = []; firstdep = []; bloom = []; meanstock = [[],[],[]]; periods = []; acpeaks = []
        recov = []; nbr_succ = []; lagpeaks = []; switches = []; inner_max = []; inner_at192 = []; outer_at150 = []
        cv = []; asym = []
        for e in range(G):
            t = np.array(R["ts"][e]); st = np.array(R["stocks"][e]); al = np.array(R["alive"][e], dtype=bool)
            dt = t[1]-t[0] if len(t) > 1 else 0.2
            live = al
            x = st / K
            dorm.append((x[live] < 1.0/K).mean())
            bloom.append((x * live).__ge__(0.5).sum(1).mean())
            for k,(a,b) in enumerate([(0,120),(120,192),(192,240)]):
                sel = (t>=a)&(t<b)
                if sel.any():
                    m = x[sel]; l = live[sel]
                    if l.any(): meanstock[k].append(m[l].mean())
            # depletion events (S crosses below 1 from above)
            below = st < 1.0
            evs = []
            for m in range(st.shape[1]):
                bm = below[:, m] & al[:, m]
                idx = np.where(bm[1:] & ~bm[:-1])[0] + 1
                for i in idx:
                    evs.append((t[i], m))
                    # recovery time to 0.5K
                    j = i
                    while j < len(t) and (st[j, m] < 0.5*K) and al[j, m]: j += 1
                    if j < len(t) and al[j, m]: recov.append(t[j]-t[i])
            dep.append(len(evs)); firstdep.append(min([tt for tt,_ in evs]) if evs else np.nan)
            evs.sort()
            nm = st.shape[1]
            for (t1,m1),(t2,m2) in zip(evs[:-1], evs[1:]):
                if t2 - t1 < 30 and m1 != m2:
                    nbr_succ.append(1.0 if (m2-m1) % nm in (1, nm-1) else 0.0)
            # per-mine period from detrended stock
            for m in range(nm):
                sel = al[:, m]
                if sel.sum() < 200: continue
                xs = x[sel, m]
                if xs.min() < 0.5:
                    p, pk = acf_period(xs, dt)
                    if p: periods.append(p); acpeaks.append(pk)
                cv.append(xs.std())
            # neighbour lag correlation of dS (wave signature): ring neighbours m, m+1
            lags = np.arange(-100, 101)  # in frames (0.2 s)
            for m in range(nm):
                m2 = (m+1) % nm
                sel = al[:, m] & al[:, m2]
                if sel.sum() < 400: continue
                a = np.diff(st[sel, m]); b = np.diff(st[sel, m2])
                if a.std() < 1e-6 or b.std() < 1e-6: continue
                a = (a-a.mean())/a.std(); b = (b-b.mean())/b.std()
                cc = [np.mean(a[max(0,-L):len(a)-max(0,L)] * b[max(0,L):len(b)-max(0,-L)]) for L in lags]
                cc = np.array(cc); L = lags[np.argmax(np.abs(cc))]
                lagpeaks.append((L*dt, cc[np.argmax(np.abs(cc))]))
                asym.append(np.abs(cc[lags>0]).max() - np.abs(cc[lags<0]).max())
            # mine switches per player
            mpos = np.array(R["meta"][e]["mine_pos"])
            last = {}; sw = 0
            for fi, frame in enumerate(R["bodies"][e]):
                for (mem, bx, by, pool) in frame:
                    rr = 0.045*np.sqrt(len(mem)) + 0.08 + 0.01
                    d = np.hypot(mpos[:,0]-bx, mpos[:,1]-by)
                    mm = int(np.argmin(d)) if d.min() < rr else -1
                    if mm < 0: continue
                    for p in mem:
                        if p in last and last[p] != mm: sw += 1
                        last[p] = mm
            switches.append(sw / len(R["seats"]))
            inner = [m for m in range(nm) if np.hypot(*mpos[m]) < 0.5]; outer = [m for m in range(nm) if np.hypot(*mpos[m]) >= 0.5]
            sel = (t >= 120) & (t < 192)
            if sel.any(): inner_max.append(x[sel][:, inner].max())
            i192 = np.searchsorted(t, 191.5)
            if i192 < len(t): inner_at192.append(x[i192][inner].mean())
            i150 = np.searchsorted(t, 150)
            if i150 < len(t): outer_at150.append(x[i150][outer].mean())
        r = dict(
            games=G, length=float(np.mean([len(x) for x in R["ts"]])*0.2),
            dormant_frac=float(np.mean(dorm)), bloom_count=float(np.mean(bloom)),
            mean_stock_0_120=float(np.mean(meanstock[0])) if meanstock[0] else None,
            mean_stock_120_192=float(np.mean(meanstock[1])) if meanstock[1] else None,
            mean_stock_192_240=float(np.mean(meanstock[2])) if meanstock[2] else None,
            depletions_per_round=float(np.mean(dep)), first_depletion=float(np.nanmean(firstdep)) if not all(np.isnan(firstdep)) else None,
            recovery_to_half_K_s=(float(np.median(recov)), float(np.percentile(recov, 90))) if recov else None,
            n_recov=len(recov),
            successive_depletions_on_ring_neighbour=(float(np.mean(nbr_succ)), len(nbr_succ)) if nbr_succ else None,
            chance_neighbour=2.0/7,
            stock_period_s=(float(np.median(periods)), float(np.mean(acpeaks)), len(periods)) if periods else None,
            stock_sd_over_K=float(np.mean(cv)) if cv else None,
            nbr_lag_peak=(float(np.median([l for l,_ in lagpeaks])), float(np.mean([abs(c) for _,c in lagpeaks]))) if lagpeaks else None,
            nbr_lag_asym=float(np.mean(asym)) if asym else None,
            mine_switches_per_player=float(np.mean(switches)),
            inner_max_over_K_autumn=float(np.mean(inner_max)) if inner_max else None,
            inner_stock_over_K_at_192=float(np.mean(inner_at192)) if inner_at192 else None,
            outer_stock_over_K_at_150=float(np.mean(outer_at150)) if outer_at150 else None,
            progress=float(np.mean([np.mean(ep["progress"]) for ep in R["eps"]])),
            banks=float(np.mean([ep["banks"] for ep in R["eps"]])),
        )
        res[ln] = r
        print(f"--- {name} {ln}")
        for k, v in r.items(): print(f"   {k}: {v}")
    json.dump(res, open(f"{S}/analysis_{name}.json", "w"), indent=1)

if __name__ == "__main__":
    for nm in sys.argv[1:]: analyze(nm)
