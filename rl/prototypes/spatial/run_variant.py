"""Run one rule variant: 6-seat ladder (24 paired rounds) + 20-player crowd statistics (8 rounds).
Usage: UG_LIB=... UG_CXXFLAGS="-DCUR_U0=0.2" python3 run_variant.py NAME"""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # scratch 'ungroup' package (patched loader)
sys.path.insert(1, '/home/user/ungroup_sfml/rl')  # ladder_native
from ungroup.native import NativeBatch, Config
import ladder_native
name = sys.argv[1]
out = {}
t0 = time.time()
# ---------------- ladder
lineups = {
 'six_solo': ['solo']*6, 'six_bail': ['bail']*6, 'six_loyal': ['loyal']*6,
 '3bail_3loyal': ['bail']*3+['loyal']*3, '3loyal_3solo': ['loyal']*3+['solo']*3,
 '2b2l2kidnap': ['bail']*2+['loyal']*2+['kidnap']*2, '2b2l2rammer': ['bail']*2+['loyal']*2+['rammer']*2,
 '1loyal_5bail': ['loyal']+['bail']*5, '2loyal_4solo': ['loyal']*2+['solo']*4,
}
lad = {}
for k, seats in lineups.items():
    r = ladder_native.run(seats, 24, 1000, Config(), quiet=True)
    lad[k] = dict(len=r['length'], early=r['finished_early'], grp=r['avg_group'], merges=r['merges'], leaves=r['leaves'],
                  spills=r['spills'], banks=r['banks'], alliances=r['alliances'], adur=r['alliance_dur'],
                  prog={s: (round(v['progress'],3), round(v['se'],3), round(v['win_per_seat'],2)) for s, v in r['by_type'].items()})
out['ladder'] = lad
# ---------------- crowd stats, 20 players, equal lineup
def crowd(n, E, seed0):
    seats = (['solo','bail','loyal']*8)[:n]
    cfg = Config(n_players=n)
    b = NativeBatch(E, cfg, seed=seed0, decide_every=5)
    for e in range(E): b.set_seats(e, seats); b.reset(e, seed0+e)
    acts = np.zeros((E, n, 4), dtype=np.int32)
    done = [False]*E; eps = {}
    series = [dict(t=[], stock=[], occ=[], L=[], mining=0, nbody=0, frames=0) for _ in range(E)]
    metas = [b.meta(e) for e in range(E)]
    k = 0
    while not all(done):
        _, _, _, ep = b.step(acts, auto_reset=False)
        for d in ep: eps[d['env']] = d; done[d['env']] = True
        k += 1
        if k % 2: continue
        for e in range(E):
            if done[e] and eps[e]['env'] == e and len(series[e]['t']) and series[e]['t'][-1] >= eps[e]['length'] - 0.4: continue
            fr = b.frame(e); mp = np.array(metas[e]['mine_pos'])
            S = series[e]; S['t'].append(fr['t']); S['stock'].append(fr['mines'])
            occ = np.zeros(len(mp)); num = 0.0; den = 0.0
            for bd in fr['bodies']:
                nb = len(bd['m']); r = 0.045*np.sqrt(nb); x = np.array([bd['x'], bd['y']]); v = np.array([bd['vx'], bd['vy']])
                dist = np.linalg.norm(mp - x, axis=1); touching = dist < r + 0.09
                occ[touching] += nb; S['mining'] += nb if touching.any() else 0; S['nbody'] += nb
                sp = np.linalg.norm(v); rr = np.linalg.norm(x)
                if sp > 0.02 and rr > 0.05: num += nb*(x[0]*v[1]-x[1]*v[0]); den += nb*rr*sp
            S['occ'].append(occ.tolist()); S['L'].append(num/den if den > 0 else 0.0); S['frames'] += 1
    res = []
    for e in range(E):
        S = series[e]; t = np.array(S['t']); st = np.array(S['stock']); occ = np.array(S['occ']); L = np.array(S['L'])
        dtf = np.median(np.diff(t)) if len(t) > 1 else 0.33
        # chirality: 30 s windows
        w = max(1, int(round(30/dtf))); wins = [L[i:i+w].mean() for i in range(0, len(L)-w+1, w)]
        chir = float(np.mean(np.abs(wins))) if wins else 0.0
        chir_all = float(abs(L.mean()))
        # neighbour-lag correlation of detrended occupancy between angular neighbours m, m+1
        M = st.shape[1]; lagF = int(round(10/dtf)); wave = []; peak = []
        def dtr(x): return x - x.mean()
        for m in range(M):
            a = dtr(occ[:, m]); c = dtr(occ[:, (m+1) % M])
            if a.std() < 1e-9 or c.std() < 1e-9: continue
            def xc(l):  # corr(a[t], c[t+l])
                if l >= 0: x, y = a[:len(a)-l], c[l:]
                else: x, y = a[-l:], c[:len(c)+l]
                return float(np.corrcoef(x, y)[0, 1]) if len(x) > 10 else 0.0
            wave.append(xc(lagF) - xc(-lagF))
            lags = range(-3*lagF, 3*lagF+1, max(1, lagF//5)); vals = [xc(l) for l in lags]
            j = int(np.argmax(np.abs(vals))); peak.append((list(lags)[j]*dtf, vals[j]))
        # depletion propagation
        dep = []
        for m in range(M):
            s = st[:, m]; idx = np.where((s[1:] < 1.0) & (s[:-1] >= 1.0))[0] + 1
            for i in idx: dep.append((t[i], m))
        prop = 0
        for (ti, m) in dep:
            for mm in ((m+1) % M, (m-1) % M):
                if any(mm == m2 and 0 < t2 - ti <= 20 for (t2, m2) in dep): prop += 1; break
        d = eps[e]
        res.append(dict(seed=seed0+e, length=d['length'], progress=d['progress'], spills=d['spills'], merges=d['merges'], leaves=d['leaves'],
                        banks=d['banks'], chir30=chir, chir_all=chir_all, wave=float(np.mean(wave)) if wave else 0.0,
                        wave_abs=float(np.mean(np.abs(wave))) if wave else 0.0, peak_lag=float(np.mean([abs(p[0]) for p in peak])) if peak else 0.0,
                        peak_corr=float(np.mean([p[1] for p in peak])) if peak else 0.0, depletions=len(dep), dep_prop=prop/len(dep) if dep else 0.0,
                        mean_stock=float(st.mean()), mining_frac=S['mining']/max(1, S['nbody'])))
    return res
out['crowd20'] = crowd(20, 8, 5000)
out['crowd12'] = crowd(12, 8, 5000)
out['secs'] = time.time() - t0
json.dump(out, open(os.path.join(HERE, f'res_{name}.json'), 'w'), indent=1)
print(name, 'done in %.0f s' % out['secs'])
