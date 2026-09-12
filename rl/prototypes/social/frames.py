import sys, os, json, math
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
variant = sys.argv[1]
sys.path.insert(0, "rl")
import ungroup.native as native
native.LIB = os.path.join(SC, f"lib_{variant}.so")
from ungroup.native import Config, NativeBatch

LINEUPS = {
  "12eq": ["solo","bail","loyal"]*4,
  "12bh": ["bail","bail","bail","bail","solo","loyal"]*2,
  "20eq": (["solo","bail","loyal"]*7)[:20],
  "6b3l3": ["bail"]*3+["loyal"]*3,
}
ROUNDS = 6
def acf(x, lag):
    x = np.asarray(x, float); x = x - x.mean()
    if x.std() < 1e-9 or lag >= len(x): return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))
out = {}
for lk, seats in LINEUPS.items():
    n = len(seats)
    cfg = Config().replace(n_players=n)
    b = NativeBatch(ROUNDS, cfg, seed=5000)
    for e in range(ROUNDS):
        b.set_seats(e, seats); b.reset(e, 5000 + n + e)
    acts = np.zeros((ROUNDS, n, 4), dtype=np.int32)
    metas = [b.meta(e) for e in range(ROUNDS)]
    done = [False]*ROUNDS
    rec = [dict(brand=[], groups=[], events=[], t=[], heads=[], prev_brand=np.zeros(n), bodies=[]) for _ in range(ROUNDS)]
    step = 0
    while not all(done):
        _, _, d, eps = b.step(acts, auto_reset=False)
        for e in range(ROUNDS):
            if done[e]: continue
            f = b.frame(e)
            r = rec[e]
            br = np.array([p.get("brand", 0.0) for p in f["players"]])
            r["t"].append(f["t"]); r["brand"].append(br); r["groups"].append(sum(1 for bd in f["bodies"] if len(bd["m"]) > 1))
            r["events"].append(f["events"]); r["bodies"].append([(tuple(bd["m"]), bd.get("head", -1), bd["x"], bd["y"], bd["pool"]) for bd in f["bodies"]])
            r["banked"] = [p["banked"] for p in f["players"]]
        for ep in eps: done[ep["env"]] = True
        step += 1
    res = dict(rounds=[])
    for e in range(ROUNDS):
        r = rec[e]; T = len(r["t"]); dt = r["t"][1]-r["t"][0]
        B = np.array(r["brand"]); frac = (B > 0).mean(axis=1)
        # brand acquisitions: increase in brand at a step; source = leave event by that player in the same step else contagion
        acq_leave = acq_cont = 0; chains = []
        for k in range(1, T):
            evs = r["events"][k]
            leavers = {ev["player"] for ev in evs if ev["kind"] == "leave"}
            for i in range(n):
                if B[k, i] > B[k-1, i] + 1e-6:
                    if i in leavers: acq_leave += 1
                    else: acq_cont += 1
        # oscillation of branded fraction and of group count
        g = np.array(r["groups"], float)
        lags = range(1, min(T-2, int(60/dt)))
        ac_frac = [acf(frac, L) for L in lags]; ac_g = [acf(g, L) for L in lags]
        def first_min(a):
            if not a: return (0, 0.0)
            k = int(np.argmin(a)); return (float((k+1)*dt), float(a[k]))
        # crown fairness & rotation, bond mult, door-tax leaves
        banks = [(k, ev) for k in range(T) for ev in r["events"][k] if ev["kind"] == "bank" and len(ev["group"]) > 1]
        fair = 0; mults = []; totals = []
        prog_at = {}
        for k, ev in banks:
            grp = ev["group"]; recv = ev["player"]
            # progress before the bank: from banked at frame k-1 is unavailable; approximate using bodies list order: use meta needs and banked in frame k minus amount
            mults.append(ev.get("mult", 1 + 0.15*(len(grp)-1))); totals.append(ev.get("total", 0))
        heads_per_life = []
        life = {}  # member set -> set of heads
        for k in range(T):
            for m, h, x, y, pool in r["bodies"][k]:
                if len(m) > 1: life.setdefault(m, set()).add(h)
        heads_per_life = [len(v) for v in life.values()]
        leaves = [(k, ev) for k in range(T) for ev in r["events"][k] if ev["kind"] == "leave"]
        door = 0
        pads = metas[e]["pads"]
        for k, ev in leaves:
            # position of the leaver's former group at step k-1 and the head pad distance
            i = ev["player"]
            for m, h, x, y, pool in r["bodies"][max(0,k-1)]:
                if i in m and len(m) > 1 and h >= 0:
                    Rk = 1.0 if r["t"][k] < 120 else max(0.5, 1.0 - 0.5*(r["t"][k]-120)/120)
                    # pad position uses R - 0.08 approx
                    pr = max(Rk - 0.08, 0.1); px, py = pr*math.cos(pads[h]), pr*math.sin(pads[h])
                    if math.hypot(x-px, y-py) < 0.3: door += 1
        # loyal segregation: fraction of loyal grouped time with a branded co-member; loyal-bail co-membership share
        loyal = [i for i, s in enumerate(seats) if s == "loyal"]; bail = [i for i, s in enumerate(seats) if s == "bail"]
        lg = lgb = llb = 0
        for k in range(T):
            for m, h, x, y, pool in r["bodies"][k]:
                if len(m) < 2: continue
                for i in m:
                    if seats[i] != "loyal": continue
                    lg += 1
                    if any(B[k, j] > 0 for j in m if j != i): lgb += 1
                    if any(seats[j] == "bail" for j in m if j != i): llb += 1
        res["rounds"].append(dict(
            length=r["t"][-1], brand_mean=float(frac.mean()), brand_max=float(frac.max()),
            brand_acq_leave=acq_leave, brand_acq_contagion=acq_cont,
            brand_acf_min=first_min(ac_frac), group_acf_min=first_min(ac_g),
            group_mean=float(g.mean()), group_sd=float(g.std()),
            group_banks=len(banks), mult_mean=float(np.mean(mults)) if mults else 0.0, mult_full=float(np.mean([m >= 1+0.15*(len(ev["group"])-1)-1e-6 for (k, ev), m in zip(banks, mults)])) if banks else 0.0,
            heads_per_life_mean=float(np.mean(heads_per_life)) if heads_per_life else 0.0, lives=len(heads_per_life),
            leaves=len(leaves), door_leaves=door,
            loyal_grouped_steps=lg, loyal_with_branded=lgb/lg if lg else 0.0, loyal_with_bail=llb/lg if lg else 0.0,
            final_prog={s: float(np.mean([np.mean([min(r["banked"][i][t]/metas[e]["needs"][i][t],1) for t in range(4)]) for i in range(n) if seats[i]==s])) for s in set(seats)},
        ))
    # aggregate
    keys = [k for k in res["rounds"][0] if isinstance(res["rounds"][0][k], (int, float))]
    agg = {k: float(np.mean([rr[k] for rr in res["rounds"]])) for k in keys}
    agg["brand_acf_min"] = [float(np.mean([rr["brand_acf_min"][0] for rr in res["rounds"]])), float(np.mean([rr["brand_acf_min"][1] for rr in res["rounds"]]))]
    agg["group_acf_min"] = [float(np.mean([rr["group_acf_min"][0] for rr in res["rounds"]])), float(np.mean([rr["group_acf_min"][1] for rr in res["rounds"]]))]
    agg["final_prog"] = {s: float(np.mean([rr["final_prog"][s] for rr in res["rounds"]])) for s in set(seats)}
    out[lk] = agg
    print(variant, lk, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}), flush=True)
    # save one branded-fraction series for plotting
    np.save(os.path.join(SC, f"brandseries_{variant}_{lk}.npy"), np.array([(B > 0).mean(axis=1) for B in [np.array(rec[0]["brand"])]][0]))
    np.save(os.path.join(SC, f"groupseries_{variant}_{lk}.npy"), np.array(rec[0]["groups"], float))
json.dump(out, open(os.path.join(SC, f"frames_{variant}.json"), "w"), indent=1)
