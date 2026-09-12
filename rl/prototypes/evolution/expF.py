"""Density league (birth/death/copy/migrate) on the patched core with optional capital and persistent ledger."""
import sys, json, math, time, argparse
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution")
from native_ev import Config, EvoBatch2
ap = argparse.ArgumentParser()
ap.add_argument("--types", default="solo,bail,loyal"); ap.add_argument("--gens", type=int, default=60); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--B", type=float, default=0.75); ap.add_argument("--D", type=float, default=0.5); ap.add_argument("--mig", type=float, default=0.05)
ap.add_argument("--capital", type=float, default=0.0); ap.add_argument("--persist", type=int, default=0); ap.add_argument("--mut", type=float, default=0.03)
ap.add_argument("--L", type=int, default=8); ap.add_argument("--T", type=float, default=0.05); ap.add_argument("--tag", default="")
a = ap.parse_args()
TYPES = a.types.split(","); L = a.L
rng = np.random.default_rng(a.seed)
# a seat is a dict: type, inherit (4), lineage id
def new_seat(t): return dict(type=int(t), inherit=[0.0] * 4)
lobbies = [[new_seat(rng.integers(len(TYPES))) for _ in range(6)] for _ in range(L)]
prev = [None] * L  # (batch, ids) for ledger remap
hist = []; t0 = time.time()
for g in range(a.gens):
    batches = []; results = []
    for e, lob in enumerate(lobbies):
        n = len(lob); seed = 50000 + 1000 * a.seed + 31 * g + e
        b = EvoBatch2(1, Config().replace(n_players=n), seed=seed)
        b.set_seats(0, [TYPES[s["type"]] for s in lob]); b.set_evo(0, inherit_cap=0.5, persist=bool(a.persist))
        if a.persist and prev[e] is not None:
            ob, oids, olen = prev[e]
            mapping = [oids.index(s.get("id")) if s.get("id") in oids else -1 for s in lob]
            b.remap_history(ob, 0, 0, mapping, olen)
        for i, s in enumerate(lob): b.set_inherit(0, i, s["inherit"])
        b.reset(0, seed)
        start = [b.start_progress(0, i) for i in range(n)]
        acts = np.zeros((1, n, 4), dtype=np.int32)
        while True:
            _, _, _, ep = b.step(acts, auto_reset=False)
            if ep: break
        ep = ep[0]; fr = b.frame(0); me = b.meta(0)
        for i, s in enumerate(lob):
            sur = np.maximum(np.array(fr["players"][i]["banked"]) - np.array(me["needs"][i]), 0)
            s["inherit"] = [float(x) for x in (a.capital * sur)]
        batches.append(b); results.append((ep, start))
    allt = np.array([s["type"] for lob in lobbies for s in lob]); allp = np.array([p for ep, _ in results for p in ep["progress"]])
    alls = np.array([x for _, st in results for x in st])
    freq = [float((allt == k).mean()) for k in range(len(TYPES))]
    pay = [float(allp[allt == k].mean()) if (allt == k).any() else float("nan") for k in range(len(TYPES))]
    sizes = [len(l) for l in lobbies]
    # evolve
    nb = nd = 0; newl = []
    for e, (lob, (ep, start)) in enumerate(zip(lobbies, results)):
        prog = np.array(ep["progress"]); n = len(lob)
        for s in lob: s.setdefault("id", None)
        ids = [id(s) for s in lob]
        for s, i_ in zip(lob, ids): s["id"] = i_
        prev[e] = (batches[e], ids, ep["length"])
        new = [dict(s) for s in lob]
        for i in range(n):
            if n < 2: break
            j = rng.integers(n - 1); j = j + (j >= i)
            if rng.random() < 1 / (1 + math.exp(-(prog[j] - prog[i]) / a.T)): new[i]["type"] = lob[j]["type"]
        births = []
        for i in range(n):
            if prog[i] >= a.B:
                c = dict(new[i]); c["inherit"] = [0.5 * x for x in c["inherit"]]; c["id"] = None; births.append(c)
        surv = [new[i] for i in range(n) if prog[i] >= a.D]
        out = surv + births; nb += len(births); nd += n - len(surv)
        for s in out:
            if rng.random() < a.mut: s["type"] = int(rng.integers(len(TYPES)))
        while len(out) < 3: out.append(new_seat(rng.integers(len(TYPES))))
        if len(out) > 32: out = [out[k] for k in rng.permutation(len(out))[:32]]
        newl.append(out)
    # migration to a neighbouring lobby (arrivals carry capital but no ledger)
    for e in range(L):
        for s in list(newl[e]):
            if rng.random() < a.mig and len(newl[e]) > 3:
                e2 = (e + int(rng.choice([-1, 1]))) % L
                if len(newl[e2]) < 32:
                    newl[e].remove(s); s["id"] = None; newl[e2].append(s)
    loyal_like = [k for k, t in enumerate(TYPES) if t in ("loyal", "grudge")]
    coop_by_lobby = [float(np.mean([s["type"] in loyal_like for s in l])) for l in lobbies]
    rec = dict(gen=g, N=int(sum(sizes)), sizes=sizes, freq=freq, pay=pay, births=nb, deaths=nd, coop_by_lobby=coop_by_lobby,
               early=float(np.mean([not ep["timeout_win"] for ep, _ in results])), spills=float(np.mean([ep["spills"] for ep, _ in results])),
               mean_start=float(alls.mean()), mean_prog=float(allp.mean()))
    hist.append(rec)
    print(f"g={g:3d} N={rec['N']:3d} sizes={sizes} freq " + " ".join(f"{TYPES[k][:3]}={freq[k]:.2f}" for k in range(len(TYPES))) +
          f" pay={['%.2f' % p for p in pay]} b={nb} d={nd} early={rec['early']:.2f} start={rec['mean_start']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    lobbies = newl
json.dump(hist, open(f"/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expF_{a.tag}.json", "w"))
