"""Per-seat banked surplus (banked - need, per type) at round end, by lineup."""
import sys, json
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
def surplus(seats, games=24, seed=1000):
    n = len(seats); cfg = Config().replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed)
    for e in range(games):
        b.set_seats(e, seats); b.reset(e, seed + e)
    acts = np.zeros((games, n, 4), dtype=np.int32)
    done = set()
    res = {s: [] for s in set(seats)}
    lens = []
    while len(done) < games:
        _, _, _, ep = b.step(acts, auto_reset=False)
        for d in ep:
            e = d["env"]
            if e in done: continue
            done.add(e); lens.append(d["length"])
            fr = b.frame(e); me = b.meta(e)
            for i, s in enumerate(seats):
                banked = np.array(fr["players"][i]["banked"]); need = np.array(me["needs"][i])
                sur = np.maximum(banked - need, 0)
                prim = int(np.argmax(need))
                res[s].append(dict(surplus_total=float(sur.sum()), surplus_primary=float(sur[prim]), surplus_secondary=float(sur.sum() - sur[prim]),
                                   n_complete_types=int((banked >= need).sum()), banked_total=float(banked.sum()), progress=float(np.mean(np.minimum(banked / need, 1)))))
    out = {}
    for s, rows in res.items():
        out[s] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[s]["frac_surplus_ge6"] = float(np.mean([r["surplus_total"] >= 6 for r in rows]))
        out[s]["frac_surplus_ge12"] = float(np.mean([r["surplus_total"] >= 12 for r in rows]))
    return out
lineups = {"6loyal": ["loyal"]*6, "6bail": ["bail"]*6, "6solo": ["solo"]*6, "4solo_2loyal": ["solo"]*4+["loyal"]*2,
           "3bail_3loyal": ["bail"]*3+["loyal"]*3, "2loyal": ["loyal"]*2, "3loyal_3solo": ["loyal"]*3+["solo"]*3}
out = {}
for k, s in lineups.items():
    out[k] = surplus(s)
    for t, v in out[k].items():
        print(f"{k:14s} {t:6s} " + " ".join(f"{a}={b:.2f}" for a, b in v.items()), flush=True)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expS.json", "w"), indent=1)
