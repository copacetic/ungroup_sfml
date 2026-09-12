# Endgame geometry luck: distance from each player's pad (at final R=0.5) to the nearest surviving mine of their primary type,
# versus final progress and winning. 24 rounds x 2 lineups.
import sys, math, json
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
def study(seats, games=24, seed=3000):
    n = len(seats); cfg = Config().replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed)
    for e in range(games): b.set_seats(e, seats); b.reset(e, seed+e)
    metas = [b.meta(e) for e in range(games)]
    acts = np.zeros((games, n, 4), dtype=np.int32); done = [None]*games
    while any(d is None for d in done):
        _, _, _, eps = b.step(acts, auto_reset=False)
        for d in eps:
            if done[d["env"]] is None: done[d["env"]] = d
    rows = []
    for e in range(games):
        m = metas[e]; R = 0.5; rp = R - 0.06 - 0.02
        for i in range(n):
            pad = (rp*math.cos(m["pads"][i]), rp*math.sin(m["pads"][i]))
            prim = int(np.argmax(m["needs"][i]))
            dmin = 9; dany = 9
            for k, (mx, my) in enumerate(m["mine_pos"]):
                if math.hypot(mx, my) + 0.08 > R: continue  # dead by the end
                d = math.hypot(mx-pad[0], my-pad[1]); dany = min(dany, d)
                if m["mine_type"][k] == prim: dmin = min(dmin, d)
            rows.append(dict(env=e, seat=i, type=seats[i], d_primary=dmin, d_any=dany, prog=done[e]["progress"][i], win=int(done[e]["winner"] == i)))
    return rows
out = {}
for name, seats in {"6 bail": ["bail"]*6, "3 bail 3 loyal": ["bail"]*3+["loyal"]*3}.items():
    rows = study(seats); out[name] = rows
    d = np.array([r["d_primary"] for r in rows]); p = np.array([r["prog"] for r in rows]); w = np.array([r["win"] for r in rows])
    near = d < np.median(d)
    # winner: was the winner's pad closer to its primary surviving mine than the round median?
    wins_near = 0; tot = 0
    for e in set(r["env"] for r in rows):
        rr = [r for r in rows if r["env"] == e]; med = np.median([r["d_primary"] for r in rr])
        wr = [r for r in rr if r["win"]][0]; tot += 1; wins_near += wr["d_primary"] < med
    print(f"{name}: corr(d_primary, progress) = {np.corrcoef(d, p)[0,1]:+.2f}; progress near/far half = {p[near].mean():.3f}/{p[~near].mean():.3f}; "
          f"win rate near/far half = {w[near].mean():.2f}/{w[~near].mean():.2f}; winner had below-median distance in {wins_near}/{tot} rounds; d range {d.min():.2f}-{d.max():.2f}")
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/skill-ceiling/geom.json","w"))
