"""Oracle coalition of n policy seats: merge at the origin, mine the finisher's need / bank multiplier, bank once. Reports finish time."""
import sys, math, itertools, json
import numpy as np
sys.path.insert(0, "rl")
sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy")
from ungroup.native import Config, NativeBatch
from oracle_solo import steer, pad_pos

def play(n, seed, cfg=None, verbose=False):
    cfg = (cfg or Config()).replace(n_players=n)
    b = NativeBatch(1, cfg, seed=seed, decide_every=1)
    b.set_seats(0, ["policy"] * n); b.reset(0, seed)
    meta = b.meta(0)
    needs = np.array(meta["needs"]); pads = meta["pads"]
    mpos = np.array(meta["mine_pos"]); mtype = np.array(meta["mine_type"])
    rn = cfg.solo_radius * math.sqrt(n); speed = cfg.base_speed / math.sqrt(n)
    touch = rn + cfg.mine_radius + 0.005
    mult = 1 + cfg.group_bank_bonus * (n - 1)
    mines_of = {t: [m for m in range(cfg.n_mines) if mtype[m] == t] for t in range(4)}
    def leg(a, b_, ca, cb): return max(0.0, np.linalg.norm(a - b_) - ca - cb)
    # choose finisher + route minimising path length from the origin through one mine per type to the finisher's pad
    best = None
    for f in range(n):
        padf = pad_pos(20.0, pads[f], cfg)
        for perm in itertools.permutations(range(4)):
            for choice in itertools.product([0, 1], repeat=4):
                ms = [mines_of[t][c] for t, c in zip(perm, choice)]
                L = leg(np.zeros(2), mpos[ms[0]], 0, touch)
                for a, c in zip(ms, ms[1:]): L += leg(mpos[a], mpos[c], touch, touch)
                L += leg(mpos[ms[-1]], padf, touch, rn + cfg.pad_radius)
                if best is None or L < best[0]: best = (L, f, ms)
    plan_len, fin, route = best
    target = needs[fin] / mult
    acts = np.zeros((1, n, 4), dtype=np.int32); acts[0, :, 0] = 9; acts[0, :, 1] = 1
    phase, k, t_merged = "merge", 0, None
    ep = None; log = []
    while ep is None:
        f = b.frame(0); t = f["t"]; bodies = f["bodies"]
        dirs = {}
        if phase == "merge":
            big = max(bodies, key=lambda bb: len(bb["m"]))
            if len(big["m"]) == n:
                phase = "route"; t_merged = t
            elif len(big["m"]) == 1:
                for bb in bodies:
                    for i in bb["m"]: dirs[i] = -np.array([bb["x"], bb["y"]])
            else:
                anchor = np.array([big["x"], big["y"]])
                for bb in bodies:
                    for i in bb["m"]: dirs[i] = (np.zeros(2) if bb is big else anchor - np.array([bb["x"], bb["y"]]))
        if phase in ("route", "bank"):
            bb = bodies[0]; pos = np.array([bb["x"], bb["y"]]); pool = np.array(bb["pool"])
            if phase == "route":
                while k < len(route) and pool[mtype[route[k]]] >= target[mtype[route[k]]] + 0.05: k += 1
                if k >= len(route): phase = "bank"
            if phase == "route":
                m = route[k]; tgt = steer(pos, mpos[m], mpos, touch + 0.03, exclude=m)
            else:
                tgt = steer(pos, pad_pos(t, pads[fin], cfg), mpos, touch + 0.03)
            for i in range(n): dirs[i] = tgt - pos
        for i in range(n):
            d = dirs.get(i, np.zeros(2)); b.set_direction(0, i, float(d[0]), float(d[1]))
        _, _, _, eps = b.step(acts, auto_reset=False, decide_every=1)
        if verbose and int(round(t * 30)) % 15 == 0:
            log.append(dict(t=round(t, 1), phase=phase, k=k, bodies=[(bb["m"], round(bb["x"], 2), round(bb["y"], 2), [round(x, 1) for x in bb["pool"]]) for bb in bodies]))
        if eps: ep = eps[0]
        if t > 200: break
    return dict(n=n, seed=seed, finisher=fin, route=route, plan_len=round(plan_len, 3), t_merged=t_merged,
                length=ep["length"] if ep else None, winner=ep["winner"] if ep else None, progress=ep["progress"] if ep else None,
                merges=ep["merges"] if ep else None, spills=ep["spills"] if ep else None, log=log)

if __name__ == "__main__":
    n = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    res = [play(n, s, verbose=(s == 1000)) for s in range(1000, 1000 + N)]
    for r in res: print(r["n"], r["seed"], "fin", r["finisher"], "route", r["route"], "plan", r["plan_len"], "merged@", None if r["t_merged"] is None else round(r["t_merged"], 2), "len", r["length"], "winner", r["winner"], "spills", r["spills"], "prog", None if r["progress"] is None else [round(p, 2) for p in r["progress"]])
    L = np.array([r["length"] for r in res if r["length"]])
    print(f"n={n} MEAN finish {L.mean():.2f}s sd {L.std(ddof=1) if len(L) > 1 else 0:.2f} min {L.min():.2f} max {L.max():.2f} ok {len(L)}/{len(res)} mean merged {np.mean([r['t_merged'] for r in res if r['t_merged'] is not None]):.2f}")
    json.dump(res, open(f"/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy/oracle_group_{n}.json", "w"), indent=1)
