"""Oracle solo: one policy seat, set_direction + move 9, single terminal bank (plus re-bank if time allows)."""
import sys, math, itertools, json
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch

SPEED, LERP_LAG = 0.45, 0.17
def R_of(t, cfg):
    frac = t / cfg.time_limit
    if frac <= cfg.shrink_start: return 1.0
    k = (frac - cfg.shrink_start) / (1 - cfg.shrink_start)
    return 1.0 + (cfg.final_radius - 1.0) * min(k, 1.0)
def pad_pos(t, ang, cfg):
    r = max(R_of(t, cfg) - cfg.pad_radius - 0.02, 0.1)
    return np.array([r * math.cos(ang), r * math.sin(ang)])

def steer(pos, tgt, mpos, clear, exclude=None):
    """If the straight segment pos->tgt passes through a non-target mine, aim at a tangent waypoint instead."""
    d = tgt - pos; L = np.linalg.norm(d)
    if L < 1e-9: return tgt
    u = d / L
    best = None
    for m in range(len(mpos)):
        if m == exclude: continue
        w = mpos[m] - pos; s = float(w @ u)
        if s <= 0 or s >= L: continue
        perp = w - s * u; pd = np.linalg.norm(perp)
        if pd < clear and (best is None or s < best[0]):
            side = perp / pd if pd > 1e-6 else np.array([-u[1], u[0]])
            best = (s, mpos[m] - side * clear * 1.1)
    return tgt if best is None else best[1]

def play(seed, cfg=None, margin=1.5, verbose=False):
    cfg = (cfg or Config()).replace(n_players=1)
    b = NativeBatch(1, cfg, seed=seed, decide_every=1)
    b.set_seats(0, ["policy"]); b.reset(0, seed)
    meta = b.meta(0)
    need = np.array(meta["needs"][0]); pad_ang = meta["pads"][0]
    mpos = np.array(meta["mine_pos"]); mtype = np.array(meta["mine_type"])
    primary = int(need.argmax()); secs = [t for t in range(4) if t != primary]
    r1 = cfg.solo_radius
    touch = r1 + cfg.mine_radius + 0.005   # inside the mining zone (< r+mine_radius+0.01)
    # plan: choose order of the three secondary mines (inner or outer) then the primary inner mine last
    inner = {t: [m for m in range(cfg.n_mines) if mtype[m] == t and np.linalg.norm(mpos[m]) < 0.5][0] for t in range(4)}
    outer = {t: [m for m in range(cfg.n_mines) if mtype[m] == t and np.linalg.norm(mpos[m]) > 0.5][0] for t in range(4)}
    start = np.array([0.72 * math.cos(pad_ang), 0.72 * math.sin(pad_ang)])
    pad_end = pad_pos(238.0, pad_ang, cfg)
    def leg(a, b_, ca, cb): return max(0.0, np.linalg.norm(a - b_) - ca - cb)
    best = None
    for perm in itertools.permutations(secs):
        for choice in itertools.product([0, 1], repeat=3):
            ms = [inner[t] if c == 0 else outer[t] for t, c in zip(perm, choice)] + [inner[primary]]
            L = leg(start, mpos[ms[0]], 0, touch)
            for a, c in zip(ms, ms[1:]): L += leg(mpos[a], mpos[c], touch, touch)
            L += leg(mpos[ms[-1]], pad_end, touch, r1 + cfg.pad_radius)
            if best is None or L < best[0]: best = (L, ms)
    plan_len, route = best
    targets = [(m, need[mtype[m]] if mtype[m] != primary else 1e9) for m in route]
    acts = np.zeros((1, 1, 4), dtype=np.int32); acts[0, 0, 0] = 9
    phase, k = "mine", 0
    log = []; banks = []
    ep = None
    while ep is None:
        f = b.frame(0); t = f["t"]; body = f["bodies"][0]
        pos = np.array([body["x"], body["y"]]); pool = np.array(body["pool"])
        # deadline check: time to reach the pad (position at arrival)
        pp = pad_pos(t, pad_ang, cfg)
        d = max(0.0, np.linalg.norm(pos - pp) - r1 - cfg.pad_radius); eta = d / SPEED + LERP_LAG
        pp = pad_pos(t + eta, pad_ang, cfg)
        d = max(0.0, np.linalg.norm(pos - pp) - r1 - cfg.pad_radius); eta = d / SPEED + LERP_LAG
        if phase == "mine":
            if t + eta + margin >= cfg.time_limit and pool.sum() >= 0.5: phase = "bank"
            else:
                while k < len(targets) - 1 and pool[mtype[targets[k][0]]] >= targets[k][1] + 0.05: k += 1
                m = targets[k][0]
                tgt = mpos[m]
                if k == len(targets) - 1 and pool.sum() >= 0.5:
                    # last mine: if the pad is within reach of the mine zone anyway, keep mining and bank at the end only
                    pass
        if phase == "bank":
            tgt = pp
        tgt = steer(pos, tgt, mpos, r1 + cfg.mine_radius + 0.03, exclude=(None if phase == "bank" else m))
        v = tgt - pos
        b.set_direction(0, 0, float(v[0]), float(v[1]))
        _, _, _, eps = b.step(acts, auto_reset=False, decide_every=1)
        for e in f.get("events", []):
            if e.get("kind") == "bank": banks.append((t, e.get("amount")))
        if verbose and (int(round(t * 30)) % 300 == 0 or (t > 225 and int(round(t * 30)) % 15 == 0)):
            log.append(dict(t=round(t, 1), pos=pos.round(3).tolist(), pool=pool.round(2).tolist(), phase=phase, k=k))
        if eps: ep = eps[0]
        if phase == "bank" and pool.sum() < 0.5 and t > 1:
            phase = "mine"  # banked; go back to mining if any time remains
    f = b.frame(0)
    banked = np.array(f["players"][0]["banked"]); pool = np.array(f["bodies"][0]["pool"])
    prog = float(np.mean(np.minimum(banked / need, 1)))
    return dict(seed=seed, progress=prog, ep_progress=ep["progress"][0], banked=banked.round(2).tolist(), leftover_pool=pool.round(2).tolist(),
                need=need.tolist(), plan_len=round(plan_len, 3), banks=[(round(t, 1), a) for t, a in banks], route=route, log=log)

if __name__ == "__main__":
    seeds = list(range(1000, 1000 + int(sys.argv[1]) if len(sys.argv) > 1 else 1012))
    res = [play(s, verbose=(s == seeds[0])) for s in seeds]
    for r in res: print(r["seed"], f"prog={r['progress']:.3f}", "banked", r["banked"], "need", r["need"], "left", r["leftover_pool"], "banks", r["banks"], "plan_len", r["plan_len"])
    for l in res[0]["log"]: print(l)
    p = np.array([r["progress"] for r in res])
    print(f"MEAN progress {p.mean():.4f} sd {p.std(ddof=1):.4f} n={len(p)}")
    json.dump(res, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy/oracle_solo.json", "w"), indent=1)
