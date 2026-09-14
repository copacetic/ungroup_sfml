"""Coalition of n policy seats that brings EVERY member to progress ~1: bank each member up to need-0.2 (so nobody
finishes early), then top up the last one. Reports the time at which all n are within 0.2 units of full."""
import sys, math, itertools, json
import numpy as np
sys.path.insert(0, "rl")
sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy")
from ungroup.native import Config, NativeBatch
from oracle_solo import steer, pad_pos

def plan_route(pos, target, mines_of, mpos, alive, touch, padp, rpad):
    def leg(a, b_, ca, cb): return max(0.0, np.linalg.norm(a - b_) - ca - cb)
    types = [t for t in range(4) if target[t] > 0.05]
    best = None
    for perm in itertools.permutations(types):
        for choice in itertools.product([0, 1], repeat=len(types)):
            ms = [mines_of[t][c] for t, c in zip(perm, choice)]
            if any(not alive[m] for m in ms): continue
            L = leg(pos, mpos[ms[0]], 0, touch)
            for a, c in zip(ms, ms[1:]): L += leg(mpos[a], mpos[c], touch, touch)
            L += leg(mpos[ms[-1]], padp, touch, rpad)
            if best is None or L < best[0]: best = (L, ms)
    return best[1]

def play(n, seed, cfg=None, hold=1.5, debug=False):
    cfg = (cfg or Config()).replace(n_players=n)
    b = NativeBatch(1, cfg, seed=seed, decide_every=1)
    b.set_seats(0, ["policy"] * n); b.reset(0, seed)
    meta = b.meta(0)
    needs = np.array(meta["needs"]); pads = meta["pads"]
    mpos = np.array(meta["mine_pos"]); mtype = np.array(meta["mine_type"])
    rn = cfg.solo_radius * math.sqrt(n); touch = rn + cfg.mine_radius + 0.005
    mult = 1 + cfg.group_bank_bonus * (n - 1)
    mines_of = {t: [m for m in range(cfg.n_mines) if mtype[m] == t] for t in range(4)}
    acts = np.zeros((1, n, 4), dtype=np.int32); acts[0, :, 0] = 9; acts[0, :, 1] = 1
    phase = "merge"; done_members = []; cur = None; route = []; k = 0; target = None
    ep = None; t_all = None; bank_times = []
    while ep is None:
        f = b.frame(0); t = f["t"]; bodies = f["bodies"]
        alive = f["alive"]
        dirs = {}
        if phase == "merge":
            big = max(bodies, key=lambda bb: len(bb["m"]))
            if len(big["m"]) == n: phase = "pick"
            elif len(big["m"]) == 1:
                for bb in bodies:
                    for i in bb["m"]: dirs[i] = -np.array([bb["x"], bb["y"]])
            else:
                anchor = np.array([big["x"], big["y"]])
                for bb in bodies:
                    for i in bb["m"]: dirs[i] = (np.zeros(2) if bb is big else anchor - np.array([bb["x"], bb["y"]]))
        if phase != "merge":
            bb = bodies[0]; pos = np.array([bb["x"], bb["y"]]); pool = np.array(bb["pool"])
            banked = np.array([p["banked"] for p in f["players"]])
        if phase == "pick":
            remaining = [i for i in range(n) if i not in done_members]
            if not remaining: break
            # nearest pad first; the last member gets the full amount (ends the round)
            cur = min(remaining, key=lambda i: np.linalg.norm(pad_pos(t, pads[i], cfg) - pos))
            last = len(remaining) == 1
            holdv = np.zeros(4)
            rate = cfg.mine_rate * n ** cfg.mine_exp
            if not last: holdv[int(needs[cur].argmax())] = hold + 0.45 * rate * mult
            want = needs[cur] - banked[cur] - holdv
            target = np.maximum(want / mult - (0 if last else 0.45 * rate), 0)
            # never route through a mine that overlaps the pad of a member who is already done (it would bank every tick)
            alive = list(alive)
            for j in done_members:
                for mm in range(cfg.n_mines):
                    if np.linalg.norm(mpos[mm] - pad_pos(t, pads[j], cfg)) < touch + rn + cfg.pad_radius + 0.03: alive[mm] = 0
            route = plan_route(pos, target, mines_of, mpos, alive, touch, pad_pos(t + 10, pads[cur], cfg), rn + cfg.pad_radius)
            k = 0; phase = "route"
            if debug: print('pick', cur, 'need', needs[cur].tolist(), 'banked', banked[cur].tolist(), 'target', target.round(2).tolist(), 'route', route, 't', round(t,1))
        if phase == "route":
            while k < len(route) and pool[mtype[route[k]]] >= target[mtype[route[k]]] + 0.05: k += 1
            if k >= len(route): phase = "bank"
            else:
                m = route[k]; tgt = steer(pos, mpos[m], mpos, touch + 0.03, exclude=m)
        if phase == "bank":
            tgt = steer(pos, pad_pos(t, pads[cur], cfg), mpos, touch + 0.03)
            if pool.sum() < 0.5 and any(e.get("kind") == "bank" for e in f.get("events", [])) or (pool.sum() < 0.5 and banked[cur].sum() > 0.5):
                done_members.append(cur); bank_times.append((cur, round(t, 1))); phase = "pick"
                if len(done_members) == n - 1: t_all_but_one = t
        if phase != "merge":
            for i in range(n): dirs[i] = tgt - pos
        for i in range(n):
            d = dirs.get(i, np.zeros(2)); b.set_direction(0, i, float(d[0]), float(d[1]))
        _, _, _, eps = b.step(acts, auto_reset=False, decide_every=1)
        if debug and phase != 'merge' and int(round(t*30)) % 15 == 0: print(round(t,1), phase, k, pos.round(2).tolist(), pool.tolist(), list(f.keys()) if t < 3 else '')
        if debug:
            for e in f.get('events', []):
                if e.get('kind') in ('bank', 'win'): print(round(t,1), e)
        if eps: ep = eps[0]
        if t > 239: break
    return dict(n=n, seed=seed, length=ep["length"] if ep else None, winner=ep["winner"] if ep else None,
                progress=[round(p, 3) for p in ep["progress"]] if ep else None, banks=bank_times, merges=ep["merges"] if ep else None)

if __name__ == "__main__":
    n = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    res = [play(n, s) for s in range(1000, 1000 + N)]
    for r in res: print(r)
    L = np.array([r["length"] for r in res if r["length"]]); P = np.array([np.mean(r["progress"]) for r in res if r["progress"]])
    print(f"n={n} ALL-FINISH: mean length {L.mean():.1f}s (min {L.min():.1f} max {L.max():.1f}), mean progress {P.mean():.3f}, ok {len(L)}/{len(res)}")
    json.dump(res, open(f"/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy/oracle_rotation_{n}.json", "w"), indent=1)
