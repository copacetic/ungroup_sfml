"""Mine field: diffusion along the ring and persistence across rounds, 24-player loyal-heavy lobby."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution")
from native_ev import Config, EvoBatch2
def run(seats, diff, persist, rounds=4, games=6, seed=7000):
    n = len(seats); b = EvoBatch2(games, Config().replace(n_players=n), seed=seed)
    for e in range(games): b.set_seats(e, seats); b.set_mines(e, mine_diff=diff, persist_mines=persist)
    acts = np.zeros((games, n, 4), dtype=np.int32)
    rows = []
    for r in range(rounds):
        for e in range(games): b.reset(e, seed + e)  # same seed -> same layout; stock persists if enabled
        start_stock = np.array([b.frame(e)["mines"] for e in range(games)])
        series = [[] for _ in range(games)]; done = {}
        tick = 0
        while len(done) < games:
            _, _, _, ep = b.step(acts, auto_reset=False)
            tick += 1
            if tick % 2 == 0:
                for e in range(games):
                    if e not in done: series[e].append(b.frame(e)["mines"])
            for d in ep: done.setdefault(d["env"], d)
        prog = np.array([done[e]["progress"] for e in range(games)])
        # statistics
        dry = []; travel = []; xc = []; meanst = []
        for e in range(games):
            S = np.array(series[e])  # (T, 8) every 12 ticks = 0.4 s
            if len(S) < 10: continue
            meanst.append(S[:, :].mean())
            d = (S < 1.0)
            dry.append(d.mean())
            # trough travel: for each mine, correlation of its stock change with the neighbour's change lagged by k samples
            dS = np.diff(S, axis=0)
            for k in (0, 5, 12, 25):  # 0, 2, 5, 10 s
                c = []
                for m in range(8):
                    q = (m + 1) % 8
                    a_ = dS[:len(dS) - k, m]; b_ = dS[k:, q]
                    if a_.std() > 1e-9 and b_.std() > 1e-9: c.append(np.corrcoef(a_, b_)[0, 1])
                xc.append((k, float(np.mean(c)) if c else 0.0))
            # dry-event succession: after mine m goes dry, is the next new dry mine within 30 s a ring neighbour?
            ev = []
            for tt in range(1, len(S)):
                for m in range(8):
                    if d[tt, m] and not d[tt - 1, m]: ev.append((tt, m))
            succ = []
            for i, (tt, m) in enumerate(ev):
                nxt = [(t2, m2) for (t2, m2) in ev[i + 1:] if t2 - tt <= 75 and m2 != m]
                if nxt: succ.append(int(abs(nxt[0][1] - m) % 8 in (1, 7)))
            if succ: travel.append(np.mean(succ))
        xcd = {}
        for k, v in xc: xcd.setdefault(k, []).append(v)
        row = dict(round=r, start_stock=float(start_stock.mean()), mean_stock=float(np.mean(meanst)), dry_frac=float(np.mean(dry)),
                   neighbour_succession=float(np.mean(travel)) if travel else None, xcorr={k: float(np.mean(v)) for k, v in xcd.items()},
                   prog={s: float(prog[:, [i for i, x in enumerate(seats) if x == s]].mean()) for s in set(seats)},
                   early=float(np.mean([not done[e]["timeout_win"] for e in range(games)])))
        rows.append(row)
        print(f"diff={diff} persist={persist} round {r}: start_stock={row['start_stock']:.1f} mean_stock={row['mean_stock']:.1f} dry={row['dry_frac']:.2f} "
              f"neighbour_succ={row['neighbour_succession']} xcorr={ {k: round(v, 2) for k, v in row['xcorr'].items()} } prog={ {k: round(v, 2) for k, v in row['prog'].items()} } early={row['early']:.2f}", flush=True)
    return rows
seats = ["loyal"] * 16 + ["solo"] * 4 + ["bail"] * 4
out = {}
for diff, persist in [(0.0, False), (0.05, False), (0.2, False), (0.0, True), (0.05, True)]:
    out[f"diff{diff}_persist{persist}"] = run(seats, diff, persist)
out["6p_diff0.05"] = run(["loyal"] * 3 + ["bail"] * 3, 0.05, False, rounds=1, games=24)
out["6p_diff0"] = run(["loyal"] * 3 + ["bail"] * 3, 0.0, False, rounds=1, games=24)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expM.json", "w"), indent=1)
