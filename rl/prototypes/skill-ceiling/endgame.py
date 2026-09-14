# Endgame lead stability: does the leader at 120 s (shrink start), 192 s (outer mines die) and 220 s win?
import sys, json
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
def study(seats, games=24, seed=2000):
    n = len(seats); cfg = Config().replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed)
    for e in range(games): b.set_seats(e, seats); b.reset(e, seed+e)
    acts = np.zeros((games, n, 4), dtype=np.int32)
    marks = [60, 120, 192, 220]; lead = {m: [None]*games for m in marks}; snap = {m: [None]*games for m in marks}
    finals = [None]*games; changes = [0]*games; lastlead = [None]*games; changes_after192 = [0]*games
    done = [False]*games; tprev = [0.0]*games
    while not all(done):
        _, _, _, eps = b.step(acts, auto_reset=False)
        for e in range(games):
            if done[e]: continue
            t = b.time(e); pr = [b.progress(e, i) for i in range(n)]
            L = int(np.argmax(pr))
            if lastlead[e] is not None and L != lastlead[e]:
                changes[e] += 1
                if t > 192: changes_after192[e] += 1
            lastlead[e] = L
            for m in marks:
                if lead[m][e] is None and t >= m: lead[m][e] = L; snap[m][e] = list(pr)
        for d in eps:
            e = d["env"]
            if not done[e]:
                done[e] = True; finals[e] = dict(winner=d["winner"], timeout=d["timeout_win"], length=d["length"], progress=d["progress"])
    res = {}
    for m in marks:
        ok = [i for i in range(games) if lead[m][i] is not None]
        res[f"leader_at_{m}_wins"] = float(np.mean([lead[m][i] == finals[i]["winner"] for i in ok])) if ok else None
        res[f"rounds_reaching_{m}"] = len(ok)
        if ok:
            gaps = [sorted(snap[m][i])[-1]-sorted(snap[m][i])[-2] for i in ok]
            res[f"lead_margin_at_{m}"] = float(np.mean(gaps))
    res["timeout_frac"] = float(np.mean([f["timeout"] for f in finals]))
    res["mean_len"] = float(np.mean([f["length"] for f in finals]))
    res["lead_changes_per_round"] = float(np.mean(changes)); res["lead_changes_after_192"] = float(np.mean(changes_after192))
    res["final_margin"] = float(np.mean([sorted(f["progress"])[-1]-sorted(f["progress"])[-2] for f in finals]))
    # progress gained in the last 48 s by the eventual winner
    gain = [finals[i]["progress"][finals[i]["winner"]] - snap[192][i][finals[i]["winner"]] for i in range(games) if snap[192][i] is not None]
    res["winner_gain_after_192"] = float(np.mean(gain)) if gain else None
    return res
out = {}
for name, seats in {"6 bail": ["bail"]*6, "3 bail 3 loyal": ["bail"]*3+["loyal"]*3, "6 loyal": ["loyal"]*6, "2 bail 2 loyal 2 rammer": ["bail"]*2+["loyal"]*2+["rammer"]*2}.items():
    r = study(seats); out[name] = r
    print(name, json.dumps(r))
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/skill-ceiling/endgame.json","w"), indent=1)
