import sys, json, gzip, pickle, numpy as np
S = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"
sys.path.insert(0, S + "/rl")
from ungroup.native import Config, NativeBatch
sys.path.insert(0, S); from eco_ladder import LINEUPS

def record(name, kw, lineups, games=12, seed=1000):
    cfg0 = Config().replace(**kw)
    out = {}
    for ln in lineups:
        seats = LINEUPS[ln]; n = len(seats)
        cfg = cfg0.replace(n_players=n)
        b = NativeBatch(games, cfg, seed=seed)
        for e in range(games):
            b.set_seats(e, seats); b.reset(e, seed + e)
        acts = np.zeros((games, n, 4), dtype=np.int32)
        stocks = [[] for _ in range(games)]; alive = [[] for _ in range(games)]; ts = [[] for _ in range(games)]
        bodies = [[] for _ in range(games)]; events = [[] for _ in range(games)]
        done = [False]*games; eps = {}
        meta = [b.meta(e) for e in range(games)]
        while not all(done):
            _, _, _, ep = b.step(acts, auto_reset=False, decide_every=6)
            for e in range(games):
                if done[e]: continue
                fr = b.frame(e)
                ts[e].append(fr["t"]); stocks[e].append(fr["mines"]); alive[e].append(fr["alive"])
                bodies[e].append([(bd["m"], bd["x"], bd["y"], sum(bd["pool"])) for bd in fr["bodies"]])
                events[e].extend([dict(ev, t=fr["t"]) for ev in fr.get("events", [])])
            for d in ep:
                if not done[d["env"]]:
                    done[d["env"]] = True; eps[d["env"]] = d
        out[ln] = dict(ts=ts, stocks=stocks, alive=alive, bodies=bodies, events=events, meta=meta, eps=[eps[e] for e in range(games)], cfg=kw, seats=seats)
        print(name, ln, "recorded", flush=True)
    pickle.dump(out, gzip.open(f"{S}/rec_{name}.pkl.gz", "wb"))

if __name__ == "__main__":
    name = sys.argv[1]; kw = {}
    lineups = ["6solo", "6bail", "6loyal", "3bail3loyal", "2loyal4solo"]
    for a in sys.argv[2:]:
        if "=" in a: k, v = a.split("="); kw[k] = float(v)
        else: lineups = a.split(",")
    record(name, kw, lineups)
