"""Record large-lobby rounds: frames every 10 ticks (2 x 5-tick steps, events merged) for 8 envs."""
import sys, os, time, pickle, gzip, json
sys.path.insert(0, "rl")
import numpy as np
from ungroup.native import Config, NativeBatch
from ladder_native import summarize

OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/lobbies"

def lineup(n, kind):
    if kind == "equal":
        base = ["solo", "bail", "loyal"]
    elif kind == "loyal":
        base = ["loyal", "loyal", "loyal", "loyal", "solo", "bail"]
    elif kind == "bail":
        base = ["bail", "bail", "bail", "bail", "solo", "loyal"]
    elif kind == "policy":
        base = ["policy", "bail", "policy", "loyal", "policy", "solo", "policy", "bail", "policy", "loyal"]
    return [base[i % len(base)] for i in range(n)]

def run(seats, games, seed, cfg, policy=None, sub_steps=2, de=5):
    n = len(seats)
    cfg = cfg.replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed)
    for e in range(games):
        b.set_seats(e, seats); b.reset(e, seed + e)
    obs = b.observe()
    frames = [[] for _ in range(games)]
    metas = [b.meta(e) for e in range(games)]
    for e in range(games):
        f0 = b.frame(e); f0["events"] = []; frames[e].append(f0)
    eps = {}
    acts = np.zeros((games, n, 4), dtype=np.int32)
    t0 = time.time()
    while len(eps) < games:
        pend = [[] for _ in range(games)]
        for s in range(sub_steps):
            if policy is not None:
                import torch
                with torch.no_grad():
                    a, _ = policy.act(torch.from_numpy(obs.reshape(games * n, -1)))
                acts = a.numpy().reshape(games, n, 4).astype(np.int32)
            obs, _, _, ep = b.step(acts, auto_reset=False, decide_every=de)
            for d in ep:
                if d["env"] not in eps: eps[d["env"]] = d
            for e in range(games):
                if e in eps and eps[e].get("_closed"): continue
                fr = b.frame(e)
                pend[e].extend(fr["events"])
                if s == sub_steps - 1 or e in eps:
                    fr["events"] = pend[e]; pend[e] = []
                    frames[e].append(fr)
                    if e in eps: eps[e]["_closed"] = True
    dt = time.time() - t0
    return dict(seats=seats, cfg=cfg, frames=frames, metas=metas, eps=[eps[e] for e in range(games)],
                summary=summarize(seats, [eps[e] for e in range(games)]), wall=dt)

if __name__ == "__main__":
    n = int(sys.argv[1]); kind = sys.argv[2]; games = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seed = 5000 + n
    cfg = Config()
    policy = None
    if kind == "policy":
        import torch
        torch.set_num_threads(1)
        from train_v2 import load_checkpoint
        policy, pcfg, ck = load_checkpoint("rl/models/v4_200.pt")
        print("checkpoint cfg diff vs default:", {k: (getattr(pcfg, k), getattr(cfg, k)) for k in Config.__dataclass_fields__ if getattr(pcfg, k) != getattr(cfg, k)})
    seats = lineup(n, kind)
    res = run(seats, games, seed, cfg, policy)
    print(f"n={n} kind={kind} games={games} wall={res['wall']:.1f}s frames/env={[len(f) for f in res['frames']]}")
    s = res["summary"]
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("by_type", "seats")})
    for k, v in s["by_type"].items(): print(f"  {k:7s} progress {v['progress']:.3f} +- {v['se']:.3f} win/seat {v['win_per_seat']:.3f}")
    with gzip.open(f"{OUT}/rec_{n}_{kind}.pkl.gz", "wb") as f:
        pickle.dump(res, f)
