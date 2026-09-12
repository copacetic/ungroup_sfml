import sys, json, os, math
sys.path.insert(0, "rl")
import numpy as np, torch
torch.set_num_threads(1)
from ungroup.native import Config, record_game
from train_v2 import load_checkpoint
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from record import lineup
OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/lobbies"
policy, pcfg, ck = load_checkpoint("rl/models/v4_200.pt")
def act(obs):
    with torch.no_grad():
        x, _ = policy.act(torch.from_numpy(obs))
    return x.numpy()
def score(rep):
    spills = [e for f in rep["frames"] for e in f["events"] if e["kind"] == "spill"]
    cas = 0; cur = []
    for ev in spills:
        if cur and any(ev["t"] - p["t"] <= 2 and math.hypot(ev["x"] - p["x"], ev["y"] - p["y"]) <= 0.2 for p in cur): cur.append(ev)
        else:
            if len(cur) >= 2: cas += 1
            cur = [ev]
    if len(cur) >= 2: cas += 1
    largest = [max(len(b["m"]) for b in f["bodies"]) for f in rep["frames"]]
    mega = sum(1 for k in range(1, len(largest)) if largest[k] >= 4 and largest[k - 1] < 4)
    picks = max(len(f["picks"]) for f in rep["frames"])
    return dict(spills=len(spills), cascades=cas, mega_spans=mega, maxsize=max(largest), picks_max=picks,
                score=3 * cas + 2 * mega + len(spills) / 5 + picks / 2)
for kind, seeds in (("policy", [5026, 5027, 5024]), ("equal", [5023])):
    seats = lineup(20, kind)
    names = [f"Agent {i}" if s == "policy" else f"{s.capitalize()} {i}" for i, s in enumerate(seats)]
    best = None
    for sd in seeds:
        rep = record_game(act, seats, Config(n_players=20), seed=sd, names=names, meta={"lineup": kind, "checkpoint": "v4_200.pt"})
        sc = score(rep); print(kind, sd, sc, "winner", rep["winner"], "timeout", rep["meta"]["timeout_win"], "t", rep["frames"][-1]["t"])
        if best is None or sc["score"] > best[1]["score"]: best = (rep, sc, sd)
    path = f"{OUT}/replay_20_{kind}_seed{best[2]}.json"
    json.dump(best[0], open(path, "w"), separators=(",", ":"))
    print("wrote", path, os.path.getsize(path) / 1e6, "MB")
