import sys, os, json
import numpy as np
SC = os.path.dirname(os.path.abspath(__file__))
variant = sys.argv[1]
sys.path.insert(0, "rl")
import ungroup.native as native
native.LIB = os.path.join(SC, f"lib_{variant}.so")
from ungroup.native import Config, record_game
seats = ["bail","bail","bail","bail","solo","loyal"]*2
zero = lambda obs: np.zeros((len(seats), 4), dtype=np.int32)
g = record_game(zero, seats, Config(n_players=len(seats)), seed=5012, names=[f"{s}{i}" for i, s in enumerate(seats)])
out = os.path.join(SC, f"replay_12bh_{variant}_seed5012.json")
json.dump(g, open(out, "w"))
ev = [e for f in g["frames"] for e in f["events"]]
from collections import Counter
print(variant, "frames", len(g["frames"]), "winner", g["winner"], "progress", [round(p, 2) for p in g["meta"]["progress"]], Counter(e["kind"] for e in ev))
# brand timeline summary (who is branded when) for the first 120 s
tl = []
for f in g["frames"][::30]:
    br = [i for i, p in enumerate(f["players"]) if p.get("brand", 0) > 0]
    heads = sorted(set(b.get("head", -1) for b in f["bodies"] if len(b["m"]) > 1))
    tl.append((round(f["t"]), br, heads))
print(tl[:40])
