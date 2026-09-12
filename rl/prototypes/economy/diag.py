import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy"); sys.path.insert(0, "rl")
import oracle_solo as o, numpy as np, math
from ungroup.native import Config, NativeBatch
# replicate the loop with fine logging near the end for seed 1008
seed = int(sys.argv[1])
cfg = Config().replace(n_players=1)
b = NativeBatch(1, cfg, seed=seed, decide_every=1); b.set_seats(0, ["policy"]); b.reset(0, seed)
meta = b.meta(0); pad_ang = meta["pads"][0]; mpos = np.array(meta["mine_pos"])
print("pad angle", pad_ang, "pad_end", o.pad_pos(238, pad_ang, cfg), "inner mines", [m.round(3).tolist() for m in mpos if np.linalg.norm(m) < 0.5])
r = o.play(seed, verbose=True)
print(r["route"], r["banks"], r["progress"])
for l in r["log"][-40:]: print(l)
