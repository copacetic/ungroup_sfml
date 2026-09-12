import sys, os
import numpy as np, torch
SC = os.path.dirname(os.path.abspath(__file__))
variant = sys.argv[1]
sys.path.insert(0, "rl")
import ungroup.native as native
native.LIB = os.path.join(SC, f"lib_{variant}.so")
from ungroup.native import Config, NativeBatch
from train_v2 import load_checkpoint
torch.set_num_threads(1)
for ck in ["rl/models/v8_300.pt", "rl/models/v4_200.pt"]:
    policy, cfg, _ = load_checkpoint(ck)
    seats = ["policy","policy","bail","bail","loyal","loyal"]
    n = 6; E = 16
    cfg = Config().replace(n_players=n)
    b = NativeBatch(E, cfg, seed=1000)
    for e in range(E): b.set_seats(e, seats); b.reset(e, 1000+e)
    obs, _, _, _ = b.step(np.zeros((E, n, 4), dtype=np.int32), auto_reset=False)
    eps = []
    while len(eps) < E:
        with torch.no_grad():
            a, _ = policy.act(torch.from_numpy(obs.reshape(E*n, -1)))
        acts = a.numpy().reshape(E, n, 4).astype(np.int32)
        acts[:, 2:, :] = 0
        obs, _, _, ep = b.step(acts, auto_reset=False)
        for d in ep:
            if all(d["env"] != x["env"] for x in eps): eps.append(d)
    prog = np.array([d["progress"] for d in eps])
    print(f"{variant} {os.path.basename(ck)} policy {prog[:, :2].mean():.3f} bail {prog[:, 2:4].mean():.3f} loyal {prog[:, 4:].mean():.3f} early {np.mean([not d['timeout_win'] for d in eps]):.2f}", flush=True)
