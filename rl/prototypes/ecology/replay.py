import sys, json, numpy as np
S="/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"
sys.path.insert(0, S+"/rl")
from ungroup.native import Config, record_game
cfg = Config().replace(mine_cap=6, bloom_rate=0.1, yield_stock_exp=0, mine_rate=0.12, seed_rate=0.15, seed_floor=0.05, migrate_rate=0.2)
seats = ["loyal"]*6
rec = record_game(lambda obs: np.zeros((len(seats),4), dtype=np.int32), seats, cfg, seed=1000, names=[f"loyal{i}" for i in range(6)])
json.dump(rec, open(f"{S}/replay_D_6loyal_seed1000.json","w"))
seats = ["solo","bail","loyal"]*4
rec = record_game(lambda obs: np.zeros((len(seats),4), dtype=np.int32), seats, cfg.replace(n_players=12), seed=1000, names=[f"{s}{i}" for i,s in enumerate(seats)])
json.dump(rec, open(f"{S}/replay_D_12equal_seed1000.json","w"))
print("replays ok")
