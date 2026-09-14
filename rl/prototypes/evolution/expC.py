import sys, json
sys.path.insert(0, "rl")
from ladder_native import run
from ungroup.native import Config
lineups = {"6loyal": ["loyal"]*6, "6bail": ["bail"]*6, "6solo": ["solo"]*6,
           "3bail_3loyal": ["bail"]*3+["loyal"]*3, "4bail_2loyal": ["bail"]*4+["loyal"]*2,
           "4solo_2loyal": ["solo"]*4+["loyal"]*2, "5loyal_1bail": ["loyal"]*5+["bail"]}
grid = [(30,0.5),(10,0.5),(4,0.5),(30,0.15),(10,0.15),(4,0.15),(2,0.15)]
out = {}
for cap, regen in grid:
    cfg = Config().replace(mine_cap=float(cap), mine_regen=regen)
    for k, s in lineups.items():
        r = run(s, 24, 1000, cfg, quiet=True)
        key = f"cap{cap}_regen{regen}_{k}"
        out[key] = r
        line = " ".join(f"{t} {v['progress']:.3f} win {v['win_per_seat']:.2f}" for t, v in r["by_type"].items())
        print(f"cap={cap:>3} regen={regen:<5} {k:14s} early={r['finished_early']:.2f} len={r['length']:.0f} group={r['avg_group']:.2f} | {line}", flush=True)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expC.json","w"), indent=1)
