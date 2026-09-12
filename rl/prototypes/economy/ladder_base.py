import sys, json
sys.path.insert(0, "rl")
from ladder_native import run, fmt
out = {}
for seats in [["solo"], ["loyal","loyal"], ["loyal"]*3, ["loyal"]*6, ["solo"]*6, ["bail"]*2]:
    r = run(seats, games=24, seed=1000, quiet=True)
    print(fmt(r))
    for k, v in r["by_type"].items():
        print(f"   {k}: progress={v['progress']:.3f} se={v['se']:.3f} win/seat={v['win_per_seat']:.2f}")
    out[",".join(seats)] = {k: v for k, v in r.items() if k != "seats"}
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy/ladder_base.json", "w"), indent=1)
