import sys, os, json
SC = os.path.dirname(os.path.abspath(__file__))
variant, seed, which = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, "rl")
import ungroup.native as native
native.LIB = os.path.join(SC, f"lib_{variant}.so")
from ladder_native import run
from ungroup.native import Config
full = {
 "6solo": ["solo"]*6, "6bail": ["bail"]*6, "6loyal": ["loyal"]*6,
 "3b3l": ["bail"]*3+["loyal"]*3, "4b2l": ["bail"]*4+["loyal"]*2, "5b1l": ["bail"]*5+["loyal"],
 "1b5l": ["bail"]+["loyal"]*5, "1s5l": ["solo"]+["loyal"]*5, "4s2l": ["solo"]*4+["loyal"]*2,
 "2b2l2k": ["bail"]*2+["loyal"]*2+["kidnap"]*2, "2b2l2r": ["bail"]*2+["loyal"]*2+["rammer"]*2,
 "3b3s": ["bail"]*3+["solo"]*3,
}
key = {k: full[k] for k in ["6bail","6loyal","3b3l","5b1l","1b5l","1s5l"]}
lineups = full if which == "full" else key
for k, seats in lineups.items():
    r = run(seats, 24, seed, Config(), quiet=True)
    line = f"{variant:7s} s{seed} {k:7s} early={r['finished_early']:.2f} len={r['length']:.0f} grp={r['avg_group']:.2f} merges={r['merges']:.1f} leaves={r['leaves']:.1f} banks={r['banks']:.1f} all={r['alliances']:.1f} dur={r['alliance_dur']:.0f}s |"
    for t, v in r["by_type"].items():
        line += f" {t} {v['progress']:.3f}±{v['se']:.3f} w{v['win_per_seat']:.2f}"
    print(line, flush=True)
