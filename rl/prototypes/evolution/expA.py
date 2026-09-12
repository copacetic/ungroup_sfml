import sys, json
sys.path.insert(0, "rl")
from ladder_native import run, fmt
from ungroup.native import Config
lineups = {
 "5loyal_1kidnap": ["loyal"]*5+["kidnap"],
 "4loyal_2kidnap": ["loyal"]*4+["kidnap"]*2,
 "3loyal_3kidnap": ["loyal"]*3+["kidnap"]*3,
 "6kidnap": ["kidnap"]*6,
 "5kidnap_1bail": ["kidnap"]*5+["bail"],
 "5kidnap_1loyal": ["kidnap"]*5+["loyal"],
 "5kidnap_1solo": ["kidnap"]*5+["solo"],
 "4bail_2loyal": ["bail"]*4+["loyal"]*2,
 "4solo_2loyal": ["solo"]*4+["loyal"]*2,
 "5bail_1loyal": ["bail"]*5+["loyal"],
 "5solo_1loyal": ["solo"]*5+["loyal"],
 "6bail": ["bail"]*6,
 "6loyal": ["loyal"]*6,
}
out = {}
for k, s in lineups.items():
    r = run(s, 24, 1000, Config(), quiet=True)
    out[k] = r
    print("==", k); print(fmt(r))
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expA.json","w"), indent=1)
