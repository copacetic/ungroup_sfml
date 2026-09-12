import sys, json, time
S = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"
sys.path.insert(0, S + "/rl")
from ladder_native import run
from ungroup.native import Config
LINEUPS = {
 "6solo": ["solo"]*6, "6bail": ["bail"]*6, "6loyal": ["loyal"]*6,
 "3bail3loyal": ["bail"]*3+["loyal"]*3, "3bail3solo": ["bail"]*3+["solo"]*3,
 "2loyal4solo": ["loyal"]*2+["solo"]*4, "1solo": ["solo"], "2loyal": ["loyal"]*2,
 "2bail2loyal2kidnap": ["bail"]*2+["loyal"]*2+["kidnap"]*2,
 "2bail2loyal2rammer": ["bail"]*2+["loyal"]*2+["rammer"]*2,
 "1loyal5bail": ["loyal"]+["bail"]*5, "12equal": ["solo","bail","loyal"]*4, "12loyal": (["loyal"]*4+["solo","bail"])*2, "1bail5loyal": ["bail"]+["loyal"]*5,
}
def main(name, kw, games=24, seed=1000, lineups=None):
    cfg = Config().replace(**kw)
    out = {}
    for ln in (lineups or LINEUPS):
        seats = LINEUPS[ln]
        r = run(seats, games, seed, cfg, quiet=True)
        row = {k: round(r[k], 3) for k in ("finished_early","length","avg_group","merges","leaves","spills","banks")}
        row["by"] = {t: (round(v["progress"],3), round(v["se"],3)) for t, v in r["by_type"].items()}
        out[ln] = row
        print(f"{name:12s} {ln:20s} early {row['finished_early']:.2f} len {row['length']:5.0f} grp {row['avg_group']:.2f} merges {row['merges']:5.1f} leaves {row['leaves']:5.1f} spills {row['spills']:4.1f} banks {row['banks']:5.1f} | " + " ".join(f"{t} {p:.3f}+-{se:.3f}" for t,(p,se) in row["by"].items()), flush=True)
    json.dump(out, open(f"{S}/ladder_{name}.json","w"), indent=1)
    return out
if __name__ == "__main__":
    name = sys.argv[1]
    kw = {}
    for a in sys.argv[2:]:
        k, v = a.split("=")
        kw[k] = float(v)
    main(name, kw)
