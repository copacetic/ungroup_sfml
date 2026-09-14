import sys, json, csv, time, itertools
sys.path.insert(0, "rl")
from ladder_native import run
from ungroup.native import Config

OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/payoff-landscape/"
GAMES, SEED = 16, 3000
cfg = Config()
rows = []
t0 = time.time()
lineups = []
for s in range(7):
    for b in range(7 - s):
        l = 6 - s - b
        lineups.append(("comp", ["solo"] * s + ["bail"] * b + ["loyal"] * l))
for inv in ["kidnap", "rammer"]:
    for res in ["solo", "bail", "loyal"]:
        lineups.append(("inv1", [inv] + [res] * 5))
for res in ["solo", "bail", "loyal"]:
    lineups.append(("inv2", ["kidnap", "rammer"] + [res] * 4))

for kind, seats in lineups:
    r = run(seats, GAMES, SEED, cfg, quiet=True)
    row = dict(kind=kind, seats=",".join(seats),
               n_solo=seats.count("solo"), n_bail=seats.count("bail"), n_loyal=seats.count("loyal"),
               n_kidnap=seats.count("kidnap"), n_rammer=seats.count("rammer"),
               finished_early=r["finished_early"], length=r["length"], avg_group=r["avg_group"],
               merges=r["merges"], leaves=r["leaves"], spills=r["spills"], banks=r["banks"],
               group_banks=r["group_banks"], alliances=r["alliances"], alliance_dur=r["alliance_dur"],
               alliances_long=r["alliances_long"])
    for t in ["solo", "bail", "loyal", "kidnap", "rammer"]:
        v = r["by_type"].get(t)
        row[f"{t}_prog"] = v["progress"] if v else ""
        row[f"{t}_se"] = v["se"] if v else ""
        row[f"{t}_win"] = v["win_per_seat"] if v else ""
    rows.append(row)
    print(f"{time.time()-t0:6.1f}s {kind} {row['seats']:40s} " + " ".join(
        f"{t}={r['by_type'][t]['progress']:.3f}+-{r['by_type'][t]['se']:.3f}/w{r['by_type'][t]['win_per_seat']:.2f}" for t in r["by_type"]), flush=True)

with open(OUT + "compositions_seed3000.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(rows, open(OUT + "compositions_seed3000.json", "w"), indent=1)
print("done", time.time() - t0)
