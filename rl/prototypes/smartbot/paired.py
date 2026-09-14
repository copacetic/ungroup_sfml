"""Paired margins (smart minus resident, per round) with standard errors for the final bot, 24 rounds, seed 1000."""
import json
import sys
import os
import numpy as np
sys.path.insert(0, "/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4/rl")
from ungroup.native import Config, NativeBatch  # noqa

LINEUPS = [
    ["smart"] * 3 + ["bail"] * 3, ["smart"] * 3 + ["loyal"] * 3, ["smart"] * 3 + ["solo"] * 3, ["smart"] * 6,
    ["smart"] + ["loyal"] * 5, ["smart"] + ["bail"] * 5, ["smart"] * 2 + ["bail"] * 2 + ["loyal"] * 2,
    ["smart"] * 3 + ["rammer"] * 3, ["smart"] * 3 + ["kidnap"] * 3,
]
rows = []
for seats in LINEUPS:
    n = len(seats); E = 24
    b = NativeBatch(E, Config().replace(n_players=n), seed=1000)
    for e in range(E):
        b.set_seats(e, seats); b.reset(e, 1000 + e)
    acts = np.zeros((E, n, 4), dtype=np.int32)
    eps = {}
    while len(eps) < E:
        _, _, _, ep = b.step(acts, auto_reset=False)
        for d in ep:
            eps.setdefault(d["env"], d)
    prog = np.array([eps[e]["progress"] for e in range(E)])
    si = [i for i, s in enumerate(seats) if s == "smart"]
    smart = prog[:, si].mean(axis=1)
    row = dict(lineup="+".join(seats), smart=float(smart.mean()), smart_se=float(smart.std(ddof=1) / np.sqrt(E)),
               early=float(np.mean([not eps[e]["timeout_win"] for e in range(E)])), length=float(np.mean([eps[e]["length"] for e in range(E)])),
               smart_wins=float(sum(1 for e in range(E) if eps[e]["winner"] in si) / E),
               smart_min=float(prog[:, si].min(axis=1).mean()), smart_max=float(prog[:, si].max(axis=1).mean()))
    for name in sorted(set(seats) - {"smart"}, key=seats.index):
        oi = [i for i, s in enumerate(seats) if s == name]
        other = prog[:, oi].mean(axis=1)
        diff = smart - other
        row[name] = float(other.mean()); row[name + "_se"] = float(other.std(ddof=1) / np.sqrt(E))
        row["margin_vs_" + name] = float(diff.mean()); row["margin_se_vs_" + name] = float(diff.std(ddof=1) / np.sqrt(E))
    rows.append(row)
    print(json.dumps(row))
d = os.path.dirname(os.path.abspath(__file__))
json.dump(rows, open(os.path.join(d, "paired_v4.json"), "w"), indent=1)
