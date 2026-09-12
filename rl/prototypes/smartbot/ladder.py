import json
import sys
import os

sys.path.insert(0, "/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4/rl")
from ladder_native import run, fmt  # noqa: E402

LINEUPS = [
    ["smart"] * 3 + ["bail"] * 3,
    ["smart"] * 3 + ["loyal"] * 3,
    ["smart"] * 3 + ["solo"] * 3,
    ["smart"] * 6,
    ["smart"] + ["loyal"] * 5,
    ["smart"] + ["bail"] * 5,
    ["smart"] * 2 + ["bail"] * 2 + ["loyal"] * 2,
    ["smart"] * 3 + ["rammer"] * 3,
]
extra = [
    ["smart"] * 3 + ["kidnap"] * 3,
]

label = sys.argv[1]
games = int(sys.argv[2]) if len(sys.argv) > 2 else 24
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
out = []
for seats in LINEUPS + (extra if "--extra" in sys.argv else []):
    r = run(seats, games, seed, quiet=True)
    r.pop("sps", None)
    print(fmt(r))
    out.append(r)
d = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(d, f"results_{label}.json"), "w"), indent=1)
