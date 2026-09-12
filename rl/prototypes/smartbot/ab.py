"""A/B a compile flag set on a few lineups. Usage: ab.py <label> [flags...]"""
import subprocess
import sys
import os

D = os.path.dirname(os.path.abspath(__file__))
label = sys.argv[1]
flags = sys.argv[2:]
subprocess.check_call([sys.executable, D + '/swap.py', D + '/bot_v3.cpp'] + flags, stdout=subprocess.DEVNULL)
sys.path.insert(0, "/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4/rl")
from ladder_native import run  # noqa: E402

lineups = [["smart"] * 3 + ["bail"] * 3, ["smart"] + ["bail"] * 5, ["smart"] * 6, ["smart"] * 3 + ["loyal"] * 3,
           ["smart"] * 3 + ["kidnap"] * 3]
print(label, flags)
for seats in lineups:
    r = run(seats, 24, 1000, quiet=True)
    parts = [f"{k} {v['progress']:.3f}+-{v['se']:.3f}" for k, v in r["by_type"].items()]
    print(f"  {'+'.join(sorted(set(seats), key=seats.index)):14s} early={r['finished_early']:.2f} leaves={r['leaves']:.1f} spills={r['spills']:.1f} group={r['avg_group']:.2f} | " + "  ".join(parts))
