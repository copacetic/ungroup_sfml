"""Count what the smart seats do over a few rounds: leaves, time in groups, time fleeing/stalled."""
import sys
import json
import numpy as np
sys.path.insert(0, "/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4/rl")
from ungroup.native import Config, NativeBatch  # noqa

seats = sys.argv[1].split(",")
games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
n = len(seats)
cfg = Config().replace(n_players=n)
b = NativeBatch(games, cfg, seed=1000)
for e in range(games):
    b.set_seats(e, seats)
    b.reset(e, 1000 + e)
acts = np.zeros((games, n, 4), dtype=np.int32)
ev = {}
grp_time = np.zeros(n)
steps = 0
done = [False] * games
leave_ctx = []
while not all(done):
    for e in range(games):
        if done[e]:
            continue
        f = b.frame(e)
        for k in f["events"]:
            ev[k["kind"]] = ev.get(k["kind"], 0) + 1
            if k["kind"] == "leave" and seats[k["player"]] == "smart":
                leave_ctx.append((round(k["t"]), k["from_size"], [round(x, 1) for x in k["share"]]))
        for body in f["bodies"]:
            for m in body["m"]:
                grp_time[m] += len(body["m"]) > 1
    _, _, dn, eps = b.step(acts, auto_reset=False)
    steps += 1
    for d in eps:
        done[d["env"]] = True
        print("env", d["env"], "len", round(d["length"]), "progress", [round(p, 2) for p in d["progress"]], "leaves", d["leaves"], "spills", d["spills"])
print("events per round:", {k: v / games for k, v in ev.items()})
print("group time fraction per seat:", [round(x / steps, 2) for x in grp_time])
print("smart leaves (t, from_size, share):", leave_ctx[:30])
