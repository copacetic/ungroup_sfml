"""Timeline of seat 0 (smart) over a few rounds: merges, leaves, banks; plus final progress and time."""
import sys
import numpy as np
sys.path.insert(0, "/home/user/ungroup_sfml/.claude/worktrees/wf_09edbbde-75a-4/rl")
from ungroup.native import Config, NativeBatch  # noqa

seats = sys.argv[1].split(",")
games = int(sys.argv[2]) if len(sys.argv) > 2 else 2
n = len(seats)
cfg = Config().replace(n_players=n)
b = NativeBatch(games, cfg, seed=1000)
for e in range(games):
    b.set_seats(e, seats)
    b.reset(e, 1000 + e)
acts = np.zeros((games, n, 4), dtype=np.int32)
done = [False] * games
tl = {e: [] for e in range(games)}
solo_t = np.zeros(games); grp_t = np.zeros(games); carry = np.zeros(games); steps = np.zeros(games)
while not all(done):
    for e in range(games):
        if done[e]:
            continue
        f = b.frame(e)
        for body in f["bodies"]:
            if 0 in body["m"]:
                steps[e] += 1
                if len(body["m"]) == 1: solo_t[e] += 1
                else: grp_t[e] += 1
                carry[e] += sum(body["pool"]) / len(body["m"])
        for k in f["events"]:
            if k["kind"] == "merge" and (0 in k["a"] or 0 in k["b"]):
                tl[e].append(f"{k['t']:.0f}:merge{k['a']}+{k['b']}")
            if k["kind"] == "leave" and (k["player"] == 0):
                tl[e].append(f"{k['t']:.0f}:ILEAVE{k['from_size']} {sum(k['share']):.1f}")
            if k["kind"] == "leave" and k["player"] != 0:
                bi = [bb for bb in f["bodies"] if 0 in bb["m"]]
                tl[e].append(f"{k['t']:.0f}:leave{k['player']} {sum(k['share']):.1f}")
            if k["kind"] == "bank" and 0 in k["group"]:
                tl[e].append(f"{k['t']:.0f}:bank->{k['player']} {sum(k['amount']):.1f} g{len(k['group'])}")
            if k["kind"] == "spill" and (0 in k["a"] or 0 in k["b"]):
                tl[e].append(f"{k['t']:.0f}:spill{k['units']}")
            if k["kind"] in ("win", "timeout"):
                tl[e].append(f"{k['t']:.0f}:{k['kind']}{k['player']}")
    _, _, dn, eps = b.step(acts, auto_reset=False)
    for d in eps:
        done[d["env"]] = True
        e = d["env"]
        print(f"env {e} len {d['length']:.0f} progress {[round(p, 2) for p in d['progress']]} solo_frac {solo_t[e]/steps[e]:.2f} mean_carry {carry[e]/steps[e]:.1f}")
        print("   " + " ".join(tl[e]))
