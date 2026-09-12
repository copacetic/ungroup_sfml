"""Persistence check: same seats, consecutive rounds, ledger carried over. 24 lobbies per lineup, 4 rounds."""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution")
from native_ev import Config, EvoBatch
def series(seats, rounds=4, games=24, seed=1000, persist=True, capital=0.0, cap=0.5, cash_at=0.4):
    n = len(seats); b = EvoBatch(games, Config().replace(n_players=n), seed=seed)
    for e in range(games): b.set_seats(e, seats); b.set_evo(e, inherit_cap=cap, persist=persist, cash_at=cash_at)
    acts = np.zeros((games, n, 4), dtype=np.int32)
    rows = []
    for r in range(rounds):
        for e in range(games): b.reset(e, seed + 100 * r + e)
        start = np.array([[b.start_progress(e, i) for i in range(n)] for e in range(games)])
        done = {}
        while len(done) < games:
            _, _, _, ep = b.step(acts, auto_reset=False)
            for d in ep: done.setdefault(d["env"], d)
        prog = np.array([done[e]["progress"] for e in range(games)])
        wins = np.zeros(n)
        for e in range(games): wins[done[e]["winner"]] += 1
        early = np.mean([not done[e]["timeout_win"] for e in range(games)])
        row = {"round": r, "early": float(early), "start": {}, "prog": {}, "win": {}}
        for s in sorted(set(seats), key=seats.index):
            idx = [i for i, x in enumerate(seats) if x == s]
            row["prog"][s] = float(prog[:, idx].mean()); row["win"][s] = float(wins[idx].sum() / (len(idx) * games)); row["start"][s] = float(start[:, idx].mean())
        rows.append(row)
        # inheritance for next round
        for e in range(games):
            fr = b.frame(e); me = b.meta(e)
            for i in range(n):
                sur = np.maximum(np.array(fr["players"][i]["banked"]) - np.array(me["needs"][i]), 0)
                b.set_inherit(e, i, capital * sur)
        print(f"{'+'.join(seats)} persist={persist} capital={capital} round {r}: early={early:.2f} " +
              " ".join(f"{s}: start {row['start'][s]:.2f} prog {row['prog'][s]:.3f} win {row['win'][s]:.2f}" for s in row["prog"]), flush=True)
    return rows
out = {}
for seats, persist, capital in [
    (["bail"]*4 + ["grudge"]*2, True, 0.0), (["bail"]*4 + ["grudge"]*2, False, 0.0),
    (["bail"]*5 + ["grudge"], True, 0.0), (["grudge"]*6, True, 0.0), (["grudge"]*5 + ["bail"], True, 0.0),
    (["grudge"]*3 + ["loyal"]*3, True, 0.0), (["solo"]*4 + ["grudge"]*2, True, 0.0),
    (["loyal"]*6, False, 0.5), (["loyal"]*5 + ["cash"], False, 0.5), (["loyal"]*5 + ["solo"], False, 0.5),
    (["cash"]*6, False, 0.5), (["cash"]*4 + ["loyal"]*2, False, 0.5), (["cash"]*3 + ["solo"]*3, False, 0.5),
    (["loyal"]*3 + ["bail"]*3, False, 0.5), (["solo"]*4 + ["cash"]*2, False, 0.5), (["loyal"]*5 + ["cash"], False, 0.0)]:
    out["+".join(seats) + f"_p{int(persist)}_c{capital}"] = series(seats, persist=persist, capital=capital)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expP.json", "w"), indent=1)
