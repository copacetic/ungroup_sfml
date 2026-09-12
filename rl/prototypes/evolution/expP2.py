import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution")
sys.argv = ["x"]
exec(open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expP.py").read().split("out = {}")[0])
out = {}
for seats, persist, capital in [
    (["bail"]*4 + ["grudge"]*2, True, 0.0), (["bail"]*5 + ["grudge"], True, 0.0), (["grudge"]*6, True, 0.0),
    (["grudge"]*5 + ["bail"], True, 0.0), (["grudge"]*4 + ["bail"]*2, True, 0.0), (["grudge"]*3 + ["bail"]*3, True, 0.0),
    (["grudge"]*3 + ["loyal"]*3, True, 0.0), (["solo"]*4 + ["grudge"]*2, True, 0.0), (["grudge"]*2 + ["bail"]*2 + ["loyal"]*2, True, 0.0),
    (["loyal"]*5 + ["bail"], True, 0.5), (["grudge"]*4 + ["bail"]*2, True, 0.5)]:
    out["+".join(seats) + f"_p{int(persist)}_c{capital}"] = series(seats, rounds=5, persist=persist, capital=capital)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/expP2.json", "w"), indent=1)
