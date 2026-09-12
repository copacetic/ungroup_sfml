"""Where does the scripted solo bot (alone) lose progress? Record banked per type, leftover pool, contact time."""
import sys, json, numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
rows = []
for seed in range(1000, 1012):
    cfg = Config().replace(n_players=1)
    b = NativeBatch(1, cfg, seed=seed, decide_every=1); b.set_seats(0, ["solo"]); b.reset(0, seed)
    meta = b.meta(0); need = np.array(meta["needs"][0]); mpos = np.array(meta["mine_pos"])
    acts = np.zeros((1, 1, 4), dtype=np.int32)
    contact = 0.0; banks = []; ep = None; last_bank_t = 0
    while ep is None:
        f = b.frame(0); bb = f["bodies"][0]; pos = np.array([bb["x"], bb["y"]])
        if min(np.linalg.norm(mpos - pos, axis=1)) < 0.045 + 0.08 + 0.01: contact += 1 / 30
        for e in f.get("events", []):
            if e.get("kind") == "bank": banks.append((round(f["t"], 1), e["amount"]))
        _, _, _, eps = b.step(acts, auto_reset=False, decide_every=1)
        if eps: ep = eps[0]
    f = b.frame(0); banked = np.array(f["players"][0]["banked"]); pool = np.array(f["bodies"][0]["pool"])
    prog = np.mean(np.minimum(banked / need, 1))
    # counterfactuals: (i) the same total units allocated secondaries-first; (ii) plus the leftover pool banked
    tot = banked.sum(); prim = int(need.argmax())
    def best_prog(u):
        sec = min(u, 18); return (sec / 6 * (1 / 4) if sec < 18 else 0.75) + min(max(u - 18, 0), 18) / 18 / 4
    rows.append(dict(seed=seed, progress=round(float(prog), 3), banked=banked.round(1).tolist(), need=need.tolist(), leftover=round(float(pool.sum()), 1),
                     contact_s=round(contact, 1), mined=round(tot + pool.sum(), 1), n_banks=len(banks), last_bank=banks[-1][0] if banks else None,
                     prog_if_realloc=round(best_prog(tot), 3), prog_if_realloc_and_banked=round(best_prog(tot + pool.sum()), 3)))
for r in rows: print(r)
import statistics as st
for k in ("progress", "leftover", "contact_s", "mined", "n_banks", "prog_if_realloc", "prog_if_realloc_and_banked"):
    print(k, round(st.mean(r[k] for r in rows), 3))
json.dump(rows, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy/solo_bot_attrib.json", "w"), indent=1)
