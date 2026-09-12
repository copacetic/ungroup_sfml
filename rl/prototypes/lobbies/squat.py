import sys, gzip, pickle, math, numpy as np
sys.path.insert(0, "rl")
OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/lobbies"
for name in ["12_equal", "20_equal", "20_loyal", "20_bail", "20_policy", "32_equal", "32_loyal", "32_bail"]:
    with gzip.open(f"{OUT}/rec_{name}.pkl.gz") as f: rec = pickle.load(f)
    cfg = rec["cfg"]
    dep_occ = []; dep_n = []; crowd = []; longest_dep = []
    for e, frames in enumerate(rec["frames"]):
        mp = np.array(rec["metas"][e]["mine_pos"]); run = np.zeros(cfg.n_mines); best = 0
        for f in frames:
            occ = np.zeros(cfg.n_mines); nbod = np.zeros(cfg.n_mines)
            for b in f["bodies"]:
                r = cfg.solo_radius * math.sqrt(len(b["m"])); d = np.linalg.norm(mp - [b["x"], b["y"]], axis=1)
                for mi in range(cfg.n_mines):
                    if f["alive"][mi] and d[mi] < r + cfg.mine_radius + 0.01: occ[mi] += len(b["m"]); nbod[mi] += 1
            for mi in range(cfg.n_mines):
                if f["alive"][mi] and f["mines"][mi] < 1.0:
                    dep_occ.append(occ[mi] > 0); dep_n.append(occ[mi]); run[mi] += 1 / 3; best = max(best, run[mi])
                else: run[mi] = 0
            crowd.extend(nbod[nbod > 0])
        longest_dep.append(best)
    print(f"{name}: depleted-mine frames {len(dep_occ)}; occupied while depleted {np.mean(dep_occ) if dep_occ else float('nan'):.2f}; mean players on a depleted mine {np.mean(dep_n) if dep_n else 0:.2f}; bodies per occupied mine {np.mean(crowd):.2f} (p90 {np.percentile(crowd, 90):.0f}); longest continuous depletion {np.mean(longest_dep):.0f}s")
