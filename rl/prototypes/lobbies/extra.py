"""Follow-up checks: pad/mine overlap under shrink, bank cycle period, spill geography, depletion synchrony, mega-group endings."""
import sys, gzip, pickle, math, json
import numpy as np
sys.path.insert(0, "rl")
OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/lobbies"
def pad_pos(angle, R, pr=0.06):
    r = max(R - pr - 0.02, 0.1); return np.array([r * math.cos(angle), r * math.sin(angle)])
res = {}
for name in ["12_equal", "12_loyal", "12_bail", "20_equal", "20_loyal", "20_bail", "20_policy", "32_equal", "32_loyal", "32_bail"]:
    with gzip.open(f"{OUT}/rec_{name}.pkl.gz") as f: rec = pickle.load(f)
    cfg = rec["cfg"]; n = cfg.n_players
    out = dict(name=name)
    # 1. pad-mine overlap: for each env, per frame, count bodies that are at a member pad AND touching an alive mine
    both = {"<120": [], "120-190": [], ">190": []}; bank_at_mine = 0; banks = 0
    padmine_dist = {"t=0": [], "t=200": [], "t=239": []}
    interbank = []; spill_dmine = []; spill_dpad = []; spill_dcentre = []; spill_t = []
    depl_times = []
    mega_end = {"leave": 0, "round_end": 0, "other": 0}
    for e, frames in enumerate(rec["frames"]):
        meta = rec["metas"][e]; mine_pos = np.array(meta["mine_pos"]); pads = meta["pads"]
        last_bank = {}
        prev_largest = 1; prev_sizes = None
        for f in frames:
            R = f["R"]; t = f["t"]
            bodies = f["bodies"]; pos = np.array([[b["x"], b["y"]] for b in bodies])
            cnt = 0
            atmine = set()
            for bi, b in enumerate(bodies):
                r = cfg.solo_radius * math.sqrt(len(b["m"]))
                dm = np.linalg.norm(mine_pos - pos[bi], axis=1)
                hit = any(f["alive"][mi] and dm[mi] < r + cfg.mine_radius + 0.01 for mi in range(cfg.n_mines))
                atpad = any(np.linalg.norm(pad_pos(pads[i], R) - pos[bi]) < cfg.pad_radius + r for i in b["m"])
                if hit: atmine.update(b["m"])
                if hit and atpad: cnt += len(b["m"])
            key = "<120" if t < 120 else ("120-190" if t < 190 else ">190")
            both[key].append(cnt / n)
            for key2, tt in (("t=0", 0), ("t=200", 200), ("t=239", 239)):
                if abs(t - tt) < 0.2:
                    for i in range(n):
                        pp = pad_pos(pads[i], R)
                        d = min(np.linalg.norm(mine_pos[mi] - pp) for mi in range(cfg.n_mines) if f["alive"][mi]) if any(f["alive"]) else np.nan
                        padmine_dist[key2].append(d)
            for ev in f["events"]:
                if ev["kind"] == "bank":
                    banks += 1
                    if ev["player"] in atmine: bank_at_mine += 1
                    if ev["player"] in last_bank: interbank.append(ev["t"] - last_bank[ev["player"]])
                    last_bank[ev["player"]] = ev["t"]
                if ev["kind"] == "spill":
                    p = np.array([ev["x"], ev["y"]])
                    spill_dmine.append(min(np.linalg.norm(mine_pos - p, axis=1)))
                    spill_dpad.append(min(np.linalg.norm(pad_pos(pads[i], R) - p) for i in range(n)))
                    spill_dcentre.append(np.linalg.norm(p) / R); spill_t.append(ev["t"])
            sizes = [len(b["m"]) for b in bodies]; largest = max(sizes)
            if prev_largest >= 4 and largest < 4:
                kinds = [ev["kind"] for ev in f["events"]]
                mega_end["leave" if "leave" in kinds else "other"] += 1
            prev_largest = largest
        if prev_largest >= 4: mega_end["round_end"] += 1
        # depletion synchrony: stock time series correlation between mines
        st = np.array([f["mines"] for f in frames]); ts = np.array([f["t"] for f in frames])
        if name.startswith("32") or name.startswith("20"):
            d = np.diff(st, axis=0)
            c = np.corrcoef(d.T); iu = np.triu_indices(cfg.n_mines, 1)
            out.setdefault("stock_change_corr", []).append(float(np.nanmean(c[iu])))
            # same-type pairs vs different
            same = [c[i, j] for i, j in zip(*iu) if (i // 2) == (j // 2)]; diff = [c[i, j] for i, j in zip(*iu) if (i // 2) != (j // 2)]
            out.setdefault("corr_same_type", []).append(float(np.nanmean(same))); out.setdefault("corr_diff_type", []).append(float(np.nanmean(diff)))
        # depletion "wave": fraction of mines below 1 unit per frame; max simultaneous depleted
        dep = (st < 1.0) & np.array([f["alive"] for f in frames])
        out.setdefault("max_simult_depleted", []).append(int(dep.sum(1).max()))
        out.setdefault("frac_time_any_depleted", []).append(float(dep.any(1).mean()))
        out.setdefault("frac_time_half_depleted", []).append(float((dep.sum(1) >= 4).mean()))
    out["padmine_overlap_frac"] = {k: float(np.mean(v)) if v else None for k, v in both.items()}
    out["bank_at_mine_frac"] = bank_at_mine / max(1, banks); out["banks_total"] = banks
    out["padmine_dist"] = {k: float(np.mean(v)) if v else None for k, v in padmine_dist.items()}
    out["interbank_median"] = float(np.median(interbank)) if interbank else None
    out["interbank_median_excl_lt2"] = float(np.median([x for x in interbank if x > 2])) if interbank else None
    out["spill_dmine_median"] = float(np.median(spill_dmine)) if spill_dmine else None
    out["spill_within_mine_reach_frac"] = float(np.mean(np.array(spill_dmine) < 0.08 + 0.1)) if spill_dmine else None
    out["spill_dpad_median"] = float(np.median(spill_dpad)) if spill_dpad else None
    out["spill_rel_radius_median"] = float(np.median(spill_dcentre)) if spill_dcentre else None
    out["mega_end"] = mega_end
    for k in ("stock_change_corr", "corr_same_type", "corr_diff_type", "max_simult_depleted", "frac_time_any_depleted", "frac_time_half_depleted"):
        if k in out: out[k] = float(np.mean(out[k]))
    res[name] = out
    print(json.dumps(out))
json.dump(res, open(f"{OUT}/extra.json", "w"), indent=1)
