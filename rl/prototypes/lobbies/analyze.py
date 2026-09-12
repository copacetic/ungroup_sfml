import sys, os, gzip, pickle, json, math
import numpy as np
sys.path.insert(0, "rl")
OUT = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/lobbies"
rng = np.random.default_rng(0)
_nn_cache = {}

def nn_baseline(N, R):
    key = (N, round(R, 2))
    if key not in _nn_cache:
        vals = []
        for _ in range(60):
            r = R * np.sqrt(rng.random(N)); a = rng.random(N) * 2 * np.pi
            p = np.stack([r * np.cos(a), r * np.sin(a)], 1)
            d = np.linalg.norm(p[:, None] - p[None], axis=2); np.fill_diagonal(d, np.inf)
            vals.append(d.min(1).mean())
        _nn_cache[key] = float(np.mean(vals))
    return _nn_cache[key]

def pad_pos(angle, R, pad_radius=0.06):
    r = max(R - pad_radius - 0.02, 0.1)
    return np.array([r * math.cos(angle), r * math.sin(angle)])

def analyze(path):
    with gzip.open(path) as f: rec = pickle.load(f)
    seats = rec["seats"]; n = len(seats); cfg = rec["cfg"]
    G = len(rec["frames"])
    per_env = []
    for e in range(G):
        frames = rec["frames"][e]; meta = rec["metas"][e]
        mine_pos = np.array(meta["mine_pos"]); pads = meta["pads"]
        primary = [int(np.argmax(nd)) for nd in meta["needs"]]
        T = len(frames)
        ts = np.array([f["t"] for f in frames])
        m = dict(t=ts, n_bodies=np.zeros(T), largest=np.zeros(T), n_groups=np.zeros(T), grouped_frac=np.zeros(T),
                 nn=np.zeros(T), nn_base=np.zeros(T), mine_frac=np.zeros(T), picks=np.zeros(T), R=np.zeros(T),
                 stock=np.zeros((T, cfg.n_mines)), alive=np.zeros((T, cfg.n_mines)), occ=np.zeros((T, cfg.n_mines)),
                 radial=np.zeros(T), stunned_frac=np.zeros(T), carried=np.zeros(T),
                 size_hist=np.zeros((T, cfg.max_group + 1)), intent_same_grp=[], intent_same_nn=[], prim_same_grp=[], intent_expect=[])
        prev_pairs = set(); form = []; diss = []; pair_start = {}; alliance_durs = []
        camp_runs = []; camp_state = {}  # player -> (start_t, own)
        mine_seq = {i: [] for i in range(n)}  # sequence of distinct mines touched
        cur_mine = {i: -1 for i in range(n)}
        mega_spans = []; mega_on = None
        spills = []; events_all = []
        for k, f in enumerate(frames):
            R = f["R"]; m["R"][k] = R
            bodies = f["bodies"]; nb = len(bodies)
            m["n_bodies"][k] = nb
            sizes = np.array([len(b["m"]) for b in bodies])
            m["largest"][k] = sizes.max(); m["n_groups"][k] = (sizes >= 2).sum(); m["grouped_frac"][k] = sizes[sizes >= 2].sum() / n
            for s in sizes: m["size_hist"][k, min(s, cfg.max_group)] += 1
            pos = np.array([[b["x"], b["y"]] for b in bodies])
            if nb >= 2:
                d = np.linalg.norm(pos[:, None] - pos[None], axis=2); np.fill_diagonal(d, np.inf)
                nnd = d.min(1); m["nn"][k] = nnd.mean(); m["nn_base"][k] = nn_baseline(nb, R)
                nn_idx = d.argmin(1)
            else:
                m["nn"][k] = m["nn_base"][k] = np.nan
            m["radial"][k] = np.mean([np.linalg.norm(p) * len(b["m"]) for p, b in zip(pos, bodies)]) * nb / n
            m["stunned_frac"][k] = sum(len(b["m"]) for b in bodies if b["stun"] > 0) / n
            m["carried"][k] = sum(sum(b["pool"]) for b in bodies)
            m["picks"][k] = len(f["picks"])
            m["stock"][k] = f["mines"]; m["alive"][k] = f["alive"]
            touching = 0
            intents = [p["intent"] for p in f["players"]]
            pi = np.bincount(intents, minlength=4) / n; m["intent_expect"].append(float((pi ** 2).sum()))
            for bi, b in enumerate(bodies):
                r = cfg.solo_radius * math.sqrt(len(b["m"]))
                dm = np.linalg.norm(mine_pos - pos[bi], axis=1)
                hit = [mi for mi in range(cfg.n_mines) if f["alive"][mi] and dm[mi] < r + cfg.mine_radius + 0.01]
                if hit:
                    touching += len(b["m"]); m["occ"][k, hit[0]] += len(b["m"])
                for i in b["m"]:
                    mi = hit[0] if hit else -1
                    if mi >= 0 and mi != cur_mine[i]:
                        mine_seq[i].append((f["t"], mi))
                    if mi >= 0: cur_mine[i] = mi
                # pad camping: within pad reach of a member's pad, slow
                speed = math.hypot(b["vx"], b["vy"])
                at_own = any(np.linalg.norm(pad_pos(pads[i], R) - pos[bi]) < cfg.pad_radius + r for i in b["m"])
                at_other = (not at_own) and any(np.linalg.norm(pad_pos(pads[i], R) - pos[bi]) < cfg.pad_radius + r for i in range(n) if i not in b["m"])
                for i in b["m"]:
                    key = i
                    state = "own" if (at_own and speed < 0.05) else ("other" if (at_other and speed < 0.05) else None)
                    if key in camp_state and camp_state[key][1] != state:
                        st, kind = camp_state.pop(key)
                        if kind: camp_runs.append((kind, f["t"] - st, st, len(b["m"])))
                    if key not in camp_state and state:
                        camp_state[key] = (f["t"], state)
                if len(b["m"]) >= 2:
                    ms = b["m"]
                    for a in range(len(ms)):
                        for c in range(a + 1, len(ms)):
                            m["intent_same_grp"].append(intents[ms[a]] == intents[ms[c]])
                            m["prim_same_grp"].append(primary[ms[a]] == primary[ms[c]])
            m["mine_frac"][k] = touching / n
            if nb >= 2:
                for bi, b in enumerate(bodies):
                    if len(b["m"]) == 1 and len(bodies[nn_idx[bi]]["m"]) == 1:
                        m["intent_same_nn"].append(intents[b["m"][0]] == intents[bodies[nn_idx[bi]]["m"][0]])
            # pairs
            pairs = set()
            for b in bodies:
                ms = sorted(b["m"])
                for a in range(len(ms)):
                    for c in range(a + 1, len(ms)): pairs.add((ms[a], ms[c]))
            new = pairs - prev_pairs; gone = prev_pairs - pairs
            form.append(len(new)); diss.append(len(gone))
            for p in new: pair_start[p] = f["t"]
            for p in gone: alliance_durs.append(f["t"] - pair_start.pop(p, f["t"]))
            prev_pairs = pairs
            # mega group spans (largest >= 4)
            if sizes.max() >= 4 and mega_on is None: mega_on = (f["t"], int(sizes.max()))
            if sizes.max() >= 4 and mega_on is not None: mega_on = (mega_on[0], max(mega_on[1], int(sizes.max())))
            if sizes.max() < 4 and mega_on is not None: mega_spans.append((mega_on[0], f["t"] - mega_on[0], mega_on[1])); mega_on = None
            for ev in f["events"]:
                events_all.append(ev)
                if ev["kind"] == "spill": spills.append(ev)
        if mega_on is not None: mega_spans.append((mega_on[0], ts[-1] - mega_on[0], mega_on[1]))
        for p, st in pair_start.items(): alliance_durs.append(ts[-1] - st)
        for key, (st, kind) in camp_state.items():
            if kind: camp_runs.append((kind, ts[-1] - st, st, 0))
        # spill cascades: chain if within 2 s and 0.2 of a previous spill
        cascades = []; cur = []
        for ev in spills:
            if cur and any(ev["t"] - p["t"] <= 2.0 and math.hypot(ev["x"] - p["x"], ev["y"] - p["y"]) <= 0.2 for p in cur):
                cur.append(ev)
            else:
                if len(cur) >= 2: cascades.append(cur)
                cur = [ev]
        if len(cur) >= 2: cascades.append(cur)
        # depletion events: stock crossing below 1.0 from above
        depl = []
        for mi in range(cfg.n_mines):
            s = m["stock"][:, mi]
            for k in range(1, T):
                if s[k - 1] >= 1.0 and s[k] < 1.0: depl.append((ts[k], mi))
        m.update(form=np.array(form), diss=np.array(diss), alliance_durs=alliance_durs, camp_runs=camp_runs, mine_seq=mine_seq,
                 mega_spans=mega_spans, spills=spills, cascades=cascades, depl=depl, events=events_all, ep=rec["eps"][e], T=T)
        per_env.append(m)
    return rec, per_env

def tgrid(per_env, key, step=1.0):
    """Resample a per-frame series onto a common time grid (mean across envs, nan after episode end)."""
    grid = np.arange(0, 240.01, step)
    out = np.full((len(per_env), len(grid)), np.nan)
    for e, m in enumerate(per_env):
        v = m[key]
        idx = np.searchsorted(m["t"], grid, side="right") - 1
        ok = grid <= m["t"][-1]
        out[e, ok] = v[idx[ok]]
    return grid, out

def summarize(name, rec, per_env):
    n = len(rec["seats"]); cfg = rec["cfg"]
    r = dict(name=name, n=n, seats=rec["seats"], games=len(per_env))
    def mean_over(key, lo=0, hi=240):
        vals = []
        for m in per_env:
            sel = (m["t"] >= lo) & (m["t"] < hi)
            if sel.any(): vals.append(np.nanmean(m[key][sel]))
        return float(np.mean(vals)) if vals else float("nan")
    r["length"] = float(np.mean([m["t"][-1] for m in per_env]))
    r["n_bodies"] = mean_over("n_bodies"); r["n_bodies_early"] = mean_over("n_bodies", 0, 120); r["n_bodies_late"] = mean_over("n_bodies", 120, 240)
    r["largest_mean"] = mean_over("largest"); r["largest_max"] = int(max(m["largest"].max() for m in per_env))
    r["n_groups"] = mean_over("n_groups"); r["grouped_frac"] = mean_over("grouped_frac")
    sh = np.concatenate([m["size_hist"] for m in per_env]).sum(0); sh_players = sh * np.arange(cfg.max_group + 1)
    r["size_dist_players"] = (sh_players / sh_players.sum()).round(3).tolist()
    r["nn_ratio"] = float(np.nanmean(np.concatenate([m["nn"] / m["nn_base"] for m in per_env])))
    r["nn_ratio_early"] = float(np.nanmean(np.concatenate([(m["nn"] / m["nn_base"])[m["t"] < 120] for m in per_env])))
    r["nn_ratio_late"] = float(np.nanmean(np.concatenate([(m["nn"] / m["nn_base"])[m["t"] >= 120] for m in per_env])))
    r["mine_frac"] = mean_over("mine_frac"); r["mine_frac_early"] = mean_over("mine_frac", 0, 120); r["mine_frac_late"] = mean_over("mine_frac", 120, 240)
    r["picks"] = mean_over("picks"); r["picks_max"] = float(np.mean([m["picks"].max() for m in per_env]))
    r["stunned_frac"] = mean_over("stunned_frac")
    r["form_per_min"] = float(np.mean([m["form"].sum() / (m["t"][-1] / 60) for m in per_env]))
    r["diss_per_min"] = float(np.mean([m["diss"].sum() / (m["t"][-1] / 60) for m in per_env]))
    durs = np.concatenate([m["alliance_durs"] for m in per_env])
    r["pair_dur_median"] = float(np.median(durs)) if len(durs) else 0; r["pair_dur_mean"] = float(np.mean(durs)) if len(durs) else 0
    r["pair_dur_p90"] = float(np.percentile(durs, 90)) if len(durs) else 0
    # mega groups
    spans = [s for m in per_env for s in m["mega_spans"]]
    r["mega_spans_per_round"] = len(spans) / len(per_env); r["mega_dur_mean"] = float(np.mean([s[1] for s in spans])) if spans else 0
    r["mega_dur_max"] = float(max([s[1] for s in spans])) if spans else 0; r["mega_size_max"] = int(max([s[2] for s in spans])) if spans else 0
    r["time_with_mega_frac"] = float(np.mean([np.mean(m["largest"] >= 4) for m in per_env]))
    r["time_with_size6_frac"] = float(np.mean([np.mean(m["largest"] >= 6) for m in per_env]))
    # camping
    runs = [c for m in per_env for c in m["camp_runs"] if c[1] >= 3.0]
    own = [c for c in runs if c[0] == "own"]; oth = [c for c in runs if c[0] == "other"]
    r["camp_own_runs_per_round"] = len(own) / len(per_env); r["camp_own_dur_mean"] = float(np.mean([c[1] for c in own])) if own else 0
    r["camp_own_dur_max"] = float(max([c[1] for c in own])) if own else 0
    r["camp_own_player_seconds_per_round"] = sum(c[1] for c in own) / len(per_env)
    r["camp_own_after120_frac"] = float(np.mean([c[2] >= 120 for c in own])) if own else 0
    r["camp_other_runs_per_round"] = len(oth) / len(per_env)
    # spills / cascades
    r["spills_per_round"] = float(np.mean([len(m["spills"]) for m in per_env]))
    r["spills_early"] = float(np.mean([sum(1 for s in m["spills"] if s["t"] < 120) for m in per_env]))
    r["spills_late"] = float(np.mean([sum(1 for s in m["spills"] if s["t"] >= 120) for m in per_env]))
    r["spill_units_per_round"] = float(np.mean([sum(s["units"] for s in m["spills"]) for m in per_env]))
    cas = [c for m in per_env for c in m["cascades"]]
    r["cascades_per_round"] = len(cas) / len(per_env)
    r["cascade_len_mean"] = float(np.mean([len(c) for c in cas])) if cas else 0; r["cascade_len_max"] = max([len(c) for c in cas]) if cas else 0
    r["spills_in_cascades_frac"] = (sum(len(c) for c in cas) / max(1, sum(len(m["spills"]) for m in per_env)))
    # depletion
    depl = [d for m in per_env for d in m["depl"]]
    r["depletions_per_round"] = len(depl) / len(per_env)
    r["depl_outer"] = sum(1 for d in depl if d[1] % 2 == 0) / len(per_env); r["depl_inner"] = sum(1 for d in depl if d[1] % 2 == 1) / len(per_env)
    r["first_depletion_t"] = float(np.nanmean([min([d[0] for d in m["depl"]]) if m["depl"] else np.nan for m in per_env]))
    r["rounds_with_depletion"] = float(np.mean([bool(m["depl"]) for m in per_env]))
    st = np.stack([m["stock"][:500].mean(1) for m in per_env if m["T"] >= 500]) if any(m["T"] >= 500 for m in per_env) else None
    r["stock_mean_0_120"] = float(np.mean([m["stock"][m["t"] < 120].mean() for m in per_env]))
    r["stock_min_frame_mean"] = float(np.mean([m["stock"][m["t"] < 190].mean(1).min() for m in per_env]))
    # migration
    switches = [len(seq) - 1 for m in per_env for seq in m["mine_seq"].values() if len(seq) >= 1]
    r["mine_switches_per_player"] = float(np.mean(switches)) if switches else 0
    # ring occupancy
    occ = np.concatenate([m["occ"] for m in per_env]); r["outer_occ_frac"] = float(occ[:, 0::2].sum() / max(1, occ.sum()))
    # segregation
    g = np.concatenate([m["intent_same_grp"] for m in per_env]) if any(len(m["intent_same_grp"]) for m in per_env) else np.array([])
    r["intent_same_in_group"] = float(g.mean()) if len(g) else np.nan
    r["intent_same_expected"] = float(np.mean(np.concatenate([m["intent_expect"] for m in per_env])))
    p = np.concatenate([m["prim_same_grp"] for m in per_env]) if any(len(m["prim_same_grp"]) for m in per_env) else np.array([])
    r["primary_same_in_group"] = float(p.mean()) if len(p) else np.nan
    nnp = np.concatenate([m["intent_same_nn"] for m in per_env]) if any(len(m["intent_same_nn"]) for m in per_env) else np.array([])
    r["intent_same_nn_solo"] = float(nnp.mean()) if len(nnp) else np.nan
    # oscillation: autocorrelation of n_groups detrended (per env), dominant period
    grid, ng = tgrid(per_env, "n_groups", 1.0)
    per = []; ac = []
    for e in range(len(per_env)):
        v = ng[e]; v = v[~np.isnan(v)]
        if len(v) < 60: continue
        v = v - np.convolve(v, np.ones(31) / 31, mode="same")
        v = v[15:-15]
        f = np.abs(np.fft.rfft(v - v.mean())) ** 2; freqs = np.fft.rfftfreq(len(v), 1.0)
        sel = (freqs > 1 / 120) & (freqs < 0.5)
        per.append(1 / freqs[sel][np.argmax(f[sel])])
        a = np.correlate(v - v.mean(), v - v.mean(), "full")[len(v) - 1:]; a /= a[0]
        ac.append(a[:60])
    r["osc_period_s"] = float(np.median(per)) if per else np.nan
    ac = np.mean(ac, 0) if ac else np.zeros(60)
    r["osc_ac_first_min"] = float(ac[1:].min()); r["osc_ac_first_min_lag"] = int(ac[1:].argmin() + 1)
    r["osc_ac_second_peak"] = float(ac[ac[1:].argmin() + 1:].max()) if len(ac) > ac[1:].argmin() + 2 else np.nan
    r["n_groups_std_within_round"] = float(np.mean([np.nanstd(ng[e]) for e in range(len(per_env))]))
    # shrink effects
    r["radial_mean_100_120"] = mean_over("radial", 100, 120); r["radial_mean_200_240"] = mean_over("radial", 200, 240)
    r["R_at_200"] = 1 + (0.5 - 1) * ((200 / 240 - 0.5) / 0.5)
    r["mine_dead_t"] = float(np.mean([np.mean([ev["t"] for ev in m["events"] if ev["kind"] == "mine_dead"] or [np.nan]) for m in per_env]))
    def rate(kind, lo, hi):
        return float(np.mean([sum(1 for ev in m["events"] if ev["kind"] == kind and lo <= ev["t"] < hi) / ((min(hi, m["t"][-1]) - lo) / 60) if m["t"][-1] > lo else np.nan for m in per_env]))
    for kind in ("merge", "leave", "spill", "bank"):
        r[f"{kind}_per_min_0_120"] = rate(kind, 0, 120); r[f"{kind}_per_min_120_190"] = rate(kind, 120, 190); r[f"{kind}_per_min_190_240"] = rate(kind, 190, 240)
    r["progress"] = {k: round(v["progress"], 3) for k, v in rec["summary"]["by_type"].items()}
    r["finished_early"] = rec["summary"]["finished_early"]
    return r

if __name__ == "__main__":
    names = [(12, "equal"), (12, "loyal"), (12, "bail"), (20, "equal"), (20, "loyal"), (20, "bail"), (20, "policy"), (32, "equal"), (32, "loyal"), (32, "bail")]
    allres = {}; series = {}
    for n, kind in names:
        rec, per_env = analyze(f"{OUT}/rec_{n}_{kind}.pkl.gz")
        r = summarize(f"{n}_{kind}", rec, per_env)
        allres[f"{n}_{kind}"] = r
        s = {}
        for key in ("n_bodies", "largest", "n_groups", "grouped_frac", "mine_frac", "picks", "stunned_frac", "carried", "radial"):
            grid, v = tgrid(per_env, key, 1.0); s[key] = np.nanmean(v, 0).tolist()
        grid, v = tgrid(per_env, "nn", 1.0); grid, vb = tgrid(per_env, "nn_base", 1.0); s["nn_ratio"] = np.nanmean(v / vb, 0).tolist()
        # per mine stock (mean over envs), ordered outer/inner
        st = np.full((len(per_env), 241, rec["cfg"].n_mines), np.nan)
        for e, m in enumerate(per_env):
            idx = np.searchsorted(m["t"], grid, side="right") - 1; ok = grid <= m["t"][-1]
            st[e, ok] = m["stock"][idx[ok]]
        s["stock_outer"] = np.nanmean(st[:, :, 0::2], (0, 2)).tolist(); s["stock_inner"] = np.nanmean(st[:, :, 1::2], (0, 2)).tolist()
        s["stock_env0"] = np.nan_to_num(st[0]).tolist()
        s["n_groups_env0"] = np.nan_to_num(tgrid(per_env, "n_groups", 1.0)[1][0]).tolist()
        s["largest_env0"] = np.nan_to_num(tgrid(per_env, "largest", 1.0)[1][0]).tolist()
        # spill time histogram (per 10 s)
        s["spills_hist"] = np.histogram([sp["t"] for m in per_env for sp in m["spills"]], bins=np.arange(0, 241, 10))[0].tolist()
        s["form_hist"] = (np.histogram(np.concatenate([np.repeat(m["t"], m["form"].astype(int)) for m in per_env]), bins=np.arange(0, 241, 10))[0] / len(per_env)).tolist()
        series[f"{n}_{kind}"] = s
        # per-round striking score for 20 player rounds
        if n == 20:
            for e, m in enumerate(per_env):
                sc = 3 * len(m["cascades"]) + sum(s_[2] for s_ in m["mega_spans"]) + len(m["spills"]) / 5 + m["picks"].max() / 2
                print(f"  round {kind} env {e} seed {5000+n+e}: spills {len(m['spills'])} cascades {len(m['cascades'])} mega spans {len(m['mega_spans'])} maxsize {int(m['largest'].max())} picks_max {int(m['picks'].max())} len {m['t'][-1]:.0f} score {sc:.1f}")
        print(f"== {n} {kind}")
        for k, v in r.items():
            if k not in ("seats",): print(f"   {k}: {v}")
    with open(f"{OUT}/results.json", "w") as f: json.dump(allres, f, indent=1, default=float)
    with open(f"{OUT}/series.json", "w") as f: json.dump(series, f)
