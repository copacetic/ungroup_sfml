"""
Automatic health checks for a training run: the failures that took hours to notice by hand.

  python3 rl/health.py check --checkpoint rl/checkpoints/v4/v2_150.pt   # one snapshot, prints PASS/WARN lines
  python3 rl/health.py watch --run rl/checkpoints/v4                    # evaluates every new snapshot as it appears

Per snapshot (paired seeds, --games rounds per lineup, about two minutes on one core):

  strength      three agents vs three bail and vs three solo: progress margin with standard error.
                The training log pools mixed lineups and league opponents, so it hides "loses to a bot".
  cooperation   six agents in self-play: progress against the six-bail ladder baseline (agents that score
                below bots playing themselves are destroying each other), alliance mean duration, alliances
                over 10 s.
  spills        one recorded self-play round with every spill classified by the loads before impact:
                ram (empty-handed aggressor on a laden victim), clash (both laden), bump (light bodies,
                little or nothing dropped). A run whose spills are mostly bumps has a movement problem,
                not a ramming problem.
  heads         from log.csv at that update: P(join | solo) and P(leave | grouped) inside (0.02, 0.98).

Gates are absolute (a snapshot past --min-update that loses to a bot) and relative (a metric that fell
more than a fraction from the best snapshot so far). Rows go to <run>/health.csv and alerts to
<run>/alerts.log, so the trend is visible without re-running anything.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_native import run as ladder_run  # noqa: E402
from play_v2 import play  # noqa: E402
from train_v2 import load_checkpoint  # noqa: E402
from ungroup.native import record_game  # noqa: E402

torch.set_num_threads(1)

FIELDS = ["update", "samples", "margin_bail", "se_bail", "margin_solo", "se_solo", "self_prog", "bail_baseline",
          "alliance_dur", "alliances_long", "spills", "ram", "clash", "bump", "p_join_solo", "p_leave_grouped", "warnings"]


def classify_spills(rep):
    """Count spills in a recorded round by the bodies' loads in the frame before impact."""
    frames = rep["frames"]
    ram = clash = bump = 0
    for fi, f in enumerate(frames):
        for e in f["events"]:
            if e["kind"] != "spill":
                continue
            pf = frames[max(0, fi - 1)]

            def body_of(members):
                for b in pf["bodies"]:
                    if set(b["m"]) & set(members):
                        return b
                return None

            ba, bb = body_of(e["a"]), body_of(e["b"])
            if ba is None or bb is None:
                continue
            pa, pb = sum(ba["pool"]), sum(bb["pool"])
            nx, ny = bb["x"] - ba["x"], bb["y"] - ba["y"]
            d = math.hypot(nx, ny) or 1.0
            nx, ny = nx / d, ny / d
            va, vb = ba["vx"] * nx + ba["vy"] * ny, -(bb["vx"] * nx + bb["vy"] * ny)
            agg, vic = (pa, pb) if va > vb else (pb, pa)
            if agg < 1.0 and vic >= 3:
                ram += 1
            elif min(pa, pb) >= 3:
                clash += 1
            else:
                bump += 1
    return ram, clash, bump


def heads_at(run_dir, update):
    """P(join | solo) and P(leave | grouped) from log.csv, averaged over the 10 updates up to `update`."""
    path = os.path.join(run_dir, "log.csv")
    if not os.path.exists(path):
        return float("nan"), float("nan")
    rows = [r for r in csv.DictReader(open(path)) if r.get("p_join_solo") and update - 10 < int(r["update"]) <= update]
    if not rows:
        return float("nan"), float("nan")
    return (float(np.mean([float(r["p_join_solo"]) for r in rows])), float(np.mean([float(r["p_leave_grouped"]) for r in rows])))


def bail_baseline(cfg, games, seed):
    r = ladder_run(["bail"] * 6, games, seed, cfg.replace(n_players=6), quiet=True)
    return r["by_type"]["bail"]["progress"]


def check(path, games=16, seed=9000, baseline=None, best=None, min_update=200, run_dir=None):
    policy, cfg, ck = load_checkpoint(path)
    m = re.search(r"_(\d+)\.pt$", os.path.basename(path))
    update = int(m.group(1)) if m else -1
    row = dict(update=update, samples=ck.get("samples", 0))

    r = play(policy, ["policy"] * 3 + ["bail"] * 3, games, seed, cfg)
    row["margin_bail"] = r["by_type"]["policy"]["progress"] - r["by_type"]["bail"]["progress"]
    row["se_bail"] = math.hypot(r["by_type"]["policy"]["se"], r["by_type"]["bail"]["se"])
    r = play(policy, ["policy"] * 3 + ["solo"] * 3, games, seed, cfg)
    row["margin_solo"] = r["by_type"]["policy"]["progress"] - r["by_type"]["solo"]["progress"]
    row["se_solo"] = math.hypot(r["by_type"]["policy"]["se"], r["by_type"]["solo"]["se"])
    r = play(policy, ["policy"] * 6, games, seed, cfg)
    row["self_prog"] = r["by_type"]["policy"]["progress"]
    row["alliance_dur"] = r["alliance_dur"]
    row["alliances_long"] = r["alliances_long"]
    row["spills"] = r["spills"]
    row["bail_baseline"] = baseline if baseline is not None else bail_baseline(cfg, games, seed)

    def act(obs):
        with torch.no_grad():
            a, _ = policy.act(torch.from_numpy(obs))
        return a.numpy()

    rep = record_game(act, ["policy"] * 6, cfg.replace(n_players=6), seed=seed + 777)
    row["ram"], row["clash"], row["bump"] = classify_spills(rep)
    row["p_join_solo"], row["p_leave_grouped"] = heads_at(run_dir or os.path.dirname(path), update)

    warns = []
    mature = update >= min_update
    if mature and row["margin_bail"] < -0.03:
        warns.append(f"loses to bail by {-row['margin_bail']:.2f}")
    if mature and row["margin_solo"] < -0.03:
        warns.append(f"loses to solo by {-row['margin_solo']:.2f}")
    if row["self_prog"] < row["bail_baseline"] - 0.05:
        warns.append(f"self-play progress {row['self_prog']:.2f} below six-bail baseline {row['bail_baseline']:.2f}: agents hurt each other")
    if mature and row["alliance_dur"] < 8:
        warns.append(f"alliances last {row['alliance_dur']:.1f}s")
    total = row["ram"] + row["clash"] + row["bump"]
    if total >= 10 and row["bump"] / total > 0.6:
        warns.append(f"{row['bump']}/{total} spills are light bumps: movement tax, not ramming")
    for k in ("p_join_solo", "p_leave_grouped"):
        v = row[k]
        if not math.isnan(v) and not 0.02 < v < 0.98:
            warns.append(f"{k}={v:.3f} collapsed")
    if best:
        if row["margin_bail"] < best["margin_bail"] - 0.05:
            warns.append(f"margin vs bail fell from best {best['margin_bail']:.2f} to {row['margin_bail']:.2f}")
        if best["alliance_dur"] > 0 and row["alliance_dur"] < 0.7 * best["alliance_dur"]:
            warns.append(f"alliance duration fell from best {best['alliance_dur']:.1f}s to {row['alliance_dur']:.1f}s")
        if row["spills"] > 1.5 * max(best["spills"], 5):
            warns.append(f"spills rose from {best['spills']:.1f} to {row['spills']:.1f}")
    row["warnings"] = "; ".join(warns)
    return row


def fmt_row(row):
    s = (f"update {row['update']:>5} ({row['samples'] / 1e6:.1f}M): vs bail {row['margin_bail']:+.3f}±{row['se_bail']:.3f}  "
         f"vs solo {row['margin_solo']:+.3f}±{row['se_solo']:.3f}  self {row['self_prog']:.3f} (six bail {row['bail_baseline']:.3f})  "
         f"alliances {row['alliance_dur']:.1f}s long {row['alliances_long']:.1f}  spills {row['spills']:.1f} "
         f"[ram {row['ram']} clash {row['clash']} bump {row['bump']}]  join|solo {row['p_join_solo']:.2f} leave|grouped {row['p_leave_grouped']:.2f}")
    return s + ("\n    WARN " + "\n    WARN ".join(row["warnings"].split("; ")) if row["warnings"] else "\n    PASS")


def load_health(run_dir):
    path = os.path.join(run_dir, "health.csv")
    if not os.path.exists(path):
        return []
    return [{k: (float(v) if k not in ("warnings",) and v not in ("", "nan") else v) for k, v in r.items()} for r in csv.DictReader(open(path))]


def append_health(run_dir, row):
    path = os.path.join(run_dir, "health.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    if row["warnings"]:
        with open(os.path.join(run_dir, "alerts.log"), "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} update {row['update']}: {row['warnings']}\n")


def best_of(rows):
    if not rows:
        return None
    return dict(margin_bail=max(r["margin_bail"] for r in rows), alliance_dur=max(r["alliance_dur"] for r in rows),
                spills=min(r["spills"] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--games", type=int, default=16)
        p.add_argument("--seed", type=int, default=9000)
        p.add_argument("--min-update", type=int, default=200, help="absolute strength gates apply from this update")
        if name == "check":
            p.add_argument("--checkpoint", required=True)
        else:
            p.add_argument("--run", required=True)
            p.add_argument("--poll", type=int, default=60, help="seconds between directory scans")
            p.add_argument("--once", action="store_true", help="evaluate pending snapshots and exit")
    a = ap.parse_args()

    if a.cmd == "check":
        run_dir = os.path.dirname(a.checkpoint)
        best = best_of(load_health(run_dir))
        row = check(a.checkpoint, a.games, a.seed, best=best, min_update=a.min_update, run_dir=run_dir)
        print(fmt_row(row))
        append_health(run_dir, row)
        return

    baseline = None
    while True:
        done = {int(r["update"]) for r in load_health(a.run)}
        pending = sorted(glob.glob(os.path.join(a.run, "v2_*.pt")), key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))
        pending = [p for p in pending if int(re.search(r"_(\d+)\.pt$", p).group(1)) not in done]
        for path in pending:
            if baseline is None:
                _, cfg, _ = load_checkpoint(path)
                baseline = bail_baseline(cfg, a.games, a.seed)
            best = best_of(load_health(a.run))
            row = check(path, a.games, a.seed, baseline=baseline, best=best, min_update=a.min_update, run_dir=a.run)
            print(fmt_row(row), flush=True)
            append_health(a.run, row)
        if a.once:
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
