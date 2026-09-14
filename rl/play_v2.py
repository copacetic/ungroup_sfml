"""
Evaluate, sweep, and record v2 checkpoints on the canonical core.

  python3 rl/play_v2.py eval   --checkpoint rl/checkpoints/v2/latest.pt --games 96
  python3 rl/play_v2.py sweep  --checkpoint rl/checkpoints/v2/latest.pt --games 48
  python3 rl/play_v2.py record --checkpoint rl/checkpoints/v2/latest.pt --seats policy,policy,policy,policy,policy,policy --tries 6 --out replay.json

Evaluation uses paired seeds (the same round seeds for every lineup) and reports progress with
standard errors, wins relative to chance, and the alliance statistics from the core.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_native import fmt, summarize  # noqa: E402
from train_v2 import load_checkpoint  # noqa: E402
from ungroup.native import Config, NativeBatch, record_game  # noqa: E402

torch.set_num_threads(2)


def play(policy, seats, games, seed, cfg, deterministic=False):
    n = len(seats)
    cfg = cfg.replace(n_players=n)
    b = NativeBatch(games, cfg, seed=seed, decide_every=getattr(policy, "decide_every", 6))
    for e in range(games):
        b.set_seats(e, seats)
        b.reset(e, seed + e)
    obs = b.observe()
    eps = []
    done_envs = set()
    while len(done_envs) < games:
        with torch.no_grad():
            a, _ = policy.act(torch.from_numpy(obs.reshape(games * n, -1)), deterministic=deterministic)
        obs, _, _, ep = b.step(a.numpy().reshape(games, n, 4), auto_reset=False)
        for d in ep:
            if d["env"] not in done_envs:
                done_envs.add(d["env"]); eps.append(d)
    return summarize(seats, eps)


LINEUPS = {
    "self-play": ["policy"] * 6,
    "vs bail": ["policy"] * 3 + ["bail"] * 3,
    "vs solo": ["policy"] * 3 + ["solo"] * 3,
    "vs loyal": ["policy"] * 3 + ["loyal"] * 3,
    "vs kidnap (held out)": ["policy"] * 3 + ["kidnap"] * 3,
    "vs rammer (held out)": ["policy"] * 3 + ["rammer"] * 3,
    "mixed": ["policy"] * 2 + ["bail", "loyal", "kidnap", "rammer"],
}

SWEEP = [("base", {}), ("mine_rate x0.8", {"mine_rate": 0.096}), ("mine_rate x1.25", {"mine_rate": 0.15}),
         ("leave_time 1.5", {"leave_time": 1.5}), ("leave_time 3", {"leave_time": 3.0}), ("need 14", {"need_primary": 14, "need_secondary": 4}),
         ("need 22", {"need_primary": 22, "need_secondary": 7}), ("forfeit 0.3", {"leave_forfeit": 0.3}), ("no shrink", {"shrink_start": 1.0}),
         ("spill 0.3", {"spill_min_speed": 0.3})]


def apply_sets(cfg, sets):
    for kv in sets or []:
        k, v = kv.split("=")
        f = Config.__dataclass_fields__[k]
        cfg = cfg.replace(**{k: int(v) if f.type is int else float(v)})
    return cfg


def cmd_eval(a):
    policy, cfg, ck = load_checkpoint(a.checkpoint)
    cfg = apply_sets(cfg, a.set)
    print(f"checkpoint {a.checkpoint}: {ck.get('samples', 0)} samples, git {ck.get('git')}, config mine_rate={cfg.mine_rate} forfeit={cfg.leave_forfeit}")
    for name, seats in LINEUPS.items():
        r = play(policy, seats, a.games, a.seed, cfg, a.deterministic)
        print(f"== {name}\n{fmt(r)}")


def cmd_sweep(a):
    policy, cfg, _ = load_checkpoint(a.checkpoint)
    print("lineup: 3 policy + 3 bail, and the bail-vs-solo ladder under the same config")
    for name, over in SWEEP:
        c = cfg.replace(**over)
        r = play(policy, ["policy"] * 3 + ["bail"] * 3, a.games, a.seed, c)
        pol, bail = r["by_type"]["policy"]["progress"], r["by_type"]["bail"]["progress"]
        from ladder_native import run
        lad = run(["bail"] * 3 + ["solo"] * 3, a.games, a.seed, c, quiet=True)
        gap = lad["by_type"]["bail"]["progress"] - lad["by_type"]["solo"]["progress"]
        print(f"{name:16s} policy {pol:.3f} bail {bail:.3f} margin {pol - bail:+.3f} | ladder bail-solo gap {gap:+.3f} | early_finish {r['finished_early']:.2f} alliances {r['alliances']:.1f} dur {r['alliance_dur']:.0f}s")


def cmd_record(a):
    policy, cfg, ck = load_checkpoint(a.checkpoint)
    cfg = apply_sets(cfg, a.set)
    seats = a.seats.split(",")
    names = [f"Agent {i}" if s == "policy" else f"{s.capitalize()} bot {i}" for i, s in enumerate(seats)]

    def act(obs):
        with torch.no_grad():
            x, _ = policy.act(torch.from_numpy(obs), deterministic=a.deterministic)
        return x.numpy()

    best, best_score = None, -1e9
    for k in range(a.tries):
        rep = record_game(act, seats, cfg.replace(n_players=len(seats)), seed=a.seed + k, names=names, decide_every=getattr(policy, "decide_every", 6),
                          meta={"checkpoint": os.path.basename(a.checkpoint), "samples": ck.get("samples", 0)})
        frames = rep["frames"]
        kinds = {}
        for f in frames:
            for e in f["events"]:
                kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        long_alliance = 0
        # score: real alliances (merges that are not fast re-merges), group banks, moderate spills, early finish
        score = 3 * kinds.get("merge", 0) + 2 * kinds.get("bank", 0) + kinds.get("spill", 0) + (30 if not rep["meta"]["timeout_win"] else 0)
        print(f"seed {a.seed + k}: winner={rep['winner']} timeout={rep['meta']['timeout_win']} t={frames[-1]['t']:.0f}s events={kinds} score={score}")
        if score > best_score:
            best, best_score = rep, score
    with open(a.out, "w") as f:
        json.dump(best, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("eval", "sweep", "record"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--games", type=int, default=48)
        p.add_argument("--seed", type=int, default=5000)
        p.add_argument("--deterministic", action="store_true")
        p.add_argument("--set", action="append", help="override a Config field for the evaluation, e.g. --set head_steer=2")
        if name == "record":
            p.add_argument("--seats", default="policy,policy,policy,policy,policy,policy")
            p.add_argument("--tries", type=int, default=4)
            p.add_argument("--out", default="replay.json")
    a = ap.parse_args()
    {"eval": cmd_eval, "sweep": cmd_sweep, "record": cmd_record}[a.cmd](a)
