"""
Play one recorded game with a trained policy (and optionally scripted bots in some seats) and
write a replay JSON that the viewer can load.

Usage:
  python3 rl/record_replay.py --checkpoint rl/checkpoints/policy_latest.pt --players 6 --out replay.json
  python3 rl/record_replay.py --checkpoint ... --seats policy,policy,policy,policy,bail,solo
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ppo import load_policy  # noqa: E402
from ungroup import Config, UngroupEnv  # noqa: E402
from ungroup.bots import BOTS  # noqa: E402


def play(checkpoint, seats, seed=0, deterministic=False, decide_every=2, cfg_kwargs=None):
    cfg = Config(n_players=len(seats), **(cfg_kwargs or {}))
    env = UngroupEnv(cfg, seed=seed, decide_every=decide_every, record=True)
    obs = env.reset(seed=seed)
    policy = load_policy(checkpoint, env.obs_dim)
    bots = [None if s == "policy" else BOTS[s]() for s in seats]
    done = False
    torch.manual_seed(seed)
    while not done:
        act, _, _ = policy.act(torch.from_numpy(obs), deterministic=deterministic)
        act = act.numpy()
        for i, b in enumerate(bots):
            if b is not None:
                act[i] = b.act(env.game, i)
        obs, _, done, _ = env.step(act)
    g = env.game
    names = []
    for i, s in enumerate(seats):
        names.append(f"Agent {i}" if s == "policy" else f"{s.capitalize()} bot {i}")
    rep = g.replay(names=names, meta={"checkpoint": os.path.basename(checkpoint), "seats": seats, "seed": seed,
                                      "progress": [g.progress(i) for i in range(len(seats))]})
    return rep


def summarize(rep):
    frames = rep["frames"]
    kinds = {}
    for f in frames:
        for e in f["events"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    avg_group = np.mean([np.mean([len(b["m"]) for b in f["bodies"]] if f["bodies"] else [1]) for f in frames])
    return dict(length=frames[-1]["t"], winner=rep["winner"], events=kinds, frames=len(frames),
                progress=[round(x, 2) for x in rep["meta"]["progress"]], mean_body_size=round(float(avg_group), 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--players", type=int, default=6)
    ap.add_argument("--seats", default=None, help="comma list of policy|solo|bail|random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--out", default="replay.json")
    ap.add_argument("--tries", type=int, default=1, help="record several seeds and keep the most eventful finished game")
    a = ap.parse_args()
    seats = a.seats.split(",") if a.seats else ["policy"] * a.players
    best, best_score = None, -1
    for k in range(a.tries):
        rep = play(a.checkpoint, seats, seed=a.seed + k, deterministic=a.deterministic)
        s = summarize(rep)
        score = (s["events"].get("merge", 0) + s["events"].get("leave", 0) + 2 * s["events"].get("spill", 0)
                 + (50 if s["winner"] >= 0 else 0))
        print(f"seed {a.seed + k}: {s} score={score}")
        if score > best_score:
            best, best_score = rep, score
    with open(a.out, "w") as f:
        json.dump(best, f, separators=(",", ":"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")
