"""
Play Ungroup v2 in a browser: a WebSocket server that runs one room on the C++ core with human seats,
scripted bots, and (optionally) trained agents.

  python3 rl/play_server.py --port 8765 --bots bail,loyal --checkpoint rl/checkpoints/v2/latest.pt --agents 2
  then open rl/viewer/play.html (or http://<host>:8766/) and share the link with friends.

Every connected browser gets a seat; the remaining seats are filled with agents and bots. Rounds
restart automatically. Each finished round is saved as a replay JSON in --replays.
"""

import argparse
import asyncio
import http.server
import json
import os
import socketserver
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ungroup.native import PRESETS, Config, NativeBatch, preset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class Room:
    def __init__(self, cfg, seats, policy=None, frame_every=3, decide_every=6):
        self.cfg = cfg
        self.seats = list(seats)          # 'human' / 'policy' / bot name per seat
        self.n = len(seats)
        self.policy = policy
        self.frame_every = frame_every
        self.decide_every = decide_every
        self.batch = NativeBatch(1, cfg.replace(n_players=self.n), seed=int(time.time()) % 100000, decide_every=1)
        self.human_input = {i: dict(dir=(0.0, 0.0), join=False, leave=False, intent=0) for i in range(self.n) if seats[i] == "human"}
        self.round_id = 0
        self.frames = []
        self.actions = []
        self.pending_events = []
        self.tick = 0
        self.act = np.zeros((self.n, 4), dtype=np.int32)
        self.reset()

    def core_seats(self):
        return ["policy" if s in ("human", "policy") else s for s in self.seats]

    def reset(self):
        self.round_id += 1
        self.batch.set_seats(0, self.core_seats())
        self.batch.reset(0, int(time.time() * 1000) % 1000000007)
        self.obs = self.batch.observe()
        self.frames = [self.batch.frame(0)]
        self.actions = []
        self.pending_events = []
        self.tick = 0
        self.act[:] = 0
        for i in self.human_input:
            self.human_input[i]["intent"] = 0

    def meta(self):
        m = self.batch.meta(0)
        return dict(type="hello", seats=self.seats, needs=m["needs"], pads=m["pads"], mine_pos=m["mine_pos"], mine_type=m["mine_type"],
                    config=self.cfg.replace(n_players=self.n).__dict__, round=self.round_id, frame_every=self.frame_every)

    def step_tick(self):
        """Advance one physics tick; returns a frame dict every frame_every ticks, else None."""
        if self.tick % self.decide_every == 0:
            if self.policy is not None and any(s == "policy" for s in self.seats):
                import torch
                with torch.no_grad():
                    a, _ = self.policy.act(torch.from_numpy(self.obs[0]))
                a = a.numpy()
                for i, s in enumerate(self.seats):
                    if s == "policy":
                        self.act[i] = a[i]
            self.actions.append(self.act.tolist())
        for i, inp in self.human_input.items():
            dx, dy = inp["dir"]
            self.batch.set_direction(0, i, dx, dy)
            self.act[i] = (9, 1 if inp["join"] else 0, 1 if inp["leave"] else 0, inp["intent"])
            inp["intent"] = 0  # intent is a one-shot request
        self.obs, _, _, _ = self.batch.step(self.act[None], auto_reset=False, decide_every=1)
        self.tick += 1
        fr = self.batch.frame(0)
        self.pending_events.extend(fr["events"])
        if self.tick % self.frame_every == 0 or self.batch.done(0):
            fr["events"] = self.pending_events
            self.pending_events = []
            self.frames.append(fr)
            return fr
        return None

    def done(self):
        return self.batch.done(0)

    def replay(self):
        m = self.batch.meta(0)
        names = []
        for i, s in enumerate(self.seats):
            names.append(f"Human {i}" if s == "human" else f"Agent {i}" if s == "policy" else f"{s.capitalize()} bot {i}")
        return {"config": self.cfg.replace(n_players=self.n).__dict__, "seed": m["seed"], "names": names,
                "meta": {"progress": [self.batch.progress(0, i) for i in range(self.n)], "seats": self.seats, "timeout_win": m["timeout_win"],
                         "decide_every": self.decide_every, "frame_every": self.frame_every},
                "needs": m["needs"], "pads": m["pads"], "mine_pos": m["mine_pos"], "mine_type": m["mine_type"],
                "winner": m["winner"], "frames": self.frames, "actions": self.actions}


async def main():
    import websockets

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--http-port", type=int, default=8766, help="serves rl/viewer/play.html; 0 to disable")
    ap.add_argument("--humans", type=int, default=2, help="seats reserved for humans")
    ap.add_argument("--agents", type=int, default=0, help="seats filled by the checkpoint policy")
    ap.add_argument("--bots", default="bail,loyal,bail", help="comma list filling the remaining seats")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--replays", default=os.path.join(HERE, "replays"))
    ap.add_argument("--set", action="append")
    ap.add_argument("--preset", default="legacy", choices=sorted(PRESETS), help="named rule set (legacy, crown, bloom, life, series)")
    args = ap.parse_args()

    cfg = preset(args.preset)
    for kv in args.set or []:
        k, v = kv.split("=")
        f = Config.__dataclass_fields__[k]
        cfg = cfg.replace(**{k: int(v) if f.type is int else float(v)})
    policy = None
    if args.checkpoint:
        from train_v2 import load_checkpoint
        policy, _, _ = load_checkpoint(args.checkpoint)
    seats = ["human"] * args.humans + ["policy"] * (args.agents if policy else 0) + [b for b in args.bots.split(",") if b]
    room = Room(cfg, seats, policy, decide_every=getattr(policy, "decide_every", 6))
    os.makedirs(args.replays, exist_ok=True)
    clients = {}   # websocket -> seat index
    free = [i for i, s in enumerate(seats) if s == "human"]

    if args.http_port:
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=os.path.join(HERE, "viewer"), **k)
        httpd = socketserver.TCPServer(("", args.http_port), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"page: http://localhost:{args.http_port}/play.html?ws=ws://localhost:{args.port}")

    async def handler(ws):
        if not free:
            await ws.send(json.dumps({"type": "full"}))
            return
        seat = free.pop(0)
        clients[ws] = seat
        await ws.send(json.dumps(dict(room.meta(), seat=seat)))
        try:
            async for msg in ws:
                d = json.loads(msg)
                if d.get("type") == "input":
                    inp = room.human_input[seat]
                    inp["dir"] = (float(d.get("dx", 0)), float(d.get("dy", 0)))
                    inp["join"] = bool(d.get("join", False))
                    inp["leave"] = bool(d.get("leave", False))
                    if d.get("intent"):
                        inp["intent"] = int(d["intent"])
        finally:
            clients.pop(ws, None)
            free.append(seat)
            free.sort()
            room.human_input[seat] = dict(dir=(0.0, 0.0), join=False, leave=False, intent=0)

    async def loop():
        dt = room.cfg.dt
        next_t = time.perf_counter()
        while True:
            fr = room.step_tick()
            if fr is not None and clients:
                msg = json.dumps(dict(fr, type="frame"))
                await asyncio.gather(*[c.send(msg) for c in list(clients)], return_exceptions=True)
            if room.done():
                rep = room.replay()
                path = os.path.join(args.replays, f"round_{int(time.time())}.json")
                with open(path, "w") as f:
                    json.dump(rep, f, separators=(",", ":"))
                end = json.dumps({"type": "end", "winner": rep["winner"], "timeout": rep["meta"]["timeout_win"], "progress": rep["meta"]["progress"]})
                await asyncio.gather(*[c.send(end) for c in list(clients)], return_exceptions=True)
                await asyncio.sleep(6)
                room.reset()
                hello = room.meta()
                await asyncio.gather(*[c.send(json.dumps(dict(hello, seat=s))) for c, s in list(clients.items())], return_exceptions=True)
                next_t = time.perf_counter()
                continue
            next_t += dt
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = time.perf_counter()

    async with websockets.serve(handler, "", args.port):
        print(f"ws: ws://localhost:{args.port}  seats: {seats}")
        await loop()


if __name__ == "__main__":
    asyncio.run(main())
