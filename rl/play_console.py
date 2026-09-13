"""
Play Ungroup from a terminal, one coarse decision at a time (for reviewing the design by playing it).

  python3 rl/play_console.py --preset life --seats me,loyal,bail,grudge,solo,bail --port 8123 &
  curl -s localhost:8123/state                         # compact situation report
  curl -s "localhost:8123/act?go=mine3&join=1&secs=6"   # go to mine 3, joinable on, advance 6 s
  curl -s "localhost:8123/act?go=pad&secs=8"           # go to my pad
  curl -s "localhost:8123/act?go=head&secs=5"          # go to the group head's pad
  curl -s "localhost:8123/act?go=body0&secs=4"         # go to the nearest other body
  curl -s "localhost:8123/act?leave=1&secs=3"          # hold leave
  curl -s "localhost:8123/act?intent=2&secs=1"         # declare intent for type 2 (0-3)
  curl -s "localhost:8123/act?go=stop&secs=10"
  curl -s localhost:8123/new?seed=7                    # new round (same seats)
  curl -s localhost:8123/replay > round.json           # replay of the current round so far

The game runs on the canonical C++ core; the seat named 'me' is the external seat you control. Bots act
every 6 ticks. The report shows what a player can see: public state only (no other needs).
"""

import argparse
import json
import math
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_native import parse_sets  # noqa: E402
from ungroup.native import NativeBatch  # noqa: E402

TYPES = "ABCD"


class Console:
    def __init__(self, seats, cfg, seed):
        self.seats = seats
        self.me = seats.index("me")
        self.n = len(seats)
        self.cfg = cfg.replace(n_players=self.n)
        self.batch = NativeBatch(1, self.cfg, seed=seed, decide_every=1)
        self.core_seats = ["policy" if s == "me" else s for s in seats]
        self.act = np.zeros((1, self.n, 4), dtype=np.int32)
        self.lock = threading.Lock()
        self.new(seed)

    def new(self, seed):
        self.seed = seed
        self.batch.set_seats(0, self.core_seats)
        self.batch.reset(0, seed)
        self.meta = self.batch.meta(0)
        self.act[:] = 0
        self.frames = [self.batch.frame(0)]
        self.events = []
        self.tick = 0

    def step(self, secs):
        steps = max(1, int(round(secs * 30 / 6)))   # bots decide every 6 ticks, like the ladder and the trainer
        new_events = []
        for k in range(steps):
            if self.batch.done(0):
                break
            self.batch.step(self.act, auto_reset=False, decide_every=6)
            self.tick += 6
            fr = self.batch.frame(0)
            new_events.extend(fr["events"])
            fr["events"] = list(fr["events"])
            self.frames.append(fr)
        self.act[0, self.me, 3] = 0  # intent is one-shot
        self.events = new_events
        return new_events

    # ------------------------------------------------------------ report
    def report(self):
        fr = self.batch.frame(0)
        m = self.meta
        me = self.me
        my_body = next(b for b in fr["bodies"] if me in b["m"])
        pos = np.array([my_body["x"], my_body["y"]])
        needs = m["needs"][me]
        banked = fr["players"][me]["banked"]
        prog = [min(banked[t] / needs[t], 1.0) for t in range(4)]
        pad = self.pad_pos(me, fr["R"])
        lines = []
        lines.append(f"t={fr['t']:.0f}s of {self.cfg.time_limit:.0f}  arena R={fr['R']:.2f}  {'ROUND OVER winner=' + str(self.batch.winner(0)) if self.batch.done(0) else ''}")
        lines.append(f"ME seat {me}: pos ({pos[0]:+.2f},{pos[1]:+.2f})  pad at ({pad[0]:+.2f},{pad[1]:+.2f}) dist {np.linalg.norm(pad - pos):.2f}  progress {sum(prog) / 4:.2f}")
        lines.append("   needs " + " ".join(f"{TYPES[t]}:{banked[t]:.0f}/{needs[t]}" for t in range(4)) +
                     f"   intent {TYPES[fr['players'][me]['intent']]}  joinable {'ON' if fr['players'][me]['join'] else 'off'}  " +
                     (f"leaving {fr['players'][me]['leaving']:.1f}s  " if fr['players'][me]['leaving'] >= 0 else "") +
                     (f"cooldown {fr['players'][me]['cd']:.1f}s  " if fr['players'][me]['cd'] > 0 else "") +
                     (f"BRANDED {fr['players'][me]['brand']:.0f}s" if fr['players'][me]['brand'] > 0 else ""))
        pool = my_body["pool"]
        if len(my_body["m"]) > 1:
            head = my_body["head"]
            lines.append(f"   GROUP of {len(my_body['m'])}: members {my_body['m']}  head {head}{' (me)' if head == me else ''} pays at pad dist {np.linalg.norm(self.pad_pos(head, fr['R']) - pos):.2f}  pool " +
                         " ".join(f"{TYPES[t]}:{pool[t]:.1f}" for t in range(4)) + f" (my share {sum(pool) / len(my_body['m']):.1f})" +
                         (f"  stunned {my_body['stun']:.1f}s" if my_body["stun"] > 0 else ""))
        else:
            lines.append("   SOLO carrying " + " ".join(f"{TYPES[t]}:{pool[t]:.1f}" for t in range(4)) + (f"  stunned {my_body['stun']:.1f}s" if my_body["stun"] > 0 else ""))
        # others by distance (the observation's slot order)
        others = []
        for j in range(self.n):
            if j == me:
                continue
            bj = next(b for b in fr["bodies"] if j in b["m"])
            d = np.linalg.norm(np.array([bj["x"], bj["y"]]) - pos)
            others.append((d, j, bj))
        others.sort(key=lambda x: (x[0], x[1]))
        lines.append("OTHERS (nearest first; body0..3 are the macro targets):")
        for k, (d, j, bj) in enumerate(others):
            p = fr["players"][j]
            pj = self.batch.progress(0, j)
            same = me in bj["m"]
            st = ("group %d" % len(bj["m"])) if len(bj["m"]) > 1 else "solo"
            flags = []
            if p["join"]: flags.append("joinable")
            if p["leaving"] >= 0: flags.append("leaving")
            if p["brand"] > 0: flags.append(f"branded{p['brand']:.0f}")
            if p["leaver"]: flags.append("public-leaver")
            if len(bj["m"]) > 1 and bj["head"] == j: flags.append("head")
            if same: flags.append("WITH ME")
            lines.append(f"   body{k if k < 4 else '-'} {self.seats[j]}#{j}: dist {d:.2f} {st} carry {sum(bj['pool']):.0f} intent {TYPES[p['intent']]} progress {pj:.2f} {' '.join(flags)}")
        lines.append("MINES (macro target mineN):")
        for mi, (mx, my) in enumerate(m["mine_pos"]):
            d = math.hypot(mx - pos[0], my - pos[1])
            alive = fr["alive"][mi]
            lines.append(f"   mine{mi} type {TYPES[m['mine_type'][mi]]} dist {d:.2f} stock {fr['mines'][mi]:.1f}{'' if alive else ' DEAD'}")
        if fr["picks"]:
            lines.append(f"FLOOR: {len(fr['picks'])} pickups, nearest {min(math.hypot(x - pos[0], y - pos[1]) for x, y, _ in fr['picks']):.2f}")
        if self.events:
            lines.append("EVENTS since last step:")
            for e in self.events[-12:]:
                lines.append("   " + self.describe(e))
        board = sorted(((self.batch.progress(0, j), j) for j in range(self.n)), reverse=True)
        lines.append("BOARD: " + "  ".join(f"{self.seats[j]}#{j} {p:.2f}" for p, j in board))
        return "\n".join(lines) + "\n"

    def pad_pos(self, i, R):
        r = max(R - self.cfg.pad_radius - 0.02, 0.1)
        a = self.meta["pads"][i]
        return np.array([r * math.cos(a), r * math.sin(a)])

    def describe(self, e):
        nm = lambda i: f"{self.seats[i]}#{i}"
        k = e["kind"]
        if k == "merge": return f"{e['t']:.0f}s MERGE {[nm(i) for i in e['a']]} + {[nm(i) for i in e['b']]}"
        if k == "leave": return f"{e['t']:.0f}s LEAVE {nm(e['player'])} took {sum(e['share']):.1f}"
        if k == "bank": return f"{e['t']:.0f}s BANK {nm(e['player'])} +{sum(e['amount']):.1f} (group {len(e['group'])})"
        if k == "spill": return f"{e['t']:.0f}s SPILL {[nm(i) for i in e['a']]} x {[nm(i) for i in e['b']]} {e['units']} units"
        if k == "crown": return f"{e['t']:.0f}s CROWN {nm(e['player'])} heads {[nm(i) for i in e['group']]}"
        if k == "mine_dead": return f"{e['t']:.0f}s MINE {e['mine']} died"
        if k in ("win", "timeout"): return f"{e['t']:.0f}s {k.upper()} {nm(e['player'])}"
        if k == "leave_start": return f"{e['t']:.0f}s {nm(e['player'])} starts leaving"
        return f"{e['t']:.0f}s {k}"

    def apply(self, q):
        go = q.get("go", [None])[0]
        a = self.act[0, self.me]
        if go is not None:
            if go == "stop": a[0] = 0
            elif go == "pad": a[0] = 18
            elif go == "head": a[0] = 19
            elif go.startswith("mine"): a[0] = 10 + int(go[4:])
            elif go.startswith("body"): a[0] = 20 + int(go[4:])
        if "join" in q: a[1] = int(q["join"][0])
        if "leave" in q: a[2] = int(q["leave"][0])
        if "intent" in q: a[3] = int(q["intent"][0]) + 1
        secs = float(q.get("secs", ["5"])[0])
        return self.step(secs)

    def replay(self):
        return {"config": self.cfg.__dict__, "seed": self.seed, "names": [f"{s}#{i}" for i, s in enumerate(self.seats)],
                "meta": {"seats": self.seats, "decide_every": 6, "frame_every": 6, "timeout_win": self.meta["timeout_win"],
                         "progress": [self.batch.progress(0, i) for i in range(self.n)]},
                "needs": self.meta["needs"], "pads": self.meta["pads"], "mine_pos": self.meta["mine_pos"], "mine_type": self.meta["mine_type"],
                "winner": self.batch.winner(0), "frames": self.frames, "actions": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="life")
    ap.add_argument("--set", action="append")
    ap.add_argument("--seats", default="me,loyal,bail,grudge,solo,bail")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--port", type=int, default=8123)
    a = ap.parse_args()
    cfg = parse_sets(a.set, a.preset)
    con = Console(a.seats.split(","), cfg, a.seed)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            with con.lock:
                if u.path == "/state":
                    body = con.report()
                elif u.path == "/act":
                    con.apply(q)
                    body = con.report()
                elif u.path == "/new":
                    con.new(int(q.get("seed", [con.seed + 1])[0]))
                    body = con.report()
                elif u.path == "/replay":
                    body = json.dumps(con.replay(), separators=(",", ":"))
                else:
                    body = "unknown\n"
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"console on http://localhost:{a.port}  seats {con.seats}  me = seat {con.me}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
