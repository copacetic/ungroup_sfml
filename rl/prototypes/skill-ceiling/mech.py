# Mechanics checks with policy seats and set_direction: (a) pair stall when members push opposite, (b) trio 2-vs-1 speed,
# (c) lone pusher steers a group of 6, (d) relative approach speed of a solo chasing a fleeing group (spill threshold 0.40).
import sys, math
import numpy as np
sys.path.insert(0, "rl")
from ungroup.native import Config, NativeBatch
def make(n, seed=7):
    cfg = Config().replace(n_players=n)
    b = NativeBatch(1, cfg, seed=seed, decide_every=1); b.set_seats(0, ["policy"]*n); b.reset(0, seed); return b
def step(b, n, joins, dirs, ticks):
    acts = np.zeros((1, n, 4), dtype=np.int32)
    for i in range(n):
        acts[0, i] = (9, joins[i], 0, 0); b.set_direction(0, i, *dirs[i])
    for _ in range(ticks): b.step(acts, auto_reset=False, decide_every=1)
    return b.frame(0)
def group_up(b, n, ticks=400):
    # everybody joinable, everybody steers to the centre until one body remains
    for _ in range(ticks):
        fr = b.frame(0)
        if len(fr["bodies"]) == 1: return fr
        dirs = [(-bd["x"], -bd["y"]) for bd in fr["bodies"] for _m in bd["m"]]
        dmap = {}
        for bd in fr["bodies"]:
            for m in bd["m"]: dmap[m] = (-bd["x"], -bd["y"])
        step(b, n, [1]*n, [dmap[i] for i in range(n)], 1)
    return b.frame(0)
def speed(fr): bd = fr["bodies"][0]; return math.hypot(bd["vx"], bd["vy"])
# (a) pair
b = make(2); fr = group_up(b, 2); assert len(fr["bodies"]) == 1, fr["bodies"]
fr = step(b, 2, [1,1], [(1,0),(1,0)], 45); v_agree = speed(fr)
fr = step(b, 2, [1,1], [(1,0),(-1,0)], 45); v_oppose = speed(fr)
fr = step(b, 2, [1,1], [(1,0),(0,0)], 45); v_one = speed(fr)
print(f"pair: both push east {v_agree:.3f}  opposite {v_oppose:.3f}  one pushes, one follows {v_one:.3f}  (0.45/sqrt2={0.45/math.sqrt(2):.3f})")
# (b) trio 2 vs 1
b = make(3); fr = group_up(b, 3); assert len(fr["bodies"]) == 1
fr = step(b, 3, [1]*3, [(1,0),(1,0),(1,0)], 45); v3 = speed(fr)
fr = step(b, 3, [1]*3, [(1,0),(1,0),(-1,0)], 45); v21 = speed(fr)
fr = step(b, 3, [1]*3, [(1,0),(1,0),(0,1)], 45); v_side = speed(fr); bd = fr["bodies"][0]; ang = math.degrees(math.atan2(bd["vy"], bd["vx"]))
print(f"trio: agree {v3:.3f}  2 vs 1 opposite {v21:.3f} ({v21/v3:.2f}x)  2 east + 1 north: speed {v_side:.3f} heading {ang:.0f} deg")
# (c) six: one pusher, five followers; and 5 east vs 1 north
b = make(6); fr = group_up(b, 6, 900); print("six merged bodies:", len(fr["bodies"]))
if len(fr["bodies"]) == 1:
    fr = step(b, 6, [1]*6, [(1,0)]+[(0,0)]*5, 45); v_lone = speed(fr)
    fr = step(b, 6, [1]*6, [(1,0)]*5+[(0,1)], 45); bd = fr["bodies"][0]; ang = math.degrees(math.atan2(bd["vy"], bd["vx"]))
    fr = step(b, 6, [1]*6, [(1,0)]*5+[(-1,0)], 45); v51 = speed(fr)
    print(f"six: lone pusher steers at {v_lone:.3f} (0.45/sqrt6={0.45/math.sqrt(6):.3f}); 5 east + 1 north heading {ang:.0f} deg; 5 vs 1 opposite {v51:.3f}")
# (d) approach speed of a solo at 0.45 against a group of n fleeing at 0.45/sqrt(n): spill needs > 0.40
for n in [1,2,3,4,6]:
    v = 0.45/math.sqrt(n)
    print(f"solo rams group of {n}: stationary target rel={0.45:.2f}; fleeing rel={0.45-v:.3f}; head-on rel={0.45+v:.3f}; units if stationary={min(6, round(6*(0.45-0.40)+1))}, head-on={min(6, round(6*(0.45+v-0.40)+1))}")
# (e) keyboard: 8-way steering hit tolerance
for d in [0.2, 0.5, 1.0]:
    print(f"solo->pad from {d}: angular tolerance {math.degrees(math.atan((0.045+0.06)/d)):.1f} deg (keyboard step 45 deg); pad reach {(0.045+0.06):.3f}")
# (f) mine economics
for n in [1,2,3,4,6]:
    rate = 0.12*n*n; net = rate-0.5
    print(f"group {n}: yield {rate:.2f}/s ({rate/n:.3f}/s per member), drains a full mine in {30/net:.0f}s" if net>0 else f"group {n}: yield {rate:.2f}/s ({rate/n:.3f}/s per member), never drains (regen 0.5/s)")
print("time to mine 36 units solo:", 36/0.12, "s; pair share:", 36/(0.48/2), "s; trio:", 36/(1.08/3), "s; six:", 36/(4.32/6), "s")
print("arena crossing solo 2.0/0.45 =", 2/0.45, "s; six-group", 2/(0.45/math.sqrt(6)), "s")
print("outer mines die at t =", 120 + 120*(1-0.70)/0.5, "s (R<0.70)")
# (g) leave share arithmetic
for desc, wl, wp in [("leaver intent A, partner intent B", 3, 1), ("both intent A", 3, 3)]:
    print(f"pair leave, {desc}: leaver takes {3/(3+wp)*0.85*100:.0f}% of A, {1/(1+ (3 if wp==1 else 1))*0.85*100:.0f}% of the partner's intent type, {0.5*0.85*100:.0f}% of neutral types")
