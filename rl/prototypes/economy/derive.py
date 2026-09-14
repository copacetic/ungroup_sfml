"""Closed-form resource economy of Ungroup v2 (default Config)."""
import math, json
base, lerp, r1, mr, cap, regen, rate1, T = 0.45, 6.0, 0.045, 0.08, 30.0, 0.5, 0.12, 240.0
need = 36.0; need_p, need_s = 18.0, 6.0
lag = 1 / lerp   # ~0.17 s lost per acceleration (v(t) = v0 (1 - e^{-6t}))
out = {}
print("== per-mine drain by group size ==")
for n in range(1, 7):
    rate = rate1 * n ** 2
    net = rate - regen
    t_dry = cap / net if net > 0 else math.inf
    yield_dry = cap + regen * t_dry if net > 0 else math.inf
    print(f"n={n}: rate {rate:.2f}/s ({rate/n:.2f} per member), net drain {net:+.2f}/s, stock lasts {t_dry:.1f}s, units before regen-limited {yield_dry:.1f}")
    out[f"drain_n{n}"] = dict(rate=rate, net=net, t_dry=t_dry)
print("\n== fastest first finish (one member reaches progress 1) ==")
# geometry: inner mines on r=0.35 at 90 deg spacing (chord 0.495), outer on r=0.62; contact zone r_n + 0.08 + 0.01;
# start at r=0.72 on the pad angle; pads at r=0.92 (R=1). Merge: everyone converges on the origin: 0.72/0.45 + lag.
for n in (1, 2, 3, 6):
    rn = r1 * math.sqrt(n); v = base / math.sqrt(n); zone = rn + mr + 0.01; mult = 1 + 0.15 * (n - 1)
    rate = rate1 * n ** 2
    pool_needed = need / mult
    t_contact = pool_needed / rate
    t_merge = 0 if n == 1 else 0.72 / base + 2 * lag
    # legs: origin->inner mine (0.35 - zone), 3 inter-mine chords (0.495 - 2*zone, but the body sits on the arrival side: ~+0.1), last mine->pad (0.92-0.35 - zone - (rn+0.06))
    legs = [max(0.35 - zone, 0)] + 3 * [max(0.495 - 2 * zone, 0) + 0.10] + [0.57 - zone - (rn + 0.06)]
    t_travel = sum(l / v + lag for l in legs)
    t_finish = t_merge + t_contact + t_travel
    # progress at 240 s for a lone body (no bonus): secondaries first
    units_240 = rate * (T - (sum(legs[:-1]) / v + 4 * lag + 1.5)) if n == 1 else None
    prog = None
    if n == 1:
        u = units_240; prog = min(u, 18) / 24 + max(min(u - 18, 18), 0) / 72
    print(f"n={n}: pool needed {pool_needed:.2f} (x{mult:.2f} bonus) -> contact {t_contact:.1f}s + merge {t_merge:.1f}s + travel {t_travel:.1f}s = first finish {t_finish:.1f}s"
          + (f"; cannot finish in 240 s: units at 240 s = {units_240:.1f} -> progress {prog:.3f}" if n == 1 else ""))
    out[f"first_finish_n{n}"] = dict(pool_needed=pool_needed, t_contact=t_contact, t_merge=t_merge, t_travel=t_travel, t_finish=t_finish, units_240=units_240, progress_240=prog)
print("\n== everyone finishes (rotating banks, stay below 1 until the end) ==")
for n in (2, 3, 6):
    rn = r1 * math.sqrt(n); v = base / math.sqrt(n); zone = rn + mr + 0.01; mult = 1 + 0.15 * (n - 1); rate = rate1 * n ** 2
    pool_total = n * need / mult
    t_contact = pool_total / rate
    bank_leg = (0.57 - zone - (rn + 0.06)) / v + lag
    t_banks = n * 2 * bank_leg
    t_legs = n * 3 * ((max(0.495 - 2 * zone, 0) + 0.10) / v + lag)
    t_merge = 0.72 / base + 2 * lag
    tot = t_merge + t_contact + t_banks + t_legs
    print(f"n={n}: pool total {pool_total:.1f} -> contact {t_contact:.1f}s, {n} bank round trips {t_banks:.1f}s, mine legs {t_legs:.1f}s, merge {t_merge:.1f}s = {tot:.1f}s; stock drained per visit? n>=3 only after {out[f'drain_n{n}']['t_dry']:.1f}s at one mine")
    out[f"all_finish_n{n}"] = dict(pool_total=pool_total, t_contact=t_contact, t_banks=t_banks, t_legs=t_legs, total=tot)
print("\n== arena yield in 240 s vs lobby need ==")
t_outer_dead = 120 + 120 * (1 - 0.70) / 0.5   # R < 0.62 + 0.08 -> k = 0.6 -> t = 120 + 72
supply = 8 * cap + 4 * regen * T + 4 * regen * t_outer_dead
print(f"outer mines die at t={t_outer_dead:.0f}s; gross stock+regen available = {supply:.0f} units; lobby need 6x36 = 216 raw, or {216/1.75:.0f} pool units for a six that banks at x1.75")
print(f"extraction caps: six solos {6*rate1*T:.0f} units (contact-rate bound, < 216); one six-body {rate1*36:.2f}/s while stock lasts, then regen-bound 0.5/s per mine (4 mines alive late = 2.0/s)")
print(f"regen alone over 240 s (4 inner mines) = {4*regen*T:.0f} units; the whole lobby's need could be met by regen of the inner ring alone in {216/(4*regen):.0f}s of perfect multi-mine extraction")
out["supply"] = dict(t_outer_dead=t_outer_dead, supply=supply, six_solo_cap=6 * rate1 * T)
json.dump(out, open("/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/economy/derive.json", "w"), indent=1)
