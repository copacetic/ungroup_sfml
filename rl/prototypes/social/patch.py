import sys, re
p = sys.argv[1]
s = open(p).read()
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:60], s.count(old))
    s = s.replace(old, new)

# --- constants (overridable with -D)
rep("constexpr double PI = 3.14159265358979323846;",
"""constexpr double PI = 3.14159265358979323846;
#ifndef HEAD_VEST
#define HEAD_VEST 10.0      // s a newcomer must be bonded to every member before it can be crowned
#endif
#ifndef BOND_FULL
#define BOND_FULL 30.0      // s of co-membership at which a bond pays the full 0.15 bank bonus
#endif
#ifndef BRAND_MIN
#define BRAND_MIN 4.0       // units taken by a leaver that earn a brand
#endif
#ifndef BRAND_BASE
#define BRAND_BASE 20.0
#endif
#ifndef BRAND_PER_UNIT
#define BRAND_PER_UNIT 2.0
#endif
#ifndef BRAND_MAX
#define BRAND_MAX 60.0
#endif
#ifndef SHUN_DIST
#define SHUN_DIST 0.25
#endif
""")
# --- body head
rep("    double stun = 0;  // seconds of stun remaining\n    int n() const",
    "    double stun = 0;  // seconds of stun remaining\n    int head = -1;    // contract receiver (crown)\n    int n() const")
# --- brand state
rep("    double pair_ended[MAX_PLAYERS][MAX_PLAYERS];       // end time of the last co-membership (-1 none)",
    "    double pair_ended[MAX_PLAYERS][MAX_PLAYERS];       // end time of the last co-membership (-1 none)\n    double brand[MAX_PLAYERS];                          // public betrayal mark, seconds remaining")
rep("                last_left_me[i][j] = -1; partner_cd[i][j] = -1; pair_since[i][j] = -1; pair_ended[i][j] = -1;\n            }",
    "                last_left_me[i][j] = -1; partner_cd[i][j] = -1; pair_since[i][j] = -1; pair_ended[i][j] = -1;\n            }\n        for (int i = 0; i < MAX_PLAYERS; i++) brand[i] = 0;")
# --- helpers: owed, vested, crown
rep("    bool all_joinable(const Body& b) const {",
"""    double owed(int j) const {  // units j watched partners receive minus units j received in groups
        double s = 0;
        for (int i = 0; i < cfg.n_players; i++) if (i != j) s += banked_while[j][i] - banked_while[i][j];
        return s;
    }
    bool vested(const Body& b, int j) const {
        for (int k : b.members) if (k != j && (pair_since[j][k] < 0 || t - pair_since[j][k] < HEAD_VEST)) return false;
        return true;
    }
    bool branded(int j) const { return brand[j] > 0; }
    bool body_branded(const Body& b) const { for (int j : b.members) if (branded(j)) return true; return false; }
    // Crown: lowest progress among vested, unbranded members (ledger tie-break: most owed first, then index).
    // If nobody qualifies, keep the current head when still a member; else relax brand, then vesting.
    void crown(Body& b) {
        if (b.n() <= 1) { b.head = b.n() == 1 ? b.members[0] : -1; return; }
        auto pick = [&](bool need_vest, bool need_clean) {
            int best = -1;
            for (int j : b.members) {
                if (need_vest && !vested(b, j)) continue;
#ifdef M_BRAND
                if (need_clean && branded(j)) continue;
#endif
                if (best < 0) { best = j; continue; }
                double pj = progress(j), pb = progress(best);
                if (pj < pb - 1e-9 || (std::fabs(pj - pb) <= 1e-9 && (owed(j) > owed(best) + 1e-9 || (std::fabs(owed(j) - owed(best)) <= 1e-9 && j < best)))) best = j;
            }
            return best;
        };
        int h = pick(true, true);
        if (h < 0 && b.head >= 0 && std::find(b.members.begin(), b.members.end(), b.head) != b.members.end() && !branded(b.head)) h = b.head;
        if (h < 0) h = pick(true, false);
        if (h < 0 && b.head >= 0 && std::find(b.members.begin(), b.members.end(), b.head) != b.members.end()) h = b.head;
        if (h < 0) h = pick(false, true);
        if (h < 0) h = pick(false, false);
        if (h != b.head) event("\\"kind\\":\\"crown\\",\\"player\\":" + std::to_string(h) + ",\\"group\\":" + ids(b.members));
        b.head = h;
    }
    bool all_joinable(const Body& b) const {""")
# --- detach: brand + recrown
rep("        stats.leaves++;\n        stats.units_taken += taken;",
"""        stats.leaves++;
        stats.units_taken += taken;
#ifdef M_BRAND
        if (taken >= BRAND_MIN) brand[i] = std::min(BRAND_MAX, std::max(brand[i], BRAND_BASE + BRAND_PER_UNIT * taken));
#endif
        bodies.back().head = i;
        crown(bodies[bi]);""")
# detach: b invalid after push_back, so re-index: bi is still valid index since push_back appends. OK.
# --- merge: contagion + crown
rep("        nb.push_back(m);\n        bodies = nb;\n        stats.merges++;",
"""#ifdef M_CONTAGION
        { double ma_b = 0, mb_b = 0; for (int i : a.members) ma_b = std::max(ma_b, brand[i]); for (int j : b.members) mb_b = std::max(mb_b, brand[j]);
          for (int i : a.members) brand[i] = std::max(brand[i], 0.5 * mb_b); for (int j : b.members) brand[j] = std::max(brand[j], 0.5 * ma_b); }
#endif
        m.head = (a.n() >= b.n()) ? a.head : b.head;
        nb.push_back(m);
        bodies = nb;
        crown(bodies.back());
        stats.merges++;""")
# --- bank: lock to head, bond-scaled bonus, redemption
rep("""            for (int i : b.members) {
                Vec pad = pad_pos(i);
                if ((b.pos - pad).norm() < r + cfg.pad_radius) {
                    double amount[TYPES];
                    double total = b.pool_total();
                    double mult = 1.0 + cfg.group_bank_bonus * (b.n() - 1);""",
"""            for (int i : b.members) {
#ifdef M_CROWN
                if (b.n() > 1 && i != b.head) continue;
#endif
                Vec pad = pad_pos(i);
                if ((b.pos - pad).norm() < r + cfg.pad_radius) {
                    double amount[TYPES];
                    double total = b.pool_total();
                    double mult = 1.0 + cfg.group_bank_bonus * (b.n() - 1);
#ifdef M_BOND
                    { double bond = 0; for (int j : b.members) if (j != i) bond += std::min(1.0, std::max(0.0, t - pair_since[i][j]) / BOND_FULL); mult = 1.0 + cfg.group_bank_bonus * bond; }
#endif
#ifdef M_BRAND
                    for (int j : b.members) if (j != i && brand[j] > 0) brand[j] = std::max(0.0, brand[j] - total);  // redemption: 1 s per unit banked for others
#endif""")
rep("""                    stats.banks++;
                    if (b.n() > 1) stats.group_banks++;""",
"""                    stats.banks++;
                    if (b.n() > 1) stats.group_banks++;
                    stats.bond_mult += mult; stats.bond_banks++;
                    crown(b);""")
rep("    double units_taken = 0;  // units carried away by leavers",
    "    double units_taken = 0;  // units carried away by leavers\n    double bond_mult = 0; int bond_banks = 0;")
# --- brand decay in update_timers
rep("        for (Player& p : players) if (p.join_cooldown > 0) p.join_cooldown = std::max(0.0, p.join_cooldown - cfg.dt);",
    "        for (Player& p : players) if (p.join_cooldown > 0) p.join_cooldown = std::max(0.0, p.join_cooldown - cfg.dt);\n        for (int i = 0; i < cfg.n_players; i++) if (brand[i] > 0) brand[i] = std::max(0.0, brand[i] - cfg.dt);")
# --- bots: bail/loyal/kidnap group bank target = head's pad (kidnap keeps dragging to its own pad)
rep("            if (b.pool_total() >= bank_at) { target = my_pad; has = true; }\n            if (my_share >= bail_at",
    "            if (b.pool_total() >= bank_at) { target = pad_pos(b.head >= 0 ? b.head : i); has = true; }\n            if (my_share >= bail_at")
rep("""            double best = 2.0;
            for (int j : b.members) { double pj = progress(j); if (pj < best) { best = pj; target = pad_pos(j); has = true; } }""",
"""            double best = 2.0;
            for (int j : b.members) { double pj = progress(j); if (pj < best) { best = pj; target = pad_pos(j); has = true; } }
#ifdef M_CROWN
            if (b.head >= 0) target = pad_pos(b.head);
#endif""")
# loyal shun: refuse branded bodies nearby, don't seek them
rep("""                int k = nearest_joinable_body(i, b.pos, 0.6);
                if (k >= 0 && players[i].join_cooldown <= 0) { target = bodies[k].pos; has = true; }
            }
        }
        if (!has) {
            if (b.n() > 1) target = group_mine_target(b, &has); else target = mine_target(i, b.pos, &has);
            if (!has) target = pad_pos(i);
        }
        act[0] = direction_to_move(target - b.pos);
        act[1] = 1;
        act[2] = 0;
        act[3] = intent_for(i) + 1;
    }

    // Kidnapper""",
"""                int k = nearest_joinable_body(i, b.pos, 0.6);
#ifdef M_SHUN
                if (k >= 0 && body_branded(bodies[k])) k = -1;
#endif
                if (k >= 0 && players[i].join_cooldown <= 0) { target = bodies[k].pos; has = true; }
            }
        }
        if (!has) {
            if (b.n() > 1) target = group_mine_target(b, &has); else target = mine_target(i, b.pos, &has);
            if (!has) target = pad_pos(i);
        }
        int joinable = 1;
#ifdef M_SHUN
        for (const Body& o : bodies) if (&o != &b && body_branded(o) && (o.pos - b.pos).norm() < SHUN_DIST + radius(b.n()) + radius(o.n())) joinable = 0;
#endif
        act[0] = direction_to_move(target - b.pos);
        act[1] = joinable;
        act[2] = 0;
        act[3] = intent_for(i) + 1;
    }

    // Kidnapper""")
# --- frame json: head and brand
rep("""            s += "{\\"m\\":" + ids(b.members) + buf + nums(b.pool, TYPES, 1) + "}";""",
    """            s += "{\\"m\\":" + ids(b.members) + buf + nums(b.pool, TYPES, 1) + ",\\"head\\":" + std::to_string(b.head) + "}";""")
rep("""            snprintf(buf, sizeof buf, ",\\"intent\\":%d,\\"join\\":%s,\\"leaving\\":%.1f,\\"cd\\":%.1f,\\"dir\\":[%.2f,%.2f]}", p.intent,
                     p.joinable ? "true" : "false", p.leave_timer >= 0 ? p.leave_timer : -1.0, p.join_cooldown, p.dir.x, p.dir.y);""",
    """            snprintf(buf, sizeof buf, ",\\"intent\\":%d,\\"join\\":%s,\\"leaving\\":%.1f,\\"cd\\":%.1f,\\"dir\\":[%.2f,%.2f],\\"brand\\":%.1f,\\"owed\\":%.1f}", p.intent,
                     p.joinable ? "true" : "false", p.leave_timer >= 0 ? p.leave_timer : -1.0, p.join_cooldown, p.dir.x, p.dir.y, brand[i], owed(i));""")
# initial crown for solos at reset
rep("            b.members = {i};\n            b.pos = {0.72", "            b.members = {i};\n            b.head = i;\n            b.pos = {0.72")
open(p, "w").write(s)
print("patched")
