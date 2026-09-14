
    // Smart: a best-response probe built only on public information (positions, velocities, dirs,
    // memberships, joinable flags, intents, banked amounts, pools, leave timers, pads, leave history).
    // Cooperates on the loyal protocol (bank at the pad of the member furthest behind) but banks by
    // useful units with a threshold that grows with n, follows loyal partners' steering in mixed
    // groups, leaves defectors (a partner pushing to its own pad off protocol), flees bodies that
    // close fast while laden, banks before merging with strangers, and takes its share late in a pair.
    static constexpr double SM_THRESH0 = 10.0, SM_THRESH_PER = 4.0;   // group bank threshold 10 + 4 (n - 1) useful units
    static constexpr double SM_BLACKLIST = 20.0;                        // seconds to refuse a body containing someone who left me
    static constexpr int SM_MAX_GROUP = 6;
    static constexpr double SM_SPEED = 0.30;                            // conservative travel speed for end-of-round timing
    bool smart_bad(int i, int j) const { return last_left_me[i][j] >= 0 && t - last_left_me[i][j] < SM_BLACKLIST; }
    bool smart_proven(int i, int j) const { return last_left_me[i][j] < 0 && (banked_while[j][i] > 0 || comember_time[i][j] > 20.0); }
    double smart_leave_share(int i, const Body& b) const {
        double s = 0;
        for (int t = 0; t < TYPES; t++) {
            double w = 0;
            for (int j : b.members) w += (players[j].intent == t) ? cfg.intent_weight : 1.0;
            s += b.pool[t] * (((players[i].intent == t) ? cfg.intent_weight : 1.0) / w) * (1.0 - cfg.leave_forfeit);
        }
        return s;
    }
    // Nearest live mine of a type that would still be useful to receiver r given the pool.
    Vec smart_mine(const Body& b, int r, double mult, bool* found) const {
        bool useful[TYPES], any = false;
        for (int t = 0; t < TYPES; t++) { useful[t] = players[r].banked[t] + b.pool[t] * mult < players[r].need[t]; any |= useful[t]; }
        int m = any ? nearest_mine(b.pos, useful) : -1;
        if (m < 0) return group_mine_target(b, found);
        *found = true;
        return mine_pos[m];
    }
    // Is any non-mergeable body closing on me fast enough to matter? Returns the flee direction.
    bool smart_hunted(const Body& b, Vec* away) const {
        for (const Body& o : bodies) {
            if (&o == &b || can_merge(b, o)) continue;
            Vec d = b.pos - o.pos;
            double dist = d.norm();
            if (dist < 1e-6 || dist > 0.3) continue;
            if ((o.vel - b.vel).dot(d) / dist > 0.3) { *away = d * (1.0 / dist); return true; }
        }
        return false;
    }
    void bot_smart(int i, int* act) const {
        const Body& b = bodies[body_index(i)];
        const Player& me = players[i];
        double time_left = cfg.time_limit - t;
        double pool = b.pool_total();
        Vec my_pad = pad_pos(i);
        double my_pad_d = (my_pad - b.pos).norm();
        bool has = false;
        Vec target;
        int joinable = b.n() < SM_MAX_GROUP ? 1 : 0;
        int leave = 0;
        int want = 0;  // intent: the type with the largest remaining need
        for (int k = 1; k < TYPES; k++) if (me.need[k] - me.banked[k] > me.need[want] - me.banked[want]) want = k;
        Vec away;
        bool hunted = pool >= 2.0 && smart_hunted(b, &away);
        if (b.n() > 1) {
            int behind = b.members[0];
            for (int j : b.members) if (progress(j) < progress(behind)) behind = j;
            double mult = 1.0 + cfg.group_bank_bonus * (b.n() - 1);
            Vec rpad = pad_pos(behind);
            double useful = 0, remaining = 0;
            for (int k = 0; k < TYPES; k++) {
                double rem = std::max(0.0, players[behind].need[k] - players[behind].banked[k]);
                useful += std::min(b.pool[k] * mult, rem); remaining += rem;
            }
            double thresh = std::min(SM_THRESH0 + SM_THRESH_PER * (b.n() - 1), remaining - 0.5);
            double rpad_d = (rpad - b.pos).norm();
            bool loyal_banks = pool >= 0.28 * need_total(behind);  // the loyal protocol's bank trigger
            Vec T_l = loyal_banks ? rpad : group_mine_target(b, &has);
            bool want_bank = useful >= thresh || (useful >= 1.0 && time_left < rpad_d / SM_SPEED + 4.0);
            Vec T_s = want_bank ? rpad : smart_mine(b, behind, mult, &has);
            // Defector: a partner pushing to its own pad, off protocol, with a laden pool.
            bool defect = false, follow = false;
            Vec dl = T_l - b.pos, ds = T_s - b.pos;
            for (int j : b.members) {
                if (j == i || players[j].dir.norm() < 0.5) continue;
                Vec tj = pad_pos(j) - b.pos;
                double dj = tj.norm();
                double to_l = dl.norm() > 1e-6 ? players[j].dir.dot(dl) / dl.norm() : 1.0;
                double to_s = ds.norm() > 1e-6 ? players[j].dir.dot(ds) / ds.norm() : 1.0;
                if (j != behind && pool >= 4.0 && dj > 1e-6 && players[j].dir.dot(tj) / dj > 0.85 && to_l < 0.5 && to_s < 0.5) defect = true;
                if (to_l > 0.7 && to_s < 0.7) follow = true;
            }
            double share = smart_leave_share(i, b);
            bool late = b.n() == 2 && behind != i && time_left < 25.0 && share >= 4.0 && my_pad_d / SM_SPEED + cfg.leave_time + 2.0 < time_left;
            bool keep = me.leave_timer >= 0 && pool >= 1.0;
            if (defect || late || keep) {
                leave = 1;
                Vec u = b.vel.norm() > 0.05 ? b.vel * -1.0 : (rpad - b.pos) * -1.0;  // stall the group
                target = b.pos + u; has = true;
            } else if (hunted) { target = b.pos + away; has = true; }
            else { target = follow ? T_l : T_s; has = true; }
        } else {
            // Solo: bank when full, when time is short, or when carrying near strangers (do not feed a stranger).
            bool stranger_near = false;
            for (const Body& o : bodies) if (&o != &b && (o.pos - b.pos).norm() < 0.3 && all_joinable(o)) {
                bool proven = true;
                for (int j : o.members) proven &= smart_proven(i, j);
                stranger_near |= !proven;
            }
            bool must_bank = pool >= 0.5 && time_left < my_pad_d / SM_SPEED + 4.0;
            if (pool >= 0.22 * need_total(i) || must_bank || (pool >= 3.0 && stranger_near)) { target = my_pad; has = true; }
            if (hunted) {
                Vec pd = my_pad - b.pos;
                bool bank_ok = pool >= 3.0 && pd.norm() > 1e-6 && pd.dot(away) / pd.norm() > -0.2;
                target = bank_ok ? my_pad : b.pos + away; has = true;
            }
            if (pool >= 3.0 && stranger_near) joinable = 0;
            // Choose a partner: near, not blacklisted, not about to finish, intent unlike my need.
            if (!has && me.join_cooldown <= 0) {
                double best = 1e9;
                for (const Body& o : bodies) {
                    if (&o == &b || !all_joinable(o) || o.n() >= SM_MAX_GROUP) continue;
                    double d = (o.pos - b.pos).norm();
                    if (d > 0.7) continue;
                    double score = d;
                    bool ok = true;
                    for (int j : o.members) {
                        if (smart_bad(i, j) || partner_cd[i][j] > t) ok = false;
                        if (progress(j) >= 0.8) score += 0.5;
                        if (players[j].intent == want) score += 0.15;
                        if (smart_proven(i, j)) score -= 0.2;
                    }
                    if (ok && score < best) { best = score; target = o.pos; has = true; }
                }
            }
            for (const Body& o : bodies) {  // refuse a blacklisted or nearly finished body about to touch
                if (&o == &b || (o.pos - b.pos).norm() > 0.2) continue;
                for (int j : o.members) if (smart_bad(i, j) || progress(j) >= 0.8) joinable = 0;
            }
            if (!has) for (const Pickup& pk : picks) if ((pk.pos - b.pos).norm() < 0.12) { target = pk.pos; has = true; break; }
        }
        if (!has) {
            target = mine_target(i, b.pos, &has);
            if (!has) target = my_pad;
        }
        act[0] = direction_to_move(target - b.pos);
        act[1] = joinable;
        act[2] = leave;
        act[3] = want + 1;
    }
