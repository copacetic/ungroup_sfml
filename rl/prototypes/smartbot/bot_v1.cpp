
    // Smart: a best-response probe. Cooperates on the loyal protocol (join, mine the consensus mine,
    // bank at the pad of the member furthest behind) while partners do, but uses public history and
    // public steering to leave defectors, blacklists anyone who left it, prefers partners whose
    // intent differs from its own need, flees fast approaching bodies while laden, and takes its
    // share late in the round when the next group bank would not be its own.
    static constexpr double SMART_TRAVEL_SPEED = 0.30;  // conservative speed for end-of-round timing
    bool smart_trusts(int i, int j) const { return last_left_me[i][j] < 0 && took_from[i][j] <= 0; }
    double smart_leave_share(int i, const Body& b) const {
        double s = 0;
        for (int t = 0; t < TYPES; t++) {
            double w = 0;
            for (int j : b.members) w += (players[j].intent == t) ? cfg.intent_weight : 1.0;
            double wi = (players[i].intent == t) ? cfg.intent_weight : 1.0;
            s += b.pool[t] * (wi / w) * (1.0 - cfg.leave_forfeit);
        }
        return s;
    }
    int smart_behind(const Body& b) const {
        int behind = b.members[0];
        for (int j : b.members) if (progress(j) < progress(behind)) behind = j;
        return behind;
    }
    void bot_smart(int i, int* act) const {
        const int max_group_bot = 6;
        const Body& b = bodies[body_index(i)];
        const Player& me = players[i];
        double need_tot = need_total(i);
        double bank_at = 0.28 * need_tot;
        double time_left = cfg.time_limit - t;
        bool has = false;
        Vec target;
        int joinable = b.n() < max_group_bot ? 1 : 0;
        int leave = 0;
        // intent: the type with the largest remaining need (only changes while solo)
        int want = 0;
        for (int k = 1; k < TYPES; k++)
            if (me.need[k] - me.banked[k] > me.need[want] - me.banked[want]) want = k;
        if (b.n() > 1) {
            int behind = smart_behind(b);
            double pool = b.pool_total();
            Vec vpad = pad_pos(behind);
            // Defector detection: a partner pushing toward its own pad off-protocol with a laden pool,
            // or a partner one bank from finishing the round near its pad, or an untrusted partner.
            bool defect = false;
            for (int j : b.members) {
                if (j == i) continue;
                const Player& pj = players[j];
                Vec tj = pad_pos(j) - b.pos;
                double dj = tj.norm();
                bool to_own_pad = j != behind && dj > 1e-6 && pj.dir.norm() > 0.5 && pj.dir.dot(tj) / dj > 0.8 && dj < 0.7;
                bool finishing = j != behind && progress(j) >= 0.75 && dj < 0.5;
                if (pool >= 3.0 && (to_own_pad || finishing)) defect = true;
                if (!smart_trusts(i, j)) defect = true;
            }
            double share = smart_leave_share(i, b);
            double my_pad_d = (pad_pos(i) - b.pos).norm();
            // Late in the round: if the group's next bank is not mine and my share is worth banking, take it.
            bool late = behind != i && time_left < 30.0 && share >= 3.0 && my_pad_d / SMART_TRAVEL_SPEED + cfg.leave_time + 3.0 < time_left;
            bool keep_leaving = me.leave_timer >= 0 && pool >= 1.0;
            if (defect || late || keep_leaving) {
                leave = 1;
                // Stall the group while the timer runs: push against the group's motion.
                Vec u = b.vel.norm() > 0.05 ? b.vel * -1.0 : (vpad - b.pos) * -1.0;
                target = b.pos + u; has = true;
            } else if (pool >= bank_at) { target = vpad; has = true; }
        } else {
            double pool = b.pool_total();
            double my_pad_d = (pad_pos(i) - b.pos).norm();
            bool must_bank = pool >= 0.5 && time_left < my_pad_d / SMART_TRAVEL_SPEED + 4.0;
            if (pool >= bank_at || must_bank) { target = pad_pos(i); has = true; }
            // Flee a non-mergeable body closing fast while laden (rammers).
            if (pool >= 1.0) {
                for (const Body& o : bodies) {
                    if (&o == &b || can_merge(b, o)) continue;
                    Vec d = b.pos - o.pos;
                    double dist = d.norm();
                    if (dist < 1e-6 || dist > 0.35) continue;
                    double closing = (o.vel - b.vel).dot(d) * (1.0 / dist);
                    if (closing > 0.25) { target = b.pos + d * (1.0 / dist); has = true; break; }
                }
            }
            // Choose a partner: nearby, trusted, not about to finish, intent unlike my need.
            if (!has && me.join_cooldown <= 0) {
                double best = 1e9;
                for (int k = 0; k < (int)bodies.size(); k++) {
                    const Body& o = bodies[k];
                    if (&o == &b || !all_joinable(o) || o.n() >= max_group_bot) continue;
                    double d = (o.pos - b.pos).norm();
                    if (d > 0.7) continue;
                    double score = d;
                    bool ok = true;
                    for (int j : o.members) {
                        if (!smart_trusts(i, j) || partner_cd[i][j] > t) ok = false;
                        if (progress(j) >= 0.8) score += 0.5;
                        if (players[j].intent == want) score += 0.15;
                    }
                    if (ok && score < best) { best = score; target = o.pos; has = true; }
                }
            }
            // Refuse to merge with an untrusted body that is about to touch.
            for (int k = 0; k < (int)bodies.size() && joinable; k++) {
                const Body& o = bodies[k];
                if (&o == &b || (o.pos - b.pos).norm() > 0.2) continue;
                for (int j : o.members) if (!smart_trusts(i, j) || progress(j) >= 0.8) joinable = 0;
            }
            // Free units on the floor next to me.
            if (!has) for (const Pickup& pk : picks) if ((pk.pos - b.pos).norm() < 0.12) { target = pk.pos; has = true; break; }
        }
        if (!has) {
            if (b.n() > 1) target = group_mine_target(b, &has); else target = mine_target(i, b.pos, &has);
            if (!has) target = pad_pos(i);
        }
        act[0] = direction_to_move(target - b.pos);
        act[1] = joinable;
        act[2] = leave;
        act[3] = want + 1;
    }
