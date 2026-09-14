import re, subprocess, os
SRC = "rl/native/ungroup.cpp"
D = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/evolution/native"
s = open(SRC).read()
def rep(old, new, count=1):
    global s
    assert s.count(old) >= 1, old[:60]
    s = s.replace(old, new, count)
# 1. seat types
rep("SEAT_KIDNAP = 5, SEAT_RAMMER = 6 };", "SEAT_KIDNAP = 5, SEAT_RAMMER = 6, SEAT_GRUDGE = 7, SEAT_CASH = 8 };")
# 2. state
rep("    double pair_ended[MAX_PLAYERS][MAX_PLAYERS];       // end time of the last co-membership (-1 none)\n",
    "    double pair_ended[MAX_PLAYERS][MAX_PLAYERS];       // end time of the last co-membership (-1 none)\n"
    "    // --- evolution prototype ---\n"
    "    double inherit[MAX_PLAYERS][TYPES] = {};  // banked units a seat starts the next round with (capital)\n"
    "    double inherit_cap = 0.5;                 // at most this fraction of each need may be inherited\n"
    "    bool persist = false;                     // keep the pairwise ledger across resets (shifted, decayed)\n"
    "    double ledger_decay = 0.5;\n"
    "    double grudge_window = 500.0, cash_at = 0.4;\n"
    "    double mine_diff = 0.0;                   // per-second fraction of the stock difference that flows between ring neighbours\n"
    "    bool persist_mines = false; double mrot_keep = -1;  // keep mine positions and stock across resets\n")
# 3. reset: capture t_prev, apply inherit, persist ledger
rep("    void reset(uint64_t sd) {\n        seed = sd;\n        rng.seed(sd);\n        sample_cfg();\n        t = 0;",
    "    void reset(uint64_t sd) {\n        double t_prev = t;\n        seed = sd;\n        rng.seed(sd);\n        sample_cfg();\n        t = 0;")
rep("            players[i].need[primary] = cfg.need_primary;\n",
    "            players[i].need[primary] = cfg.need_primary;\n"
    "            for (int t = 0; t < TYPES; t++) players[i].banked[t] = std::min(inherit[i][t], inherit_cap * players[i].need[t]);\n")
rep("""        for (int i = 0; i < MAX_PLAYERS; i++)
            for (int j = 0; j < MAX_PLAYERS; j++) {
                comember_time[i][j] = 0; took_from[i][j] = 0; banked_while[i][j] = 0;
                last_left_me[i][j] = -1; partner_cd[i][j] = -1; pair_since[i][j] = -1; pair_ended[i][j] = -1;
            }""",
"""        for (int i = 0; i < MAX_PLAYERS; i++)
            for (int j = 0; j < MAX_PLAYERS; j++) {
                if (persist) {
                    comember_time[i][j] *= ledger_decay; took_from[i][j] *= ledger_decay; banked_while[i][j] *= ledger_decay;
                    if (last_left_me[i][j] <= -1e8 || last_left_me[i][j] == -1.0) last_left_me[i][j] = -1e9;  // never
                    else last_left_me[i][j] -= t_prev;
                    partner_cd[i][j] -= t_prev; pair_since[i][j] = -1; pair_ended[i][j] = -1;
                } else {
                    comember_time[i][j] = 0; took_from[i][j] = 0; banked_while[i][j] = 0;
                    last_left_me[i][j] = -1; partner_cd[i][j] = -1; pair_since[i][j] = -1; pair_ended[i][j] = -1;
                }
            }""")
rep("""        mine_pos.assign(cfg.n_mines, Vec());
        mine_type.assign(cfg.n_mines, 0);
        mine_stock.assign(cfg.n_mines, cfg.mine_cap);
        mine_alive.assign(cfg.n_mines, 1);
        double mrot = uniform(0, 2 * PI);""",
"""        bool keep = persist_mines && (int)mine_stock.size() == cfg.n_mines && mrot_keep >= 0;
        std::vector<double> old_stock = mine_stock;
        mine_pos.assign(cfg.n_mines, Vec());
        mine_type.assign(cfg.n_mines, 0);
        mine_stock.assign(cfg.n_mines, cfg.mine_cap);
        mine_alive.assign(cfg.n_mines, 1);
        double mrot = uniform(0, 2 * PI);
        if (keep) { mrot = mrot_keep; mine_stock = old_stock; } else mrot_keep = mrot;""")
rep("""    void regen() {
        for (int m = 0; m < cfg.n_mines; m++)
            if (mine_alive[m]) mine_stock[m] = std::min(cfg.mine_cap, mine_stock[m] + cfg.mine_regen * cfg.dt);
    }""",
"""    void regen() {
        for (int m = 0; m < cfg.n_mines; m++)
            if (mine_alive[m]) mine_stock[m] = std::min(cfg.mine_cap, mine_stock[m] + cfg.mine_regen * cfg.dt);
        if (mine_diff > 0) {  // ore flows along the ring toward drained neighbours (explicit Euler, symmetric, conserving)
            int M = cfg.n_mines;
            std::vector<double> flow(M, 0.0);
            for (int m = 0; m < M; m++) {
                int q = (m + 1) % M;
                if (!mine_alive[m] || !mine_alive[q]) continue;
                double f = mine_diff * cfg.dt * (mine_stock[m] - mine_stock[q]);
                flow[m] -= f; flow[q] += f;
            }
            for (int m = 0; m < M; m++) mine_stock[m] = std::min(cfg.mine_cap, std::max(0.0, mine_stock[m] + flow[m]));
        }
    }""")
# 4. bots
rep("    void bot_action(int seat_type, int i, int* act) const {",
"""    // Grudge: loyal, but refuses to merge with anyone who publicly left a partner within grudge_window s
    // (the ledger survives resets when persist is on, so a leave in round r is remembered in round r+1).
    bool public_leaver(int j) const {
        for (int k = 0; k < cfg.n_players; k++)
            if (k != j && last_left_me[k][j] > -1e8 && last_left_me[k][j] != -1.0 && t - last_left_me[k][j] < grudge_window) return true;
        return false;
    }
    bool body_has_leaver(const Body& b) const { for (int j : b.members) if (public_leaver(j)) return true; return false; }
    void bot_grudge(int i, int* act) const {
        const Body& b = bodies[body_index(i)];
        double bank_at = 0.28 * need_total(i);
        bool has = false;
        Vec target;
        int joinable = 1;
        if (b.n() > 1 && b.pool_total() >= bank_at) {
            double best = 2.0;
            for (int j : b.members) { double pj = progress(j); if (pj < best) { best = pj; target = pad_pos(j); has = true; } }
        } else if (b.n() == 1) {
            if (b.pool_total() >= bank_at) { target = pad_pos(i); has = true; }
            else if (players[i].join_cooldown <= 0) {
                double bd = 0.6;
                for (const Body& o : bodies) {
                    if (&o == &b || !all_joinable(o) || body_has_leaver(o)) continue;
                    double d = (o.pos - b.pos).norm();
                    if (d < bd) { bd = d; target = o.pos; has = true; }
                }
            }
        }
        for (const Body& o : bodies)
            if (&o != &b && (o.pos - b.pos).norm() < 0.25 + radius(b.n()) && body_has_leaver(o)) joinable = 0;
        if (!has) {
            if (b.n() > 1) target = group_mine_target(b, &has); else target = mine_target(i, b.pos, &has);
            if (!has) target = pad_pos(i);
        }
        act[0] = direction_to_move(target - b.pos);
        act[1] = joinable;
        act[2] = 0;
        act[3] = intent_for(i) + 1;
    }
    // Cash: loyal while behind, solo (never joinable, leaves any group) once its own progress is at least cash_at.
    void bot_cash(int i, int* act) const {
        if (progress(i) >= cash_at) bot_solo(i, act); else bot_loyal(i, act);
    }
    void bot_action(int seat_type, int i, int* act) const {""")
rep("            case SEAT_RAMMER: bot_rammer(i, act); break;",
    "            case SEAT_RAMMER: bot_rammer(i, act); break;\n            case SEAT_GRUDGE: bot_grudge(i, act); break;\n            case SEAT_CASH: bot_cash(i, act); break;")
# 5. C API
rep("double ugb_progress(void* h, int env, int player)",
"""void ugb_set_inherit(void* h, int env, int seat, const double* v) { Game& g = ((Batch*)h)->games[env]; for (int t = 0; t < TYPES; t++) g.inherit[seat][t] = v[t]; }
void ugb_set_evo(void* h, int env, double inherit_cap, int persist, double ledger_decay, double grudge_window, double cash_at) {
    Game& g = ((Batch*)h)->games[env]; g.inherit_cap = inherit_cap; g.persist = persist != 0; g.ledger_decay = ledger_decay; g.grudge_window = grudge_window; g.cash_at = cash_at; }
void ugb_clear_history(void* h, int env, int seat) {
    Game& g = ((Batch*)h)->games[env];
    for (int j = 0; j < MAX_PLAYERS; j++) {
        g.comember_time[seat][j] = g.comember_time[j][seat] = 0; g.took_from[seat][j] = g.took_from[j][seat] = 0;
        g.banked_while[seat][j] = g.banked_while[j][seat] = 0; g.last_left_me[seat][j] = g.last_left_me[j][seat] = -1e9;
        g.partner_cd[seat][j] = g.partner_cd[j][seat] = -1; }
}
void ugb_set_mines(void* h, int env, double mine_diff, int persist_mines) { Game& g = ((Batch*)h)->games[env]; g.mine_diff = mine_diff; g.persist_mines = persist_mines != 0; }
void ugb_remap_history(void* ho, int envo, void* hn, int envn, const int* map, int n_new, double t_shift) {
    Game& a = ((Batch*)ho)->games[envo]; Game& b = ((Batch*)hn)->games[envn];
    for (int i = 0; i < n_new; i++) for (int j = 0; j < n_new; j++) {
        int io = map[i], jo = map[j];
        if (io < 0 || jo < 0 || i == j) { b.comember_time[i][j] = 0; b.took_from[i][j] = 0; b.banked_while[i][j] = 0; b.last_left_me[i][j] = -1e9; b.partner_cd[i][j] = -1; continue; }
        b.comember_time[i][j] = a.comember_time[io][jo]; b.took_from[i][j] = a.took_from[io][jo]; b.banked_while[i][j] = a.banked_while[io][jo];
        double l = a.last_left_me[io][jo];
        b.last_left_me[i][j] = (l <= -1e8 || l == -1.0) ? -1e9 : l - t_shift;
        b.partner_cd[i][j] = a.partner_cd[io][jo] - t_shift;
    }
}
double ugb_start_progress(void* h, int env, int player) { Game& g = ((Batch*)h)->games[env]; double s = 0; for (int t = 0; t < TYPES; t++) s += std::min(g.inherit[player][t], g.inherit_cap * g.players[player].need[t]) / g.players[player].need[t]; return s / TYPES; }
double ugb_progress(void* h, int env, int player)""")
open(f"{D}/ungroup_ev.cpp", "w").write(s)
subprocess.check_call(["g++", "-O3", "-march=native", "-std=c++17", "-fopenmp", "-shared", "-fPIC", "-o", f"{D}/libungroup_ev.so", f"{D}/ungroup_ev.cpp"])
print("built")
