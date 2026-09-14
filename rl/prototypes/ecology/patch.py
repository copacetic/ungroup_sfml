import re, sys
S = "/tmp/claude-0/-home-user-ungroup-sfml/90f81c2f-9967-548b-bb7f-131c873074fa/scratchpad/ecology"
cpp = open(S + "/rl/native/ungroup.cpp").read()
py = open(S + "/rl/ungroup/native.py").read()

def rep(s, old, new, count=1):
    assert s.count(old) >= 1, old
    return s.replace(old, new, count)

# --- Cfg fields (appended) ---
cpp = rep(cpp, "    double carried_shaping = 2.0, win_bonus = 10.0, lose_penalty = 2.0, relative_reward = 0.5;\n};\nconstexpr int CFG_LEN = 37;",
"""    double carried_shaping = 2.0, win_bonus = 10.0, lose_penalty = 2.0, relative_reward = 0.5;
    // ecology (all zero = legacy rules)
    double bloom_rate = 0.0;      // r: logistic growth r*S*(1-S/K) per second (replaces constant regen when > 0)
    double yield_stock_exp = 0.0; // yield multiplier (S/K)^e (0 = constant yield while stock > 0)
    double seed_rate = 0.0;       // sigma: per neighbour, seed inflow sigma*min(S_j/K,1)*(1-S_m/K)
    double seed_floor = 0.0;      // s_min: inflow from the seed bank s_min*(1-S_m/K) even with dead neighbours
    double seed_range = 0.5;      // mines closer than this are neighbours
    double migrate_rate = 0.0;    // mu: autumn migration of excess stock from a withering mine to its twin
    double bloom_max = 1.5;       // stock may exceed K up to bloom_max*K (superbloom), then decays back
};
constexpr int CFG_LEN = 45;""")
cpp = rep(cpp, "    c.carried_shaping = a[k++]; c.win_bonus = a[k++]; c.lose_penalty = a[k++]; c.relative_reward = a[k++];\n}",
"""    c.carried_shaping = a[k++]; c.win_bonus = a[k++]; c.lose_penalty = a[k++]; c.relative_reward = a[k++];
    c.bloom_rate = a[k++]; c.yield_stock_exp = a[k++]; c.seed_rate = a[k++]; c.seed_floor = a[k++];
    c.seed_range = a[k++]; c.migrate_rate = a[k++]; c.bloom_max = a[k++];
}""")
cpp = rep(cpp, "    a[k++] = c.carried_shaping; a[k++] = c.win_bonus; a[k++] = c.lose_penalty; a[k++] = c.relative_reward;\n}",
"""    a[k++] = c.carried_shaping; a[k++] = c.win_bonus; a[k++] = c.lose_penalty; a[k++] = c.relative_reward;
    a[k++] = c.bloom_rate; a[k++] = c.yield_stock_exp; a[k++] = c.seed_rate; a[k++] = c.seed_floor;
    a[k++] = c.seed_range; a[k++] = c.migrate_rate; a[k++] = c.bloom_max;
}""")

# --- state: mine capacity (autumn), neighbour list, per-round ecology stats ---
cpp = rep(cpp, "    std::vector<char> mine_alive;\n    std::deque<Pickup> picks;",
"""    std::vector<char> mine_alive;
    std::vector<double> mine_capk;              // K_m(t): current carrying capacity (autumn withering)
    std::vector<std::vector<int>> mine_nbr;     // neighbour lists for seeding
    std::deque<Pickup> picks;""")
cpp = rep(cpp, "        mine_alive.assign(cfg.n_mines, 1);\n        double mrot",
"""        mine_alive.assign(cfg.n_mines, 1);
        mine_capk.assign(cfg.n_mines, cfg.mine_cap);
        double mrot""")
cpp = rep(cpp, "        bodies.clear();\n        for (int i = 0; i < n; i++) {\n            Body b;",
"""        mine_nbr.assign(cfg.n_mines, {});
        for (int m = 0; m < cfg.n_mines; m++)
            for (int j = 0; j < cfg.n_mines; j++)
                if (j != m && (mine_pos[m] - mine_pos[j]).norm() < cfg.seed_range) mine_nbr[m].push_back(j);
        bodies.clear();
        for (int i = 0; i < n; i++) {
            Body b;""")

# --- yield proportional to stock ---
cpp = rep(cpp, """                    double rate = cfg.mine_rate * std::pow((double)b.n(), cfg.mine_exp);
                    double amount = std::min(rate * cfg.dt, mine_stock[m]);""",
"""                    double rate = cfg.mine_rate * std::pow((double)b.n(), cfg.mine_exp);
                    if (cfg.yield_stock_exp > 0) rate *= std::pow(std::min(mine_stock[m] / cfg.mine_cap, cfg.bloom_max), cfg.yield_stock_exp);
                    double amount = std::min(rate * cfg.dt, mine_stock[m]);""")

# --- growth: logistic + seeding ---
cpp = rep(cpp, """    void regen() {
        for (int m = 0; m < cfg.n_mines; m++)
            if (mine_alive[m]) mine_stock[m] = std::min(cfg.mine_cap, mine_stock[m] + cfg.mine_regen * cfg.dt);
    }""",
"""    void regen() {
        if (cfg.bloom_rate <= 0) {
            for (int m = 0; m < cfg.n_mines; m++)
                if (mine_alive[m]) mine_stock[m] = std::min(cfg.mine_cap, mine_stock[m] + cfg.mine_regen * cfg.dt);
            return;
        }
        std::vector<double> ns(mine_stock);
        for (int m = 0; m < cfg.n_mines; m++) {
            if (!mine_alive[m]) continue;
            double K = mine_capk[m];
            if (K <= 1e-9) continue;
            double x = mine_stock[m] / K;
            double growth = cfg.bloom_rate * mine_stock[m] * (1.0 - x);      // logistic (negative above K)
            double free = std::max(0.0, 1.0 - x);
            double seed = cfg.seed_floor * free;
            for (int j : mine_nbr[m]) if (mine_alive[j]) seed += cfg.seed_rate * std::min(mine_stock[j] / cfg.mine_cap, 1.0) * free;
            ns[m] = std::max(0.0, mine_stock[m] + (growth + seed) * cfg.dt);
            ns[m] = std::min(ns[m], cfg.bloom_max * cfg.mine_cap);
        }
        mine_stock.swap(ns);
    }""")

# --- autumn: withering capacity and migration to the twin ---
cpp = rep(cpp, """        for (int m = 0; m < cfg.n_mines; m++) {
            if (mine_alive[m] && mine_pos[m].norm() + cfg.mine_radius > R) {
                mine_alive[m] = 0;
                event("\\"kind\\":\\"mine_dead\\",\\"mine\\":" + std::to_string(m));
            }
        }""",
"""        for (int m = 0; m < cfg.n_mines; m++) {
            if (!mine_alive[m]) continue;
            double dm = mine_pos[m].norm() + cfg.mine_radius;
            if (cfg.migrate_rate > 0 && dm > cfg.final_radius) {
                // Autumn: capacity ramps from K (R = 1) to 0 (R = dm); excess stock migrates to the nearest live twin.
                double k = std::max(0.0, std::min(1.0, (R - dm) / (1.0 - dm)));
                mine_capk[m] = cfg.mine_cap * k;
                double excess = mine_stock[m] - mine_capk[m];
                if (excess > 0) {
                    double flow = std::min(excess, cfg.migrate_rate * excess * cfg.dt);
                    int tw = -1; double bd = 1e9;
                    for (int j = 0; j < cfg.n_mines; j++) {
                        if (j == m || !mine_alive[j] || mine_type[j] != mine_type[m]) continue;
                        double d = (mine_pos[j] - mine_pos[m]).norm();
                        if (d < bd) { bd = d; tw = j; }
                    }
                    mine_stock[m] -= flow;
                    if (tw >= 0) mine_stock[tw] = std::min(mine_stock[tw] + flow, cfg.bloom_max * cfg.mine_cap);
                }
            }
            if (dm > R) {
                mine_alive[m] = 0;
                event("\\"kind\\":\\"mine_dead\\",\\"mine\\":" + std::to_string(m));
            }
        }""")
open(S + "/rl/native/ungroup.cpp", "w").write(cpp)

py = rep(py, "    relative_reward: float = 0.5\n",
"""    relative_reward: float = 0.5
    bloom_rate: float = 0.0
    yield_stock_exp: float = 0.0
    seed_rate: float = 0.0
    seed_floor: float = 0.0
    seed_range: float = 0.5
    migrate_rate: float = 0.0
    bloom_max: float = 1.5
""")
open(S + "/rl/ungroup/native.py", "w").write(py)
print("patched")
