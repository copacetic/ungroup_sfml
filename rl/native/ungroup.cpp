// Ungroup v2 rules core in C++: a faithful port of rl/ungroup/core.py, env.py, and bots.py.
//
// Exposed as a plain C API (see the bottom of the file) and driven from Python through
// ctypes (rl/ungroup/native.py). A "batch" owns many independent games and steps them in
// parallel with OpenMP. Observation layout, rewards, and the scripted bots match the Python
// versions exactly so checkpoints trained on either can be evaluated on the other.
//
// Build: g++ -O3 -march=native -std=c++17 -fopenmp -shared -fPIC -o libungroup.so ungroup.cpp

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int TYPES = 4;
constexpr int N_PICKUPS_OBS = 4;
constexpr double PI = 3.14159265358979323846;

struct Cfg {
    int n_players = 6, n_mines = 8;
    double dt = 0.1, time_limit = 240.0, base_speed = 0.45, vel_lerp = 6.0, solo_radius = 0.045;
    double mine_radius = 0.08, mine_cap = 30.0, mine_regen = 0.5, mine_rate = 0.15, mine_exp = 2.0;
    double pad_radius = 0.06;
    int need_primary = 18, need_secondary = 6;
    double leave_time = 2.0, spill_min_speed = 0.25, spill_k = 6.0;
    int spill_max = 6;
    double pickup_ttl = 8.0;
    int max_pickups = 64;
    double shrink_start = 0.4, final_radius = 0.35, restitution = 0.5;
    int max_group = 6;
    double carried_shaping = 2.0, win_bonus = 10.0, lose_penalty = 2.0;
};

Cfg cfg_from_array(const double* a, int len) {
    Cfg c;
    if (len < 28) return c;
    int k = 0;
    c.n_players = (int)a[k++]; c.n_mines = (int)a[k++]; c.dt = a[k++]; c.time_limit = a[k++];
    c.base_speed = a[k++]; c.vel_lerp = a[k++]; c.solo_radius = a[k++]; c.mine_radius = a[k++];
    c.mine_cap = a[k++]; c.mine_regen = a[k++]; c.mine_rate = a[k++]; c.mine_exp = a[k++];
    c.pad_radius = a[k++]; c.need_primary = (int)a[k++]; c.need_secondary = (int)a[k++];
    c.leave_time = a[k++]; c.spill_min_speed = a[k++]; c.spill_k = a[k++]; c.spill_max = (int)a[k++];
    c.pickup_ttl = a[k++]; c.max_pickups = (int)a[k++]; c.shrink_start = a[k++]; c.final_radius = a[k++];
    c.restitution = a[k++]; c.max_group = (int)a[k++]; c.carried_shaping = a[k++]; c.win_bonus = a[k++];
    c.lose_penalty = a[k++];
    return c;
}

struct Vec {
    double x = 0, y = 0;
    Vec() {}
    Vec(double a, double b) : x(a), y(b) {}
    Vec operator+(const Vec& o) const { return {x + o.x, y + o.y}; }
    Vec operator-(const Vec& o) const { return {x - o.x, y - o.y}; }
    Vec operator*(double s) const { return {x * s, y * s}; }
    double dot(const Vec& o) const { return x * o.x + y * o.y; }
    double norm() const { return std::hypot(x, y); }
};

struct Body {
    std::vector<int> members;
    Vec pos, vel;
    double pool[TYPES] = {0, 0, 0, 0};
    int n() const { return (int)members.size(); }
    double pool_total() const { return pool[0] + pool[1] + pool[2] + pool[3]; }
};

struct Player {
    double need[TYPES], banked[TYPES];
    double pad_angle = 0;
    int intent = 0;
    bool joinable = false;
    Vec dir;
    double leave_timer = -1.0;
};

struct Pickup {
    Vec pos;
    int type;
    double ttl;
};

enum SeatType { SEAT_EXTERNAL = 0, SEAT_EXTERNAL2 = 1, SEAT_SOLO = 2, SEAT_BAIL = 3 };

struct Stats {
    double group = 0;
    int steps = 0, merges = 0, leaves = 0, spills = 0, banks = 0;
};

int direction_to_move(Vec v) {
    double n = v.norm();
    if (n < 1e-6) return 0;
    double ang = std::atan2(v.y, v.x);
    long idx = (long)std::nearbyint(ang / (2 * PI / 8));  // round half to even, like Python
    idx = ((idx % 8) + 8) % 8;
    return (int)idx + 1;
}

struct Game {
    Cfg cfg;
    std::mt19937_64 rng;
    uint64_t seed = 0;
    double t = 0;
    int step_count = 0;
    bool done = false;
    int winner = -1;
    double R = 1.0;
    std::vector<Player> players;
    std::vector<Body> bodies;
    std::vector<Vec> mine_pos;
    std::vector<int> mine_type;
    std::vector<double> mine_stock;
    std::vector<char> mine_alive;
    std::deque<Pickup> picks;
    std::string events;  // JSON array items of the current tick, comma separated
    std::vector<double> prev_pot;
    std::vector<int> seats;
    Stats stats;

    double uniform(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }

    double radius(int n) const { return cfg.solo_radius * std::sqrt((double)n); }

    Vec pad_pos(int i) const {
        double r = std::max(R - cfg.pad_radius - 0.02, 0.1);
        return {r * std::cos(players[i].pad_angle), r * std::sin(players[i].pad_angle)};
    }

    int body_index(int player) const {
        for (size_t k = 0; k < bodies.size(); k++)
            for (int m : bodies[k].members)
                if (m == player) return (int)k;
        return -1;
    }

    double progress(int i) const {
        double s = 0;
        for (int t = 0; t < TYPES; t++) s += std::min(players[i].banked[t] / players[i].need[t], 1.0);
        return s / TYPES;
    }

    void event(const std::string& body) {
        if (!events.empty()) events += ",";
        char buf[64];
        snprintf(buf, sizeof buf, "{\"t\":%.2f,", t);
        events += buf + body + "}";
    }

    static std::string ids(const std::vector<int>& v) {
        std::string s = "[";
        for (size_t k = 0; k < v.size(); k++) { if (k) s += ","; s += std::to_string(v[k]); }
        return s + "]";
    }

    static std::string nums(const double* v, int n, int prec) {
        std::string s = "[";
        char buf[32];
        for (int k = 0; k < n; k++) { if (k) s += ","; snprintf(buf, sizeof buf, "%.*f", prec, v[k]); s += buf; }
        return s + "]";
    }

    void reset(uint64_t sd) {
        seed = sd;
        rng.seed(sd);
        t = 0; step_count = 0; done = false; winner = -1; R = 1.0;
        events.clear();
        stats = Stats();
        int n = cfg.n_players;
        players.assign(n, Player());
        double rot = uniform(0, 2 * PI);
        for (int i = 0; i < n; i++) {
            int primary = (int)std::uniform_int_distribution<int>(0, TYPES - 1)(rng);
            for (int t = 0; t < TYPES; t++) { players[i].need[t] = cfg.need_secondary; players[i].banked[t] = 0; }
            players[i].need[primary] = cfg.need_primary;
            players[i].pad_angle = rot + 2 * PI * i / n;
            players[i].intent = primary;
        }
        mine_pos.assign(cfg.n_mines, Vec());
        mine_type.assign(cfg.n_mines, 0);
        mine_stock.assign(cfg.n_mines, cfg.mine_cap);
        mine_alive.assign(cfg.n_mines, 1);
        double mrot = uniform(0, 2 * PI);
        for (int m = 0; m < cfg.n_mines; m++) {
            double ring = (m % 2 == 0) ? 0.62 : 0.35;
            double ang = mrot + 2 * PI * m / cfg.n_mines;
            mine_pos[m] = {ring * std::cos(ang), ring * std::sin(ang)};
            mine_type[m] = (cfg.n_mines >= 2 * TYPES) ? (m / 2) % TYPES : m % TYPES;
        }
        bodies.clear();
        for (int i = 0; i < n; i++) {
            Body b;
            b.members = {i};
            b.pos = {0.72 * std::cos(players[i].pad_angle), 0.72 * std::sin(players[i].pad_angle)};
            bodies.push_back(b);
        }
        picks.clear();
        prev_pot = potentials();
        if ((int)seats.size() != n) seats.assign(n, SEAT_EXTERNAL);
    }

    // ------------------------------------------------------------ actions

    void apply_actions(const int* act) {
        for (int i = 0; i < cfg.n_players; i++) {
            int move = act[i * 4], joinable = act[i * 4 + 1], leave = act[i * 4 + 2], intent = act[i * 4 + 3];
            Player& p = players[i];
            if (move == 0) p.dir = {0, 0};
            else { double a = 2 * PI * (move - 1) / 8; p.dir = {std::cos(a), std::sin(a)}; }
            p.joinable = joinable != 0;
            if (intent > 0) p.intent = intent - 1;
            if (leave && p.leave_timer < 0 && bodies[body_index(i)].n() > 1) {
                p.leave_timer = cfg.leave_time;
                event("\"kind\":\"leave_start\",\"player\":" + std::to_string(i));
            }
        }
    }

    // --------------------------------------------------------------- tick

    void tick() {
        if (done) return;
        events.clear();
        update_arena();
        move_bodies();
        update_leaving();
        collide_bodies();
        collide_mines();
        bank();
        collect_pickups();
        regen();
        t += cfg.dt;
        step_count++;
        for (int i = 0; i < cfg.n_players; i++) {
            bool win = true;
            for (int k = 0; k < TYPES; k++) if (players[i].banked[k] < players[i].need[k]) { win = false; break; }
            if (win) { done = true; winner = i; event("\"kind\":\"win\",\"player\":" + std::to_string(i)); break; }
        }
        if (!done && t >= cfg.time_limit) { done = true; event("\"kind\":\"timeout\""); }
    }

    void update_arena() {
        double frac = t / cfg.time_limit;
        if (frac <= cfg.shrink_start) R = 1.0;
        else {
            double k = (frac - cfg.shrink_start) / (1.0 - cfg.shrink_start);
            R = 1.0 + (cfg.final_radius - 1.0) * std::min(k, 1.0);
        }
        for (int m = 0; m < cfg.n_mines; m++) {
            if (mine_alive[m] && mine_pos[m].norm() + cfg.mine_radius > R) {
                mine_alive[m] = 0;
                event("\"kind\":\"mine_dead\",\"mine\":" + std::to_string(m));
            }
        }
    }

    void move_bodies() {
        double a = std::min(1.0, cfg.vel_lerp * cfg.dt);
        for (Body& b : bodies) {
            Vec mean;
            for (int i : b.members) mean = mean + players[i].dir;
            mean = mean * (1.0 / b.n());
            Vec target = mean * (cfg.base_speed / std::sqrt((double)b.n()));
            b.vel = b.vel + (target - b.vel) * a;
            b.pos = b.pos + b.vel * cfg.dt;
            double r = radius(b.n());
            double d = b.pos.norm();
            double limit = R - r;
            if (d > limit && d > 0) {
                Vec nrm = b.pos * (1.0 / d);
                b.pos = nrm * limit;
                double vr = b.vel.dot(nrm);
                if (vr > 0) b.vel = b.vel - nrm * vr;
            }
        }
    }

    void update_leaving() {
        for (int i = 0; i < cfg.n_players; i++) {
            Player& p = players[i];
            if (p.leave_timer < 0) continue;
            int bi = body_index(i);
            if (bodies[bi].n() == 1) { p.leave_timer = -1.0; continue; }
            p.leave_timer -= cfg.dt;
            if (p.leave_timer <= 0) { p.leave_timer = -1.0; detach(i, bi); }
        }
    }

    void detach(int i, int bi) {
        Body& b = bodies[bi];
        Player& p = players[i];
        double weights[TYPES] = {0, 0, 0, 0};
        for (int j : b.members) for (int t = 0; t < TYPES; t++) weights[t] += (players[j].intent == t) ? 2.0 : 1.0;
        double share[TYPES];
        for (int t = 0; t < TYPES; t++) {
            double w = (p.intent == t) ? 2.0 : 1.0;
            share[t] = b.pool[t] * (w / weights[t]);
            b.pool[t] -= share[t];
        }
        int from_size = b.n();
        b.members.erase(std::find(b.members.begin(), b.members.end(), i));
        Vec u = p.dir;
        if (u.norm() < 1e-6) { double ang = uniform(0, 2 * PI); u = {std::cos(ang), std::sin(ang)}; }
        else u = u * (1.0 / u.norm());
        Body nb;
        nb.members = {i};
        nb.pos = b.pos + u * (radius(b.n()) + radius(1) + 0.01);
        nb.vel = b.vel + u * 0.15;
        for (int t = 0; t < TYPES; t++) nb.pool[t] = share[t];
        bodies.push_back(nb);  // note: b reference is invalid after this line
        stats.leaves++;
        event("\"kind\":\"leave\",\"player\":" + std::to_string(i) + ",\"share\":" + nums(share, TYPES, 2) +
              ",\"from_size\":" + std::to_string(from_size));
    }

    bool joinable(const Body& b) const {
        for (int i : b.members) if (players[i].joinable) return true;
        return false;
    }

    void collide_bodies() {
        bool restart = true;
        int guard = 0;
        while (restart && guard < 20) {
            restart = false;
            guard++;
            int nb = (int)bodies.size();
            for (int ai = 0; ai < nb && !restart; ai++) {
                for (int bi = ai + 1; bi < nb; bi++) {
                    Body& a = bodies[ai];
                    Body& b = bodies[bi];
                    double ra = radius(a.n()), rb = radius(b.n());
                    Vec delta = b.pos - a.pos;
                    double dist = delta.norm();
                    if (dist >= ra + rb || dist < 1e-9) continue;
                    Vec nrm = delta * (1.0 / dist);
                    if (joinable(a) && joinable(b) && a.n() + b.n() <= cfg.max_group) {
                        merge(ai, bi);
                        restart = true;
                        break;
                    }
                    double ma = a.n(), mb = b.n();
                    double overlap = ra + rb - dist;
                    a.pos = a.pos - nrm * (overlap * (mb / (ma + mb)));
                    b.pos = b.pos + nrm * (overlap * (ma / (ma + mb)));
                    double rel = (a.vel - b.vel).dot(nrm);
                    if (rel > 0) {
                        double j = (1 + cfg.restitution) * rel / (1 / ma + 1 / mb);
                        a.vel = a.vel - nrm * (j / ma);
                        b.vel = b.vel + nrm * (j / mb);
                        if (rel > cfg.spill_min_speed) {
                            Vec contact = a.pos + nrm * ra;
                            int units = (int)std::min((double)cfg.spill_max,
                                                      std::nearbyint(cfg.spill_k * (rel - cfg.spill_min_speed) + 1));
                            int sa = spill(a, units, contact);
                            int sb = spill(b, units, contact);
                            if (sa || sb) {
                                stats.spills++;
                                char buf[96];
                                snprintf(buf, sizeof buf, ",\"units\":%d,\"speed\":%.2f,\"x\":%.3f,\"y\":%.3f", sa + sb, rel, contact.x, contact.y);
                                event("\"kind\":\"spill\",\"a\":" + ids(a.members) + ",\"b\":" + ids(b.members) + buf);
                            }
                        }
                    }
                }
            }
        }
    }

    void merge(int ai, int bi) {
        Body a = bodies[ai], b = bodies[bi];
        double ma = a.n(), mb = b.n();
        Body m;
        m.pos = (a.pos * ma + b.pos * mb) * (1.0 / (ma + mb));
        m.vel = (a.vel * ma + b.vel * mb) * (1.0 / (ma + mb));
        m.members = a.members;
        m.members.insert(m.members.end(), b.members.begin(), b.members.end());
        for (int t = 0; t < TYPES; t++) m.pool[t] = a.pool[t] + b.pool[t];
        for (int i : m.members) players[i].leave_timer = -1.0;
        std::vector<Body> nb;
        for (int k = 0; k < (int)bodies.size(); k++) if (k != ai && k != bi) nb.push_back(bodies[k]);
        nb.push_back(m);
        bodies = nb;
        stats.merges++;
        event("\"kind\":\"merge\",\"a\":" + ids(a.members) + ",\"b\":" + ids(b.members) + ",\"size\":" + std::to_string(m.n()));
    }

    int spill(Body& b, int units, Vec contact) {
        double total = b.pool_total();
        if (total <= 0 || units <= 0) return 0;
        units = std::min(units, (int)std::floor(total));
        if (units <= 0) return 0;
        int counts[TYPES] = {0, 0, 0, 0};
        std::discrete_distribution<int> dist({b.pool[0] / total, b.pool[1] / total, b.pool[2] / total, b.pool[3] / total});
        for (int k = 0; k < units; k++) counts[dist(rng)]++;
        int sum = 0;
        for (int t = 0; t < TYPES; t++) {
            counts[t] = std::min(counts[t], (int)std::floor(b.pool[t]));
            b.pool[t] -= counts[t];
            sum += counts[t];
        }
        for (int t = 0; t < TYPES; t++) {
            for (int k = 0; k < counts[t]; k++) {
                double ang = uniform(0, 2 * PI);
                double rad = uniform(0.04, 0.12);
                Vec pos = contact + Vec(rad * std::cos(ang), rad * std::sin(ang));
                double d = pos.norm();
                if (d > R - 0.02) pos = pos * ((R - 0.02) / d);
                picks.push_back({pos, t, cfg.pickup_ttl});
                if ((int)picks.size() > cfg.max_pickups) picks.pop_front();
            }
        }
        return sum;
    }

    void collide_mines() {
        for (Body& b : bodies) {
            double r = radius(b.n());
            for (int m = 0; m < cfg.n_mines; m++) {
                Vec delta = b.pos - mine_pos[m];
                double dist = delta.norm();
                if (dist >= r + cfg.mine_radius + 0.01) continue;
                if (dist < 1e-9) { delta = {1.0, 0.0}; dist = 1e-9; }
                Vec nrm = delta * (1.0 / dist);
                if (mine_alive[m] && mine_stock[m] > 0) {
                    double rate = cfg.mine_rate * std::pow((double)b.n(), cfg.mine_exp);
                    double amount = std::min(rate * cfg.dt, mine_stock[m]);
                    mine_stock[m] -= amount;
                    b.pool[mine_type[m]] += amount;
                }
                if (dist < r + cfg.mine_radius) {
                    b.pos = mine_pos[m] + nrm * (r + cfg.mine_radius);
                    double vr = b.vel.dot(nrm);
                    if (vr < 0) b.vel = b.vel - nrm * vr;
                }
            }
        }
    }

    void bank() {
        for (Body& b : bodies) {
            if (b.pool_total() < 0.5) continue;
            double r = radius(b.n());
            for (int i : b.members) {
                Vec pad = pad_pos(i);
                if ((b.pos - pad).norm() < r + cfg.pad_radius) {
                    double amount[TYPES];
                    for (int t = 0; t < TYPES; t++) { amount[t] = b.pool[t]; players[i].banked[t] += b.pool[t]; b.pool[t] = 0; }
                    stats.banks++;
                    event("\"kind\":\"bank\",\"player\":" + std::to_string(i) + ",\"amount\":" + nums(amount, TYPES, 2) +
                          ",\"group\":" + ids(b.members));
                    break;
                }
            }
        }
    }

    void collect_pickups() {
        if (picks.empty()) return;
        std::vector<char> keep(picks.size());
        for (size_t k = 0; k < picks.size(); k++) { picks[k].ttl -= cfg.dt; keep[k] = picks[k].ttl > 0; }
        for (Body& b : bodies) {
            double r = radius(b.n());
            for (size_t k = 0; k < picks.size(); k++) {
                if (!keep[k]) continue;
                if ((picks[k].pos - b.pos).norm() < r + 0.015) { b.pool[picks[k].type] += 1.0; keep[k] = 0; }
            }
        }
        std::deque<Pickup> np;
        for (size_t k = 0; k < picks.size(); k++) if (keep[k]) np.push_back(picks[k]);
        picks.swap(np);
    }

    void regen() {
        for (int m = 0; m < cfg.n_mines; m++)
            if (mine_alive[m]) mine_stock[m] = std::min(cfg.mine_cap, mine_stock[m] + cfg.mine_regen * cfg.dt);
    }

    // -------------------------------------------------------------- reward

    std::vector<double> potentials() const {
        std::vector<double> pots(cfg.n_players);
        for (int i = 0; i < cfg.n_players; i++) {
            const Body& b = bodies[body_index(i)];
            double bp = 0, cp = 0;
            for (int t = 0; t < TYPES; t++) {
                bp += std::min(players[i].banked[t] / players[i].need[t], 1.0);
                cp += std::min((players[i].banked[t] + b.pool[t] / b.n()) / players[i].need[t], 1.0);
            }
            bp /= TYPES; cp /= TYPES;
            pots[i] = 10.0 * bp + cfg.carried_shaping * (cp - bp);
        }
        return pots;
    }

    // --------------------------------------------------------- observation

    int obs_dim() const { return 29 + (cfg.n_players - 1) * 16 + cfg.n_mines * 8 + N_PICKUPS_OBS * 6; }

    void observe(float* out) const {
        int k = cfg.n_players;
        double tfrac = t / cfg.time_limit;
        std::vector<int> bidx(k);
        std::vector<double> prog(k);
        std::vector<Vec> pads(k);
        for (int i = 0; i < k; i++) { bidx[i] = body_index(i); prog[i] = progress(i); pads[i] = pad_pos(i); }
        int D = obs_dim();
        for (int i = 0; i < k; i++) {
            float* f = out + i * D;
            int c = 0;
            const Player& p = players[i];
            const Body& b = bodies[bidx[i]];
            Vec pos = b.pos;
            f[c++] = pos.x; f[c++] = pos.y; f[c++] = b.vel.x; f[c++] = b.vel.y;
            for (int t = 0; t < TYPES; t++) f[c++] = p.need[t] / cfg.need_primary;
            for (int t = 0; t < TYPES; t++) f[c++] = std::min(p.banked[t] / p.need[t], 1.5);
            for (int t = 0; t < TYPES; t++) f[c++] = (p.intent == t) ? 1.f : 0.f;
            f[c++] = p.joinable ? 1.f : 0.f;
            f[c++] = (float)b.n() / cfg.max_group;
            for (int t = 0; t < TYPES; t++) f[c++] = b.pool[t] / 20.0;
            f[c++] = p.leave_timer >= 0 ? p.leave_timer / cfg.leave_time : 0.f;
            f[c++] = R;
            f[c++] = tfrac;
            f[c++] = pads[i].x - pos.x; f[c++] = pads[i].y - pos.y;
            f[c++] = b.pool_total() / 40.0;
            f[c++] = prog[i];
            // Others sorted by distance, stable on index.
            std::vector<std::pair<double, int>> order;
            for (int j = 0; j < k; j++) if (j != i) order.push_back({(bodies[bidx[j]].pos - pos).norm(), j});
            std::stable_sort(order.begin(), order.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first < b.first; });
            for (auto& oj : order) {
                int j = oj.second;
                const Player& q = players[j];
                const Body& bj = bodies[bidx[j]];
                f[c++] = bj.pos.x - pos.x; f[c++] = bj.pos.y - pos.y;
                f[c++] = bj.vel.x; f[c++] = bj.vel.y;
                f[c++] = (float)bj.n() / cfg.max_group;
                f[c++] = q.joinable ? 1.f : 0.f;
                for (int t = 0; t < TYPES; t++) f[c++] = (q.intent == t) ? 1.f : 0.f;
                f[c++] = (bidx[j] == bidx[i]) ? 1.f : 0.f;
                f[c++] = q.leave_timer >= 0 ? 1.f : 0.f;
                f[c++] = prog[j];
                f[c++] = pads[j].x - pos.x; f[c++] = pads[j].y - pos.y;
                f[c++] = bj.pool_total() / 40.0;
            }
            for (int m = 0; m < cfg.n_mines; m++) {
                f[c++] = mine_pos[m].x - pos.x; f[c++] = mine_pos[m].y - pos.y;
                for (int t = 0; t < TYPES; t++) f[c++] = (mine_type[m] == t) ? 1.f : 0.f;
                f[c++] = mine_stock[m] / cfg.mine_cap;
                f[c++] = mine_alive[m] ? 1.f : 0.f;
            }
            std::vector<std::pair<double, int>> po;
            for (size_t q = 0; q < picks.size(); q++) po.push_back({(picks[q].pos - pos).norm(), (int)q});
            std::stable_sort(po.begin(), po.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first < b.first; });
            int cnt = std::min((int)po.size(), N_PICKUPS_OBS);
            for (int q = 0; q < cnt; q++) {
                const Pickup& pk = picks[po[q].second];
                f[c++] = pk.pos.x - pos.x; f[c++] = pk.pos.y - pos.y;
                for (int t = 0; t < TYPES; t++) f[c++] = (pk.type == t) ? 1.f : 0.f;
            }
            for (int q = cnt; q < N_PICKUPS_OBS; q++) for (int z = 0; z < 6; z++) f[c++] = 0.f;
        }
    }

    // ---------------------------------------------------------------- bots

    int nearest_mine(Vec pos, const bool* types) const {
        int best = -1;
        double bd = 1e9;
        for (int m = 0; m < cfg.n_mines; m++) {
            if (!mine_alive[m] || mine_stock[m] < 1.0) continue;
            if (types && !types[mine_type[m]]) continue;
            double d = (mine_pos[m] - pos).norm();
            if (d < bd) { best = m; bd = d; }
        }
        return best;
    }

    void needed_types(int i, bool* out, bool* any) const {
        const Body& b = bodies[body_index(i)];
        *any = false;
        for (int t = 0; t < TYPES; t++) {
            out[t] = players[i].banked[t] + b.pool[t] / b.n() < players[i].need[t];
            if (out[t]) *any = true;
        }
    }

    void bot_solo(int i, int* act) const {
        const Player& p = players[i];
        const Body& b = bodies[body_index(i)];
        int primary = 0;
        for (int t = 1; t < TYPES; t++) if (p.need[t] > p.need[primary]) primary = t;
        bool has_target = false;
        Vec target;
        if (b.pool_total() >= 8.0) { target = pad_pos(i); has_target = true; }
        else {
            bool needed[TYPES], any;
            needed_types(i, needed, &any);
            int m = nearest_mine(b.pos, any ? needed : nullptr);
            if (m < 0) m = nearest_mine(b.pos, nullptr);
            if (m >= 0) { target = mine_pos[m]; has_target = true; }
            else if (b.pool_total() > 0) { target = pad_pos(i); has_target = true; }
        }
        act[0] = has_target ? direction_to_move(target - b.pos) : 0;
        act[1] = 0;
        act[2] = b.n() > 1 ? 1 : 0;
        act[3] = primary + 1;
    }

    void bot_bail(int i, int* act) const {
        const double max_group = 3, bank_at = 10.0, bail_at = 4.0, pad_danger = 0.3;
        const Player& p = players[i];
        const Body& b = bodies[body_index(i)];
        bool needed[TYPES], any;
        needed_types(i, needed, &any);
        int intent;
        if (any) { intent = 0; while (!needed[intent]) intent++; }
        else { intent = 0; for (int t = 1; t < TYPES; t++) if (p.need[t] > p.need[intent]) intent = t; }
        int joinable = b.n() < max_group ? 1 : 0;
        int leave = 0;
        bool has_target = false;
        Vec target;
        Vec my_pad = pad_pos(i);
        if (b.n() > 1) {
            double my_share = b.pool_total() / b.n();
            for (int j : b.members) {
                if (j == i) continue;
                if ((pad_pos(j) - b.pos).norm() < pad_danger && b.pool_total() >= bail_at) leave = 1;
            }
            if (b.pool_total() >= bank_at) { target = my_pad; has_target = true; }
            if (my_share >= bail_at && (my_pad - b.pos).norm() > 0.6 && leave == 0) leave = 1;
        } else {
            if (b.pool_total() >= bail_at) { target = my_pad; has_target = true; }
        }
        if (!has_target) {
            int m = nearest_mine(b.pos, any ? needed : nullptr);
            if (m < 0) m = nearest_mine(b.pos, nullptr);
            target = m >= 0 ? mine_pos[m] : my_pad;
        }
        act[0] = direction_to_move(target - b.pos);
        act[1] = joinable;
        act[2] = leave;
        act[3] = intent + 1;
    }

    // ------------------------------------------------------------ recording

    std::string frame_json() const {
        std::string s;
        char buf[128];
        snprintf(buf, sizeof buf, "{\"t\":%.2f,\"R\":%.3f,\"bodies\":[", t, R);
        s += buf;
        for (size_t k = 0; k < bodies.size(); k++) {
            const Body& b = bodies[k];
            if (k) s += ",";
            snprintf(buf, sizeof buf, ",\"x\":%.3f,\"y\":%.3f,\"vx\":%.3f,\"vy\":%.3f,\"pool\":", b.pos.x, b.pos.y, b.vel.x, b.vel.y);
            s += "{\"m\":" + ids(b.members) + buf + nums(b.pool, TYPES, 1) + "}";
        }
        s += "],\"mines\":" + nums(mine_stock.data(), cfg.n_mines, 1) + ",\"alive\":[";
        for (int m = 0; m < cfg.n_mines; m++) { if (m) s += ","; s += mine_alive[m] ? "true" : "false"; }
        s += "],\"picks\":[";
        for (size_t k = 0; k < picks.size(); k++) {
            if (k) s += ",";
            snprintf(buf, sizeof buf, "[%.3f,%.3f,%d]", picks[k].pos.x, picks[k].pos.y, picks[k].type);
            s += buf;
        }
        s += "],\"players\":[";
        for (int i = 0; i < cfg.n_players; i++) {
            const Player& p = players[i];
            if (i) s += ",";
            s += "{\"banked\":" + nums(p.banked, TYPES, 1);
            snprintf(buf, sizeof buf, ",\"intent\":%d,\"join\":%s,\"leaving\":%s,\"dir\":[%.2f,%.2f]}", p.intent,
                     p.joinable ? "true" : "false", p.leave_timer >= 0 ? std::to_string((int)(p.leave_timer * 10 + 0.5) / 10.0).c_str() : "-1", p.dir.x, p.dir.y);
            s += buf;
        }
        s += "],\"events\":[" + events + "]}";
        return s;
    }

    std::string meta_json() const {
        std::string s = "{\"seed\":" + std::to_string(seed) + ",\"needs\":[";
        for (int i = 0; i < cfg.n_players; i++) { if (i) s += ","; s += nums(players[i].need, TYPES, 0); }
        s += "],\"pads\":[";
        char buf[64];
        for (int i = 0; i < cfg.n_players; i++) { if (i) s += ","; snprintf(buf, sizeof buf, "%.6f", players[i].pad_angle); s += buf; }
        s += "],\"mine_pos\":[";
        for (int m = 0; m < cfg.n_mines; m++) { if (m) s += ","; snprintf(buf, sizeof buf, "[%.6f,%.6f]", mine_pos[m].x, mine_pos[m].y); s += buf; }
        s += "],\"mine_type\":" + ids(mine_type) + ",\"winner\":" + std::to_string(winner) + "}";
        return s;
    }
};

struct Batch {
    std::vector<Game> games;
    int decide_every = 2;
    uint64_t next_seed = 1;
};

}  // namespace

extern "C" {

void* ugb_create(int n_envs, const double* cfg, int cfg_len, unsigned long long seed, int decide_every) {
    Batch* b = new Batch();
    b->decide_every = decide_every;
    b->next_seed = seed;
    b->games.resize(n_envs);
    for (int e = 0; e < n_envs; e++) {
        b->games[e].cfg = cfg_from_array(cfg, cfg_len);
        b->games[e].seats.assign(b->games[e].cfg.n_players, SEAT_EXTERNAL);
        b->games[e].reset(b->next_seed++);
    }
    return b;
}

void ugb_destroy(void* h) { delete (Batch*)h; }

int ugb_obs_dim(void* h) { return ((Batch*)h)->games[0].obs_dim(); }
int ugb_n_players(void* h) { return ((Batch*)h)->games[0].cfg.n_players; }

void ugb_set_seats(void* h, int env, const int* seats) {
    Game& g = ((Batch*)h)->games[env];
    for (int i = 0; i < g.cfg.n_players; i++) g.seats[i] = seats[i];
}

void ugb_reset(void* h, int env, unsigned long long seed) {
    Batch* b = (Batch*)h;
    b->games[env].reset(seed ? seed : b->next_seed++);
}

void ugb_observe(void* h, float* obs) {
    Batch* b = (Batch*)h;
    int D = b->games[0].obs_dim(), n = b->games[0].cfg.n_players;
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < (int)b->games.size(); e++) b->games[e].observe(obs + (size_t)e * n * D);
}

// Steps every game. actions: (E, n, 4) int32 for external seats (bot seats are overridden).
// Writes rewards (E, n), dones (E), next obs (E, n, D). When a game ends and auto_reset is set, it
// is reset with a fresh seed before its observation is written, and its episode summary is written
// to ep_stats (E, 8 + n): [winner, length, avg_group, merges, leaves, spills, banks, ended, progress...].
void ugb_step(void* h, const int* actions, float* rewards, unsigned char* dones, float* obs, int auto_reset,
              double* ep_stats, int decide_every_override) {
    Batch* b = (Batch*)h;
    int E = (int)b->games.size();
    int n = b->games[0].cfg.n_players;
    int D = b->games[0].obs_dim();
    int de = decide_every_override > 0 ? decide_every_override : b->decide_every;
    std::vector<uint64_t> seeds(E);
    for (int e = 0; e < E; e++) seeds[e] = b->next_seed++;
    #pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < E; e++) {
        Game& g = b->games[e];
        std::vector<int> act(actions + (size_t)e * n * 4, actions + (size_t)(e + 1) * n * 4);
        for (int i = 0; i < n; i++) {
            if (g.seats[i] == SEAT_SOLO) g.bot_solo(i, &act[i * 4]);
            else if (g.seats[i] == SEAT_BAIL) g.bot_bail(i, &act[i * 4]);
        }
        g.apply_actions(act.data());
        std::string ev;
        for (int k = 0; k < de; k++) {
            g.tick();
            if (!g.events.empty()) { if (!ev.empty()) ev += ","; ev += g.events; }
            if (g.done) break;
        }
        g.events = ev;  // events of the whole step, for frame_json when recording
        std::vector<double> pot = g.potentials();
        for (int i = 0; i < n; i++) rewards[e * n + i] = (float)(pot[i] - g.prev_pot[i]);
        g.prev_pot = pot;
        if (g.done && g.winner >= 0) {
            for (int i = 0; i < n; i++) rewards[e * n + i] -= (float)g.cfg.lose_penalty;
            rewards[e * n + g.winner] += (float)(g.cfg.lose_penalty + g.cfg.win_bonus);
        }
        g.stats.steps++;
        double gs = 0;
        for (int i = 0; i < n; i++) gs += g.bodies[g.body_index(i)].n();
        g.stats.group += gs / n;
        dones[e] = g.done ? 1 : 0;
        double* st = ep_stats + (size_t)e * (8 + n);
        st[7] = 0;
        if (g.done) {
            st[0] = g.winner; st[1] = g.t; st[2] = g.stats.group / std::max(1, g.stats.steps);
            st[3] = g.stats.merges; st[4] = g.stats.leaves; st[5] = g.stats.spills; st[6] = g.stats.banks; st[7] = 1;
            for (int i = 0; i < n; i++) st[8 + i] = g.progress(i);
            if (auto_reset) g.reset(seeds[e]);
        }
        g.observe(obs + (size_t)e * n * D);
    }
}

int ugb_frame_json(void* h, int env, char* buf, int cap) {
    std::string s = ((Batch*)h)->games[env].frame_json();
    int len = (int)s.size();
    if (len + 1 > cap) return -(len + 1);
    memcpy(buf, s.c_str(), len + 1);
    return len;
}

int ugb_meta_json(void* h, int env, char* buf, int cap) {
    std::string s = ((Batch*)h)->games[env].meta_json();
    int len = (int)s.size();
    if (len + 1 > cap) return -(len + 1);
    memcpy(buf, s.c_str(), len + 1);
    return len;
}

double ugb_progress(void* h, int env, int player) { return ((Batch*)h)->games[env].progress(player); }
int ugb_winner(void* h, int env) { return ((Batch*)h)->games[env].winner; }
int ugb_done(void* h, int env) { return ((Batch*)h)->games[env].done ? 1 : 0; }
double ugb_time(void* h, int env) { return ((Batch*)h)->games[env].t; }

}  // extern "C"
