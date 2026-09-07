// Ungroup v2 rules core (canonical implementation).
//
// This file is the single source of truth for the "carry, bank, spill" rules. The Python
// package drives it through ctypes (rl/ungroup/native.py). A batch owns many independent games
// and steps them in parallel with OpenMP. Observations, rewards, privileged critic state, the
// scripted bots, and per-round alliance statistics are all computed here.
//
// Rules (v2, after the 2026-09 design review):
// - Circles that touch merge into one group body when EVERY member of both bodies is joinable,
//   nobody involved is on a join cooldown, and no pair is on a partner cooldown.
// - Group speed = base / sqrt(n) times the mean direction of the members that are pushing
//   (members with no direction "follow" instead of braking).
// - Mines yield mine_rate * n^mine_exp per second while a body touches them (not while stunned).
// - Nothing counts until banked: touching a member's home pad banks the whole pool to them.
// - Leaving is a HELD action: the leave timer runs while the member keeps requesting it and is
//   cancelled once they have stopped requesting it for leave_hold seconds. On detaching, the leaver takes a per-type share weighted
//   intent_weight:1 toward their declared intent, minus a forfeit that stays with the group. They
//   are ejected opposite to the group's motion, their joinable flag is forced off, they cannot
//   join anyone for join_cooldown seconds and cannot rejoin their former partners for
//   partner_cooldown seconds. Merging never cancels anyone's leave timer.
// - Intent is initialised randomly, can only be changed while solo, and is public.
// - Hard collisions (approach speed above spill_min_speed) spill carried units from both bodies
//   onto the floor and stun both bodies for stun_time seconds (no movement, no mining).
// - The arena shrinks after shrink_start; mines outside it die; pads slide inward.
// - A player wins by banking their full need vector. At the time limit the player with the
//   highest progress wins (ties by banked units), so every round has a winner.
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
constexpr int MAX_SLOTS = 8;      // other-player slots in the observation (nearest others)
constexpr int MAX_PLAYERS = 32;   // hard cap on seats (pairwise history arrays)
constexpr int N_PICKUPS_OBS = 4;
constexpr int N_DR = 4;           // randomised constants exposed in the observation
constexpr double PI = 3.14159265358979323846;

// Config array layout shared with rl/ungroup/native.py (CFG_FIELDS). Keep in sync.
struct Cfg {
    int n_players = 6, n_mines = 8;
    double dt = 1.0 / 30.0, time_limit = 240.0, base_speed = 0.45, vel_lerp = 6.0, solo_radius = 0.045;
    double mine_radius = 0.08, mine_cap = 30.0, mine_regen = 0.5, mine_rate = 0.12, mine_exp = 2.0;
    double pad_radius = 0.06;
    int need_primary = 18, need_secondary = 6;
    double leave_time = 2.0, spill_min_speed = 0.40, spill_k = 6.0;
    int spill_max = 6;
    double pickup_ttl = 8.0;
    int max_pickups = 64;
    double shrink_start = 0.5, final_radius = 0.5, restitution = 0.5;
    int max_group = 6;
    double join_cooldown = 3.0, partner_cooldown = 10.0, leave_forfeit = 0.15, intent_weight = 3.0, stun_time = 1.0;
    double leave_hold = 1.0;  // seconds a leave request stays active after the last leave=1 decision
    double group_bank_bonus = 0.15;  // banked amount x (1 + bonus * (n - 1)) when a group banks
    double rammer_stun_mult = 2.5;   // the faster body in a spill is stunned this many times longer
    // reward
    double carried_shaping = 2.0, win_bonus = 10.0, lose_penalty = 2.0, relative_reward = 0.5;
};
constexpr int CFG_LEN = 37;

void cfg_fill(Cfg& c, const double* a, int len) {
    if (len < CFG_LEN) return;
    int k = 0;
    c.n_players = (int)a[k++]; c.n_mines = (int)a[k++]; c.dt = a[k++]; c.time_limit = a[k++];
    c.base_speed = a[k++]; c.vel_lerp = a[k++]; c.solo_radius = a[k++]; c.mine_radius = a[k++];
    c.mine_cap = a[k++]; c.mine_regen = a[k++]; c.mine_rate = a[k++]; c.mine_exp = a[k++];
    c.pad_radius = a[k++]; c.need_primary = (int)a[k++]; c.need_secondary = (int)a[k++];
    c.leave_time = a[k++]; c.spill_min_speed = a[k++]; c.spill_k = a[k++]; c.spill_max = (int)a[k++];
    c.pickup_ttl = a[k++]; c.max_pickups = (int)a[k++]; c.shrink_start = a[k++]; c.final_radius = a[k++];
    c.restitution = a[k++]; c.max_group = (int)a[k++];
    c.join_cooldown = a[k++]; c.partner_cooldown = a[k++]; c.leave_forfeit = a[k++]; c.intent_weight = a[k++];
    c.stun_time = a[k++]; c.leave_hold = a[k++]; c.group_bank_bonus = a[k++]; c.rammer_stun_mult = a[k++];
    c.carried_shaping = a[k++]; c.win_bonus = a[k++]; c.lose_penalty = a[k++]; c.relative_reward = a[k++];
}

void cfg_dump(const Cfg& c, double* a) {
    int k = 0;
    a[k++] = c.n_players; a[k++] = c.n_mines; a[k++] = c.dt; a[k++] = c.time_limit;
    a[k++] = c.base_speed; a[k++] = c.vel_lerp; a[k++] = c.solo_radius; a[k++] = c.mine_radius;
    a[k++] = c.mine_cap; a[k++] = c.mine_regen; a[k++] = c.mine_rate; a[k++] = c.mine_exp;
    a[k++] = c.pad_radius; a[k++] = c.need_primary; a[k++] = c.need_secondary;
    a[k++] = c.leave_time; a[k++] = c.spill_min_speed; a[k++] = c.spill_k; a[k++] = c.spill_max;
    a[k++] = c.pickup_ttl; a[k++] = c.max_pickups; a[k++] = c.shrink_start; a[k++] = c.final_radius;
    a[k++] = c.restitution; a[k++] = c.max_group;
    a[k++] = c.join_cooldown; a[k++] = c.partner_cooldown; a[k++] = c.leave_forfeit; a[k++] = c.intent_weight;
    a[k++] = c.stun_time; a[k++] = c.leave_hold; a[k++] = c.group_bank_bonus; a[k++] = c.rammer_stun_mult;
    a[k++] = c.carried_shaping; a[k++] = c.win_bonus; a[k++] = c.lose_penalty; a[k++] = c.relative_reward;
}

struct Vec {
    double x = 0, y = 0;
    Vec() {}
    Vec(double a, double b) : x(a), y(b) {}
    Vec operator+(const Vec& o) const { return {x + o.x, y + o.y}; }
    Vec operator-(const Vec& o) const { return {x - o.x, y - o.y}; }
    Vec operator*(double s) const { return {x * s, y * s}; }
    double dot(const Vec& o) const { return x * o.x + y * o.y; }
    double norm() const { return std::sqrt(x * x + y * y); }
};

struct Body {
    std::vector<int> members;
    Vec pos, vel;
    double pool[TYPES] = {0, 0, 0, 0};
    double stun = 0;  // seconds of stun remaining
    int n() const { return (int)members.size(); }
    double pool_total() const { return pool[0] + pool[1] + pool[2] + pool[3]; }
};

struct Player {
    double need[TYPES], banked[TYPES];
    double pad_angle = 0;
    int intent = 0;
    bool joinable = false;
    Vec dir;
    double leave_timer = -1.0;   // seconds remaining, < 0 when not leaving
    double leave_last_req = -1e9; // time of the last leave=1 decision
    double join_cooldown = 0;    // seconds until this player may merge again
    double group_since = 0;      // time the current membership started (t)
    double last_bank_t = 0;
};

struct Pickup {
    Vec pos;
    int type;
    double ttl;
};

enum SeatType { SEAT_EXTERNAL = 0, SEAT_EXTERNAL2 = 1, SEAT_SOLO = 2, SEAT_BAIL = 3, SEAT_LOYAL = 4, SEAT_KIDNAP = 5, SEAT_RAMMER = 6 };

struct Stats {
    double group = 0;
    int steps = 0, merges = 0, leaves = 0, spills = 0, banks = 0, group_banks = 0, remerge_fast = 0, cancels = 0;
    int alliances = 0, alliances_long = 0;
    double alliance_dur = 0;
    double units_taken = 0;  // units carried away by leavers
};

int direction_to_move(Vec v) {
    double n = v.norm();
    if (n < 1e-6) return 0;
    double ang = std::atan2(v.y, v.x);
    long idx = (long)std::nearbyint(ang / (2 * PI / 8));
    idx = ((idx % 8) + 8) % 8;
    return (int)idx + 1;
}

struct Game {
    Cfg base_cfg, cfg;
    Cfg lo, hi;
    bool randomize = false;
    std::mt19937_64 rng;
    uint64_t seed = 0;
    double t = 0;
    bool done = false;
    int winner = -1;
    bool timeout_win = false;
    double R = 1.0;
    std::vector<Player> players;
    std::vector<Body> bodies;
    std::vector<Vec> mine_pos;
    std::vector<int> mine_type;
    std::vector<double> mine_stock;
    std::vector<char> mine_alive;
    std::deque<Pickup> picks;
    std::string events;
    std::vector<double> prev_pot;
    std::vector<int> seats;
    Stats stats;
    // pairwise history (indexed [i][j] with MAX_SLOTS stride)
    double comember_time[MAX_PLAYERS][MAX_PLAYERS];
    double took_from[MAX_PLAYERS][MAX_PLAYERS];        // units j took when leaving a body containing i
    double banked_while[MAX_PLAYERS][MAX_PLAYERS];     // units banked to j while i was a member
    double last_left_me[MAX_PLAYERS][MAX_PLAYERS];     // t at which j last left a body containing i (-1 never)
    double partner_cd[MAX_PLAYERS][MAX_PLAYERS];       // t until which i may not merge with j
    double pair_since[MAX_PLAYERS][MAX_PLAYERS];       // start time of the current co-membership (-1 none)
    double pair_ended[MAX_PLAYERS][MAX_PLAYERS];       // end time of the last co-membership (-1 none)

    double uniform(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
    double radius(int n) const { return cfg.solo_radius * std::sqrt((double)n); }
    double need_total(int i) const { double s = 0; for (int t = 0; t < TYPES; t++) s += players[i].need[t]; return s; }

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
    double banked_total(int i) const { double s = 0; for (int t = 0; t < TYPES; t++) s += players[i].banked[t]; return s; }

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

    void sample_cfg() {
        cfg = base_cfg;
        if (!randomize) return;
        auto draw = [&](double a, double b) { return a >= b ? a : uniform(a, b); };
        cfg.mine_rate = draw(lo.mine_rate, hi.mine_rate);
        cfg.mine_regen = draw(lo.mine_regen, hi.mine_regen);
        cfg.leave_time = draw(lo.leave_time, hi.leave_time);
        cfg.need_primary = (int)std::nearbyint(draw(lo.need_primary, hi.need_primary));
        cfg.need_secondary = std::max(1, cfg.need_primary / 3);
    }

    void reset(uint64_t sd) {
        seed = sd;
        rng.seed(sd);
        sample_cfg();
        t = 0; done = false; winner = -1; timeout_win = false; R = 1.0;
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
            players[i].intent = (int)std::uniform_int_distribution<int>(0, TYPES - 1)(rng);
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
        for (int i = 0; i < MAX_PLAYERS; i++)
            for (int j = 0; j < MAX_PLAYERS; j++) {
                comember_time[i][j] = 0; took_from[i][j] = 0; banked_while[i][j] = 0;
                last_left_me[i][j] = -1; partner_cd[i][j] = -1; pair_since[i][j] = -1; pair_ended[i][j] = -1;
            }
        prev_pot = potentials();
        if ((int)seats.size() != n) seats.assign(n, SEAT_EXTERNAL);
    }

    // ------------------------------------------------------------ actions

    void apply_actions(const int* act) {
        for (int i = 0; i < cfg.n_players; i++) {
            int move = act[i * 4], joinable = act[i * 4 + 1], leave = act[i * 4 + 2], intent = act[i * 4 + 3];
            Player& p = players[i];
            if (move == 0) p.dir = {0, 0};
            else if (move == 9) { /* keep the direction set through set_direction (human seats) */ }
            else { double a = 2 * PI * (move - 1) / 8; p.dir = {std::cos(a), std::sin(a)}; }
            p.joinable = joinable != 0;
            int bi = body_index(i);
            int n = bodies[bi].n();
            if (intent > 0 && n == 1) p.intent = intent - 1;  // locked while grouped
            if (leave && n > 1) {
                p.leave_last_req = t;
                if (p.leave_timer < 0) { p.leave_timer = cfg.leave_time; event("\"kind\":\"leave_start\",\"player\":" + std::to_string(i)); }
            } else if (!leave && p.leave_timer >= 0 && t - p.leave_last_req > cfg.leave_hold) {
                p.leave_timer = -1.0;
                stats.cancels++;
                event("\"kind\":\"leave_cancel\",\"player\":" + std::to_string(i));
            }
        }
    }

    void set_direction(int i, double dx, double dy) {
        double n = std::sqrt(dx * dx + dy * dy);
        players[i].dir = n < 1e-6 ? Vec(0, 0) : Vec(dx / n, dy / n);
    }

    // --------------------------------------------------------------- tick

    void tick() {
        if (done) return;
        update_arena();
        move_bodies();
        update_timers();
        update_leaving();
        collide_bodies();
        collide_mines();
        bank();
        collect_pickups();
        regen();
        t += cfg.dt;
        for (int i = 0; i < cfg.n_players; i++) {
            bool win = true;
            for (int k = 0; k < TYPES; k++) if (players[i].banked[k] < players[i].need[k]) { win = false; break; }
            if (win) { done = true; winner = i; event("\"kind\":\"win\",\"player\":" + std::to_string(i)); break; }
        }
        if (!done && t >= cfg.time_limit - 1e-9) {
            done = true;
            int best = 0;
            for (int i = 1; i < cfg.n_players; i++) {
                double pi = progress(i), pb = progress(best);
                if (pi > pb + 1e-12 || (std::fabs(pi - pb) <= 1e-12 && banked_total(i) > banked_total(best))) best = i;
            }
            winner = best;
            timeout_win = true;
            event("\"kind\":\"timeout\",\"player\":" + std::to_string(best));
        }
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
            int pushing = 0;
            for (int i : b.members) {
                if (players[i].dir.norm() > 1e-6) { mean = mean + players[i].dir; pushing++; }
            }
            if (pushing > 0) mean = mean * (1.0 / pushing);
            Vec target = b.stun > 0 ? Vec(0, 0) : mean * (cfg.base_speed / std::sqrt((double)b.n()));
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

    void update_timers() {
        for (Body& b : bodies) if (b.stun > 0) b.stun = std::max(0.0, b.stun - cfg.dt);
        for (Player& p : players) if (p.join_cooldown > 0) p.join_cooldown = std::max(0.0, p.join_cooldown - cfg.dt);
        // co-membership time
        for (const Body& b : bodies)
            if (b.n() > 1)
                for (int i : b.members) for (int j : b.members) if (i != j) comember_time[i][j] += cfg.dt;
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

    void end_pair(int i, int j) {
        if (pair_since[i][j] >= 0) {
            double dur = t - pair_since[i][j];
            stats.alliances++;
            stats.alliance_dur += dur;
            if (dur >= 10.0) stats.alliances_long++;
            pair_ended[i][j] = t;
            pair_since[i][j] = -1;
        }
    }

    void detach(int i, int bi) {
        Body& b = bodies[bi];
        Player& p = players[i];
        double weights[TYPES] = {0, 0, 0, 0};
        for (int j : b.members) for (int t = 0; t < TYPES; t++) weights[t] += (players[j].intent == t) ? cfg.intent_weight : 1.0;
        double share[TYPES];
        double taken = 0;
        for (int t = 0; t < TYPES; t++) {
            double w = (p.intent == t) ? cfg.intent_weight : 1.0;
            share[t] = b.pool[t] * (w / weights[t]) * (1.0 - cfg.leave_forfeit);
            b.pool[t] -= share[t];
            taken += share[t];
        }
        int from_size = b.n();
        std::vector<int> former;
        for (int j : b.members) if (j != i) former.push_back(j);
        b.members.erase(std::find(b.members.begin(), b.members.end(), i));
        for (int j : former) {
            took_from[j][i] += taken;
            last_left_me[j][i] = t;
            partner_cd[i][j] = t + cfg.partner_cooldown;
            partner_cd[j][i] = t + cfg.partner_cooldown;
            end_pair(i, j); end_pair(j, i);
        }
        // Eject opposite to the group's motion (or a random direction when it is still).
        Vec u = b.vel * -1.0;
        if (u.norm() < 1e-3) {
            Vec mean;
            for (int j : b.members) mean = mean + players[j].dir;
            u = mean * -1.0;
        }
        if (u.norm() < 1e-6) { double ang = uniform(0, 2 * PI); u = {std::cos(ang), std::sin(ang)}; }
        else u = u * (1.0 / u.norm());
        Body nb;
        nb.members = {i};
        nb.pos = b.pos + u * (radius(b.n()) + radius(1) + 0.02);
        nb.vel = u * 0.2;
        for (int t = 0; t < TYPES; t++) nb.pool[t] = share[t];
        p.joinable = false;
        p.join_cooldown = cfg.join_cooldown;
        p.group_since = t;
        bodies.push_back(nb);  // b is invalid after this line
        stats.leaves++;
        stats.units_taken += taken;
        event("\"kind\":\"leave\",\"player\":" + std::to_string(i) + ",\"share\":" + nums(share, TYPES, 2) +
              ",\"from_size\":" + std::to_string(from_size));
    }

    bool all_joinable(const Body& b) const {
        for (int i : b.members) if (!players[i].joinable || players[i].join_cooldown > 0) return false;
        return true;
    }

    bool can_merge(const Body& a, const Body& b) const {
        if (a.n() + b.n() > cfg.max_group) return false;
        if (!all_joinable(a) || !all_joinable(b)) return false;
        for (int i : a.members) for (int j : b.members) if (partner_cd[i][j] > t) return false;
        return true;
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
                    if (can_merge(a, b)) { merge(ai, bi); restart = true; break; }
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
                            // The faster body along the normal is the aggressor and is dazed longer.
                            bool a_faster = a.vel.dot(nrm) > -b.vel.dot(nrm);
                            a.stun = std::max(a.stun, cfg.stun_time * (a_faster ? cfg.rammer_stun_mult : 1.0));
                            b.stun = std::max(b.stun, cfg.stun_time * (a_faster ? 1.0 : cfg.rammer_stun_mult));
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

    void merge(int ai, int bi) {
        Body a = bodies[ai], b = bodies[bi];
        double ma = a.n(), mb = b.n();
        Body m;
        m.pos = (a.pos * ma + b.pos * mb) * (1.0 / (ma + mb));
        m.vel = (a.vel * ma + b.vel * mb) * (1.0 / (ma + mb));
        m.stun = std::max(a.stun, b.stun);
        m.members = a.members;
        m.members.insert(m.members.end(), b.members.begin(), b.members.end());
        for (int t = 0; t < TYPES; t++) m.pool[t] = a.pool[t] + b.pool[t];
        bool fast = false;
        for (int i : a.members) for (int j : b.members) {
            if (pair_ended[i][j] >= 0 && t - pair_ended[i][j] < 3.0) fast = true;
            pair_since[i][j] = t; pair_since[j][i] = t;
        }
        for (int i : m.members) if (body_index(i) >= 0 && bodies[body_index(i)].n() == 1) players[i].group_since = t;
        std::vector<Body> nb;
        for (int k = 0; k < (int)bodies.size(); k++) if (k != ai && k != bi) nb.push_back(bodies[k]);
        nb.push_back(m);
        bodies = nb;
        stats.merges++;
        if (fast) stats.remerge_fast++;
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
                if (mine_alive[m] && mine_stock[m] > 0 && b.stun <= 0) {
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
                    double total = b.pool_total();
                    double mult = 1.0 + cfg.group_bank_bonus * (b.n() - 1);
                    for (int t = 0; t < TYPES; t++) { amount[t] = b.pool[t] * mult; players[i].banked[t] += amount[t]; b.pool[t] = 0; }
                    players[i].last_bank_t = t;
                    for (int j : b.members) if (j != i) banked_while[j][i] += total;
                    stats.banks++;
                    if (b.n() > 1) stats.group_banks++;
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
        int n = cfg.n_players;
        std::vector<double> bp(n), pots(n);
        for (int i = 0; i < n; i++) {
            const Body& b = bodies[body_index(i)];
            double bpi = 0, cp = 0;
            for (int t = 0; t < TYPES; t++) {
                bpi += std::min(players[i].banked[t] / players[i].need[t], 1.0);
                cp += std::min((players[i].banked[t] + b.pool[t] / b.n()) / players[i].need[t], 1.0);
            }
            bpi /= TYPES; cp /= TYPES;
            bp[i] = bpi;
            pots[i] = 10.0 * bpi + cfg.carried_shaping * (cp - bpi);
        }
        if (n > 1 && cfg.relative_reward > 0) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += bp[i];
            for (int i = 0; i < n; i++) pots[i] -= cfg.relative_reward * 10.0 * (sum - bp[i]) / (n - 1);
        }
        return pots;
    }

    // --------------------------------------------------------- observation

    static constexpr int OWN_DIM = 33 + N_DR;
    static constexpr int OTHER_DIM = 22;
    static constexpr int MINE_DIM = 8;
    static constexpr int PICK_DIM = 7;
    int obs_dim() const { return OWN_DIM + MAX_SLOTS * OTHER_DIM + cfg.n_mines * MINE_DIM + N_PICKUPS_OBS * PICK_DIM; }
    static constexpr int PRIV_SLOT = 17;
    int priv_dim() const { return MAX_SLOTS * PRIV_SLOT + MAX_SLOTS; }

    void dr_features(float* f) const {
        f[0] = cfg.mine_rate / 0.3;
        f[1] = cfg.mine_regen / 1.0;
        f[2] = cfg.leave_time / 4.0;
        f[3] = cfg.need_primary / 24.0;
    }

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
            f[c++] = std::min((t - p.group_since) / 60.0, 1.0);        // time in current membership state
            f[c++] = std::min((t - p.last_bank_t) / 60.0, 1.0);        // time since last bank
            f[c++] = p.join_cooldown / std::max(cfg.join_cooldown, 1e-6);
            f[c++] = b.stun / std::max(cfg.stun_time, 1e-6);
            dr_features(f + c); c += N_DR;
            // Others sorted by distance, stable on index; MAX_SLOTS slots with presence flag.
            std::vector<std::pair<double, int>> order;
            for (int j = 0; j < k; j++) if (j != i) order.push_back({(bodies[bidx[j]].pos - pos).norm(), j});
            std::stable_sort(order.begin(), order.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first < b.first; });
            int slot = 0;
            for (auto& oj : order) {
                if (slot >= MAX_SLOTS) break;
                int j = oj.second;
                const Player& q = players[j];
                const Body& bj = bodies[bidx[j]];
                f[c++] = bj.pos.x - pos.x; f[c++] = bj.pos.y - pos.y;
                f[c++] = bj.vel.x; f[c++] = bj.vel.y;
                f[c++] = (float)bj.n() / cfg.max_group;
                f[c++] = q.joinable ? 1.f : 0.f;
                for (int t = 0; t < TYPES; t++) f[c++] = (q.intent == t) ? 1.f : 0.f;
                f[c++] = (bidx[j] == bidx[i]) ? 1.f : 0.f;
                f[c++] = q.leave_timer >= 0 ? q.leave_timer / cfg.leave_time : 0.f;
                f[c++] = prog[j];
                f[c++] = pads[j].x - pos.x; f[c++] = pads[j].y - pos.y;
                f[c++] = bj.pool_total() / 40.0;
                f[c++] = 1.f;  // present
                f[c++] = std::min(comember_time[i][j] / 60.0, 1.0);
                f[c++] = std::min(took_from[i][j] / 10.0, 1.0);
                f[c++] = std::min(banked_while[i][j] / 10.0, 1.0);
                f[c++] = last_left_me[i][j] < 0 ? 1.f : (float)std::min((t - last_left_me[i][j]) / 60.0, 1.0);
                f[c++] = (partner_cd[i][j] > t || q.join_cooldown > 0) ? 1.f : 0.f;
                slot++;
            }
            for (; slot < MAX_SLOTS; slot++) for (int z = 0; z < OTHER_DIM; z++) f[c++] = 0.f;
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
                f[c++] = 1.f;
            }
            for (int q = cnt; q < N_PICKUPS_OBS; q++) for (int z = 0; z < PICK_DIM; z++) f[c++] = 0.f;
        }
    }

    // Privileged per-seat block for the critic: every seat in fixed index order (present, need,
    // banked/need, pool share, seat-type one-hot [external, snapshot, bot, unused]) + own index.
    void observe_priv(float* out) const {
        int k = cfg.n_players;
        int D = priv_dim();
        std::vector<int> bidx(k);
        for (int i = 0; i < k; i++) bidx[i] = body_index(i);
        for (int i = 0; i < k; i++) {
            float* f = out + i * D;
            int c = 0;
            for (int s = 0; s < MAX_SLOTS; s++) {
                if (s < k) {
                    const Player& q = players[s];
                    const Body& b = bodies[bidx[s]];
                    f[c++] = 1.f;
                    for (int t = 0; t < TYPES; t++) f[c++] = q.need[t] / cfg.need_primary;
                    for (int t = 0; t < TYPES; t++) f[c++] = std::min(q.banked[t] / q.need[t], 1.5);
                    for (int t = 0; t < TYPES; t++) f[c++] = b.pool[t] / b.n() / 20.0;
                    int st = seats[s] == SEAT_EXTERNAL ? 0 : seats[s] == SEAT_EXTERNAL2 ? 1 : 2;
                    for (int z = 0; z < 4; z++) f[c++] = (z == st) ? 1.f : 0.f;
                } else {
                    for (int z = 0; z < PRIV_SLOT; z++) f[c++] = 0.f;
                }
            }
            for (int s = 0; s < MAX_SLOTS; s++) f[c++] = (s == i) ? 1.f : 0.f;
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

    int primary_of(int i) const {
        int p = 0;
        for (int t = 1; t < TYPES; t++) if (players[i].need[t] > players[i].need[p]) p = t;
        return p;
    }

    int intent_for(int i) const {
        bool needed[TYPES], any;
        needed_types(i, needed, &any);
        if (any) { int k = 0; while (!needed[k]) k++; return k; }
        return primary_of(i);
    }

    Vec mine_target(int i, Vec pos, bool* found) const {
        bool needed[TYPES], any;
        needed_types(i, needed, &any);
        int m = nearest_mine(pos, any ? needed : nullptr);
        if (m < 0) m = nearest_mine(pos, nullptr);
        *found = m >= 0;
        return m >= 0 ? mine_pos[m] : Vec();
    }

    // Group consensus target: the nearest live mine of a type the member furthest behind still needs
    // (falls back to any needed type, then any mine). Every member computes the same answer.
    Vec group_mine_target(const Body& b, bool* found) const {
        int behind = b.members[0];
        for (int j : b.members) if (progress(j) < progress(behind)) behind = j;
        bool needed[TYPES], any;
        needed_types(behind, needed, &any);
        int m = nearest_mine(b.pos, any ? needed : nullptr);
        if (m < 0) {
            bool anyneed[TYPES] = {false, false, false, false}, anyany = false;
            for (int j : b.members) { bool nj[TYPES], a; needed_types(j, nj, &a); for (int t = 0; t < TYPES; t++) if (nj[t]) { anyneed[t] = true; anyany = true; } }
            m = nearest_mine(b.pos, anyany ? anyneed : nullptr);
        }
        if (m < 0) m = nearest_mine(b.pos, nullptr);
        *found = m >= 0;
        return m >= 0 ? mine_pos[m] : Vec();
    }

    int nearest_joinable_body(int i, Vec pos, double maxd) const {
        int best = -1;
        double bd = maxd;
        for (int k = 0; k < (int)bodies.size(); k++) {
            const Body& b = bodies[k];
            if (std::find(b.members.begin(), b.members.end(), i) != b.members.end()) continue;
            if (!all_joinable(b)) continue;
            double d = (b.pos - pos).norm();
            if (d < bd) { best = k; bd = d; }
        }
        return best;
    }

    void bot_solo(int i, int* act) const {
        const Body& b = bodies[body_index(i)];
        double bank_at = 0.22 * need_total(i);
        bool has = false;
        Vec target;
        if (b.pool_total() >= bank_at) { target = pad_pos(i); has = true; }
        else {
            target = mine_target(i, b.pos, &has);
            if (!has && b.pool_total() > 0) { target = pad_pos(i); has = true; }
        }
        act[0] = has ? direction_to_move(target - b.pos) : 0;
        act[1] = 0;
        act[2] = b.n() > 1 ? 1 : 0;
        act[3] = primary_of(i) + 1;
    }

    void bot_bail(int i, int* act) const {
        const int max_group_bot = 3;
        const double pad_danger = 0.3;
        const Body& b = bodies[body_index(i)];
        double bank_at = 0.28 * need_total(i), bail_at = 0.11 * need_total(i);
        int joinable = b.n() < max_group_bot ? 1 : 0;
        int leave = 0;
        bool has = false;
        Vec target;
        Vec my_pad = pad_pos(i);
        if (b.n() > 1) {
            double my_share = b.pool_total() / b.n();
            for (int j : b.members) {
                if (j == i) continue;
                if ((pad_pos(j) - b.pos).norm() < pad_danger && b.pool_total() >= bail_at) leave = 1;
            }
            if (b.pool_total() >= bank_at) { target = my_pad; has = true; }
            if (my_share >= bail_at && (my_pad - b.pos).norm() > 0.6 && leave == 0) leave = 1;
        } else {
            if (b.pool_total() >= bail_at) { target = my_pad; has = true; }
        }
        if (!has) {
            if (b.n() > 1) target = group_mine_target(b, &has); else target = mine_target(i, b.pos, &has);
            if (!has) target = my_pad;
        }
        act[0] = direction_to_move(target - b.pos);
        act[1] = joinable;
        act[2] = leave;
        act[3] = intent_for(i) + 1;
    }

    // Loyal: always joinable, never leaves, banks the group at whichever member pad is nearest.
    void bot_loyal(int i, int* act) const {
        const Body& b = bodies[body_index(i)];
        double bank_at = 0.28 * need_total(i);
        bool has = false;
        Vec target;
        if (b.n() > 1 && b.pool_total() >= bank_at) {
            // Fair rotation: bank at the pad of the member who is furthest behind.
            double best = 2.0;
            for (int j : b.members) { double pj = progress(j); if (pj < best) { best = pj; target = pad_pos(j); has = true; } }
        } else if (b.n() == 1) {
            if (b.pool_total() >= bank_at) { target = pad_pos(i); has = true; }
            else {
                int k = nearest_joinable_body(i, b.pos, 0.6);
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

    // Kidnapper: always joinable, never leaves, drags any laden group to its own pad.
    void bot_kidnap(int i, int* act) const {
        const Body& b = bodies[body_index(i)];
        bool has = false;
        Vec target;
        if (b.n() > 1 && b.pool_total() >= 6.0) { target = pad_pos(i); has = true; }
        else if (b.n() == 1) {
            if (b.pool_total() >= 6.0) { target = pad_pos(i); has = true; }
            else {
                int k = nearest_joinable_body(i, b.pos, 0.8);
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

    // Rammer: never joins, hunts the fullest body, collects the spill, banks small loads.
    void bot_rammer(int i, int* act) const {
        const Body& b = bodies[body_index(i)];
        bool has = false;
        Vec target;
        if (b.pool_total() >= 4.0) { target = pad_pos(i); has = true; }
        if (!has) {
            double bd = 0.3;
            for (const Pickup& pk : picks) { double d = (pk.pos - b.pos).norm(); if (d < bd) { bd = d; target = pk.pos; has = true; } }
        }
        if (!has) {
            double best = 2.0;
            for (const Body& o : bodies) {
                if (&o == &b) continue;
                if (o.pool_total() > best) { best = o.pool_total(); target = o.pos + o.vel * 0.3; has = true; }
            }
        }
        if (!has) { target = mine_target(i, b.pos, &has); if (!has) target = pad_pos(i); }
        act[0] = direction_to_move(target - b.pos);
        act[1] = 0;
        act[2] = b.n() > 1 ? 1 : 0;
        act[3] = primary_of(i) + 1;
    }

    void bot_action(int seat_type, int i, int* act) const {
        switch (seat_type) {
            case SEAT_SOLO: bot_solo(i, act); break;
            case SEAT_BAIL: bot_bail(i, act); break;
            case SEAT_LOYAL: bot_loyal(i, act); break;
            case SEAT_KIDNAP: bot_kidnap(i, act); break;
            case SEAT_RAMMER: bot_rammer(i, act); break;
            default: act[0] = 0; act[1] = 0; act[2] = 0; act[3] = 0;
        }
    }

    // ------------------------------------------------------------ recording

    std::string frame_json() const {
        std::string s;
        char buf[160];
        snprintf(buf, sizeof buf, "{\"t\":%.3f,\"R\":%.3f,\"bodies\":[", t, R);
        s += buf;
        for (size_t k = 0; k < bodies.size(); k++) {
            const Body& b = bodies[k];
            if (k) s += ",";
            snprintf(buf, sizeof buf, ",\"x\":%.3f,\"y\":%.3f,\"vx\":%.3f,\"vy\":%.3f,\"stun\":%.1f,\"pool\":", b.pos.x, b.pos.y, b.vel.x, b.vel.y, b.stun);
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
            snprintf(buf, sizeof buf, ",\"intent\":%d,\"join\":%s,\"leaving\":%.1f,\"cd\":%.1f,\"dir\":[%.2f,%.2f]}", p.intent,
                     p.joinable ? "true" : "false", p.leave_timer >= 0 ? p.leave_timer : -1.0, p.join_cooldown, p.dir.x, p.dir.y);
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
        s += "],\"mine_type\":" + ids(mine_type) + ",\"winner\":" + std::to_string(winner) +
             ",\"timeout_win\":" + (timeout_win ? "true" : "false") + ",\"cfg\":[";
        double arr[CFG_LEN];
        cfg_dump(cfg, arr);
        for (int k = 0; k < CFG_LEN; k++) { if (k) s += ","; snprintf(buf, sizeof buf, "%.6g", arr[k]); s += buf; }
        s += "]}";
        return s;
    }
};

struct Batch {
    std::vector<Game> games;
    int decide_every = 6;
    uint64_t next_seed = 1;
};

constexpr int STATS_BASE = 13;

}  // namespace

extern "C" {

int ugb_cfg_len() { return CFG_LEN; }
int ugb_stats_base() { return STATS_BASE; }

void* ugb_create(int n_envs, const double* cfg, int cfg_len, unsigned long long seed, int decide_every) {
    Batch* b = new Batch();
    b->decide_every = decide_every;
    b->next_seed = seed;
    b->games.resize(n_envs);
    for (int e = 0; e < n_envs; e++) {
        Game& g = b->games[e];
        cfg_fill(g.base_cfg, cfg, cfg_len);
        g.lo = g.hi = g.base_cfg;
        if (g.base_cfg.n_players > MAX_PLAYERS) g.base_cfg.n_players = MAX_PLAYERS;
        g.seats.assign(g.base_cfg.n_players, SEAT_EXTERNAL);
        g.reset(b->next_seed++);
    }
    return b;
}

void ugb_destroy(void* h) { delete (Batch*)h; }
int ugb_obs_dim(void* h) { return ((Batch*)h)->games[0].obs_dim(); }
int ugb_priv_dim(void* h) { return ((Batch*)h)->games[0].priv_dim(); }
int ugb_n_players(void* h) { return ((Batch*)h)->games[0].cfg.n_players; }

// Per-game randomisation ranges (same layout as the config array). Applies at the next reset.
void ugb_set_cfg_range(void* h, const double* lo, const double* hi, int len) {
    for (Game& g : ((Batch*)h)->games) {
        cfg_fill(g.lo, lo, len);
        cfg_fill(g.hi, hi, len);
        g.randomize = true;
    }
}

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

void ugb_observe_priv(void* h, float* priv) {
    Batch* b = (Batch*)h;
    int D = b->games[0].priv_dim(), n = b->games[0].cfg.n_players;
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < (int)b->games.size(); e++) b->games[e].observe_priv(priv + (size_t)e * n * D);
}

// What a scripted bot of the given type would do for every seat of a game (DAgger labels).
void ugb_bot_actions(void* h, int env, int seat_type, int* out) {
    Game& g = ((Batch*)h)->games[env];
    for (int i = 0; i < g.cfg.n_players; i++) g.bot_action(seat_type, i, out + i * 4);
}

void ugb_set_direction(void* h, int env, int seat, double dx, double dy) {
    ((Batch*)h)->games[env].set_direction(seat, dx, dy);
}

void ugb_cfg(void* h, int env, double* out) { cfg_dump(((Batch*)h)->games[env].cfg, out); }

// Steps every game by decide_every ticks. actions: (E, n, 4) int32 for external seats (bot seats
// are overridden). Writes rewards (E, n), dones (E), next obs (E, n, D) and, if priv != null, the
// privileged block (E, n, P). On episode end with auto_reset, the game is reset (fresh seed) before
// its observation is written and its summary goes to ep_stats (E, STATS_BASE + n + N_DR):
// [winner, length, avg_group, merges, leaves, spills, banks, ended, group_banks, remerge_fast,
//  alliances, mean_alliance_dur, alliances_long, progress..., dr constants...].
void ugb_step(void* h, const int* actions, float* rewards, unsigned char* dones, float* obs, float* priv, int auto_reset,
              double* ep_stats, int decide_every_override) {
    Batch* b = (Batch*)h;
    int E = (int)b->games.size();
    int n = b->games[0].cfg.n_players;
    int D = b->games[0].obs_dim();
    int P = b->games[0].priv_dim();
    int de = decide_every_override > 0 ? decide_every_override : b->decide_every;
    std::vector<uint64_t> seeds(E);
    for (int e = 0; e < E; e++) seeds[e] = b->next_seed++;
    int SW = STATS_BASE + n + N_DR;
    #pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < E; e++) {
        Game& g = b->games[e];
        std::vector<int> act(actions + (size_t)e * n * 4, actions + (size_t)(e + 1) * n * 4);
        for (int i = 0; i < n; i++)
            if (g.seats[i] >= SEAT_SOLO) g.bot_action(g.seats[i], i, &act[i * 4]);
        g.events.clear();
        g.apply_actions(act.data());
        for (int k = 0; k < de; k++) {
            g.tick();
            if (g.done) break;
        }
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
        double* st = ep_stats + (size_t)e * SW;
        st[7] = 0;
        if (g.done) {
            // close open alliances for the duration statistics
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j && g.pair_since[i][j] >= 0) g.end_pair(i, j);
            st[0] = g.winner; st[1] = g.t; st[2] = g.stats.group / std::max(1, g.stats.steps);
            st[3] = g.stats.merges; st[4] = g.stats.leaves; st[5] = g.stats.spills; st[6] = g.stats.banks; st[7] = g.timeout_win ? 2 : 1;
            st[8] = g.stats.group_banks; st[9] = g.stats.remerge_fast; st[10] = g.stats.alliances / 2.0;
            st[11] = g.stats.alliances ? g.stats.alliance_dur / g.stats.alliances : 0; st[12] = g.stats.alliances_long / 2.0;
            for (int i = 0; i < n; i++) st[STATS_BASE + i] = g.progress(i);
            float dr[N_DR];
            g.dr_features(dr);
            for (int k = 0; k < N_DR; k++) st[STATS_BASE + n + k] = dr[k];
            if (auto_reset) g.reset(seeds[e]);
        }
        g.observe(obs + (size_t)e * n * D);
        if (priv) g.observe_priv(priv + (size_t)e * n * P);
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
