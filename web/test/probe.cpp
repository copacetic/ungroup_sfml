// Full-precision per-tick trace of the canonical core for tick-level diffing against engine.js.
// Build: g++ -O2 -std=c++17 -ffp-contract=off -o /tmp/ungroup_probe web/test/probe.cpp   (driven by web/test/tickdiff.mjs)
// usage: probe <cfg-array-file> <seat types comma> <seed> <max_ticks> <decide_every> <rounds> <flags>
// flags: bit0 = script external seats (macro cycle / move 9 with set_direction / join / held leave / intent)
//        bit1 = dump observations at decision ticks, bit2 = dump frame_json at every tick, bit3 = macro labels for bot seats
#include "../../rl/native/ungroup.cpp"
#include <fstream>
#include <sstream>
#include <iostream>
static std::string g(double x) { char b[40]; snprintf(b, sizeof b, "%.17g", x); return b; }
int main(int argc, char** argv) {
    std::ifstream f(argv[1]); std::string s((std::istreambuf_iterator<char>(f)), {});
    std::vector<double> arr; { std::stringstream ss(s); double v; char c; while (ss >> v) { arr.push_back(v); ss >> c; } }
    std::string seatsArg = argv[2]; unsigned long long seed = strtoull(argv[3], 0, 10); int maxTicks = atoi(argv[4]);
    int de = atoi(argv[5]); int rounds = atoi(argv[6]); int flags = atoi(argv[7]);
    Game gm; cfg_fill(gm.base_cfg, arr.data(), (int)arr.size()); gm.lo = gm.hi = gm.base_cfg;
    if (gm.base_cfg.n_players > MAX_PLAYERS) gm.base_cfg.n_players = MAX_PLAYERS;
    std::vector<int> seats; { std::stringstream ss(seatsArg); std::string tok; while (std::getline(ss, tok, ',')) seats.push_back(atoi(tok.c_str())); }
    gm.seats = seats;
    int n = (int)seats.size();
    std::vector<int> act(n * 4, 0);
    std::vector<float> obs;
    printf("[");
    bool firstOut = true;
    auto dump = [&](int round, int k) {
        if (!firstOut) printf(","); firstOut = false;
        printf("{\"round\":%d,\"k\":%d,\"t\":%s,\"R\":%s,\"bodies\":[", round, k, g(gm.t).c_str(), g(gm.R).c_str());
        for (size_t b = 0; b < gm.bodies.size(); b++) {
            const Body& B = gm.bodies[b];
            printf("%s{\"m\":%s,\"x\":%s,\"y\":%s,\"vx\":%s,\"vy\":%s,\"stun\":%s,\"head\":%d,\"pool\":[%s,%s,%s,%s]}", b ? "," : "", Game::ids(B.members).c_str(),
                   g(B.pos.x).c_str(), g(B.pos.y).c_str(), g(B.vel.x).c_str(), g(B.vel.y).c_str(), g(B.stun).c_str(), B.head,
                   g(B.pool[0]).c_str(), g(B.pool[1]).c_str(), g(B.pool[2]).c_str(), g(B.pool[3]).c_str());
        }
        printf("],\"players\":[");
        for (int i = 0; i < n; i++) {
            const Player& p = gm.players[i];
            printf("%s{\"banked\":[%s,%s,%s,%s],\"intent\":%d,\"join\":%d,\"lt\":%s,\"cd\":%s,\"dx\":%s,\"dy\":%s,\"macro\":%d,\"brand\":%s,\"gs\":%s,\"lb\":%s,\"ul\":%s,\"pad\":%s,\"padx\":%s,\"pady\":%s}", i ? "," : "",
                   g(p.banked[0]).c_str(), g(p.banked[1]).c_str(), g(p.banked[2]).c_str(), g(p.banked[3]).c_str(), p.intent, p.joinable ? 1 : 0,
                   g(p.leave_timer).c_str(), g(p.join_cooldown).c_str(), g(p.dir.x).c_str(), g(p.dir.y).c_str(), p.macro, g(gm.brand[i]).c_str(), g(p.group_since).c_str(),
                   g(p.last_bank_t).c_str(), g(gm.last_unjust_leave[i]).c_str(), g(p.pad_angle).c_str(), g(gm.pad_pos(i).x).c_str(), g(gm.pad_pos(i).y).c_str());
        }
        printf("],\"pair\":[");
        for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) printf("%s[%s,%s,%s,%s,%s,%s,%s]", (i || j) ? "," : "", g(gm.comember_time[i][j]).c_str(), g(gm.took_from[i][j]).c_str(), g(gm.banked_while[i][j]).c_str(), g(gm.last_left_me[i][j]).c_str(), g(gm.partner_cd[i][j]).c_str(), g(gm.pair_since[i][j]).c_str(), g(gm.pair_ended[i][j]).c_str());
        printf("],\"mines\":[");
        for (int m = 0; m < gm.cfg.n_mines; m++) printf("%s%s", m ? "," : "", g(gm.mine_stock[m]).c_str());
        printf("],\"minepos\":[");
        for (int m = 0; m < gm.cfg.n_mines; m++) printf("%s[%s,%s]", m ? "," : "", g(gm.mine_pos[m].x).c_str(), g(gm.mine_pos[m].y).c_str());
        printf("],\"alive\":[");
        for (int m = 0; m < gm.cfg.n_mines; m++) printf("%s%d", m ? "," : "", gm.mine_alive[m] ? 1 : 0);
        printf("],\"picks\":[");
        for (size_t q = 0; q < gm.picks.size(); q++) printf("%s[%s,%s,%d,%s]", q ? "," : "", g(gm.picks[q].pos.x).c_str(), g(gm.picks[q].pos.y).c_str(), gm.picks[q].type, g(gm.picks[q].ttl).c_str());
        printf("],\"events\":[%s],\"act\":[", gm.events.c_str());
        for (int i = 0; i < n * 4; i++) printf("%s%d", i ? "," : "", act[i]);
        printf("],\"stats\":[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s]", gm.stats.merges, gm.stats.leaves, gm.stats.spills, gm.stats.banks, gm.stats.group_banks, gm.stats.remerge_fast, gm.stats.cancels, gm.stats.alliances, gm.stats.alliances_long, gm.stats.fair_banks, gm.stats.crowns, g(gm.stats.alliance_dur).c_str(), g(gm.stats.units_taken).c_str());
        if ((flags & 2) && k % de == 0) {
            obs.assign((size_t)n * gm.obs_dim(), 0.f); gm.observe(obs.data());
            printf(",\"obs\":[");
            for (size_t q = 0; q < obs.size(); q++) printf("%s%.9g", q ? "," : "", obs[q]);
            printf("]");
        }
        if (flags & 4) printf(",\"frame\":%s,\"meta\":%s", gm.frame_json().c_str(), gm.meta_json().c_str());
        if ((flags & 8) && k % de == 0) {
            printf(",\"labels\":[");
            for (int i = 0; i < n; i++) { int tmp[4]; int lab = 0; if (gm.seats[i] >= SEAT_SOLO) { gm.bot_action(gm.seats[i], i, tmp); lab = gm.macro_label(i); } printf("%s%d", i ? "," : "", lab); }
            printf("]");
        }
        printf(",\"done\":%d,\"winner\":%d,\"tw\":%d}\n", gm.done ? 1 : 0, gm.winner, gm.timeout_win ? 1 : 0);
    };
    for (int round = 0; round < rounds; round++) {
        gm.reset(seed + round);
        dump(round, 0);
        for (int k = 0; k < maxTicks && !gm.done; k++) {
            if (k % de == 0) {
                int step = k / de;
                for (int i = 0; i < n; i++) {
                    if (gm.seats[i] >= SEAT_SOLO) gm.bot_action(gm.seats[i], i, &act[i * 4]);
                    else if (flags & 1) {
                        int phase = (step + i) % 20;
                        int move;
                        if (phase < 14) move = MACRO_BASE + phase;
                        else if (phase < 18) { move = 9; gm.set_direction(i, (double)((step * 7 + i * 3) % 11 - 5), (double)((step * 5 + i) % 7 - 3)); }
                        else if (phase == 18) move = 1 + (step % 8);
                        else move = 0;
                        act[i * 4] = move; act[i * 4 + 1] = ((step + i) >> 2) & 1; act[i * 4 + 2] = (step + 7 * i) % 40 < 8 ? 1 : 0; act[i * 4 + 3] = ((step + i) % 5 == 0) ? (step % 4) + 1 : 0;
                    }
                }
                gm.events.clear();
                gm.apply_actions(act.data());
            }
            gm.tick();
            if (gm.done) { for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j && gm.pair_since[i][j] >= 0) gm.end_pair(i, j); }  // ugb_step closes open alliances
            dump(round, k + 1);
        }
    }
    printf("]\n");
    return 0;
}
