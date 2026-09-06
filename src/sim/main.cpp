/**
 * Offline simulator: runs bot-vs-bot games of Ungroup with no networking and no window.
 * Useful for balance analysis, regression testing of game rules, and as the seed of an RL
 * environment (the same loop can be driven by an external policy instead of Bot).
 *
 * Usage: ug-sim [--games N] [--max-ticks T] [--bot-period K] [--verbose] <strategy>...
 *   strategy: 0=Random 1=NearestGreedy 2=Groupie 3=NearestGreedy+always joinable
 */
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../common/bots/Bot.hpp"
#include "../common/events/EventController.hpp"
#include "../common/factories/IdFactory.hpp"
#include "../common/physics/PhysicsController.hpp"
#include "../common/systems/GameObjectController.hpp"
#include "../common/systems/GameObjectStore.hpp"
#include "../common/util/game_settings.hpp"

struct GameResult {
    bool finished = false;
    uint32_t winner_index = 0;
    uint32_t ticks = 0;
    double avg_group_size = 0;
    uint32_t join_events = 0;
    std::vector<uint32_t> total_resources;
};

static GameResult runGame(const std::vector<int>& strategies, uint32_t max_ticks,
                          uint32_t bot_period, bool verbose) {
    IdFactory::getInstance().reset();
    EventController::getInstance().reset();

    PhysicsController pc;
    GameObjectStore gos(pc);
    GameObjectController goc(gos, LevelKey::mine_ring);
    Bot bot;

    std::vector<uint32_t> player_ids;
    for (size_t i = 0; i < strategies.size(); i++) {
        player_ids.push_back(goc.createPlayerWithGroup(static_cast<uint32_t>(i)));
    }
    // Spread starting positions slightly so players don't all overlap at the center.
    for (size_t i = 0; i < player_ids.size(); i++) {
        auto& group = goc.getGroupController().getGroup(
            goc.getGroupController().getGroupId(player_ids[i]));
        float angle = 360.f * i / player_ids.size();
        group.setPosition(GAME_SETTINGS.GAME_CENTER + VectorUtil::direction(angle) * 60.f);
    }

    const sf::Int32 dt = static_cast<sf::Int32>(GAME_SETTINGS.MIN_TIME_STEP_SEC * 1000.f);
    GameResult result;
    double group_size_accum = 0;
    size_t last_active_groups = player_ids.size();

    for (uint32_t tick = 0; tick < max_ticks; tick++) {
        InputDef::PlayerInputs pi;
        if (tick % bot_period == 0) {
            for (size_t i = 0; i < player_ids.size(); i++) {
                int s = strategies[i];
                BotStrategy bs = s == 3 ? BotStrategy::NearestGreedy : static_cast<BotStrategy>(s);
                auto move = bot.getMove(player_ids[i], bs, goc);
                if (s == 3 && tick == 0) {
                    move.first.toggle_joinable = true;
                }
                if (!move.first.allFalse()) {
                    pi.player_reliable_inputs.push_back({player_ids[i], move.first});
                }
                if (!move.second.allFalse()) {
                    pi.player_unreliable_inputs.push_back({player_ids[i], move.second});
                }
            }
        }
        goc.update(pi);
        pc.update(dt);
        EventController::getInstance().forceProcessEvents();

        // Metrics: number of non-empty groups
        size_t active_groups = 0;
        for (auto gid : goc.getGroupController().getGroupIds()) {
            if (!goc.getGroupController().getGroupPlayerIds(gid).empty()) {
                active_groups++;
            }
        }
        if (active_groups < last_active_groups) {
            result.join_events += static_cast<uint32_t>(last_active_groups - active_groups);
        }
        last_active_groups = active_groups;
        group_size_accum += static_cast<double>(player_ids.size()) / active_groups;

        bool over;
        uint32_t winner;
        std::tie(over, winner) = goc.getGameOver();
        if (over) {
            result.finished = true;
            for (size_t i = 0; i < player_ids.size(); i++) {
                if (player_ids[i] == winner) {
                    result.winner_index = static_cast<uint32_t>(i);
                }
            }
            result.ticks = tick + 1;
            break;
        }
        result.ticks = tick + 1;
    }
    result.avg_group_size = group_size_accum / result.ticks;
    for (auto pid : player_ids) {
        auto r = goc.getPlayerResources(pid);
        uint32_t total = 0;
        for (auto c : r) {
            total += c;
        }
        result.total_resources.push_back(total);
    }
    if (verbose) {
        for (size_t i = 0; i < player_ids.size(); i++) {
            auto r = goc.getPlayerResources(player_ids[i]);
            auto need = goc.getPlayerController().getPlayer(player_ids[i])->getResourceCountsToWin();
            std::cout << "  player " << i << " strat " << strategies[i] << " has ";
            for (size_t k = 0; k < RESOURCE_TYPE_COUNT; k++) {
                std::cout << r[k] << "/" << need[k] << " ";
            }
            std::cout << std::endl;
        }
    }
    return result;
}

int main(int argc, char** argv) {
    uint32_t games = 1, max_ticks = 125 * 300, bot_period = 6;
    bool verbose = false;
    std::vector<int> strategies;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--games")) {
            games = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-ticks")) {
            max_ticks = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--bot-period")) {
            bot_period = std::atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--verbose")) {
            verbose = true;
        } else {
            strategies.push_back(std::atoi(argv[i]));
        }
    }
    if (strategies.empty()) {
        std::cerr << "Pass at least one strategy (0 random, 1 greedy, 2 groupie, 3 greedy+joinable)"
                  << std::endl;
        return EXIT_FAILURE;
    }

    std::map<uint32_t, uint32_t> wins;
    uint32_t finished = 0;
    double ticks_sum = 0, group_size_sum = 0, joins_sum = 0;
    for (uint32_t g = 0; g < games; g++) {
        GameResult r = runGame(strategies, max_ticks, bot_period, verbose);
        std::cout << "game " << g << ": " << (r.finished ? "winner=" + std::to_string(r.winner_index)
                                                         : std::string("no winner"))
                  << " ticks=" << r.ticks << " (" << r.ticks * GAME_SETTINGS.MIN_TIME_STEP_SEC
                  << "s) avg_group_size=" << r.avg_group_size << " joins=" << r.join_events
                  << " totals=";
        for (auto t : r.total_resources) {
            std::cout << t << " ";
        }
        std::cout << std::endl;
        if (r.finished) {
            finished++;
            wins[r.winner_index]++;
            ticks_sum += r.ticks;
        }
        group_size_sum += r.avg_group_size;
        joins_sum += r.join_events;
    }
    std::cout << "summary: finished " << finished << "/" << games;
    if (finished) {
        std::cout << " avg_time=" << (ticks_sum / finished) * GAME_SETTINGS.MIN_TIME_STEP_SEC
                  << "s";
    }
    std::cout << " avg_group_size=" << group_size_sum / games << " avg_joins=" << joins_sum / games
              << " wins:";
    for (size_t i = 0; i < strategies.size(); i++) {
        std::cout << " p" << i << "(s" << strategies[i] << ")=" << wins[i];
    }
    std::cout << std::endl;
    return EXIT_SUCCESS;
}
