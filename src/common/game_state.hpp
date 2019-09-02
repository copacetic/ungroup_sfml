#ifndef game_state_hpp
#define game_state_hpp

#include <stdio.h>
#include <vector>

#include <SFML/Network.hpp>

#include "../common/Group.hpp"
#include "../common/Mine.hpp"
#include "../common/Player.hpp"


struct GameState {
    sf::Uint32 tick;
    std::vector<GroupUpdate> group_updates;
    std::vector<MineUpdate> mine_updates;
};

sf::Packet pack_game_state(GameState game_state);
GameState unpack_game_state(sf::Packet game_state_packet);

struct ClientUpdate {
    PlayerUpdate player_update;
};

sf::Packet& operator <<(sf::Packet& packet, const ClientUpdate& client_update);
sf::Packet& operator >>(sf::Packet& packet, ClientUpdate& client_update);

#endif /* game_state_hpp */
