#ifndef network_util_hpp
#define network_util_hpp

#include <SFML/Network.hpp>
#include "game_def.hpp"

sf::Packet& operator <<(sf::Packet& packet, const RealtimeCommand& realtime_command);
sf::Packet& operator >>(sf::Packet& packet, RealtimeCommand& realtime_command);

sf::Packet& operator <<(sf::Packet& packet, const ApiCommand& api_command);
sf::Packet& operator >>(sf::Packet& packet, ApiCommand& api_command);

sf::Packet& operator <<(sf::Packet& packet, const sf::Vector2f& vector_2f);
sf::Packet& operator >>(sf::Packet& packet, sf::Vector2f& vector_2f);

sf::Packet& operator <<(sf::Packet& packet, const ClientGroupUpdate& client_group_update);
sf::Packet& operator >>(sf::Packet& packet, ClientGroupUpdate& client_group_update);

#endif /* network_util_hpp */
