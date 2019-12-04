#ifndef game_settings_hpp
#define game_settings_hpp

#include <chrono>

#include <SFML/Graphics.hpp>

#include "InputDef.hpp"

/* Input */

const InputDef::InputKeys INPUT_KEYS = {
    .up = sf::Keyboard::Up,
    .down = sf::Keyboard::Down,
    .right = sf::Keyboard::Right,
    .left = sf::Keyboard::Left,
    .joinable = sf::Keyboard::G,
    .ungroup = sf::Keyboard::Escape,
    .stop = sf::Keyboard::Space,
};

/* Game Logic */
const sf::Vector2f GAME_SIZE(2688 * 4, 1512 * 4);
const sf::Vector2f WINDOW_RESOLUTION(2688, 1512);
const float GAME_SCALE = 1.f;
const sf::Color BACKGROUND_COLOR(sf::Color::Black);

const bool USE_SHADER = true;
const bool USE_INTERPOLATION_REPLAY = false;

const int MAX_PLAYER_COUNT = 10;
const int MAX_MINE_COUNT = 4;
const float GROUP_START_OFFSET_X = 20.f;
const float GROUP_START_OFFSET_Y = 20.f;
const float MINE_START_OFFSET_X = 50.f;
const float MINE_START_OFFSET_Y = 200.f;
const float GROUP_MEMBER_SIZE = 10.f;
const float MINE_SIZE = 80.f;
const float GROUP_SPEED = 100.f;
const uint32_t MINE_RESOURCE_COUNT = 4;

/* Threads */

const std::chrono::milliseconds CLIENT_FETCH_PLAYER_ID_SLEEP(16);

const std::chrono::milliseconds SERVER_INPUT_WINDOW_SLEEP(50);

const std::chrono::milliseconds CLIENT_RELIABLE_SEND_SLEEP(16);
const std::chrono::milliseconds
    SERVER_RELIABLE_REVC_SEND_SLEEP((CLIENT_RELIABLE_SEND_SLEEP / MAX_PLAYER_COUNT) / 2);
const std::chrono::milliseconds CLIENT_RELIBALE_RECV_SLEEP(SERVER_RELIABLE_REVC_SEND_SLEEP / 2);

const std::chrono::milliseconds CLIENT_UNRELIABLE_SEND_SLEEP(16);
const std::chrono::milliseconds
    SERVER_UNRELIABLE_RECV_SLEEP((CLIENT_UNRELIABLE_SEND_SLEEP / MAX_PLAYER_COUNT) / 2);

const std::chrono::milliseconds SERVER_BROADCAST_GAME_STATE_SLEEP(16);
const std::chrono::milliseconds CLIENT_UNRELIABLE_RECV_SLEEP(SERVER_BROADCAST_GAME_STATE_SLEEP / 2);

/* Networking */
static const char* CLIENT_IP = "127.0.0.1";
static const unsigned short SERVER_TCP_PORT = 4844;
static const char* SERVER_IP = "127.0.0.1";

#endif /* game_settings_hpp */
