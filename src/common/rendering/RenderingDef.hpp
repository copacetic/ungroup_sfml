#ifndef RenderingDef_hpp
#define RenderingDef_hpp

#include <SFML/Graphics.hpp>
#include <memory>

#include "../systems/ResourceController.hpp"

namespace RenderingDef {
    const bool USE_SHADERS = true;
    const std::size_t CIRCLE_POINT_COUNT = 60;
    enum TextureKey { collision, mine_pattern };
    enum ShaderKey { none, noop, voronoi };
    enum FontKey { monogram };
    struct Shader {
        ShaderKey key = ShaderKey::none;
        std::shared_ptr<sf::Shader> shader = nullptr;
    };
    const sf::Color JOINABLE_COLOR(227, 102, 68);
    const sf::Color DARKEST_COLOR(66, 118, 118);
    const sf::Color DARK_COLOR(63, 157, 130);
    const sf::Color LIGHT_COLOR(161, 205, 115);
    const sf::Color LIGHTEST_COLOR(236, 219, 96);

    const std::array<sf::Color, RESOURCE_TYPE_COUNT> MINE_COLORS = {
        RenderingDef::DARKEST_COLOR, RenderingDef::DARK_COLOR, RenderingDef::LIGHT_COLOR,
        RenderingDef::LIGHTEST_COLOR};
}; // namespace RenderingDef

#endif /* RenderingDef_hpp */