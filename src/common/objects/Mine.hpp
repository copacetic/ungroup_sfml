#ifndef Mine_hpp
#define Mine_hpp

#include <memory>
#include <stdio.h>
#include <vector>

#include <SFML/Graphics.hpp>
#include <SFML/Network.hpp>

#include "../resources/ResourceStore.hpp"
#include "../systems/ResourceController.hpp"
#include "CircleGameObject.hpp"

struct MineUpdate {
    sf::Uint32 mine_id;
    bool is_active;
    float x_pos;
    float y_pos;
    float radius;
    sf::Uint32 shader_key;
};

sf::Packet& operator<<(sf::Packet& packet, const MineUpdate& mine_update);
sf::Packet& operator>>(sf::Packet& packet, MineUpdate& mine_update);

class Mine : public CircleGameObject {
  public:
    Mine(uint32_t id, sf::Vector2f position, float size, sf::Color color,
         ResourceType resource_type, PhysicsController& pc, ResourceStore& rs);
    ~Mine();
    Mine(const Mine& temp_obj) = delete;            // TODO: define this
    Mine& operator=(const Mine& temp_obj) = delete; // TODO: define this

    void setResourceCount(uint32_t count) {
        m_resourceCount = count;
    }

    ResourceType getResourceType() const {
        return m_resourceType;
    }

    MineUpdate getUpdate();
    void applyUpdate(MineUpdate mu);

    void draw(sf::RenderTarget& render_target);

  private:
    ResourceType m_resourceType;
    uint32_t m_resourceCount = 0;
};

#endif /* Mine_hpp */
