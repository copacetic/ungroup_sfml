#ifndef Group_hpp
#define Group_hpp

#include <stdio.h>
#include <vector>
#include <memory>

#include <SFML/Graphics.hpp>
#include "Player.hpp"
#include "../common/Circle.hpp"
#include "../common/CircleGameObject.hpp"

class Group : public CircleGameObject {
 public:
     Group(int id, sf::Vector2f position, std::shared_ptr<CircleRigidBody> crb);
     ~Group();
     Group(const Group& temp_obj) = delete;  // TODO: define this
     Group& operator=(const Group& temp_obj) = delete;  // TODO: define this

     void update();
     void addMember(std::shared_ptr<Player> player);
     static std::vector<std::shared_ptr<Group>> getActiveGroups(std::vector<std::shared_ptr<Group>>& groups);

 private:
     std::vector<std::shared_ptr<Player>> mMembers;
     void refresh();
};

#endif /* Group_hpp */
