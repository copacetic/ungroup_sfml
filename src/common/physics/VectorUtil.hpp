#ifndef VectorUtil_hpp
#define VectorUtil_hpp

#include <SFML/Graphics.hpp>
#include <memory>
#include <stdio.h>

namespace VectorUtil {
    sf::Vector2f normalize(const sf::Vector2f& source);
    float distance(const sf::Vector2f& point_a, const sf::Vector2f& point_b);
    sf::Vector2f getVector(const sf::Vector2f& point_a, const sf::Vector2f& point_b);
    float length(const sf::Vector2f& source);
    sf::Vector2f getMidpoint(const sf::Vector2f& point_a, const sf::Vector2f& point_b);
    sf::Vector2f lerp(const sf::Vector2f& a, const sf::Vector2f& b, float t);
    void print(const sf::Vector2f& v);
    float dot(const sf::Vector2f& a, const sf::Vector2f& b);
    void clamp(sf::Vector2f& a, float min, float max);
} // namespace VectorUtil

#endif /* VectorUtil_hpp */