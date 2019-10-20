#include <iostream>
#include <random>
#include <stdio.h>
#include <utility>

#include "Circle.hpp"

sf::Clock shader_clock;

Circle::Circle(float size, sf::Vector2f position, sf::Color color) : mCircleShape(size) {
    mCircleShape.setPosition(position);
    mColor = color;
    mCircleShape.setFillColor(color);

    shader_clock.restart();
}

Circle::~Circle() {}

void Circle::draw(sf::RenderTarget &target, sf::Shader *shader, bool use_shader) {
    if (use_shader) {
        shader->setUniform("u_position", getPosition());
        shader->setUniform("u_radius", getRadius());
        shader->setUniform("u_time", shader_clock.getElapsedTime().asSeconds());
        target.draw(mCircleShape, shader);
    } else {
        target.draw(mCircleShape);
    }
}

sf::Vector2f Circle::getPosition() const { return mCircleShape.getPosition(); }

float Circle::getRadius() const { return mCircleShape.getRadius(); }

void Circle::setColor(sf::Color color) {
    mColor = color;
    mCircleShape.setFillColor(color);
}

void Circle::setColor() { mCircleShape.setFillColor(mColor); }

void Circle::changeColor(sf::Color color) { mCircleShape.setFillColor(color); }

void Circle::setPosition(sf::Vector2f position) { mCircleShape.setPosition(position); }

void Circle::setRadius(int size) { mCircleShape.setRadius(size); }
