add_executable(ug-test
    tests-main.cpp
    common/events/tests-EventController.cpp
    common/factories/tests-IdFactory.cpp
    common/physics/tests-CollisionUtil.cpp
    common/objects/tests-CircleRigidBody.cpp
    server/systems/tests-ServerGameController.cpp
    client/systems/tests-ClientGameController.cpp
    client/rendering/tests-AnimatedSprite.cpp
    client/rendering/tests-AnimationController.cpp
    common/systems/tests-ResourceController.cpp
    common/metrics/tests-TemporalMetric.cpp
)

target_link_libraries(ug-test PRIVATE server-lib common-lib client-lib catch)

# Old Catch2 v2 uses SIGSTKSZ as a constant, which glibc 2.34+ no longer allows
target_compile_definitions(ug-test PRIVATE CATCH_CONFIG_NO_POSIX_SIGNALS)
