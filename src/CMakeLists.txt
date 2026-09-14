add_subdirectory(common)

IF(ONLY_CLIENT)
    add_subdirectory(client)
ELSEIF(ONLY_SERVER)
    add_subdirectory(server)
ELSE()
    add_subdirectory(client)
    add_subdirectory(server)
ENDIF()

add_subdirectory(sim)
