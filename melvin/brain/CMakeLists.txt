# Brain Module CMakeLists.txt
# Builds the brain graph system with SQLite persistence

# Set module name
set(MODULE_NAME "brain")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find SQLite3
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)

# Find nlohmann/json
find_package(nlohmann_json REQUIRED)

# Brain module source files
set(BRAIN_SOURCES
    node.cpp
    connection.cpp
    db.cpp
    graph.cpp
)

# Brain module headers
set(BRAIN_HEADERS
    node.hpp
    connection.hpp
    db.hpp
    graph.hpp
)

# Create brain module library
add_library(melvin_brain STATIC ${BRAIN_SOURCES})

# Set target properties
set_target_properties(melvin_brain PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(melvin_brain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
)

# Link libraries
target_link_libraries(melvin_brain
    melvin_common
    ${SQLITE3_LIBRARIES}
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# Compiler flags
target_compile_options(melvin_brain PRIVATE
    ${SQLITE3_CFLAGS_OTHER}
    -Wall
    -Wextra
    -Wpedantic
    -O2
)

# Install rules
install(TARGETS melvin_brain
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${BRAIN_HEADERS}
    DESTINATION include/melvin/brain
)

# Create brain module interface target
add_library(melvin::brain ALIAS melvin_brain)

# Export target for parent CMakeLists.txt
set(MELVIN_BRAIN_TARGET melvin_brain PARENT_SCOPE)

# Optional: Build tests if enabled
if(BUILD_TESTS)
    enable_testing()
    
    # Test source files
    set(BRAIN_TEST_SOURCES
        tests/test_node.cpp
        tests/test_connection.cpp
        tests/test_graph.cpp
        tests/test_db.cpp
    )
    
    # Create test executable
    add_executable(brain_tests ${BRAIN_TEST_SOURCES})
    
    # Link test executable
    target_link_libraries(brain_tests
        melvin_brain
        melvin_common
        gtest
        gtest_main
        ${SQLITE3_LIBRARIES}
        nlohmann_json::nlohmann_json
        Threads::Threads
    )
    
    # Include test directories
    target_include_directories(brain_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/tests
        ${GTEST_INCLUDE_DIRS}
    )
    
    # Add tests
    add_test(NAME BrainNodeTests COMMAND brain_tests --gtest_filter=NodeTest.*)
    add_test(NAME BrainConnectionTests COMMAND brain_tests --gtest_filter=ConnectionTest.*)
    add_test(NAME BrainGraphTests COMMAND brain_tests --gtest_filter=GraphTest.*)
    add_test(NAME BrainDatabaseTests COMMAND brain_tests --gtest_filter=DatabaseTest.*)
    
    # Set test properties
    set_tests_properties(BrainNodeTests PROPERTIES
        TIMEOUT 30
        ENVIRONMENT "MELVIN_TEST_DATA_DIR=${CMAKE_SOURCE_DIR}/tests/data"
    )
    set_tests_properties(BrainConnectionTests PROPERTIES
        TIMEOUT 30
        ENVIRONMENT "MELVIN_TEST_DATA_DIR=${CMAKE_SOURCE_DIR}/tests/data"
    )
    set_tests_properties(BrainGraphTests PROPERTIES
        TIMEOUT 60
        ENVIRONMENT "MELVIN_TEST_DATA_DIR=${CMAKE_SOURCE_DIR}/tests/data"
    )
    set_tests_properties(BrainDatabaseTests PROPERTIES
        TIMEOUT 120
        ENVIRONMENT "MELVIN_TEST_DATA_DIR=${CMAKE_SOURCE_DIR}/tests/data"
    )
endif()

# Print configuration summary
message(STATUS "Brain module configuration:")
message(STATUS "  - SQLite3: ${SQLITE3_VERSION}")
message(STATUS "  - nlohmann/json: ${nlohmann_json_VERSION}")
message(STATUS "  - Build tests: ${BUILD_TESTS}")
message(STATUS "  - Source files: ${BRAIN_SOURCES}")
message(STATUS "  - Header files: ${BRAIN_HEADERS}")
