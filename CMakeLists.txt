cmake_minimum_required(VERSION 3.16)
project(MelvinUnifiedBrain)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files for main executable
set(MAIN_SOURCES
    src/main.cpp
    src/logging.cpp
    src/websearch_fallback.cpp
)

# Create main executable
add_executable(melvin_unified ${MAIN_SOURCES})

# Link libraries
target_link_libraries(melvin_unified 
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Test executable
add_executable(test_startup tests/test_startup.cpp src/logging.cpp src/websearch_fallback.cpp)
target_link_libraries(test_startup 
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Compiler flags
target_compile_options(melvin_unified PRIVATE
    -Wall
    -Wextra
    -O3
)

target_compile_options(test_startup PRIVATE
    -Wall
    -Wextra
    -O3
)

# Set output name
set_target_properties(melvin_unified PROPERTIES OUTPUT_NAME "melvin_unified")

# Installation
install(TARGETS melvin_unified DESTINATION bin)

# Print configuration info
message(STATUS "Melvin Unified Brain Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CURL Found: ${CURL_FOUND}")
message(STATUS "  nlohmann/json Found: ${nlohmann_json_FOUND}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")

# Runtime dependency check
if(NOT CURL_FOUND)
    message(FATAL_ERROR "CURL library not found - required for web search functionality")
endif()

if(NOT nlohmann_json_FOUND)
    message(FATAL_ERROR "nlohmann_json library not found - required for JSON parsing")
endif()