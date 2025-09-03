cmake_minimum_required(VERSION 3.20)
project(Melvin VERSION 1.0.0 LANGUAGES CXX)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# Find system libraries
find_library(SOCKET_LIB socket)
find_library(UTIL_LIB util)

# External dependencies (will be installed via apt on Jetson)
# These are common Linux libraries that should be available
set(EXTERNAL_LIBS
    ${CMAKE_THREAD_LIBS_INIT}
    ${SOCKET_LIB}
    ${UTIL_LIB}
    dl
    m
)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/hardware
    ${CMAKE_CURRENT_SOURCE_DIR}/brain
    ${CMAKE_CURRENT_SOURCE_DIR}/learning
    ${CMAKE_CURRENT_SOURCE_DIR}/webserver
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# Common source files
set(COMMON_SOURCES
    common/logger.cpp
    common/config.cpp
    common/utils.cpp
)

# Hardware module
set(HARDWARE_SOURCES
    hardware/hal.cpp
    hardware/motors/can_bus.cpp
    hardware/motors/rmd_x_driver.cpp
    hardware/sensors/sensor_manager.cpp
    hardware/sensors/touch_sensor.cpp
    hardware/sensors/temperature_sensor.cpp
    hardware/sensors/camera_driver.cpp
    hardware/sensors/audio_driver.cpp
)

# Brain module sources are now defined in brain/CMakeLists.txt

# Learning module
set(LEARNING_SOURCES
    learning/learning_engine.cpp
    learning/rule_engine.cpp
    learning/confidence_calculator.cpp
    learning/ml_hooks.cpp
)

# Web server module
set(WEBSERVER_SOURCES
    webserver/http_server.cpp
    webserver/api_handlers.cpp
    webserver/websocket_handler.cpp
    webserver/static_file_server.cpp
)

# Main application
set(MAIN_SOURCES
    main.cpp
    melvin_core.cpp
)

# Create libraries
add_library(melvin_common STATIC ${COMMON_SOURCES})
add_library(melvin_hardware STATIC ${HARDWARE_SOURCES})

# Brain module (with its own CMakeLists.txt)
add_subdirectory(brain)
add_library(melvin_learning STATIC ${LEARNING_SOURCES})
add_library(melvin_webserver STATIC ${WEBSERVER_SOURCES})

# Link dependencies
target_link_libraries(melvin_hardware melvin_common ${EXTERNAL_LIBS})
target_link_libraries(melvin_brain melvin_common ${EXTERNAL_LIBS})
target_link_libraries(melvin_learning melvin_brain melvin_common ${EXTERNAL_LIBS})
target_link_libraries(melvin_webserver melvin_common ${EXTERNAL_LIBS})

# Main executable
add_executable(melvin ${MAIN_SOURCES})
target_link_libraries(melvin 
    melvin_common
    melvin_hardware
    melvin_brain
    melvin_learning
    melvin_webserver
    ${EXTERNAL_LIBS}
)

# Installation
install(TARGETS melvin DESTINATION bin)
install(DIRECTORY ui/frontend/ DESTINATION share/melvin/ui)
install(FILES melvin.service DESTINATION /etc/systemd/system)
install(FILES config/melvin.conf DESTINATION /etc/melvin)

# Create logs directory
install(DIRECTORY logs/ DESTINATION /var/melvin)

# Testing (optional)
option(BUILD_TESTS "Build unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
