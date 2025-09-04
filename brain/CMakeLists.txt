cmake_minimum_required(VERSION 3.16)
project(MelvinOptimizedBrain)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    optimized_node_system.cpp
    fast_brain_core.cpp
    python_wrapper.cpp
    connection.cpp
    node.cpp
)

# Header files
set(HEADERS
    optimized_node_system.hpp
    fast_brain_core.hpp
    brain_graph.hpp
    connection.hpp
    node.hpp
    db.hpp
    graph.hpp
)

# Create library
add_library(melvin_optimized_brain STATIC ${SOURCES} ${HEADERS})

# Set library properties
set_target_properties(melvin_optimized_brain PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(melvin_optimized_brain PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -fno-exceptions
        -fno-rtti
        -ffast-math
        -march=native
        -mtune=native
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(melvin_optimized_brain PRIVATE
        /W4
        /O2
        /arch:AVX2
        /fp:fast
    )
endif()

# Create test executable
add_executable(test_optimized_nodes test_optimized_nodes.cpp)
target_link_libraries(test_optimized_nodes melvin_optimized_brain)

# Create benchmark executable
add_executable(benchmark_nodes benchmark_nodes.cpp)
target_link_libraries(benchmark_nodes melvin_optimized_brain)

# Create Python extension (if Python is available)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
if(Python3_FOUND)
    pybind11_add_module(melvin_optimized_brain_py python_wrapper.cpp)
    target_link_libraries(melvin_optimized_brain_py melvin_optimized_brain)
endif()

# Installation
install(TARGETS melvin_optimized_brain
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${HEADERS} DESTINATION include/melvin)

# Testing
enable_testing()
add_test(NAME TestOptimizedNodes COMMAND test_optimized_nodes)

# Performance testing
add_test(NAME BenchmarkNodes COMMAND benchmark_nodes)

# Print configuration summary
message(STATUS "Melvin Optimized Brain Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Optimization Level: ${CMAKE_CXX_FLAGS}")
