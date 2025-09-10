cmake_minimum_required(VERSION 3.16)
project(MelvinUnifiedBrain)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_BUILD_TYPE Release)

# Find required libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZLIB REQUIRED zlib)
pkg_check_modules(LZMA REQUIRED liblzma)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Include directories
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${LZMA_INCLUDE_DIRS})
include_directories(${ZSTD_INCLUDE_DIRS})

# Core executables
add_executable(melvin_autonomous_learning
    melvin_autonomous_learning.cpp
    melvin_driver_enhanced.cpp
)
target_link_libraries(melvin_autonomous_learning ${ZLIB_LIBRARIES} ${LZMA_LIBRARIES} ${ZSTD_LIBRARIES})

add_executable(melvin_unified_brain
    melvin_unified_brain.cpp
    melvin_unified_system.cpp
)
target_link_libraries(melvin_unified_brain ${ZLIB_LIBRARIES} ${LZMA_LIBRARIES} ${ZSTD_LIBRARIES})

add_executable(melvin_optimized_v2
    melvin_optimized_v2.cpp
)
target_link_libraries(melvin_optimized_v2 ${ZLIB_LIBRARIES} ${LZMA_LIBRARIES} ${ZSTD_LIBRARIES})

add_executable(melvin_cli
    melvin_cli.cpp
    melvin_optimized_v2.cpp
)
target_link_libraries(melvin_cli ${ZLIB_LIBRARIES} ${LZMA_LIBRARIES} ${ZSTD_LIBRARIES})

add_executable(melvin_brain_monitor
    melvin_brain_monitor.cpp
    melvin_optimized_v2.cpp
)
target_link_libraries(melvin_brain_monitor ${ZLIB_LIBRARIES} ${LZMA_LIBRARIES} ${ZSTD_LIBRARIES})

add_executable(feed_melvin_data
    feed_melvin_data.cpp
    melvin_optimized_v2.cpp
)
target_link_libraries(feed_melvin_data ${ZLIB_LIBRARIES} ${LZMA_LIBRARIES} ${ZSTD_LIBRARIES})

# Utility executables
add_executable(check_melvin_status check_melvin_status.cpp)
add_executable(melvin_cpp_interactive melvin_cpp_interactive.cpp)
add_executable(melvin_work_analyzer melvin_work_analyzer.cpp)
