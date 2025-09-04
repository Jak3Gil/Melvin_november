cmake_minimum_required(VERSION 3.16)
project(MelvinOptimizedV2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Find required packages
find_package(PkgConfig REQUIRED)

# Find compression libraries
pkg_check_modules(ZLIB REQUIRED zlib)
pkg_check_modules(LZMA REQUIRED liblzma)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Include directories
include_directories(${ZLIB_INCLUDE_DIRS})
include_directories(${LZMA_INCLUDE_DIRS})
include_directories(${ZSTD_INCLUDE_DIRS})

# Create executables
add_executable(melvin_optimized_v2_cpp melvin_optimized_v2.cpp)
add_executable(melvin_brain_monitor_cpp melvin_brain_monitor.cpp)
add_executable(feed_melvin_data_cpp feed_melvin_data.cpp)

# Link libraries
target_link_libraries(melvin_optimized_v2_cpp 
    ${ZLIB_LIBRARIES}
    ${LZMA_LIBRARIES}
    ${ZSTD_LIBRARIES}
    pthread
)

target_link_libraries(melvin_brain_monitor_cpp 
    ${ZLIB_LIBRARIES}
    ${LZMA_LIBRARIES}
    ${ZSTD_LIBRARIES}
    pthread
)

target_link_libraries(feed_melvin_data_cpp 
    ${ZLIB_LIBRARIES}
    ${LZMA_LIBRARIES}
    ${ZSTD_LIBRARIES}
    pthread
)

# Set compiler flags
target_compile_options(melvin_optimized_v2_cpp PRIVATE
    ${ZLIB_CFLAGS_OTHER}
    ${LZMA_CFLAGS_OTHER}
    ${ZSTD_CFLAGS_OTHER}
)

target_compile_options(melvin_brain_monitor_cpp PRIVATE
    ${ZLIB_CFLAGS_OTHER}
    ${LZMA_CFLAGS_OTHER}
    ${ZSTD_CFLAGS_OTHER}
)

target_compile_options(feed_melvin_data_cpp PRIVATE
    ${ZLIB_CFLAGS_OTHER}
    ${LZMA_CFLAGS_OTHER}
    ${ZSTD_CFLAGS_OTHER}
)

# Add custom target for installation
install(TARGETS melvin_optimized_v2_cpp melvin_brain_monitor_cpp feed_melvin_data_cpp
    RUNTIME DESTINATION bin
)

# Build script is already provided
message(STATUS "Build script: build_melvin_cpp.sh")

# Print configuration info
message(STATUS "Melvin Optimized V2 C++ Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  ZLIB: ${ZLIB_VERSION}")
message(STATUS "  LZMA: ${LZMA_VERSION}")
message(STATUS "  ZSTD: ${ZSTD_VERSION}")
message(STATUS "  Optimizations: -O3 -march=native -ffast-math")
