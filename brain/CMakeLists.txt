# Brain Module CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Find required packages for the brain module
find_package(PkgConfig REQUIRED)

# Try to find optional high-performance libraries
find_package(OpenMP)
find_library(SQLITE3_LIB sqlite3)

# Check for Python and pybind11 for Python integration
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 QUIET)

# Brain module sources
set(BRAIN_SOURCES
    node.cpp
    connection.cpp
    graph.cpp
    db.cpp
    fast_brain_core.cpp
)

# Brain module headers
set(BRAIN_HEADERS
    node.hpp
    connection.hpp
    graph.hpp
    db.hpp
    brain_graph.hpp
    fast_brain_core.hpp
)

# Create the brain library
add_library(melvin_brain STATIC ${BRAIN_SOURCES})

# Include directories
target_include_directories(melvin_brain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(melvin_brain PRIVATE
        -march=native          # Use all available CPU features
        -mtune=native         # Optimize for current CPU
        -ffast-math           # Fast floating point math
        -funroll-loops        # Unroll loops for performance
        -finline-functions    # Aggressive inlining
        -fomit-frame-pointer  # Omit frame pointer for more registers
    )
    
    # Release-specific optimizations
    target_compile_options(melvin_brain PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:Release>:-flto>  # Link-time optimization
    )
endif()

# SIMD support detection and flags
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2)
    target_compile_definitions(melvin_brain PRIVATE HAVE_AVX2=1)
    target_compile_options(melvin_brain PRIVATE -mavx2)
    message(STATUS "AVX2 support enabled")
endif()

if(COMPILER_SUPPORTS_SSE42)
    target_compile_definitions(melvin_brain PRIVATE HAVE_SSE42=1)
    target_compile_options(melvin_brain PRIVATE -msse4.2)
    message(STATUS "SSE4.2 support enabled")
endif()

if(COMPILER_SUPPORTS_FMA)
    target_compile_options(melvin_brain PRIVATE -mfma)
    message(STATUS "FMA support enabled")
endif()

# OpenMP support for parallel processing
if(OpenMP_CXX_FOUND)
    target_link_libraries(melvin_brain PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(melvin_brain PRIVATE HAVE_OPENMP=1)
    message(STATUS "OpenMP support enabled")
endif()

# SQLite support
if(SQLITE3_LIB)
    target_link_libraries(melvin_brain PRIVATE ${SQLITE3_LIB})
    target_compile_definitions(melvin_brain PRIVATE HAVE_SQLITE3=1)
    message(STATUS "SQLite3 support enabled")
else()
    message(WARNING "SQLite3 not found - database features will be limited")
endif()

# JSON support (using nlohmann/json if available, fallback to simple parsing)
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(melvin_brain PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(melvin_brain PRIVATE HAVE_NLOHMANN_JSON=1)
    message(STATUS "nlohmann/json support enabled")
else()
    message(STATUS "Using fallback JSON implementation")
endif()

# Memory alignment and cache optimization
target_compile_definitions(melvin_brain PRIVATE
    CACHE_LINE_SIZE=64        # Optimize for 64-byte cache lines
    MEMORY_ALIGNMENT=32       # 32-byte alignment for SIMD
)

# Python integration (optional)
if(Python3_FOUND AND pybind11_FOUND)
    message(STATUS "Building Python wrapper for fast brain core")
    
    # Create Python module
    pybind11_add_module(fast_brain_core python_wrapper.cpp)
    
    # Link with brain library
    target_link_libraries(fast_brain_core PRIVATE melvin_brain)
    
    # Set properties
    target_compile_definitions(fast_brain_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
    
    # Install Python module
    install(TARGETS fast_brain_core DESTINATION .)
    
else()
    message(STATUS "Python wrapper not available (missing Python3 or pybind11)")
endif()

# Performance profiling support (optional)
option(ENABLE_PROFILING "Enable performance profiling" OFF)
if(ENABLE_PROFILING)
    target_compile_options(melvin_brain PRIVATE -pg)
    target_link_options(melvin_brain PRIVATE -pg)
    target_compile_definitions(melvin_brain PRIVATE ENABLE_PROFILING=1)
    message(STATUS "Profiling support enabled")
endif()

# Memory debugging support (debug builds only)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(melvin_brain PRIVATE
        DEBUG_MEMORY=1
        DEBUG_PERFORMANCE=1
    )
endif()

# Install headers
install(FILES ${BRAIN_HEADERS} DESTINATION include/melvin/brain)

# Install library
install(TARGETS melvin_brain
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Performance benchmarks (optional)
option(BUILD_BENCHMARKS "Build performance benchmarks" OFF)
if(BUILD_BENCHMARKS)
    add_executable(brain_benchmark benchmark.cpp)
    target_link_libraries(brain_benchmark PRIVATE melvin_brain)
    
    install(TARGETS brain_benchmark DESTINATION bin)
    message(STATUS "Brain benchmarks enabled")
endif()

# Print configuration summary
message(STATUS "Brain module configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  SIMD Support: ${COMPILER_SUPPORTS_AVX2} (AVX2), ${COMPILER_SUPPORTS_SSE42} (SSE4.2)")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  SQLite3: ${SQLITE3_LIB}")
message(STATUS "  Python: ${Python3_FOUND}")
message(STATUS "  pybind11: ${pybind11_FOUND}")
