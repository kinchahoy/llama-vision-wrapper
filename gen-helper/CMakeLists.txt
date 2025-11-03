cmake_minimum_required(VERSION 3.10)
project(GenerationHelper LANGUAGES C CXX)

# --- Optional ccache setup ---
option(ENABLE_CCACHE "Enable ccache if available" ON)

if(ENABLE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "ccache found: ${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else()
        message(STATUS "ccache not found, continuing without it")
    endif()
else()
    message(STATUS "ccache explicitly disabled via ENABLE_CCACHE=OFF")
endif()


# On macOS, force CMake to use the correct SDK path
if(APPLE)
    execute_process(
        COMMAND xcrun --show-sdk-path
        OUTPUT_VARIABLE CMAKE_OSX_SYSROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Using macOS SDK: ${CMAKE_OSX_SYSROOT}")
    set(CMAKE_OSX_SYSROOT ${CMAKE_OSX_SYSROOT} CACHE PATH "macOS SDK path" FORCE)
endif()

# Default to Release build type for performance unless specified otherwise
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (e.g., Debug, Release)" FORCE)
  message(STATUS "Setting build type to 'Release' as default.")
endif()

# Set C++ standard to C++17 (required by llama.cpp headers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# --- Define LlamaCpp Paths (Relative to this project) ---
# Assumes llama.cpp source is in a subdirectory named 'llama.cpp'
# Assumes llama.cpp was built *with* BUILD_SHARED_LIBS=ON in 'llama.cpp/build'
# Libraries are expected in 'llama.cpp/build'
set(LLAMA_CPP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../llama.cpp" CACHE PATH "Path to llama.cpp source directory")
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_SOURCE_DIR}/build" CACHE PATH "Path to llama.cpp build directory (containing libs)")
set(LLAMA_CPP_INCLUDE_DIR "${LLAMA_CPP_SOURCE_DIR}/include" CACHE PATH "Path to llama.h include directory")
set(LLAMA_CPP_COMMON_INCLUDE_DIR "${LLAMA_CPP_SOURCE_DIR}/common" CACHE PATH "Path to common.h include directory")
set(LLAMA_CPP_GGML_INCLUDE_DIR "${LLAMA_CPP_SOURCE_DIR}/ggml/include" CACHE PATH "Path to ggml.h include directory")

# --- Include Directories ---
# Include directories needed for our helper and for llama.h/common.h
include_directories(
    ${LLAMA_CPP_INCLUDE_DIR}
    ${LLAMA_CPP_COMMON_INCLUDE_DIR}
    ${LLAMA_CPP_GGML_INCLUDE_DIR}
)

# --- Find Required Libraries ---
# Use find_library for robustness, specify the build dir hint
# Look in both build and build/bin directories, and check various naming patterns
set(LLAMA_SEARCH_PATHS 
    ${LLAMA_CPP_BUILD_DIR}
    ${LLAMA_CPP_BUILD_DIR}/bin
    ${LLAMA_CPP_BUILD_DIR}/lib
    ${LLAMA_CPP_BUILD_DIR}/src
)

# Find llama library
find_library(LLAMA_LIBRARY 
    NAMES llama libllama llama.so libllama.so llama.dylib libllama.dylib
    PATHS ${LLAMA_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

# Find common library  
find_library(COMMON_LIBRARY 
    NAMES common libcommon common.so libcommon.so common.dylib libcommon.dylib
    PATHS ${LLAMA_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

# Find ggml libraries (often required)
find_library(GGML_LIBRARY 
    NAMES ggml libggml ggml.so libggml.so ggml.dylib libggml.dylib
    PATHS ${LLAMA_SEARCH_PATHS}
    NO_DEFAULT_PATH
)

if(NOT LLAMA_LIBRARY)
    message(FATAL_ERROR "libllama not found in ${LLAMA_CPP_BUILD_DIR} or subdirectories. Ensure llama.cpp was built with BUILD_SHARED_LIBS=ON.")
endif()
if(NOT COMMON_LIBRARY)
    message(FATAL_ERROR "libcommon not found in ${LLAMA_CPP_BUILD_DIR} or subdirectories. Ensure llama.cpp was built with BUILD_SHARED_LIBS=ON.")
endif()

message(STATUS "Found libllama: ${LLAMA_LIBRARY}")
message(STATUS "Found libcommon: ${COMMON_LIBRARY}")
if(GGML_LIBRARY)
    message(STATUS "Found libggml: ${GGML_LIBRARY}")
endif()

# --- Build Our Shared Library ---
add_library(generation_helper SHARED generation_helper.cpp generation_helper.h)

# Link our helper library against llama and common
target_link_libraries(generation_helper PRIVATE
    ${LLAMA_LIBRARY}
    ${COMMON_LIBRARY}
)

# Add ggml library if found
if(GGML_LIBRARY)
    target_link_libraries(generation_helper PRIVATE ${GGML_LIBRARY})
endif()

# --- Installation (Optional but Recommended) ---
# Install the library and header file to a known location
# Adjust CMAKE_INSTALL_PREFIX if needed when running cmake
install(TARGETS generation_helper DESTINATION lib)
install(FILES generation_helper.h DESTINATION include)
