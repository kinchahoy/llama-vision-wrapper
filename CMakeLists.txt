cmake_minimum_required(VERSION 3.10)
project(GenerationHelper LANGUAGES C CXX)
                                                                                                                                                                                 
# Set C++ standard to C++11 (required by llama.cpp headers)                                                                                                                      
set(CMAKE_CXX_STANDARD 11)                                                                                                                                                       
set(CMAKE_CXX_STANDARD_REQUIRED ON)                                                                                                                                              


# --- Define LlamaCpp Paths (Relative to this project) ---
# Assumes llama.cpp source is in a subdirectory named 'llama.cpp'
# Assumes llama.cpp was built *with* BUILD_SHARED_LIBS=ON in 'llama.cpp/build'
# Libraries are expected in 'llama.cpp/build/bin'
set(LLAMA_CPP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to llama.cpp source directory")
set(LLAMA_CPP_BUILD_DIR "${LLAMA_CPP_SOURCE_DIR}/build/bin" CACHE PATH "Path to llama.cpp build directory (containing libs)")
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
# Check both 'libllama.dylib' (macOS) and 'libllama.so' (Linux)
find_library(LLAMA_LIBRARY NAMES llama libllama PATHS ${LLAMA_CPP_BUILD_DIR} REQUIRED)
# Check both 'libcommon.dylib' (macOS) and 'libcommon.so' (Linux)
find_library(COMMON_LIBRARY NAMES common libcommon PATHS ${LLAMA_CPP_BUILD_DIR} REQUIRED)
# Add other ggml libs if common/llama don't pull them in transitively (might be needed depending on build)
# find_library(GGML_LIBRARY NAMES ggml libggml PATHS ${LLAMA_CPP_BUILD_DIR} REQUIRED)

if(NOT LLAMA_LIBRARY)
    message(FATAL_ERROR "libllama not found in ${LLAMA_CPP_BUILD_DIR}. Ensure llama.cpp was built with BUILD_SHARED_LIBS=ON.")
endif()
if(NOT COMMON_LIBRARY)
    message(FATAL_ERROR "libcommon not found in ${LLAMA_CPP_BUILD_DIR}. Ensure llama.cpp was built with BUILD_SHARED_LIBS=ON.")
endif()

message(STATUS "Found libllama: ${LLAMA_LIBRARY}")
message(STATUS "Found libcommon: ${COMMON_LIBRARY}")

# --- Build Our Shared Library ---
add_library(generation_helper SHARED generation_helper.cpp generation_helper.h)

# Link our helper library against llama and common
target_link_libraries(generation_helper PRIVATE
    ${LLAMA_LIBRARY}
    ${COMMON_LIBRARY}
    # ${GGML_LIBRARY} # Add if needed
)

# --- Installation (Optional but Recommended) ---
# Install the library and header file to a known location
# Adjust CMAKE_INSTALL_PREFIX if needed when running cmake
install(TARGETS generation_helper DESTINATION lib)
install(FILES generation_helper.h DESTINATION include)
