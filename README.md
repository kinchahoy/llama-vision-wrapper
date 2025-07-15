# Llama Vision Wrapper Setup Guide

This guide provides quick steps to set up the `llama.cpp` submodule within this project and apply a necessary patch.

## 1. Clone the Repository (if you haven't already)

If you haven't cloned this parent repository, do so first:

```bash
git clone <URL_OF_THIS_REPO>
cd <NAME_OF_THIS_REPO_DIRECTORY>

## 2. Initialize and Download the llama.cpp Submodule

The llama.cpp component is included as a Git submodule. You need to initialize and update it to download its contents.

git submodule update --init --recursive

## 3. Apply the llama.cpp Common CMake Patch

A patch is provided to modify the llama.cpp build configuration. This patch adjusts a line in llama.cpp/common/CMakeLists.txt.

Ensure you are in the root directory of this repository (where llama.cpp and patch_llama_common_for_dynamic.patch are located) and run:

patch -p1 < patch_llama_common_for_dynamic.patch

## 4. 

