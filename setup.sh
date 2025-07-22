#!/bin/bash
# filepath: /home/raist/trackedproj/llama-vision-wrapper/setup.sh

echo "=== Llama Vision Wrapper Interactive Setup ==="

# Step 2: Submodule update
echo "Initializing and updating llama.cpp submodule..."
git submodule update --init --recursive

# Step 3: Apply patch
if [ ! -f patch_llama_common_for_dynamic.patch ]; then
    echo "Patch file 'patch_llama_common_for_dynamic.patch' not found!"
    exit 1
fi
echo "Applying llama.cpp patch..."

# Check if patch is already applied
if patch -p1 --dry-run < patch_llama_common_for_dynamic.patch | grep -q "Reversed"; then
    echo "Patch appears to already be applied. Skipping..."
else
    patch -p1 < patch_llama_common_for_dynamic.patch
fi


# Step 4: Build llama.cpp
echo "Do you want to build llama.cpp with CUDA support?"
select cuda in "Yes" "No"; do
    case $cuda in
        Yes ) cuda_flag="-DGGML_CUDA=ON"; break;;
        No ) cuda_flag=""; break;;
    esac
done

echo "Building llama.cpp shared libraries..."
cd llama.cpp || exit 1
cmake -B build -DBUILD_SHARED_LIBS=ON $cuda_flag
cmake --build build --config Release -j "$JOBS"
cd ..

# Step 5: Build helper library
echo "Building generation_helper shared library..."
mkdir -p build
cd build || exit 1
cmake .. 
make
cd ..

echo "Libraries built in llama.cpp/build/bin/ and build/."

echo "Spinning up the venv using uv"

uv sync

# Step 6: Run examples
echo "Before running examples, ensure you have downloaded the required models and updated hardcoded paths in:"
echo "  - cython-src/cython-mtmd.py"
echo "  - cppyy-src/ccpyy-mtmd.py"
echo ""
echo "Choose an example implementation:"
select example in "cppyy (Dynamic Bindings)" "Cython (Compiled Extension)" "Skip"; do
    case $example in
        "cppyy (Dynamic Bindings)" )
            echo "Installing cppyy..."
            uv pip install cppyy
            echo "Running cpppy/ccpyy-mtmd.py..."
            uv run cppyy/ccpyy-mtmd.py
            break;;
        "Cython (Compiled Extension)" )
            echo "Building Cython extension..."
            cd cython
            uv run setup.py build_ext --inplace
            cd ..
            echo "Running cython/cython-mtmd.py..."
            uv run cython/cython-mtmd.py
            break;;
        "Skip" )
            echo "Setup complete. You can run the examples later."
            break;;
    esac
done

echo "=== Setup Finished ==="
