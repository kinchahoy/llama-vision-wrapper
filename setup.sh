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
patch -p1 < patch_llama_common_for_dynamic.patch

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
cmake --build build
cd ..

# Step 5: Build helper library
echo "Building generation_helper shared library..."
mkdir -p build
cd build || exit 1
cmake ..
make
cd ..

echo "Libraries built in llama.cpp/build/bin/ and build/."

# Step 6: Run examples
echo "Before running examples, ensure you have downloaded the required models and updated hardcoded paths in:"
echo "  - cython/cython-mtmd.py"
echo "  - cpppy/ccpyy-mtmd.py"
echo ""
echo "Choose an example implementation:"
select example in "cppyy (Dynamic Bindings)" "Cython (Compiled Extension)" "Skip"; do
    case $example in
        "cppyy (Dynamic Bindings)" )
            echo "Installing cppyy..."
            pip install cppyy
            echo "Running cpppy/ccpyy-mtmd.py..."
            python cpppy/ccpyy-mtmd.py
            break;;
        "Cython (Compiled Extension)" )
            echo "Building Cython extension..."
            cd cython
            python setup.py build_ext --inplace
            cd ..
            echo "Running cython/cython-mtmd.py..."
            python cython/cython-mtmd.py
            break;;
        "Skip" )
            echo "Setup complete. You can run the examples later."
            break;;
    esac
done

echo "=== Setup Finished ==="