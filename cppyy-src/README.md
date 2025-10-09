# cppyy-src: Modular Llama Vision Wrapper

A clean, modular Python wrapper for llama.cpp multimodal functionality using cppyy.

## Structure

### Core Library
- **`llama_core.py`** - Main llama.cpp interface with focused classes:
  - `LlamaBackend` - Backend lifecycle management  
  - `ModelLoader` - Model and multimodal projector loading
  - `MultimodalProcessor` - Image loading and processing
  - `TextGenerator` - Text generation with sampling

- **`utils.py`** - Shared utilities:
  - `Config` - Consolidated configuration management
  - `add_common_args()` - Common argument parser setup
  - `Timer` - Performance timing utilities  
  - `download_models()` - Model downloading from HuggingFace

### Examples
- **`cppyy-mtmd.py`** - Full-featured example with benchmarking
- **`examples/simple-generation.py`** - Basic multimodal generation
- **`examples/image-encoder.py`** - Image encoding example

## Usage

```bash
# Basic generation
uv run cppyy-src/examples/simple-generation.py --image path/to/image.jpg

# With custom model
uv run cppyy-src/cppyy-mtmd.py --repo-id "custom/model" --image image.jpg

# Encode images
uv run cppyy-src/examples/image-encoder.py image1.jpg image2.jpg --output-dir embeddings/
```

## Key Improvements

- **No code duplication** - Argument parsing, configuration, and timing utilities are shared
- **Single responsibility** - Each class has one focused purpose
- **Clean interfaces** - Clear separation between core functionality and examples
- **Easy extension** - Adding new examples or functionality is straightforward

## Configuration

All examples use the same configuration system via `Config` class:

- Model: `--repo-id`, `--model`, `--mmproj`  
- Runtime: `--n-gpu-layers`, `-t` (threads), `--verbose-cpp`
- Sampling: `--temp`, `--max-new-tokens`