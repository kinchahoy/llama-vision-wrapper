# llama_insight usecases

Example scripts that consume the `llama_insight` package.

## Structure

- **Package source**: `wrapper_src/llama_insight` exposes `Config`, `LlamaBackend`,
  model loading helpers, timers, and download utilities. Install it with `pip install .`
  (or `pip install llama_insight` once published).
- **Usecases** (this folder): runnable scripts that depend solely on the installed package.
  - `generate_simple.py` – minimal multimodal text generation
  - `generate_benchmark.py` – single-run generation with benchmark logging
  - `generate_batched.py` – batched multimodal generation across prompts/images
  - `encode_images.py` – load images, pull embeddings into memory, and save to disk
- **Shared helpers**: `wrapper_src/llama_insight/usecase_helpers.py` centralizes runtime
  setup, timing, and benchmark logging so the scripts stay small.

## Usage

```bash
# Install the package locally (needed unless you pip install from PyPI)
uv pip install -e .

# Basic generation
uv run usecases/generate_simple.py --image test-images/debug.jpg

# With custom model
uv run usecases/generate_benchmark.py --repo-id "custom/model" --image test-images/debug.jpg

# Encode images
uv run usecases/encode_images.py test-images/debug.jpg test-images/movie.jpg --output-dir embeddings/ --format npy

# Batched generation
uv run usecases/generate_batched.py --image test-images/debug.jpg --n-parallel 8
```

> Tip: if you build the native libraries outside of the repository tree, set
> `LLAMA_INSIGHT_ROOT=/path/to/llama-vision-wrapper` so the package can locate
> `llama.cpp` artifacts and `gen-helper` outputs.

## Configuration

All examples rely on `llama_insight.Config`, so the same CLI flags apply:

- Model: `--repo-id`, `--model`, `--mmproj`
- Runtime: `--n-gpu-layers`, `-t` (threads), `--verbose-cpp`
- Sampling: `--temp`, `--top-k`, `--top-p`, `--repeat-penalty`, `--max-new-tokens`
