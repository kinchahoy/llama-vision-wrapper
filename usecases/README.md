# llama_insight usecases

Example scripts that consume the `llama_insight` package.

## Structure

- **Package source**: `src/llama_insight` exposes `Config`, `LlamaBackend`,
  model loading helpers, timers, and download utilities.
- **Usecases** (this folder): runnable scripts that depend solely on the installed package.
  - `generate_simple.py` – minimal multimodal text generation
  - `generate_benchmark.py` – single-run generation with benchmark logging
  - `generate_batched.py` – batched multimodal generation across prompts/images
  - `encode_images.py` – load images, pull embeddings into memory, and save to disk
- **Shared helpers**: `src/llama_insight/usecase_helpers.py` centralizes runtime
  setup, timing, and benchmark logging so the scripts stay small.

## Usage

```bash
# Build and install the local wheel (repo standard workflow)
uv build
uv sync
uv pip install --reinstall --no-deps dist/llama_insight-*.whl

# Basic generation
uv run usecases/generate_simple.py --image test-images/debug.jpg

# With custom model
uv run usecases/generate_benchmark.py --repo-id "custom/model" --image test-images/debug.jpg

# Encode images
uv run usecases/encode_images.py test-images/debug.jpg test-images/movie.jpg --output-dir embeddings/ --format npy

# Batched generation
uv run usecases/generate_batched.py --image test-images/debug.jpg --n-parallel 8
```

> Do not run `uv add dist/*.whl` inside this repository. `uv` treats it as a
> self-dependency because the wheel name matches the current project.

## Configuration

All examples rely on `llama_insight.Config`, so the same CLI flags apply:

- Model: `--repo-id`, `--model`, `--mmproj`
- Runtime: `--n-gpu-layers`, `-t` (threads), `--verbose-cpp`
- Sampling: `--temp`, `--top-k`, `--top-p`, `--repeat-penalty`, `--max-new-tokens`
