# llama_insight usecases

Example scripts that consume the `llama_insight` package.

## Structure

- **Package source**: `wrapper_src/llama_insight` exposes `Config`, `LlamaBackend`,
  model loading helpers, timers, and download utilities. Install it with `pip install .`
  (or `pip install llama_insight` once published).
- **Usecases** (this folder): runnable scripts that depend solely on the installed package.
  - `cppyy-mtmd.py` – multimodal generation with benchmarking
  - `examples/simple-generation.py` – minimal multimodal text generation
  - `examples/image-encoder.py` – load images and demonstrate media encoding

## Usage

```bash
# Install the package locally (needed unless you pip install from PyPI)
uv pip install -e .

# Basic generation
uv run usecases/examples/simple-generation.py --image path/to/image.jpg

# With custom model
uv run usecases/cppyy-mtmd.py --repo-id "custom/model" --image image.jpg

# Encode images
uv run usecases/examples/image-encoder.py image1.jpg image2.jpg --output-dir embeddings/
```

> Tip: if you build the native libraries outside of the repository tree, set
> `LLAMA_INSIGHT_ROOT=/path/to/llama-vision-wrapper` so the package can locate
> `llama.cpp` artifacts and `gen-helper` outputs.

## Configuration

All examples rely on `llama_insight.Config`, so the same CLI flags apply:

- Model: `--repo-id`, `--model`, `--mmproj`
- Runtime: `--n-gpu-layers`, `-t` (threads), `--verbose-cpp`
- Sampling: `--temp`, `--top-k`, `--top-p`, `--repeat-penalty`, `--max-new-tokens`
