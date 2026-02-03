# Build Guide

This project builds a device-specific wheel that bundles llama.cpp shared libraries and the generation helper. Builds are driven by `uv` and the custom build backend in `build-tools/build_backend.py`.

## Quick Start

```bash
uv build
```

This will:
- Ensure `llama.cpp` sources exist (submodule or manual checkout)
- Apply the patch in `patch_llama_common_for_dynamic.patch`
- Stage headers into `wrapper_src/llama_insight/_headers/`
- Build llama.cpp + gen-helper shared libraries
- Package the shared libraries into `wrapper_src/llama_insight/libs/`

## Backend Selection (Reproducible Builds)

Backend selection order is:
1. `llama-insight.backend` config setting
2. `LLAMA_INSIGHT_BACKEND` env var
3. Auto-detect
4. Default: `cpu`

Supported backends:
`cpu`, `cuda`, `metal`, `vulkan`, `hip`, `kleidiai`, `custom`

### Examples

```bash
# Explicit CPU build
uv build --config-setting llama-insight.backend=cpu

# CUDA build
uv build --config-setting llama-insight.backend=cuda

# Metal build (macOS)
uv build --config-setting llama-insight.backend=metal
```

## Extra CMake Flags

Use `llama-insight.extra-flags` for additional `-D` flags. This value is parsed with shell-style quoting.

```bash
uv build --config-setting 'llama-insight.extra-flags=-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS'
```

## Skipping Native Build

If you already built and packaged the shared libraries, you can skip compilation and reuse them:

```bash
uv build --config-setting llama-insight.skip-native-build=true
```

## Dry Run

Prints the build steps without compiling:

```bash
uv build --config-setting llama-insight.dry-run=true
```

## Environment Overrides

These env vars are supported if you prefer them over config settings:
- `LLAMA_INSIGHT_BACKEND`
- `LLAMA_INSIGHT_EXTRA_CMAKE_FLAGS`
- `LLAMA_INSIGHT_SKIP_NATIVE_BUILD`
- `LLAMA_INSIGHT_DRY_RUN`
- `LLAMA_INSIGHT_JOBS` or `JOBS`

## Outputs

Packaged shared libraries are copied to:
- `wrapper_src/llama_insight/libs/`

Build metadata is written to:
- `wrapper_src/llama_insight/libs/build-metadata.json`

## Notes

- Use `llama-insight.backend=custom` when you want full control via `extra-flags`.
- For device-specific wheels in CI, always set `llama-insight.backend` explicitly.
