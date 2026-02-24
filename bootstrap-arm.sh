#!/usr/bin/env bash
# One-time setup for ARM64 Linux (e.g. RK3588 / OrangePi 5 Pro).
# Run this before the first 'uv sync' on a fresh install.
#
# Why: cppyy-cling has no pre-built aarch64 wheel. It must be compiled from
# source (ROOT/Cling, ~1-3 hours). cppyy-backend and cpycppyy pin an older
# cppyy-cling version (6.30.0) in their build metadata which fails with
# Clang 22+ on ARM64. This pre-installs the working version (6.32.8) so that
# pyproject.toml's no-build-isolation-package setting can use it.
#
# After this script succeeds, plain 'uv sync' and 'uv run' work normally.
# If the uv wheel cache is already warm, 'uv sync' may work without this script.
# Run it when setting up a machine with no prior uv cache.

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "This script is only needed on ARM64 Linux. Exiting."
    exit 0
fi

echo "Building cppyy-cling==6.32.8 from source (this takes 1-3 hours)..."
UV_LOCK_TIMEOUT=0 uv pip install "cppyy-cling==6.32.8"

echo ""
echo "Done. Run 'uv sync' to install the rest of the project dependencies."
