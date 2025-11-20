"""Configuration, CLI helpers, and download utilities for llama_insight."""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from huggingface_hub import hf_hub_download


@dataclass
class Config:
    """Unified configuration for llama_insight runtimes."""

    repo_id: str = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
    model: str = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
    mmproj: str = "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"

    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = 8
    n_gpu_layers: int = 0
    verbose_cpp: bool = False

    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_new_tokens: int = 256

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create a config object from argparse args."""
        return cls(
            repo_id=getattr(args, "repo_id", cls.repo_id),
            model=getattr(args, "model", cls.model),
            mmproj=getattr(args, "mmproj", cls.mmproj),
            n_ctx=getattr(args, "n_ctx", cls.n_ctx),
            n_batch=getattr(args, "n_batch", cls.n_batch),
            n_threads=getattr(args, "n_threads", cls.n_threads),
            n_gpu_layers=getattr(args, "n_gpu_layers", cls.n_gpu_layers),
            verbose_cpp=getattr(args, "verbose_cpp", cls.verbose_cpp),
            temp=getattr(args, "temp", cls.temp),
            top_k=getattr(args, "top_k", cls.top_k),
            top_p=getattr(args, "top_p", cls.top_p),
            repeat_penalty=getattr(args, "repeat_penalty", cls.repeat_penalty),
            max_new_tokens=getattr(args, "max_new_tokens", cls.max_new_tokens),
        )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add shared CLI arguments for llama.cpp-backed runners."""
    parser.add_argument(
        "--repo-id",
        "-hf",
        default="ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        help="Hugging Face repository ID containing the GGUF artifacts.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        help="GGUF model filename to download.",
    )
    parser.add_argument(
        "--mmproj",
        default="mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        help="Multimodal projector filename inside the repo.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        "-ngl",
        type=int,
        default=0,
        dest="n_gpu_layers",
        help="Number of layers to offload to GPU.",
    )
    parser.add_argument(
        "-t",
        type=int,
        default=8,
        dest="n_threads",
        help="Inference threads for both llama.cpp and mtmd.",
    )
    parser.add_argument(
        "--verbose-cpp",
        action="store_true",
        help="Enable verbose logging from the native backend.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        dest="top_k",
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        dest="top_p",
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        dest="repeat_penalty",
        help="Penalty applied to repeated tokens.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        dest="max_new_tokens",
        help="Maximum number of tokens to sample.",
    )
    return parser


class Timer:
    """Lightweight timer that collects durations (and optional token counts)."""

    def __init__(self) -> None:
        self.timings: List[Tuple[str, float, Optional[int]]] = []

    def time_operation(self, name: str, tokens: Optional[int] = None):
        """Context manager that records the duration of an operation."""
        return timed_operation(name, tokens, self.timings)

    def print_summary(self) -> None:
        """Pretty-print a summary of recorded timings."""
        if not self.timings:
            return
        print("\n--- Timing Summary ---")
        total_time = 0.0
        total_tokens = 0
        for name, duration, tokens in self.timings:
            total_time += duration
            if tokens:
                total_tokens += tokens
                rate = tokens / duration if duration > 0 else float("inf")
                print(f"⏱️ {name}: {duration:.2f}s ({tokens} tokens, {rate:.2f} tok/s)")
            else:
                print(f"⏱️ {name}: {duration:.2f}s")
        if total_tokens:
            avg_rate = total_tokens / total_time if total_time > 0 else float("inf")
            print(
                f"\nTotal: {total_time:.2f}s, {total_tokens} tokens, {avg_rate:.2f} tok/s"
            )


@contextmanager
def timed_operation(
    name: str,
    tokens: Optional[int] = None,
    timings: Optional[List[Tuple[str, float, Optional[int]]]] = None,
):
    """Context manager used by Timer and ad-hoc call-sites."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if timings is not None:
            timings.append((name, duration, tokens))
        if tokens:
            rate = tokens / duration if duration > 0 else float("inf")
            print(f"⏱️ {name}: {duration:.2f}s ({tokens} tokens, {rate:.2f} tok/s)")
        else:
            print(f"⏱️ {name}: {duration:.2f}s")


def download_model(repo_id: str, filename: str) -> str:
    """Download a single file from Hugging Face Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename)


def download_models(config: Config) -> Sequence[str]:
    """Download both the GGUF model and the multimodal projector."""
    print("--- Downloading models ---")
    with timed_operation("Model download"):
        model_path = download_model(config.repo_id, config.model)
    with timed_operation("MMPROJ download"):
        mmproj_path = download_model(config.repo_id, config.mmproj)
    print(f"Model: {model_path}")
    print(f"MMPROJ: {mmproj_path}")
    return model_path, mmproj_path


__all__ = [
    "Config",
    "Timer",
    "add_common_args",
    "timed_operation",
    "download_model",
    "download_models",
]
