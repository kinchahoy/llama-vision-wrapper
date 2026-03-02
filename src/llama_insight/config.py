"""Configuration, CLI helpers, and download utilities for llama_insight."""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from huggingface_hub import HfApi, hf_hub_download


@dataclass
class Config:
    """Unified configuration for llama_insight runtimes."""

    repo_id: str = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
    model: str = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
    mmproj: str = "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"
    mmproj_dtype: str = "f16"
    image_min_tokens: int = -1

    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = 8
    n_gpu_layers: int = 0
    verbose_cpp: bool = False

    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_new_tokens: int = 500

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create a config object from argparse args."""
        return cls(
            repo_id=getattr(args, "repo_id", cls.repo_id),
            model=getattr(args, "model", cls.model),
            mmproj=getattr(args, "mmproj", cls.mmproj),
            mmproj_dtype=getattr(args, "mmproj_dtype", cls.mmproj_dtype),
            image_min_tokens=getattr(args, "image_min_tokens", cls.image_min_tokens),
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


DEFAULT_REPO_ID = Config.repo_id
DEFAULT_MODEL_FILE = Config.model
DEFAULT_MMPROJ_FILE = Config.mmproj


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add shared CLI arguments for llama.cpp-backed runners."""
    parser.add_argument(
        "-hf",
        "--hf-repo",
        "--repo-id",
        default="ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        dest="repo_id",
        help=(
            "Hugging Face repository ID containing GGUF artifacts. "
            "Also accepts '<repo>:<quant>' (example: unsloth/Qwen3.5-2B-GGUF:Q4_K_M)."
        ),
    )
    parser.add_argument(
        "-m",
        "--model",
        "--hf-file",
        default="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        dest="model",
        help="Model path or HF filename (matches llama-cli -m).",
    )
    parser.add_argument(
        "--mmproj",
        default="mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
        help="Multimodal projector filename inside the repo.",
    )
    parser.add_argument(
        "--mmproj-dtype",
        choices=("f16", "bf16", "f32", "auto"),
        default="f16",
        dest="mmproj_dtype",
        help=(
            "Preferred dtype for auto-selected mmproj files when using --hf-repo. "
            "Defaults to f16."
        ),
    )
    parser.add_argument(
        "--image-min-tokens",
        type=int,
        default=-1,
        dest="image_min_tokens",
        help=(
            "Minimum image token count for dynamic-resolution vision models "
            "(-1 = model default)."
        ),
    )
    parser.add_argument(
        "-c",
        "--ctx-size",
        type=int,
        default=2048,
        dest="n_ctx",
        help="Context window size.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=512,
        dest="n_batch",
        help="Logical batch size.",
    )
    parser.add_argument(
        "-ngl",
        "--n-gpu-layers",
        type=int,
        default=0,
        dest="n_gpu_layers",
        help="Number of layers to offload to GPU.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=8,
        dest="n_threads",
        help="CPU threads for generation.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
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
        "-n",
        "--predict",
        "--max-new-tokens",
        type=int,
        default=256,
        dest="max_new_tokens",
        help="Number of tokens to generate.",
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


def _split_repo_and_tag(repo_id: str) -> tuple[str, str | None]:
    if ":" not in repo_id:
        return repo_id, None
    repo, tag = repo_id.rsplit(":", 1)
    repo = repo.strip()
    tag = tag.strip()
    if not repo:
        return repo_id, None
    return repo, tag or None


def _list_repo_files(repo_id: str) -> list[str]:
    return HfApi().list_repo_files(repo_id=repo_id, repo_type="model")


def _pick_model_file(candidates: list[str], quant_tag: str | None) -> str:
    if not candidates:
        raise FileNotFoundError("No GGUF model files found in the selected HF repo.")

    if not quant_tag:
        # Prefer common default quant names when no tag is provided.
        preferred = ("Q4_K_M", "Q4_K", "Q5_K_M", "Q8_0")
        for tag in preferred:
            for name in candidates:
                if tag in name.upper():
                    return name
        return sorted(candidates)[0]

    tag_up = quant_tag.upper()
    ranked: list[tuple[int, str]] = []
    for name in candidates:
        up = name.upper()
        score = 0
        if f"-{tag_up}.GGUF" in up or f"_{tag_up}.GGUF" in up:
            score = 3
        elif tag_up in up:
            score = 2
        ranked.append((score, name))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if ranked and ranked[0][0] > 0:
        return ranked[0][1]

    sample = ", ".join(sorted(candidates)[:8])
    raise FileNotFoundError(
        f"No GGUF file matching quant tag '{quant_tag}' found in repo. "
        f"Available GGUF files include: {sample}"
    )


def _mmproj_matches_dtype(filename: str, dtype: str) -> bool:
    up = filename.upper()
    if dtype == "bf16":
        return "BF16" in up
    if dtype == "f16":
        return "F16" in up and "BF16" not in up
    if dtype == "f32":
        return "F32" in up
    return False


def _pick_mmproj_file(
    candidates: list[str], quant_tag: str | None, preferred_dtype: str
) -> str:
    if not candidates:
        raise FileNotFoundError(
            "No mmproj GGUF file found in the selected HF repo. "
            "Camera/multimodal usecases require --mmproj (or a repo that includes it)."
        )
    candidates_sorted = sorted(candidates)

    if preferred_dtype != "auto":
        dtype_matches = [
            name
            for name in candidates_sorted
            if _mmproj_matches_dtype(name, preferred_dtype)
        ]
        if dtype_matches:
            if quant_tag:
                tag_up = quant_tag.upper()
                for name in dtype_matches:
                    if tag_up in name.upper():
                        return name
            return dtype_matches[0]

    if quant_tag:
        tag_up = quant_tag.upper()
        for name in candidates_sorted:
            if tag_up in name.upper():
                return name
    return candidates_sorted[0]


def _resolve_download_plan(config: Config) -> tuple[str, str, str]:
    repo_id, quant_tag = _split_repo_and_tag(config.repo_id)
    model_name = config.model
    mmproj_name = config.mmproj

    use_auto_model = quant_tag is not None and model_name == DEFAULT_MODEL_FILE
    use_auto_mmproj = mmproj_name == DEFAULT_MMPROJ_FILE
    if not use_auto_model and not use_auto_mmproj:
        return repo_id, model_name, mmproj_name

    files = _list_repo_files(repo_id)
    gguf_files = [name for name in files if name.lower().endswith(".gguf")]
    model_candidates = [
        name for name in gguf_files if "mmproj" not in name.lower()
    ]
    mmproj_candidates = [name for name in gguf_files if "mmproj" in name.lower()]

    if use_auto_model:
        model_name = _pick_model_file(model_candidates, quant_tag)
    if use_auto_mmproj:
        mmproj_name = _pick_mmproj_file(
            mmproj_candidates, quant_tag, config.mmproj_dtype
        )
    return repo_id, model_name, mmproj_name


def download_models(config: Config) -> Sequence[str]:
    """Download both the GGUF model and the multimodal projector."""
    repo_id, model_name, mmproj_name = _resolve_download_plan(config)
    print("--- Downloading models ---")
    with timed_operation("Model download"):
        model_path = download_model(repo_id, model_name)
    with timed_operation("MMPROJ download"):
        mmproj_path = download_model(repo_id, mmproj_name)
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
