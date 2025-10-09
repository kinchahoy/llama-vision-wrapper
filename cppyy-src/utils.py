"""Utilities for timing, configuration, and argument parsing."""

import time
import argparse
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Optional, List
from huggingface_hub import hf_hub_download


@dataclass
class Config:
    """Consolidated configuration for inference."""
    # Model config
    repo_id: str = "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"
    model: str = "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"
    mmproj: str = "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"
    
    # Runtime config
    n_ctx: int = 2048
    n_batch: int = 512
    n_threads: int = 8
    n_gpu_layers: int = 0
    verbose_cpp: bool = False
    
    # Sampling config
    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_new_tokens: int = 256

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """Create config from command line arguments."""
        return cls(
            repo_id=getattr(args, 'repo_id', "ggml-org/SmolVLM2-2.2B-Instruct-GGUF"),
            model=getattr(args, 'model', "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf"),
            mmproj=getattr(args, 'mmproj', "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf"),
            n_ctx=getattr(args, 'n_ctx', 2048),
            n_batch=getattr(args, 'n_batch', 512),
            n_gpu_layers=getattr(args, 'n_gpu_layers', 0),
            n_threads=getattr(args, 'n_threads', 8),
            verbose_cpp=getattr(args, 'verbose_cpp', False),
            temp=getattr(args, 'temp', 0.8),
            top_k=getattr(args, 'top_k', 40),
            top_p=getattr(args, 'top_p', 0.95),
            repeat_penalty=getattr(args, 'repeat_penalty', 1.1),
            max_new_tokens=getattr(args, 'max_new_tokens', 256),
        )


def add_common_args(parser: argparse.ArgumentParser):
    """Add common arguments to parser."""
    parser.add_argument("--repo-id", "-hf", default="ggml-org/SmolVLM2-2.2B-Instruct-GGUF", 
                       help="Hugging Face repository ID")
    parser.add_argument("--model", "-m", default="SmolVLM2-2.2B-Instruct-Q4_K_M.gguf", 
                       help="Model filename")
    parser.add_argument("--mmproj", default="mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf", 
                       help="Multimodal projector filename")
    parser.add_argument("--n-gpu-layers", "-ngl", type=int, default=0, dest="n_gpu_layers", 
                       help="GPU layers to offload")
    parser.add_argument("-t", type=int, default=8, dest="n_threads", 
                       help="Number of threads")
    parser.add_argument("--verbose-cpp", action="store_true", 
                       help="Enable C++ verbose logging")
    parser.add_argument("--temp", type=float, default=0.8, 
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, dest="top_k",
                       help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.95, dest="top_p",
                       help="Top-p sampling parameter") 
    parser.add_argument("--repeat-penalty", type=float, default=1.1, dest="repeat_penalty",
                       help="Repetition penalty")
    parser.add_argument("--max-new-tokens", type=int, default=256, 
                       help="Maximum new tokens to generate")
    return parser


class Timer:
    """Simple timing utility."""
    
    def __init__(self):
        self.timings = []
        
    def time_operation(self, name: str, tokens: Optional[int] = None):
        """Context manager for timing operations."""
        return timed_operation(name, tokens, self.timings)
        
    def print_summary(self):
        """Print timing summary."""
        if not self.timings:
            return
        print("\n--- Timing Summary ---")
        total_time = 0
        total_tokens = 0
        for name, duration, tokens in self.timings:
            total_time += duration
            if tokens:
                total_tokens += tokens
                rate = tokens / duration if duration > 0 else float('inf')
                print(f"⏱️ {name}: {duration:.2f}s ({tokens} tokens, {rate:.2f} tok/s)")
            else:
                print(f"⏱️ {name}: {duration:.2f}s")
        
        if total_tokens > 0:
            print(f"\nTotal: {total_time:.2f}s, {total_tokens} tokens, {total_tokens/total_time:.2f} tok/s")


@contextmanager
def timed_operation(name: str, tokens: Optional[int] = None, timings: Optional[List] = None):
    """Time an operation and optionally collect stats."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if timings is not None:
            timings.append((name, duration, tokens))
        
        if tokens:
            rate = tokens / duration if duration > 0 else float('inf')
            print(f"⏱️ {name}: {duration:.2f}s ({tokens} tokens, {rate:.2f} tok/s)")
        else:
            print(f"⏱️ {name}: {duration:.2f}s")


def download_model(repo_id: str, filename: str):
    """Download a model from Hugging Face Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename)


def download_models(config: Config):
    """Download both model and multimodal projector."""
    print("--- Downloading models ---")
    with timed_operation("Model download"):
        model_path = download_model(config.repo_id, config.model)
    with timed_operation("MMPROJ download"):
        mmproj_path = download_model(config.repo_id, config.mmproj)
    
    print(f"Model: {model_path}")
    print(f"MMPROJ: {mmproj_path}")
    return model_path, mmproj_path