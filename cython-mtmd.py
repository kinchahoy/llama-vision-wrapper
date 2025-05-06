"""Multimodal generation using mtmd library with Gemma-3 model and image input (Cython version)."""

import os
import time
import sys
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager

# Configuration
BASE_DIR = "/Users/raistlin/code/llama-mtmd-py"
LLAMA_CPP_DIR = f"{BASE_DIR}/llama.cpp"
MODEL_PATH = "/Users/raistlin/models/gguf/gemma-3-4b-it-q4_0.gguf"
MMPROJ_PATH = "/Users/raistlin/models/gguf/mmproj-model-f16-4B.gguf"
IMAGE_PATH = f"{BASE_DIR}/test.jpg"
PROMPT = "USER: Describe this image.\n<__image__>\nASSISTANT:"

# Runtime parameters
N_CTX = 2048
N_THREADS = 8
N_GPU_LAYERS = 0
N_BATCH = 512
MAX_NEW_TOKENS = 256

# Sampling parameters
TEMP = 0.8
TOP_K = 40
TOP_P = 0.95
REPEAT_PENALTY = 1.1

# Global list to collect timing statistics
all_timings = []

@dataclass
class TimingStats:
    """Stores timing statistics for a processing step."""
    name: str
    duration: float
    tokens: Optional[int] = None
    
    def __str__(self) -> str:
        """Format timing statistics as a string."""
        result = f"{self.name}: {self.duration:.2f}s"
        if self.tokens is not None:
            tokens_per_sec = self.tokens / self.duration if self.duration > 0 else float('inf')
            result += f" ({self.tokens} tokens, {tokens_per_sec:.2f} tokens/s)"
        return result

@contextmanager
def timed_operation(name: str, tokens: Optional[int] = None):
    """Context manager to time an operation and return statistics."""
    start_time = time.time()
    timing_ctx = TimingStats(name=name, duration=0, tokens=tokens)
    try:
        yield timing_ctx
    finally:
        timing_ctx.duration = time.time() - start_time
        all_timings.append(timing_ctx)
        print(f"⏱️ {timing_ctx}")

# Import our Cython module
try:
    import pyximport
    pyximport.install()
    from llama_mtmd_wrapper import (
        initialize_backend, cleanup_backend,
        load_model, create_context, load_mtmd_context, create_sampler,
        load_image, tokenize_input, evaluate_input, generate_tokens
    )
except ImportError as e:
    print(f"Error importing Cython module: {e}")
    print("Make sure you've built the Cython extension with:")
    print("python setup.py build_ext --inplace")
    sys.exit(1)

# Main execution
try:
    print("--- Initializing ---")
    
    with timed_operation("Backend initialization"):
        initialize_backend()
    
    # Load model and create contexts
    with timed_operation("Model loading"):
        model = load_model(MODEL_PATH, N_GPU_LAYERS)
    
    with timed_operation("Context creation"):
        ctx = create_context(model, N_CTX, N_BATCH, N_THREADS)
    
    with timed_operation("Multimodal projector loading"):
        ctx_mtmd = load_mtmd_context(MMPROJ_PATH, model, N_GPU_LAYERS > 0, N_THREADS)
    
    with timed_operation("Sampler initialization"):
        sampler = create_sampler(model, TEMP, TOP_K, TOP_P, REPEAT_PENALTY, N_CTX)
    
    # Load image
    print("Loading image...")
    with timed_operation("Image loading"):
        bitmap_info = load_image(IMAGE_PATH)
        print(f"Image loaded: {bitmap_info['width']}x{bitmap_info['height']}")
    
    # Prepare and evaluate multimodal input
    print("Evaluating multimodal input...")
    
    # Tokenize input
    with timed_operation("Tokenization"):
        chunks_info = tokenize_input(ctx_mtmd, PROMPT, bitmap_info['handle'])
    
    # Process prompt tokens
    prompt_tokens = chunks_info['n_tokens']
    with timed_operation("Prompt evaluation", tokens=prompt_tokens):
        n_past = evaluate_input(ctx_mtmd, ctx, chunks_info['handle'], N_BATCH)
    
    print(f"KV cache position (n_past): {n_past}")
    
    # Generate response
    print(f"\n--- Generating Response ({MAX_NEW_TOKENS} tokens max) ---")
    print(f"{PROMPT}", end="", flush=True)
    
    # Generate tokens
    with timed_operation("Token generation") as timing_ctx:
        result = generate_tokens(sampler, ctx, model, n_past, N_CTX, MAX_NEW_TOKENS)
        
        # Print results
        print(f"{result['text']}")
        
        # Update timing with token count
        timing_ctx.tokens = result['n_tokens']
    
    print(f"Final KV cache position (n_past): {result['final_n_past']}")
    
    # Cleanup
    with timed_operation("Backend cleanup"):
        cleanup_backend()

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"{type(e).__name__}: {e}")
    sys.exit(1)

print("\n--- End of script ---")
print("\nSummary of timing statistics:")
print("=" * 50)
for i, stat in enumerate(all_timings, 1):
    print(f"{i}. {stat}")
print("=" * 50)
print(f"Total operations: {len(all_timings)}")
total_time = sum(stat.duration for stat in all_timings)
print(f"Total measured time: {total_time:.2f}s")
