"""
Multimodal generation with benchmarking using the llama_insight package.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from llama_insight import (
        Config,
        LlamaBackend,
        ModelLoader,
        MultimodalProcessor,
        TextGenerator,
        Timer,
        add_common_args,
        download_models,
    )
except ImportError:  # pragma: no cover - dev fallback
    project_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(project_root / "wrapper_src"))
    from llama_insight import (  # type: ignore  # noqa
        Config,
        LlamaBackend,
        ModelLoader,
        MultimodalProcessor,
        TextGenerator,
        Timer,
        add_common_args,
        download_models,
    )


def main():
    """Main execution with benchmarking."""
    parser = argparse.ArgumentParser(
        description="Multimodal generation with benchmarking."
    )
    add_common_args(parser)
    parser.add_argument(
        "--image",
        nargs="+",
        type=str,
        default=["test-images/debug.jpg"],
        help="Path to one or more input images.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="USER: Describe this image.\n<__image__>\nASSISTANT:",
        help="The prompt for the model. Use <__image__> for each image.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmarking mode.",
    )
    parser.add_argument(
        "-fa",
        "--flash-attn",
        action="store_true",
        dest="flash_attn",
        help="Enable flash attention.",
    )
    parser.add_argument(
        "--save-embedding",
        type=str,
        help="Save image embedding to file.",
    )
    parser.add_argument(
        "--load-embedding",
        nargs="+",
        type=str,
        help="Load image embedding from one or more files.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)

    # Initialize benchmarking
    timer = Timer()
    benchmark_results = {
        "version": "cppyy-refactored",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "repo_id": config.repo_id,
            "model": config.model,
            "mmproj": config.mmproj,
            "n_ctx": config.n_ctx,
            "n_batch": config.n_batch,
            "n_threads": config.n_threads,
            "n_gpu_layers": config.n_gpu_layers,
            "verbose_cpp": config.verbose_cpp,
            "temp": config.temp,
            "top_k": config.top_k,
            "top_p": config.top_p,
            "repeat_penalty": config.repeat_penalty,
            "max_new_tokens": config.max_new_tokens,
            "benchmark": getattr(args, "benchmark", False),
            "flash_attn": getattr(args, "flash_attn", False),
        },
        "timings": {},
    }

    try:
        print("=== Multimodal Generation with Benchmarking ===")
        start_time = time.time()

        # Download models
        with timer.time_operation("Model download"):
            model_path, mmproj_path = download_models(config)

        # Initialize backend
        with LlamaBackend() as backend:
            loader = ModelLoader(backend.gbl)
            processor = MultimodalProcessor(backend.gbl)
            generator = TextGenerator(backend.gbl)

            # Load model
            with timer.time_operation("Model loading"):
                model = loader.load_model(model_path, config.n_gpu_layers)
                ctx = loader.create_context(
                    model, config.n_ctx, config.n_batch, config.n_threads
                )
                ctx_mtmd = loader.load_multimodal(
                    mmproj_path, model, config.n_gpu_layers > 0, config.n_threads
                )
                sampler = generator.create_sampler(
                    model,
                    config.temp,
                    config.top_k,
                    config.top_p,
                    config.repeat_penalty,
                )

            # Store model info
            vocab = backend.gbl.llama_model_get_vocab(model)
            benchmark_results["model_info"] = {
                "vocab_size": backend.gbl.llama_vocab_n_tokens(vocab),
                "context_size": backend.gbl.llama_n_ctx(ctx),
                "embedding_size": backend.gbl.llama_model_n_embd(model),
            }

            # Load images
            bitmaps = []
            with timer.time_operation("Image loading"):
                for image_path in args.image:
                    bitmap = processor.load_image(ctx_mtmd, image_path)
                    print(
                        f"Loaded: {image_path} ({backend.gbl.mtmd_bitmap_get_nx(bitmap)}x{backend.gbl.mtmd_bitmap_get_ny(bitmap)})"
                    )
                    bitmaps.append(bitmap)

            # Process multimodal input
            with timer.time_operation("Prompt tokenization"):
                chunks = processor.tokenize_prompt(ctx_mtmd, args.prompt, bitmaps)

            with timer.time_operation("Input processing"):
                n_past = processor.process_chunks(ctx, ctx_mtmd, chunks, config.n_batch)

            # Generate response
            print(f"\n=== Generating Response ({config.max_new_tokens} tokens max) ===")
            print(f"{args.prompt}", end="", flush=True)

            with timer.time_operation("Text generation", config.max_new_tokens):
                result = generator.generate(
                    sampler, ctx, model, n_past, config.n_ctx, config.max_new_tokens
                )

            # Print results
            print(f"{result.generated_text}")

            # Calculate benchmarks
            total_time = time.time() - start_time
            benchmark_results.update({
                "total_time": total_time,
                "total_tokens_generated": int(result.total_tokens_generated),
                "final_output": str(result.generated_text),
                "token_generation_rate": float(
                    result.total_tokens_generated / timer.timings[-1][1]
                    if timer.timings
                    else 0
                ),
            })

            print(f"\n=== Results ===")
            print(f"Tokens generated: {result.total_tokens_generated}")
            print(f"Final KV position: {result.final_n_past}")
            print(f"Total time: {total_time:.2f}s")

        # Print timing summary
        timer.print_summary()

        # Save benchmark results
        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print("\nBenchmark results saved to benchmark_results.json")

    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
