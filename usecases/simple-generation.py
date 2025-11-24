"""
A simple example using the llama_insight package.
"""

import argparse
import sys
from pathlib import Path

try:
    from llama_insight import (
        Config,
        LlamaBackend,
        ModelLoader,
        MultimodalProcessor,
        TextGenerator,
        add_common_args,
        download_models,
        timed_operation,
    )
except ImportError:  # pragma: no cover - local editable install fallback
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root / "wrapper_src"))
    from llama_insight import (  # type: ignore  # noqa
        Config,
        LlamaBackend,
        ModelLoader,
        MultimodalProcessor,
        TextGenerator,
        add_common_args,
        download_models,
        timed_operation,
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Simple multimodal generation example."
    )
    add_common_args(parser)
    parser.add_argument(
        "--image",
        nargs="+",
        type=str,
        default=["debug.jpg"],
        help="Path to one or more input images.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="USER: Describe this image.\n<__image__>\nASSISTANT:",
        help="The prompt for the model. Use <__image__> for each image.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)

    try:
        print("--- Initializing ---")
        print(f"Threads: {config.n_threads}")

        # Download models
        model_path, mmproj_path = download_models(config)

        # Initialize backend and load model
        with LlamaBackend() as backend:
            loader = ModelLoader(backend.gbl)
            processor = MultimodalProcessor(backend.gbl)
            generator = TextGenerator(backend.gbl)

            with timed_operation("Model loading"):
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

            # Load images
            print("--- Loading images ---")
            bitmaps = []
            for image_path in args.image:
                with timed_operation(f"Image loading: {image_path}"):
                    bitmap = processor.load_image(ctx_mtmd, image_path)
                    print(
                        f"Loaded: {image_path} ({backend.gbl.mtmd_bitmap_get_nx(bitmap)}x{backend.gbl.mtmd_bitmap_get_ny(bitmap)})"
                    )
                    bitmaps.append(bitmap)

            # Process multimodal input
            print("--- Processing multimodal input ---")
            with timed_operation("Tokenization"):
                chunks = processor.tokenize_prompt(ctx_mtmd, args.prompt, bitmaps)

            with timed_operation("Input processing"):
                n_past = processor.process_chunks(ctx, ctx_mtmd, chunks, config.n_batch)

            print(f"Final KV cache position: {n_past}")

            # Generate response
            print(f"\n--- Generating Response ({config.max_new_tokens} tokens max) ---")
            print(f"{args.prompt}", end="", flush=True)

            with timed_operation("Text generation", config.max_new_tokens):
                result = generator.generate(
                    sampler, ctx, model, n_past, config.n_ctx, config.max_new_tokens
                )

            # Print results
            print(f"{result.generated_text}")
            print(f"\nTokens generated: {result.total_tokens_generated}")
            print(f"Final KV position: {result.final_n_past}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n--- Complete ---")


if __name__ == "__main__":
    main()
