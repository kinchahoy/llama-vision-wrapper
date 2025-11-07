"""
Example to encode images into media embeddings.
"""
import argparse
import os
import sys
from pathlib import Path

try:
    from llama_insight import (
        Config,
        LlamaBackend,
        ModelLoader,
        MultimodalProcessor,
        add_common_args,
        download_models,
        timed_operation,
    )
except ImportError:  # pragma: no cover - dev fallback
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root / "wrapper_src"))
    from llama_insight import (  # type: ignore  # noqa
        Config,
        LlamaBackend,
        ModelLoader,
        MultimodalProcessor,
        add_common_args,
        download_models,
        timed_operation,
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Encode images into media embeddings."
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        type=str,
        nargs='+',
        help="Path to one or more input images.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=".",
        help="Directory to save embedding files.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)

    try:
        print("--- Initializing ---")
        os.makedirs(args.output_dir, exist_ok=True)

        # Download models
        model_path, mmproj_path = download_models(config)

        # Initialize backend
        with LlamaBackend() as backend:
            loader = ModelLoader(backend.gbl)
            processor = MultimodalProcessor(backend.gbl)
            
            with timed_operation("Model loading"):
                model = loader.load_model(model_path, config.n_gpu_layers)
                ctx_mtmd = loader.load_multimodal(mmproj_path, model, config.n_gpu_layers > 0, config.n_threads)

            # Process each image
            for image_path in args.images:
                print(f"\n--- Processing {image_path} ---")
                if not os.path.exists(image_path):
                    print(f"Warning: File not found: {image_path}")
                    continue

                # Load image
                bitmap = processor.load_image(ctx_mtmd, image_path)
                print(f"Loaded: {backend.gbl.mtmd_bitmap_get_nx(bitmap)}x{backend.gbl.mtmd_bitmap_get_ny(bitmap)}")

                # Tokenize with dummy prompt
                chunks = processor.tokenize_prompt(ctx_mtmd, "<__image__>", bitmap)

                # Find media chunk
                image_chunk = None
                for i in range(backend.gbl.mtmd_input_chunks_size(chunks)):
                    chunk = backend.gbl.mtmd_input_chunks_get(chunks, i)
                    chunk_type = backend.gbl.mtmd_input_chunk_get_type(chunk)
                    if chunk_type in [backend.gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE, backend.gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO]:
                        image_chunk = chunk
                        break

                if not image_chunk:
                    print(f"Warning: Could not create media chunk for {image_path}")
                    continue

                # Encode the chunk
                with timed_operation("Media encoding"):
                    processor.encode_media_chunk(ctx_mtmd, image_chunk)

                # Save embedding
                base_name = os.path.basename(image_path)
                file_name, _ = os.path.splitext(base_name)
                output_path = os.path.join(args.output_dir, f"{file_name}.embedding")

                # Note: We'd need to add save functionality to the processor
                print(f"Would save to: {output_path}")
                print(f"Successfully processed {image_path}")

    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)

    print("\n--- All images processed ---")


if __name__ == "__main__":
    main()
