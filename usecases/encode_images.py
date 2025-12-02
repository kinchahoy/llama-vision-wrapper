"""Encode images into media embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _shared import ensure_local_import

ensure_local_import()

from llama_insight import Config, Timer, add_common_args  # noqa: E402
from llama_insight.usecase_helpers import (  # noqa: E402
    EmbeddingResult,
    save_embedding,
    start_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode images into media embeddings.")
    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="Path to one or more input images.",
    )
    add_common_args(parser)
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Directory to save embedding files.",
    )
    parser.add_argument(
        "--format",
        choices=["npy", "f32"],
        default="npy",
        help="Embedding format to write; npy uses numpy when available, f32 is raw float32 + JSON meta.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Keep embeddings in memory only (do not write to disk).",
    )
    args = parser.parse_args()
    config = Config.from_args(args)
    timer = Timer()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, EmbeddingResult] = {}

    try:
        with start_session(config, timer=timer) as runtime:
            gbl = runtime.backend.gbl

            for image_path in args.images:
                path = Path(image_path)
                print(f"\n--- Processing {path} ---")
                if not path.exists():
                    print(f"Warning: File not found: {path}")
                    continue

                images = runtime.load_images([path], timer=timer)
                if not images:
                    continue

                chunks = runtime.tokenize_prompt("<__image__>", images, timer=timer)
                media_chunk = None
                for i in range(gbl.mtmd_input_chunks_size(chunks)):
                    chunk = gbl.mtmd_input_chunks_get(chunks, i)
                    chunk_type = gbl.mtmd_input_chunk_get_type(chunk)
                    if chunk_type in (
                        gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                        gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    ):
                        media_chunk = chunk
                        break

                if not media_chunk:
                    print(f"Warning: Could not create media chunk for {path}")
                    continue

                with timer.time_operation("Media encoding"):
                    runtime.processor.encode_media_chunk(runtime.ctx_mtmd, media_chunk)

                embedding = runtime.extract_embedding(media_chunk)
                embeddings[str(path)] = embedding

                if args.no_save:
                    print(
                        f"Stored embedding in memory only ({embedding.shape[0]} tokens x {embedding.shape[1]} dim)"
                    )
                    continue

                target = output_dir / path.stem
                prefer_numpy = args.format == "npy"
                saved_path = save_embedding(
                    embedding, target, prefer_numpy=prefer_numpy
                )
                print(
                    f"Saved embedding ({embedding.shape[0]} tokens x {embedding.shape[1]} dim) to {saved_path}"
                )
                if args.format == "npy" and saved_path.suffix != ".npy":
                    print(
                        "Note: numpy unavailable; fell back to raw float32 + JSON meta."
                    )

    except Exception as exc:  # pragma: no cover - runtime guard
        print("\n--- ERROR ---")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    timer.print_summary()
    print(f"\n--- All images processed ({len(embeddings)} stored) ---")


if __name__ == "__main__":
    main()
