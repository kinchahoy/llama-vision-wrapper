"""
Batched multimodal generation example using llama_insight.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
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
except ImportError:  # pragma: no cover - local editable install fallback
    project_root = Path(__file__).resolve().parents[2]
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


DEFAULT_PROMPTS = [
    "USER: Describe this image.\n<__image__>\nASSISTANT:",
    "USER: What colors are in the image.\n<__image__>\nASSISTANT:",
    "USER: What text is in the image, if any?\n<__image__>\nASSISTANT:",
    "USER: How many people are in the image.\n<__image__>\nASSISTANT:",
    "USER: What is the persons name?\n<__image__>\nASSISTANT:",
    "USER: Describe in french.\n<__image__>\nASSISTANT:",
    "USER: Where is this image taken.\n<__image__>\nASSISTANT:",
    "USER: Describe this image.\n<__image__>\nASSISTANT:",
]


@dataclass
class SequenceState:
    """Track the state of a single batched request."""

    index: int
    image_path: str
    prompt: str
    sampler: object
    seq_ids: object
    n_past: int
    i_batch: int
    tokens_generated: int = 0
    finished: bool = False
    pieces: list[str] = field(default_factory=list)

    @property
    def generated_text(self) -> str:
        return "".join(self.pieces)


def prepare_sequences(
    backend,
    ctx,
    ctx_mtmd,
    model,
    processor,
    generator,
    config: Config,
    images: list[tuple[str, object]],
    n_parallel: int,
    prompts: list[str],
) -> list[SequenceState]:
    """Prepare each batched sequence by reusing the same input image/prompt."""
    prepared: list[dict[str, object]] = []

    for idx in range(n_parallel):
        image_path, bitmap = images[idx % len(images)]
        prompt_text = prompts[idx % len(prompts)]
        chunks = processor.tokenize_prompt(ctx_mtmd, prompt_text, bitmap)
        n_chunks = int(backend.gbl.mtmd_input_chunks_size(chunks))
        seq_id = backend.gbl.llama_seq_id(idx)
        seq_ids = backend.gbl.std.vector[backend.gbl.llama_seq_id]([seq_id])
        sampler = generator.create_sampler(
            model,
            config.temp,
            config.top_k,
            config.top_p,
            config.repeat_penalty,
            config.n_ctx,
        )
        prepared.append(
            {
                "index": idx,
                "image_path": image_path,
                "prompt": prompt_text,
                "chunks": chunks,
                "n_chunks": n_chunks,
                "seq_id": seq_id,
                "seq_ids": seq_ids,
                "sampler": sampler,
                "n_past": 0,
                "last_logits_index": 0,
            }
        )

    max_chunks = max((prep["n_chunks"] for prep in prepared), default=0)
    for chunk_idx in range(max_chunks):
        media_jobs: list[tuple[dict[str, object], object, bool]] = []
        for prep in prepared:
            if chunk_idx >= prep["n_chunks"]:
                continue
            chunk = backend.gbl.mtmd_input_chunks_get(prep["chunks"], chunk_idx)
            chunk_type = backend.gbl.mtmd_input_chunk_get_type(chunk)
            is_last_chunk = chunk_idx == prep["n_chunks"] - 1
            if chunk_type == backend.gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
                prep["n_past"] = int(
                    processor._eval_text_chunk(  # type: ignore[attr-defined]
                        ctx,
                        chunk,
                        prep["n_past"],
                        config.n_batch,
                        logits_for_last=is_last_chunk,
                        seq_identifier=prep["seq_id"],
                    )
                )
                if is_last_chunk:
                    prep["last_logits_index"] = int(
                        getattr(processor, "last_logits_index", 0)
                    )
            elif chunk_type in (
                backend.gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
                backend.gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
            ):
                media_jobs.append((prep, chunk, is_last_chunk))
            else:
                raise RuntimeError(f"Unsupported chunk type: {chunk_type}")

        if media_jobs:
            chunk_ptrs = [job[1] for job in media_jobs]
            n_pasts = [int(job[0]["n_past"]) for job in media_jobs]
            seq_ids = [job[0]["seq_id"] for job in media_jobs]
            new_positions = processor.decode_media_chunks_batched(
                ctx,
                ctx_mtmd,
                chunk_ptrs,
                n_pasts,
                seq_ids,
                config.n_batch,
            )
            for (prep, _, is_last), new_pos in zip(media_jobs, new_positions):
                prep["n_past"] = new_pos
                if is_last:
                    prep["last_logits_index"] = 0

    sequences: list[SequenceState] = []
    for prep in prepared:
        sequences.append(
            SequenceState(
                index=prep["index"],
                image_path=prep["image_path"],
                prompt=prep["prompt"],
                sampler=prep["sampler"],
                seq_ids=prep["seq_ids"],
                n_past=int(prep["n_past"]),
                i_batch=int(prep["last_logits_index"]),
            )
        )

    return sequences


def batched_generate(backend, ctx, model, sequences, max_new_tokens: int) -> int:
    """Run a decode loop that advances all active sequences together."""
    if not sequences:
        return 0

    gbl = backend.gbl
    batch = gbl.llama_batch_init(max(1, len(sequences)), 0, len(sequences))
    vocab = gbl.llama_model_get_vocab(model)
    total_tokens = 0

    try:
        while True:
            gbl.common_batch_clear(batch)
            tokens_in_batch = 0

            for seq in sequences:
                if seq.finished:
                    continue
                if seq.tokens_generated >= max_new_tokens:
                    seq.finished = True
                    continue

                token_id = gbl.common_sampler_sample(seq.sampler, ctx, seq.i_batch)

                if gbl.llama_vocab_is_eog(vocab, token_id):
                    seq.finished = True
                    continue

                gbl.common_sampler_accept(seq.sampler, token_id, True)
                seq.pieces.append(str(gbl.common_token_to_piece(ctx, token_id)))
                seq.tokens_generated += 1

                seq_position = seq.n_past
                seq.i_batch = batch.n_tokens
                gbl.common_batch_add(batch, token_id, seq_position, seq.seq_ids, True)
                seq.n_past += 1
                tokens_in_batch += 1

            if tokens_in_batch == 0:
                break

            if gbl.llama_decode(ctx, batch) != 0:
                raise RuntimeError("llama_decode() failed during batched decoding")

            total_tokens += tokens_in_batch

    finally:
        gbl.llama_batch_free(batch)

    return total_tokens


def main():
    """CLI entry point for batched multimodal generation."""
    parser = argparse.ArgumentParser(
        description="Batched multimodal generation over multiple image queries."
    )
    add_common_args(parser)
    parser.add_argument(
        "--image",
        nargs="+",
        type=str,
        default=["test-images/debug.jpg"],
        help="Path(s) to input images. Reused cyclically if fewer than --n-parallel.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="USER: Describe this image.\n<__image__>\nASSISTANT:",
        help="Prompt template applied to every batched request.",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Optional list of prompts to cycle through; defaults to built-in variants (first/last identical).",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=8,
        dest="n_parallel",
        help="Number of parallel image queries to process.",
    )
    args = parser.parse_args()

    if args.n_parallel <= 0:
        parser.error("--n-parallel must be a positive integer")

    config = Config.from_args(args)
    prompt_variants = (
        args.prompts
        if args.prompts
        else ([args.prompt] if "--prompt" in sys.argv else DEFAULT_PROMPTS)
    )
    if not prompt_variants:
        prompt_variants = DEFAULT_PROMPTS
    timer = Timer()

    try:
        print("=== Batched multimodal generation ===")
        print(f"Parallel queries: {args.n_parallel}")

        with timer.time_operation("Model download"):
            model_path, mmproj_path = download_models(config)

        with LlamaBackend() as backend:
            loader = ModelLoader(backend.gbl)
            processor = MultimodalProcessor(backend.gbl)
            generator = TextGenerator(backend.gbl)

            with timer.time_operation("Model initialization"):
                model = loader.load_model(model_path, config.n_gpu_layers)
                batched_n_ctx = config.n_ctx * max(1, args.n_parallel)
                ctx = loader.create_context(
                    model,
                    batched_n_ctx,
                    config.n_batch,
                    config.n_threads,
                    args.n_parallel,
                )
                ctx_mtmd = loader.load_multimodal(
                    mmproj_path, model, config.n_gpu_layers > 0, config.n_threads
                )

            loaded_images: list[tuple[str, object]] = []
            with timer.time_operation("Image loading"):
                for image_path in args.image:
                    bitmap = processor.load_image(ctx_mtmd, image_path)
                    print(
                        f"Loaded {image_path} "
                        f"({backend.gbl.mtmd_bitmap_get_nx(bitmap)}x{backend.gbl.mtmd_bitmap_get_ny(bitmap)})"
                    )
                    loaded_images.append((image_path, bitmap))

            if not loaded_images:
                raise RuntimeError("No images were loaded")

            with timer.time_operation("Prompt encoding"):
                sequences = prepare_sequences(
                    backend,
                    ctx,
                    ctx_mtmd,
                    model,
                    processor,
                    generator,
                    config,
                    loaded_images,
                    args.n_parallel,
                    prompt_variants,
                )

            print("\n--- Generating batched responses ---")
            gen_start = time.time()
            total_tokens = batched_generate(
                backend, ctx, model, sequences, config.max_new_tokens
            )
            gen_duration = time.time() - gen_start
            throughput = (
                total_tokens / gen_duration if gen_duration > 0 else float("inf")
            )
            print(
                f"⏱️ Batched generation: {gen_duration:.2f}s "
                f"({total_tokens} tokens, {throughput:.2f} tok/s)"
            )
            timer.timings.append(("Batched generation", gen_duration, total_tokens))

            for seq in sequences:
                status = "finished" if seq.finished else "stopped"
                print(f"\n--- Sequence {seq.index} ({seq.image_path}) [{status}] ---")
                print(seq.prompt)
                print(seq.generated_text)
                print(f"Tokens: {seq.tokens_generated}, Final n_past: {seq.n_past}")

    except Exception as exc:  # pragma: no cover - runtime guard
        print("\n=== ERROR ===")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    timer.print_summary()
    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
