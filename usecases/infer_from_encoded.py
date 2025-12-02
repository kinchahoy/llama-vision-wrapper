"""
Stage 1 demo: cache media embeddings on disk and reuse them for faster inference.

What this script does today (KV caching comes in a later stage):
1) Run two cold passes (Prompt A + Image A, Prompt B + Image B) – full encode + generate.
   After each run, persist the encoded media embedding to disk and memoize tokenization.
2) Re-run every prompt/image permutation in three modes so you can compare timings:
   a) Cached prompt + cached media embedding (best-case).
   b) Fresh prompt tokenization + cached media embedding.
   c) Cached prompt tokenization + fresh media encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from _shared import default_image_path, ensure_local_import

ensure_local_import()

from llama_insight import Config, Timer, add_common_args  # noqa: E402
from llama_insight.usecase_helpers import DEFAULT_IMAGE_PROMPT, start_session  # noqa: E402


DEFAULT_ALT_PROMPT = (
    "USER: Give a brief, specific summary for the scene in this photo.\n"
    "<__image__>\nASSISTANT:"
)


@dataclass
class RunSpec:
    """Lightweight container tying a prompt to an image with friendly labels."""

    prompt_label: str
    prompt: str
    image_label: str
    image: Path


@dataclass
class CachedPrompt:
    """Memoized prompt tokenization paired with the source images."""

    chunks: object
    images: list


class PromptCache:
    """Tiny cache to reuse tokenized prompts when requested."""

    def __init__(self) -> None:
        self._cache: dict[str, CachedPrompt] = {}

    @staticmethod
    def _key(prompt: str, images: Iterable[Path]) -> str:
        resolved = [str(Path(img).resolve()) for img in images]
        return f"{prompt}||{'|'.join(resolved)}"

    def get(self, prompt: str, images: Iterable[Path]) -> CachedPrompt | None:
        return self._cache.get(self._key(prompt, images))

    def store(
        self,
        prompt: str,
        source_images: Iterable[Path],
        chunks: object,
        cached_images: list,
    ) -> CachedPrompt:
        cache_entry = CachedPrompt(chunks=chunks, images=list(cached_images))
        self._cache[self._key(prompt, source_images)] = cache_entry
        return cache_entry


def require_placeholder(prompt: str) -> None:
    """Ensure we have an image placeholder so tokenization yields a media chunk."""
    if "<__image__>" not in prompt:
        raise ValueError("Prompt must include the <__image__> placeholder.")


def cache_key_for_path(path: Path) -> str:
    """Stable cache key for a source file (stem + short hash of absolute path)."""
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{path.stem}-{digest}"


def media_cache_path(base_dir: Path, source: Path) -> Path:
    """Path to the cached media embedding payload."""
    return base_dir / "media" / f"{cache_key_for_path(source)}.bin"


def media_meta_path(bin_path: Path) -> Path:
    """Path to the JSON sidecar for a cached media embedding."""
    return bin_path.with_suffix(".json")


def reset_kv_cache(gbl, ctx) -> None:
    """Clear KV cache between runs so sequence positions restart from 0."""
    try:
        mem = gbl.llama_get_memory(ctx)
        gbl.llama_memory_clear(mem, True)
        print("KV cache cleared for new run via llama_memory_clear.")
    except Exception as exc:  # pragma: no cover - defensive fallback
        try:
            mem = gbl.llama_get_memory(ctx)
            if gbl.llama_memory_seq_rm(mem, -1, -1, -1):
                print("KV cache cleared for new run via llama_memory_seq_rm.")
                return
        except Exception:
            pass
        print(f"Warning: unable to clear KV cache cleanly ({exc})")


def _chunk_label(gbl, chunk) -> str:
    """Return a short label for a chunk describing type and shape."""
    chunk_type = gbl.mtmd_input_chunk_get_type(chunk)
    n_tokens = int(gbl.mtmd_input_chunk_get_n_tokens(chunk))
    if chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
        return f"text ({n_tokens} tokens)"
    if chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE:
        image_tokens = gbl.mtmd_input_chunk_get_tokens_image(chunk)
        nx = gbl.mtmd_image_tokens_get_nx(image_tokens) if image_tokens else 0
        ny = gbl.mtmd_image_tokens_get_ny(image_tokens) if image_tokens else 0
        return f"image ({nx}x{ny}, {n_tokens} tokens)"
    if chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO:
        return f"audio ({n_tokens} tokens)"
    return f"unknown ({n_tokens} tokens)"


def _debug_dump_chunks(gbl, chunks) -> None:
    """Print a one-line summary for each chunk in a batch."""
    n_chunks = gbl.mtmd_input_chunks_size(chunks)
    print(f"Chunk breakdown ({n_chunks} total):")
    for i in range(n_chunks):
        chunk = gbl.mtmd_input_chunks_get(chunks, i)
        print(f"  - [{i}] {_chunk_label(gbl, chunk)}")


def find_media_chunk(gbl, chunks) -> object | None:
    """Locate the first media chunk within a tokenized prompt."""
    for i in range(gbl.mtmd_input_chunks_size(chunks)):
        chunk = gbl.mtmd_input_chunks_get(chunks, i)
        chunk_type = gbl.mtmd_input_chunk_get_type(chunk)
        if chunk_type in (
            gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
            gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
        ):
            return chunk
    return None


def load_cached_media_embedding(runtime, chunk, cache_path: Path):
    """Load a previously saved media embedding and validate its shape."""
    gbl = runtime.backend.gbl
    emb_vec = gbl.std.vector["float"]()
    result = gbl.load_media_embedding(str(cache_path), emb_vec)
    if not result.success:
        print(f"Cache miss (corrupt?): could not load {cache_path}")
        return None

    n_tokens = int(gbl.mtmd_input_chunk_get_n_tokens(chunk))
    n_embd = int(gbl.llama_model_n_embd(runtime.model))
    expected = n_tokens * n_embd
    if emb_vec.size() != expected:
        print(
            f"Cache mismatch for {cache_path} "
            f"(expected {expected} floats, found {emb_vec.size()})"
        )
        return None

    return emb_vec


def save_media_embedding(
    runtime, chunk, cache_path: Path, *, source: Path, timer: Timer
) -> Path | None:
    """Persist the encoded media embedding using the C++ helper."""
    gbl = runtime.backend.gbl
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    n_tokens = int(gbl.mtmd_input_chunk_get_n_tokens(chunk))
    n_embd = int(gbl.llama_model_n_embd(runtime.model))
    image_tokens = gbl.mtmd_input_chunk_get_tokens_image(chunk)
    nx = gbl.mtmd_image_tokens_get_nx(image_tokens) if image_tokens else 0
    ny = gbl.mtmd_image_tokens_get_ny(image_tokens) if image_tokens else 0
    use_mrope = bool(gbl.mtmd_decode_use_mrope(runtime.ctx_mtmd))

    with timer.time_operation(f"Save media embedding ({cache_path.name})"):
        ok = bool(
            gbl.save_media_embedding_c(
                str(cache_path),
                nx,
                ny,
                use_mrope,
                n_tokens,
                n_embd,
                gbl.mtmd_get_output_embd(runtime.ctx_mtmd),
            )
        )

    if not ok:
        print(f"Warning: failed to save embedding to {cache_path}")
        return None

    meta = {
        "image_path": str(source),
        "tokens": n_tokens,
        "embedding_dim": n_embd,
        "nx": nx,
        "ny": ny,
        "use_mrope_pos": use_mrope,
    }
    media_meta_path(cache_path).write_text(json.dumps(meta, indent=2))
    print(f"Cached media embedding -> {cache_path}")
    return cache_path


def process_chunks_with_media_cache(
    runtime, chunks, *, timer: Timer | None = None, media_embedding=None
) -> int:
    """Process text/media chunks, optionally bypassing media encode via cached embedding."""
    gbl = runtime.backend.gbl
    n_past = 0
    seq_identifier = gbl.llama_seq_id(0)
    n_chunks = gbl.mtmd_input_chunks_size(chunks)

    # _debug_dump_chunks(gbl, chunks)
    print(
        f"Begin chunk processing: seq_id={int(seq_identifier)}, "
        f"start n_past={n_past}, cached_media={'yes' if media_embedding else 'no'}"
    )

    for i in range(n_chunks):
        chunk = gbl.mtmd_input_chunks_get(chunks, i)
        chunk_type = gbl.mtmd_input_chunk_get_type(chunk)
        is_last = i == n_chunks - 1
        n_tokens = int(gbl.mtmd_input_chunk_get_n_tokens(chunk))

        if chunk_type == gbl.MTMD_INPUT_CHUNK_TYPE_TEXT:
            print(f"  -> text chunk {i}: tokens={n_tokens}, n_past_in={n_past}")
            n_past = runtime.processor._eval_text_chunk(  # type: ignore[attr-defined]
                runtime.ctx,
                chunk,
                n_past,
                runtime.config.n_batch,
                logits_for_last=is_last,
                seq_identifier=seq_identifier,
            )
            print(f"     text chunk {i}: n_past_out={n_past}")
        elif chunk_type in (
            gbl.MTMD_INPUT_CHUNK_TYPE_IMAGE,
            gbl.MTMD_INPUT_CHUNK_TYPE_AUDIO,
        ):
            label = _chunk_label(gbl, chunk)
            print(
                f"  -> media chunk {i}: {label}, "
                f"n_past_in={n_past}, using_cache={'yes' if media_embedding else 'no'}"
            )
            if media_embedding is None:
                ctx = (
                    timer.time_operation("Media encoding (cache miss)")
                    if timer
                    else nullcontext()
                )
                with ctx:
                    runtime.processor.encode_media_chunk(runtime.ctx_mtmd, chunk)
                n_past = runtime.processor._decode_media_chunk(  # type: ignore[attr-defined]
                    runtime.ctx,
                    runtime.ctx_mtmd,
                    chunk,
                    n_past,
                    runtime.config.n_batch,
                    seq_identifier=seq_identifier,
                )
            else:
                decode_fn = getattr(gbl, "decode_media_chunk_from_embd", None)
                if decode_fn:
                    new_n_past = gbl.std.array[gbl.llama_pos, 1]()
                    new_n_past[0] = gbl.llama_pos(n_past)
                    ret = decode_fn(
                        runtime.ctx_mtmd,
                        runtime.ctx,
                        chunk,
                        media_embedding.data(),
                        n_past,
                        seq_identifier,
                        runtime.config.n_batch,
                        new_n_past.data(),
                    )
                    if ret != 0:
                        raise RuntimeError("Failed to decode cached media chunk")
                    n_past = int(new_n_past[0])
                else:
                    runtime.processor.loaded_embedding = media_embedding
                    n_past = runtime.processor._decode_media_chunk(  # type: ignore[attr-defined]
                        runtime.ctx,
                        runtime.ctx_mtmd,
                        chunk,
                        n_past,
                        runtime.config.n_batch,
                        seq_identifier=seq_identifier,
                    )
                    runtime.processor.loaded_embedding = None
            print(f"     media chunk {i}: n_past_out={n_past}")

    print(
        f"Finished chunk processing: seq_id={int(seq_identifier)}, final n_past={n_past}"
    )
    return n_past


def run_inference(
    runtime,
    spec: RunSpec,
    cache_dir: Path,
    timer: Timer,
    *,
    allow_cached_media: bool,
    prompt_cache: PromptCache | None,
    use_prompt_cache: bool,
) -> None:
    """Single inference pass plus optional cache lookup/save."""
    print(f"\n--- Inference: {spec.prompt_label} + {spec.image_label} ---")
    gbl = runtime.backend.gbl
    if not spec.image.exists():
        print(f"Warning: skipping missing image {spec.image}")
        return

    require_placeholder(spec.prompt)
    reset_kv_cache(gbl, runtime.ctx)

    prompt_cache_entry: CachedPrompt | None = None
    chunks = None
    images = None
    if prompt_cache and use_prompt_cache:
        with timer.time_operation("Prompt cache lookup"):
            prompt_cache_entry = prompt_cache.get(spec.prompt, [spec.image])
        if prompt_cache_entry:
            print(
                f"Reusing cached tokenization: {spec.prompt_label} + {spec.image_label}"
            )
            images = prompt_cache_entry.images
            chunks = prompt_cache_entry.chunks

    if chunks is None:
        images = runtime.load_images([spec.image], timer=timer)
        if not images:
            print(f"Warning: failed to load image {spec.image}")
            return

        chunks = runtime.tokenize_prompt(spec.prompt, images, timer=timer)
        if prompt_cache:
            prompt_cache_entry = prompt_cache.store(
                spec.prompt, [spec.image], chunks, images
            )

    active_images = prompt_cache_entry.images if prompt_cache_entry else images
    if not active_images:
        print(f"Warning: no images available for {spec.image}")
        return
    _ = active_images  # keep references alive for the duration of this run

    media_chunk = find_media_chunk(gbl, chunks)
    if not media_chunk:
        print(f"Warning: no media chunk found for {spec.image}")
        return

    cache_path = media_cache_path(cache_dir, spec.image)
    cached_embedding = None
    if allow_cached_media and cache_path.exists():
        cached_embedding = load_cached_media_embedding(runtime, media_chunk, cache_path)
        if cached_embedding:
            print(f"Reusing cached embedding from {cache_path}")
            n_embd = int(gbl.llama_model_n_embd(runtime.model))
            n_tokens = len(cached_embedding) // n_embd if n_embd else 0
            print(
                f"Cached embedding details: tokens={n_tokens}, "
                f"embedding_dim={n_embd}, cache_file={cache_path.name}"
            )
        else:
            print(f"Cache miss or validation failure for {cache_path}")

    prompt_state = "cached prompt" if prompt_cache_entry else "fresh prompt"
    media_state = "cached media" if cached_embedding else "cold media"
    with timer.time_operation(f"Prompt processing ({prompt_state}, {media_state})"):
        n_past = process_chunks_with_media_cache(
            runtime, chunks, timer=timer, media_embedding=cached_embedding
        )

    with timer.time_operation(
        f"Text generation: {spec.prompt_label} + {spec.image_label}",
        tokens=runtime.config.max_new_tokens,
    ):
        sampler = runtime.create_sampler()
        result = runtime.generator.generate(
            sampler,
            runtime.ctx,
            runtime.model,
            n_past,
            runtime.ctx_size,
            runtime.config.max_new_tokens,
        )

    print(result.generated_text)
    print(
        f"Tokens generated: {result.total_tokens_generated} | Final KV position: {result.final_n_past}"
    )

    if cached_embedding is None:
        save_media_embedding(
            runtime, media_chunk, cache_path, source=spec.image, timer=timer
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate inference using cached media embeddings (stage 1)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--image",
        nargs=2,
        metavar=("IMAGE_A", "IMAGE_B"),
        default=[default_image_path(), "test-images/movie.jpg"],
        help="Two images used for the demo.",
    )
    parser.add_argument(
        "--prompt-a",
        default=DEFAULT_IMAGE_PROMPT,
        help="Prompt text for the first cold run (must include <__image__>).",
    )
    parser.add_argument(
        "--prompt-b",
        default=DEFAULT_ALT_PROMPT,
        help="Prompt text for the second cold run (must include <__image__>).",
    )
    parser.add_argument(
        "--cache-dir",
        default="embeddings/cache",
        help="Directory for saved media embeddings.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)
    timer = Timer()

    images = [Path(p) for p in args.image]
    prompts = [
        ("Prompt A", args.prompt_a),
        ("Prompt B", args.prompt_b),
    ]
    cache_dir = Path(args.cache_dir)

    cold_runs = [
        RunSpec("Prompt A", prompts[0][1], "Image 1", images[0]),
        RunSpec("Prompt B", prompts[1][1], "Image 2", images[1]),
    ]

    permutations: list[RunSpec] = [
        RunSpec(p_label, prompt, f"Image {idx + 1}", img)
        for p_label, prompt in prompts
        for idx, img in enumerate(images)
    ]
    prompt_cache = PromptCache()

    try:
        with start_session(config, timer=timer) as runtime:
            print("\n--- Cold runs (populate cache) ---")
            for run in cold_runs:
                run_inference(
                    runtime,
                    run,
                    cache_dir,
                    timer,
                    allow_cached_media=False,
                    prompt_cache=prompt_cache,
                    use_prompt_cache=False,
                )

            print("\n--- Cached permutations (cached prompt + cached media) ---")
            for run in permutations:
                run_inference(
                    runtime,
                    run,
                    cache_dir,
                    timer,
                    allow_cached_media=True,
                    prompt_cache=prompt_cache,
                    use_prompt_cache=True,
                )

            print("\n--- Fresh prompt + cached media permutations ---")
            for run in permutations:
                run_inference(
                    runtime,
                    run,
                    cache_dir,
                    timer,
                    allow_cached_media=True,
                    prompt_cache=prompt_cache,
                    use_prompt_cache=False,
                )

            print("\n--- Cached prompt + fresh media encode permutations ---")
            for run in permutations:
                run_inference(
                    runtime,
                    run,
                    cache_dir,
                    timer,
                    allow_cached_media=False,
                    prompt_cache=prompt_cache,
                    use_prompt_cache=True,
                )

    except Exception as exc:  # pragma: no cover - runtime guard
        print("\n--- ERROR ---")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    timer.print_summary()
    print(
        "\nStage 1 complete — media embeddings are cached on disk. KV cache reuse comes next."
    )


if __name__ == "__main__":
    main()
