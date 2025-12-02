"""Single-run multimodal generation with optional benchmarking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _shared import default_image_path, ensure_local_import

ensure_local_import()

from llama_insight import Config, Timer, add_common_args  # noqa: E402
from llama_insight.usecase_helpers import (  # noqa: E402
    DEFAULT_IMAGE_PROMPT,
    append_run_record,
    build_run_record,
    start_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multimodal generation with benchmark logging."
    )
    add_common_args(parser)
    parser.add_argument(
        "--image",
        nargs="+",
        default=[default_image_path()],
        help="Path to one or more input images.",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_IMAGE_PROMPT,
        help="Prompt for the model; include <__image__> per image.",
    )
    parser.add_argument(
        "--results-file",
        default="benchmark_results.json",
        help="Path to append benchmark JSON (set --no-record to skip).",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not append the run to benchmark_results.json.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)
    timer = Timer()

    try:
        with start_session(config, timer=timer) as runtime:
            model_info = {
                "vocab_size": runtime.backend.gbl.llama_vocab_n_tokens(
                    runtime.backend.gbl.llama_model_get_vocab(runtime.model)
                ),
                "context_size": runtime.backend.gbl.llama_n_ctx(runtime.ctx),
                "embedding_size": runtime.backend.gbl.llama_model_n_embd(runtime.model),
            }
            images = runtime.load_images(args.image, timer=timer)

            print(f"\n=== Generating ({config.max_new_tokens} tokens max) ===")
            print(f"{args.prompt}", end="", flush=True)
            result = runtime.generate_text(
                prompt=args.prompt,
                images=images,
                timer=timer,
                max_new_tokens=config.max_new_tokens,
            )

        print(result.generated_text)
        print(f"\nTokens generated: {result.total_tokens_generated}")
        print(f"Final KV position: {result.final_n_past}")

        gen_duration = next(
            (
                duration
                for name, duration, _ in timer.timings
                if name == "Text generation"
            ),
            None,
        )
        throughput = (
            result.total_tokens_generated / gen_duration
            if gen_duration and result.total_tokens_generated
            else None
        )

        if not args.no_record:
            record = build_run_record(
                version="generate_benchmark",
                config=config,
                timer=timer,
                tokens_generated=int(result.total_tokens_generated),
                throughput=throughput,
                extra={
                    "final_output": str(result.generated_text),
                    "final_n_past": result.final_n_past,
                    "model_info": model_info,
                },
            )
            append_run_record(Path(args.results_file), record)
            print(f"\nBenchmark results saved to {args.results_file}")

    except Exception as exc:  # pragma: no cover - runtime guard
        print("\n=== ERROR ===")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    timer.print_summary()


if __name__ == "__main__":
    main()
