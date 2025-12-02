"""Simple multimodal generation using the llama_insight package."""

from __future__ import annotations

import argparse
import sys

from _shared import default_image_path, ensure_local_import

ensure_local_import()

from llama_insight import Config, Timer, add_common_args  # noqa: E402
from llama_insight.usecase_helpers import (  # noqa: E402
    DEFAULT_IMAGE_PROMPT,
    start_session,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal multimodal text generation.")
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
        help="Prompt text; include <__image__> for each image.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)
    timer = Timer()

    try:
        with start_session(config, timer=timer) as runtime:
            images = runtime.load_images(args.image, timer=timer)
            print(f"\n--- Generating Response ({config.max_new_tokens} tokens max) ---")
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
    except Exception as exc:  # pragma: no cover - runtime guard
        print("\n--- ERROR ---")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

    timer.print_summary()


if __name__ == "__main__":
    main()
