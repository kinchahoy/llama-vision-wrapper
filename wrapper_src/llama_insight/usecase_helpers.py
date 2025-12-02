"""Utilities that keep the runnable examples small and consistent."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from array import array
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import Config, Timer, download_models, timed_operation
from .core import LlamaBackend, ModelLoader, MultimodalProcessor, TextGenerator

DEFAULT_IMAGE_PROMPT = "USER: Describe this image.\n<__image__>\nASSISTANT:"


def _time_block(
    name: str, timer: Timer | None = None, tokens: int | None = None
):  # pragma: no cover - trivial passthrough
    """Use the shared Timer when provided, otherwise fall back to bare timing."""
    return (
        timer.time_operation(name, tokens) if timer else timed_operation(name, tokens)
    )


@dataclass
class LoadedImage:
    """Simple container for the decoded bitmap alongside its source path."""

    path: str
    bitmap: object


@dataclass
class EmbeddingResult:
    """Container for an embedding vector and its shape (tokens x dim)."""

    vector: List[float]
    shape: tuple[int, int]
    saved_path: Path | None = None

    def to_numpy(self):
        """Return a NumPy view if available."""
        try:  # Lazy import to avoid hard dependency.
            import numpy as np  # type: ignore
        except Exception:
            return None
        return np.array(self.vector, dtype=np.float32).reshape(self.shape)

    def save(self, path: str | Path, prefer_numpy: bool = True) -> Path:
        """Persist the embedding to disk (prefers .npy, falls back to raw + JSON)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if prefer_numpy:
            arr = self.to_numpy()
            if arr is not None:
                target = path.with_suffix(".npy")
                try:
                    import numpy as np  # type: ignore

                    np.save(target, arr, allow_pickle=False)
                except Exception:
                    pass
                else:
                    self.saved_path = target
                    return target

        raw_path = path.with_suffix(".f32")
        raw_arr = array("f", self.vector)
        with raw_path.open("wb") as f:
            raw_arr.tofile(f)
        meta_path = path.with_suffix(".f32.json")
        meta_path.write_text(
            json.dumps(
                {"shape": list(self.shape), "dtype": "float32", "path": str(raw_path)}
            )
        )
        self.saved_path = raw_path
        return raw_path


@dataclass
class UsecaseRuntime:
    """Holds the shared llama.cpp handles used by the examples."""

    config: Config
    backend: LlamaBackend
    loader: ModelLoader
    processor: MultimodalProcessor
    generator: TextGenerator
    model: object
    ctx: object
    ctx_mtmd: object
    ctx_size: int

    def create_sampler(self):
        """Create a sampler tuned to the current config."""
        return self.generator.create_sampler(
            self.model,
            self.config.temp,
            self.config.top_k,
            self.config.top_p,
            self.config.repeat_penalty,
            self.ctx_size,
        )

    def load_images(
        self, image_paths: Sequence[str | Path], *, timer: Timer | None = None
    ) -> List[LoadedImage]:
        """Load images and report their dimensions."""
        images: List[LoadedImage] = []
        for path in image_paths:
            path_str = str(path)
            with _time_block(f"Image loading: {path_str}", timer):
                bitmap = self.processor.load_image(self.ctx_mtmd, path_str)
            print(
                f"Loaded {path_str} "
                f"({self.backend.gbl.mtmd_bitmap_get_nx(bitmap)}x{self.backend.gbl.mtmd_bitmap_get_ny(bitmap)})"
            )
            images.append(LoadedImage(path=path_str, bitmap=bitmap))
        return images

    def tokenize_prompt(
        self,
        prompt: str,
        images: Sequence[LoadedImage] | Sequence[object],
        *,
        timer: Timer | None = None,
    ):
        """Tokenize a multimodal prompt while tracking time."""
        bitmaps = [
            img.bitmap if isinstance(img, LoadedImage) else img for img in images
        ]
        with _time_block("Prompt tokenization", timer):
            return self.processor.tokenize_prompt(self.ctx_mtmd, prompt, bitmaps)

    def process_chunks(
        self, chunks, *, timer: Timer | None = None, n_batch: int | None = None
    ) -> int:
        """Process text/media chunks into the KV cache."""
        with _time_block("Input processing", timer):
            return self.processor.process_chunks(
                self.ctx, self.ctx_mtmd, chunks, n_batch or self.config.n_batch
            )

    def generate_text(
        self,
        prompt: str,
        images: Sequence[LoadedImage] | Sequence[object],
        *,
        timer: Timer | None = None,
        max_new_tokens: int | None = None,
    ):
        """Tokenize, process, and generate text for a prompt with images."""
        chunks = self.tokenize_prompt(prompt, images, timer=timer)
        n_past = self.process_chunks(chunks, timer=timer)
        sampler = self.create_sampler()
        max_tokens = max_new_tokens or self.config.max_new_tokens
        with _time_block("Text generation", timer, tokens=max_tokens):
            return self.generator.generate(
                sampler, self.ctx, self.model, n_past, self.ctx_size, max_tokens
            )

    def extract_embedding(self, chunk) -> "EmbeddingResult":
        """Copy the most recent media embedding into Python memory."""
        gbl = self.backend.gbl
        n_tokens = int(gbl.mtmd_input_chunk_get_n_tokens(chunk))
        n_embd = int(gbl.llama_model_n_embd(self.model))
        total = n_tokens * n_embd
        ptr = gbl.mtmd_get_output_embd(self.ctx_mtmd)
        vector = [float(ptr[i]) for i in range(total)]
        return EmbeddingResult(vector=vector, shape=(n_tokens, n_embd))


@contextmanager
def start_session(
    config: Config,
    *,
    n_parallel: int = 1,
    ctx_multiplier: int | None = None,
    timer: Timer | None = None,
    record_downloads: bool = True,
) -> Iterable[UsecaseRuntime]:
    """Download weights, start a backend, and yield a ready-to-use runtime."""
    download_ctx = (
        _time_block("Model download", timer) if record_downloads else nullcontext()
    )
    with download_ctx:
        model_path, mmproj_path = download_models(config)

    ctx_size = config.n_ctx * (ctx_multiplier or max(1, n_parallel))
    with LlamaBackend() as backend:
        loader = ModelLoader(backend.gbl)
        processor = MultimodalProcessor(backend.gbl)
        generator = TextGenerator(backend.gbl)

        with _time_block("Model loading", timer):
            model = loader.load_model(model_path, config.n_gpu_layers)
            ctx = loader.create_context(
                model, ctx_size, config.n_batch, config.n_threads, n_parallel
            )
            ctx_mtmd = loader.load_multimodal(
                mmproj_path, model, config.n_gpu_layers > 0, config.n_threads
            )

        yield UsecaseRuntime(
            config=config,
            backend=backend,
            loader=loader,
            processor=processor,
            generator=generator,
            model=model,
            ctx=ctx,
            ctx_mtmd=ctx_mtmd,
            ctx_size=ctx_size,
        )


def append_run_record(results_path: str | Path, record: dict) -> None:
    """Append a benchmark record to a JSON file (list-backed)."""
    path = Path(results_path)
    try:
        existing = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    if isinstance(existing, dict):
        existing = [existing]
    elif not isinstance(existing, list):
        existing = []

    existing.append(record)
    path.write_text(json.dumps(existing, indent=2))


def environment_metadata() -> dict:
    """Collect lightweight host information for run logs."""
    return {
        "githash": current_git_hash(),
        "hostname": socket.gethostname(),
        "command": sys.argv,
        "recorded_at": datetime.now().isoformat(),
    }


def current_git_hash() -> str | None:
    """Return the current git revision if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def snapshot_config(config: Config) -> dict:
    """Serialize the current Config for logging."""
    return asdict(config)


def build_run_record(
    *,
    version: str,
    config: Config,
    timer: Timer,
    extra: dict | None = None,
    tokens_generated: int | None = None,
    throughput: float | None = None,
) -> dict:
    """Create a standardized run record that can be persisted."""
    record = {
        "version": version,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": snapshot_config(config),
        "timings": timer.timings,
    }
    if tokens_generated is not None:
        record["total_tokens"] = int(tokens_generated)
    if throughput is not None:
        record["throughput"] = throughput
    if extra:
        record.update(extra)
    record.update(environment_metadata())
    return record


def save_embedding(embedding: EmbeddingResult, path: str | Path, **kwargs) -> Path:
    """Convenience wrapper around EmbeddingResult.save()."""
    return embedding.save(path, **kwargs)


__all__ = [
    "DEFAULT_IMAGE_PROMPT",
    "EmbeddingResult",
    "LoadedImage",
    "UsecaseRuntime",
    "append_run_record",
    "build_run_record",
    "current_git_hash",
    "environment_metadata",
    "save_embedding",
    "snapshot_config",
    "start_session",
]
