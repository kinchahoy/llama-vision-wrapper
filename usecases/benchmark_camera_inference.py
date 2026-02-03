"""Benchmark continuous camera capture and inference with optimization strategies."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from _shared import ensure_local_import

ensure_local_import()

from llama_insight import Config, Timer, add_common_args  # noqa: E402
from llama_insight.usecase_helpers import (  # noqa: E402
    DEFAULT_IMAGE_PROMPT,
    start_session,
)

# Import camera module from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera  # noqa: E402


class PipelineMode(Enum):
    """Different pipeline optimization strategies."""


### TODO Need to figure out the sensible types of pipelining to test knowing that capturing a frame is super fast, encoding is slowish and generation is also slow.
@dataclass
class FrameData:
    """Container for frame and its path."""

    frame_id: int
    image_path: Path
    capture_time: float


@dataclass
class BenchmarkStats:
    """Statistics for pipeline performance."""

    total_frames_captured: int = 0
    total_frames_processed: int = 0
    total_frames_dropped: int = 0
    total_capture_time: float = 0.0
    total_encode_time: float = 0.0
    total_generate_time: float = 0.0
    total_tokens_generated: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def fps(self) -> float:
        """Calculate frames per sec"""


class CameraThread(threading.Thread):
    """Dedicated thread for camera capture."""

    def __init__(
        self,
        camera: Camera,
        output_queue: queue.Queue,
        output_dir: Path,
        max_frames: int,
        stats: BenchmarkStats,
    ):
        super().__init__(daemon=True)
        self.camera = camera
        self.output_queue = output_queue
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.stats = stats
        self.running = False
        self.frame_id = 0

    def run(self) -> None:
        """Continuously capture frames."""
        self.running = True
        print("[Camera] Started capture thread")

        while self.running and self.frame_id < self.max_frames:
            start = time.time()
            image_path = self.output_dir / f"frame_{self.frame_id:06d}.jpg"

            success, _ = self.camera.capture(image_path)
            capture_time = time.time() - start

            if success:
                frame_data = FrameData(
                    frame_id=self.frame_id,
                    image_path=image_path,
                    capture_time=start,
                )
                try:
                    self.output_queue.put(frame_data, block=False)
                    self.stats.total_frames_captured += 1
                    self.stats.total_capture_time += capture_time
                    self.frame_id += 1
                except queue.Full:
                    self.stats.total_frames_dropped += 1
                    image_path.unlink(missing_ok=True)
            else:
                print(f"[Camera] Failed to capture frame {self.frame_id}")

        print(f"[Camera] Stopped after {self.frame_id} frames")

    def stop(self) -> None:
        """Signal thread to stop."""
        self.running = False


def reset_kv_cache(runtime) -> None:
    """Clear KV cache between frames."""
    try:
        gbl = runtime.backend.gbl
        mem = gbl.llama_get_memory(runtime.ctx)
        gbl.llama_memory_clear(mem, True)
    except Exception:
        pass  # Ignore errors, just try to clear


def process_frame(
    runtime,
    frame: FrameData,
    prompt: str,
    max_tokens: int,
    stats: BenchmarkStats,
    verbose: bool = False,
) -> str:
    """Process a single frame: load -> encode -> generate."""
    ### Inmplement based on generate_batched.py
