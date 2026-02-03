"""Benchmark latest-frame camera capture to inference latency."""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from _shared import ensure_local_import

ensure_local_import()

from llama_insight import Config, add_common_args  # noqa: E402
from llama_insight.usecase_helpers import (  # noqa: E402
    DEFAULT_IMAGE_PROMPT,
    start_session,
)

# Import camera module from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from camera import Camera  # noqa: E402


@dataclass
class CapturedFrame:
    """Metadata for a captured frame on disk."""

    frame_id: int
    path: Path
    capture_start: float
    capture_end: float
    slot: int


@dataclass
class InferenceRecord:
    """Per-frame inference timing and output stats."""

    frame_id: int
    capture_end: float
    start_time: float
    end_time: float
    load_time: float
    generate_time: float
    frame_age: float
    total_latency: float
    tokens_generated: int


class LatestFrameBuffer:
    """Hold the most recent captured frame and protect shared slots."""

    def __init__(self, slots: Sequence[Path]) -> None:
        if not slots:
            raise ValueError("At least one slot path is required.")
        self._slots = list(slots)
        self._condition = threading.Condition()
        self._latest: Optional[CapturedFrame] = None
        self._in_use_slot: Optional[int] = None
        self._last_slot = 0
        self._stopped = False

    def next_capture_slot(self) -> tuple[int, Path]:
        """Pick a slot that is not currently in use by inference."""
        with self._condition:
            candidates = [idx for idx in range(len(self._slots))]
            if self._in_use_slot is not None and len(self._slots) > 1:
                candidates = [idx for idx in candidates if idx != self._in_use_slot]
            if self._last_slot in candidates and len(candidates) > 1:
                candidates = [idx for idx in candidates if idx != self._last_slot]
            slot_idx = candidates[0]
            self._last_slot = slot_idx
            return slot_idx, self._slots[slot_idx]

    def update_latest(self, frame: CapturedFrame) -> None:
        with self._condition:
            self._latest = frame
            self._condition.notify_all()

    def acquire_latest(self, last_frame_id: Optional[int]) -> Optional[CapturedFrame]:
        """Block until a new frame arrives, then mark its slot as in-use."""
        with self._condition:
            while True:
                if self._stopped:
                    return None
                if self._latest and self._latest.frame_id != last_frame_id:
                    self._in_use_slot = self._latest.slot
                    return self._latest
                self._condition.wait(timeout=0.5)

    def release_slot(self, slot: int) -> None:
        with self._condition:
            if self._in_use_slot == slot:
                self._in_use_slot = None

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


class CaptureWorker(threading.Thread):
    """Continuously capture frames to disk and publish the latest."""

    def __init__(
        self,
        buffer: LatestFrameBuffer,
        camera_index: int,
        width: int,
        height: int,
        max_frames: int,
        capture_interval_ms: float,
    ) -> None:
        super().__init__(daemon=True)
        self.buffer = buffer
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.capture_interval_ms = capture_interval_ms
        self.total_captured = 0
        self.total_failed = 0
        self.total_capture_time = 0.0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._stop_event = threading.Event()
        self._frame_id = 0

    def run(self) -> None:
        self.start_time = time.perf_counter()
        next_capture = self.start_time
        with Camera(self.camera_index, self.width, self.height) as cam:
            while not self._stop_event.is_set():
                if self.max_frames and self._frame_id >= self.max_frames:
                    break
                now = time.perf_counter()
                if self.capture_interval_ms > 0:
                    sleep_for = next_capture - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                capture_start = time.perf_counter()
                slot_idx, path = self.buffer.next_capture_slot()
                success, _ = cam.capture(path)
                capture_end = time.perf_counter()
                capture_time = capture_end - capture_start
                self.total_capture_time += capture_time
                if success:
                    frame = CapturedFrame(
                        frame_id=self._frame_id,
                        path=path,
                        capture_start=capture_start,
                        capture_end=capture_end,
                        slot=slot_idx,
                    )
                    self.buffer.update_latest(frame)
                    self.total_captured += 1
                    self._frame_id += 1
                else:
                    self.total_failed += 1
                if self.capture_interval_ms > 0:
                    next_capture = capture_start + self.capture_interval_ms / 1000.0
        self.end_time = time.perf_counter()
        self.buffer.stop()

    def stop(self) -> None:
        self._stop_event.set()


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Compute a percentile with linear interpolation."""
    if not values:
        return None
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lower = int(k)
    upper = min(lower + 1, len(sorted_vals) - 1)
    if lower == upper:
        return sorted_vals[lower]
    weight = k - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def reset_kv_cache(runtime) -> None:
    """Clear KV cache between frames."""
    try:
        gbl = runtime.backend.gbl
        mem = gbl.llama_get_memory(runtime.ctx)
        gbl.llama_memory_clear(mem, True)
    except Exception:
        pass


def load_image_bitmap(runtime, image_path: Path) -> object:
    """Load an image bitmap without extra console logging."""
    return runtime.processor.load_image(runtime.ctx_mtmd, str(image_path))


def process_frame(
    runtime,
    frame: CapturedFrame,
    prompt: str,
    max_new_tokens: int,
    reset_kv: bool,
) -> tuple[InferenceRecord, object]:
    """Run inference for a single frame and return timing stats."""
    if reset_kv:
        reset_kv_cache(runtime)
    start_time = time.perf_counter()
    bitmap = load_image_bitmap(runtime, frame.path)
    load_done = time.perf_counter()
    result = runtime.generate_text(
        prompt=prompt,
        images=[bitmap],
        max_new_tokens=max_new_tokens,
    )
    end_time = time.perf_counter()
    record = InferenceRecord(
        frame_id=frame.frame_id,
        capture_end=frame.capture_end,
        start_time=start_time,
        end_time=end_time,
        load_time=load_done - start_time,
        generate_time=end_time - load_done,
        frame_age=start_time - frame.capture_end,
        total_latency=end_time - frame.capture_end,
        tokens_generated=int(getattr(result, "total_tokens_generated", 0)),
    )
    return record, result


def summarize_records(records: Iterable[InferenceRecord]) -> dict[str, float]:
    """Aggregate inference timing statistics."""
    record_list = list(records)
    latencies = [r.total_latency for r in record_list]
    ages = [r.frame_age for r in record_list]
    load_times = [r.load_time for r in record_list]
    gen_times = [r.generate_time for r in record_list]
    tokens = [r.tokens_generated for r in record_list]
    return {
        "frames": len(record_list),
        "avg_latency_ms": statistics.mean(latencies) * 1000 if latencies else 0.0,
        "p50_latency_ms": (percentile(latencies, 50) or 0.0) * 1000,
        "p90_latency_ms": (percentile(latencies, 90) or 0.0) * 1000,
        "p99_latency_ms": (percentile(latencies, 99) or 0.0) * 1000,
        "avg_frame_age_ms": statistics.mean(ages) * 1000 if ages else 0.0,
        "avg_load_ms": statistics.mean(load_times) * 1000 if load_times else 0.0,
        "avg_generate_ms": statistics.mean(gen_times) * 1000 if gen_times else 0.0,
        "avg_tokens": statistics.mean(tokens) if tokens else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark latest-frame camera capture to inference latency."
    )
    add_common_args(parser)
    parser.add_argument(
        "-p",
        "--prompt",
        default=DEFAULT_IMAGE_PROMPT,
        help="Prompt text; include <__image__> for each image.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera capture width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera capture height (default: 480).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Number of inference iterations to run (default: 20).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=2,
        help="Warmup iterations excluded from metrics (default: 2).",
    )
    parser.add_argument(
        "--mode",
        choices=["latest", "sequential"],
        default="latest",
        help="Benchmark mode: latest (capture thread) or sequential.",
    )
    parser.add_argument(
        "--capture-interval-ms",
        type=float,
        default=0.0,
        help="Delay between captures in ms (0 = as fast as possible).",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_frames",
        help="Directory for captured frame files (default: benchmark_frames).",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Keep the slot files on disk instead of cleaning up.",
    )
    parser.add_argument(
        "--no-reset-kv-cache",
        action="store_true",
        help="Disable KV cache clearing between frames.",
    )
    parser.add_argument(
        "--print-output",
        action="store_true",
        help="Print the generated text for each frame.",
    )
    args = parser.parse_args()
    config = Config.from_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.num_frames <= 0:
        parser.error("--num-frames must be a positive integer")
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")

    try:
        records: list[InferenceRecord] = []
        skipped_frames = 0
        last_frame_id: Optional[int] = None
        total_iterations = args.num_frames + args.warmup_frames

        reset_kv = not args.no_reset_kv_cache

        if args.mode == "latest":
            slots = [output_dir / "slot_0.jpg", output_dir / "slot_1.jpg"]
            buffer = LatestFrameBuffer(slots)
            capture_worker = CaptureWorker(
                buffer=buffer,
                camera_index=args.camera_index,
                width=args.width,
                height=args.height,
                max_frames=0,
                capture_interval_ms=args.capture_interval_ms,
            )
            print("\n--- Starting capture thread ---")
            capture_worker.start()
            with start_session(config) as runtime:
                for iteration in range(total_iterations):
                    frame = buffer.acquire_latest(last_frame_id)
                    if frame is None:
                        break
                    if last_frame_id is not None:
                        skipped = max(0, frame.frame_id - last_frame_id - 1)
                        skipped_frames += skipped
                    record, result = process_frame(
                        runtime,
                        frame,
                        args.prompt,
                        config.max_new_tokens,
                        reset_kv,
                    )
                    buffer.release_slot(frame.slot)
                    last_frame_id = frame.frame_id
                    if iteration >= args.warmup_frames:
                        records.append(record)
                        if args.print_output:
                            print(f"\n--- Frame {record.frame_id} ---")
                            print(result.generated_text)
            capture_worker.stop()
            capture_worker.join(timeout=2.0)
            buffer.stop()

            if not args.save_frames:
                for slot in slots:
                    if slot.exists():
                        slot.unlink()
        else:
            capture_count = 0
            with Camera(args.camera_index, args.width, args.height) as cam:
                with start_session(config) as runtime:
                    for iteration in range(total_iterations):
                        capture_start = time.perf_counter()
                        if args.save_frames:
                            path = output_dir / f"frame_{capture_count:06d}.jpg"
                        else:
                            path = output_dir / "slot_0.jpg"
                        success, _ = cam.capture(path)
                        capture_end = time.perf_counter()
                        if not success:
                            continue
                        frame = CapturedFrame(
                            frame_id=capture_count,
                            path=path,
                            capture_start=capture_start,
                            capture_end=capture_end,
                            slot=0,
                        )
                        capture_count += 1
                        record, result = process_frame(
                            runtime,
                            frame,
                            args.prompt,
                            config.max_new_tokens,
                            reset_kv,
                        )
                        if iteration >= args.warmup_frames:
                            records.append(record)
                            if args.print_output:
                                print(f"\n--- Frame {record.frame_id} ---")
                                print(result.generated_text)
            if not args.save_frames:
                slot = output_dir / "slot_0.jpg"
                if slot.exists():
                    slot.unlink()

        summary = summarize_records(records)
        print("\n--- Benchmark Summary ---")
        print(f"Mode: {args.mode}")
        print(f"Processed frames: {summary['frames']}")
        print(f"Warmup frames: {args.warmup_frames}")
        if args.mode == "latest":
            print(f"Frames skipped: {skipped_frames}")
            if capture_worker.start_time and capture_worker.end_time:
                duration = max(1e-6, capture_worker.end_time - capture_worker.start_time)
                capture_fps = capture_worker.total_captured / duration
                avg_capture_ms = (
                    (capture_worker.total_capture_time / capture_worker.total_captured)
                    * 1000
                    if capture_worker.total_captured
                    else 0.0
                )
                print(
                    f"Captured frames: {capture_worker.total_captured} "
                    f"(fps {capture_fps:.2f}, avg capture {avg_capture_ms:.1f} ms)"
                )
        print(
            "Latency ms (capture->output): "
            f"avg {summary['avg_latency_ms']:.1f}, "
            f"p50 {summary['p50_latency_ms']:.1f}, "
            f"p90 {summary['p90_latency_ms']:.1f}, "
            f"p99 {summary['p99_latency_ms']:.1f}"
        )
        print(
            "Frame age ms (capture->inference start): "
            f"avg {summary['avg_frame_age_ms']:.1f}"
        )
        print(
            "Stage timing ms: "
            f"load avg {summary['avg_load_ms']:.1f}, "
            f"generate avg {summary['avg_generate_ms']:.1f}"
        )
        print(f"Avg tokens generated: {summary['avg_tokens']:.1f}")

    except Exception as exc:  # pragma: no cover - runtime guard
        print("\n--- ERROR ---")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
