#!/usr/bin/env python3
"""
Side bench for the mock imaging pipeline: how fast can we synthesise a frame, and how fast can
we store one, and what does the pipeline do end to end.

Nothing here touches the server or mock_camera.py; the candidate synthesis strategies are
re-implemented in this file so they can be tried before any of them replaces the mock's.

  uv run scripts/bench_mock_pipeline.py synth               # per-frame cost of each strategy
  uv run scripts/bench_mock_pipeline.py store               # tifffile cost per compression, thread scaling
  uv run scripts/bench_mock_pipeline.py pipeline            # producer + store queue, images/s per combination
  uv run scripts/bench_mock_pipeline.py all --frames 100

Strategies (all 1024x1024 uint16, brightfield = "cells", fluorescence = "spots", like the mock):
  baseline  the mock's current _generate_synthetic_image
  windows   same picture, but features are drawn inside their bounding box instead of a
            full-frame mask each, and noise comes from numpy's Generator in float32
  bank      like windows, but the background noise is a random crop of a noise bank made once
  prefetch  'bank' generated one frame ahead on a worker thread, for the batch case where the
            next request (channel, exposure, gain) is known
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import queue
import statistics
import sys
import tempfile
import threading
import time
import typing as tp

import numpy as np
import tifffile

H = W = 1024
BANK_MARGIN = 128  # noise bank is (H+margin) x (W+margin); frames are random crops of it


# ---------------------------------------------------------------------------------------------
# request description: what a batch producer would know about the next frame


class FrameRequest(tp.NamedTuple):
    channel_handle: str
    exposure_time_ms: float
    analog_gain: float
    frame_number: int

    @property
    def brightfield(self) -> bool:
        return self.channel_handle.startswith("bfled")


def channel_cycle(channels: list[str], n: int, exposure_ms: float = 5.0, gain: float = 0.0) -> list[FrameRequest]:
    return [FrameRequest(channels[i % len(channels)], exposure_ms, gain, i // len(channels)) for i in range(n)]


# ---------------------------------------------------------------------------------------------
# strategies


class Baseline:
    name = "baseline"

    def __init__(self) -> None:
        from seafront.hardware.camera.mock_camera import MockCamera

        self.cam = MockCamera()

    def frame(self, r: FrameRequest) -> np.ndarray:
        return self.cam._generate_synthetic_image(r.exposure_time_ms, r.analog_gain, r.frame_number, channel_handle=r.channel_handle)


def _features(r: FrameRequest, rng: np.random.Generator) -> list[tuple[int, int, int, float, float]]:
    """(cx, cy, radius, signal, noise_sd) for each cell/spot, same layout rules as the mock"""
    time_factor = np.sin(r.frame_number * 0.1) * 0.2 + 1.0
    exposure_factor = r.exposure_time_ms / 50.0
    out: list[tuple[int, int, int, float, float]] = []
    h = hash(r.channel_handle)
    if r.brightfield:
        num = 20 + int(5 * np.sin(r.frame_number * 0.05))
        for i in range(num):
            cx = int(100 + (i * 100) % (W - 100) + 10 * np.sin(r.frame_number * 0.02 + i))
            cy = int(100 + ((i * 73) % (H - 100)) + 10 * np.cos(r.frame_number * 0.03 + i))
            radius = 25 + int(10 * np.sin(r.frame_number * 0.04 + i))
            if 0 <= cx < W and 0 <= cy < H:
                out.append((cx, cy, radius, 8000 * time_factor * exposure_factor * (0.8 + 0.4 * rng.random()), 500.0))
    else:
        num = 25 + int(10 * np.sin(r.frame_number * 0.08))
        for i in range(num):
            blink = max(0.0, np.sin(r.frame_number * 0.1 + i * 0.5))
            if blink < 0.2:
                continue
            cx = int((i * 347 + h * 191) % W + 8 * np.sin(r.frame_number * 0.03 + i * 0.7)) % W
            cy = int((i * 431 + h * 239) % H + 8 * np.cos(r.frame_number * 0.04 + i * 0.7)) % H
            radius = 15 + int(10 * np.sin(r.frame_number * 0.06 + i))
            out.append((cx, cy, radius, 12000 * time_factor * exposure_factor * blink * (0.7 + 0.6 * rng.random()), 1000.0))
    return out


def _draw_features(img: np.ndarray, feats: list[tuple[int, int, int, float, float]], rng: np.random.Generator, clip_hi: int) -> None:
    """add each feature inside its bounding box only (the mock evaluates a full-frame mask per feature)"""
    for cx, cy, radius, signal, sd in feats:
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, W)
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius**2
        n = int(mask.sum())
        if n == 0:
            continue
        add = np.clip(signal + rng.standard_normal(n, dtype=np.float32) * sd, 0, clip_hi).astype(np.uint16)
        win = img[y0:y1, x0:x1]
        win[mask] += add


def _finish(base: np.ndarray, gain_db: float) -> np.ndarray:
    gain = 10.0 ** (gain_db / 10.0)
    if abs(gain - 1.0) < 1e-9:
        return base  # the mock's float32 round trip is a no-op at 0 dB
    return np.clip(base.astype(np.float32) * gain, 0, 65535).astype(np.uint16)


class Windows:
    name = "windows"

    def frame(self, r: FrameRequest) -> np.ndarray:
        rng = np.random.default_rng((r.frame_number * 73 + hash(r.channel_handle)) % 2**31)
        time_factor = np.sin(r.frame_number * 0.1) * 0.2 + 1.0
        exposure_factor = r.exposure_time_ms / 50.0
        mean, sd, clip_hi = (2000, 500, 50000) if r.brightfield else (300, 100, 40000)
        noise = rng.standard_normal((H, W), dtype=np.float32)
        noise *= sd
        noise += mean * time_factor * exposure_factor
        base = np.clip(noise, 0, 65535).astype(np.uint16)
        _draw_features(base, _features(r, rng), rng, clip_hi)
        return _finish(base, r.analog_gain)


class Bank:
    """background noise from a bank generated once per channel class; per frame: random crop + features"""

    name = "bank"

    def __init__(self, bank_frames: int = 2) -> None:
        rng = np.random.default_rng(12345)
        self.banks: dict[bool, list[np.ndarray]] = {}
        for bf, (mean, sd) in {True: (2000, 500), False: (300, 100)}.items():
            self.banks[bf] = [
                np.clip(rng.standard_normal((H + BANK_MARGIN, W + BANK_MARGIN), dtype=np.float32) * sd + mean, 0, 65535).astype(np.uint16)
                for _ in range(bank_frames)
            ]

    def frame(self, r: FrameRequest) -> np.ndarray:
        rng = np.random.default_rng((r.frame_number * 73 + hash(r.channel_handle)) % 2**31)
        bank = self.banks[r.brightfield][r.frame_number % len(self.banks[r.brightfield])]
        oy, ox = rng.integers(0, BANK_MARGIN, size=2)
        base = bank[oy : oy + H, ox : ox + W].copy()
        # intensity varies with time/exposure like the mock; apply as a cheap integer scale when it matters
        time_factor = np.sin(r.frame_number * 0.1) * 0.2 + 1.0
        scale = time_factor * (r.exposure_time_ms / 50.0)
        if abs(scale - 1.0) > 0.05:
            base = np.clip(base.astype(np.float32) * scale, 0, 65535).astype(np.uint16)
        _draw_features(base, _features(r, rng), rng, 50000 if r.brightfield else 40000)
        return _finish(base, r.analog_gain)


class Prefetch:
    """runs another strategy one frame ahead on a worker thread; the consumer hands over the plan"""

    name = "prefetch"

    def __init__(self, inner: tp.Any, plan: list[FrameRequest], depth: int = 2) -> None:
        self.inner = inner
        self.plan = plan
        self.q: queue.Queue[tuple[FrameRequest, np.ndarray]] = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        for r in self.plan:
            self.q.put((r, self.inner.frame(r)))

    def frame(self, r: FrameRequest) -> np.ndarray:
        got, img = self.q.get()
        assert got == r, "consumer diverged from the plan"
        return img


def make_strategies(plan: list[FrameRequest], names: list[str]) -> list[tp.Any]:
    out: list[tp.Any] = []
    for n in names:
        if n == "baseline":
            out.append(Baseline())
        elif n == "windows":
            out.append(Windows())
        elif n == "bank":
            out.append(Bank())
        elif n == "prefetch":
            out.append(Prefetch(Bank(), plan))
        else:
            raise SystemExit(f"unknown strategy {n}")
    return out


# ---------------------------------------------------------------------------------------------
# benches


def bench_synth(frames: int, names: list[str], channels: list[str]) -> dict[str, float]:
    plan = channel_cycle(channels, frames)
    results: dict[str, float] = {}
    print(f"synthesis, {frames} frames over channels {channels}")
    print(f"  {'strategy':10s} {'ms/frame':>9s} {'p50':>7s} {'max':>7s}  mean intensity (bf / fluo)")
    for strat in make_strategies(plan, names):
        times: list[float] = []
        means: dict[bool, list[float]] = {True: [], False: []}
        for r in plan:
            t = time.perf_counter()
            img = strat.frame(r)
            times.append((time.perf_counter() - t) * 1000)
            means[r.brightfield].append(float(img.mean()))
        results[strat.name] = statistics.mean(times)
        bf = statistics.mean(means[True]) if means[True] else float("nan")
        fl = statistics.mean(means[False]) if means[False] else float("nan")
        print(f"  {strat.name:10s} {statistics.mean(times):9.1f} {statistics.median(times):7.1f} {max(times):7.1f}  {bf:8.0f} / {fl:6.0f}")
    return results


def _write(path: str, img: np.ndarray, compression: str | None, maxworkers: int = 1, rowsperstrip: int | None = None) -> int:
    kw: dict[str, tp.Any] = {"photometric": "minisblack"}
    if compression:
        kw.update(compression=compression, compressionargs={}, maxworkers=maxworkers)
        if rowsperstrip:
            kw["rowsperstrip"] = rowsperstrip
    tifffile.imwrite(path, img, **kw)
    return os.path.getsize(path)


def bench_store(frames: int, workers_list: list[int]) -> None:
    img = Windows().frame(FrameRequest("fluo405", 5.0, 0.0, 0))
    with tempfile.TemporaryDirectory() as d:
        print(f"storage, {frames} writes of one 1024x1024 uint16 frame, single thread")
        print(f"  {'compression':22s} {'ms/image':>9s} {'MB':>6s}")
        for label, comp, mw, rps in [("none", None, 1, None), ("LZW", "LZW", 1, None), ("zlib", "zlib", 1, None),
                                     ("LZW strips+4 threads", "LZW", 4, 64), ("zlib strips+4 threads", "zlib", 4, 64)]:
            t = time.perf_counter()
            for i in range(frames):
                size = _write(f"{d}/{label.replace(' ', '_')}{i}.tiff", img, comp, mw, rps)
            print(f"  {label:22s} {(time.perf_counter() - t) / frames * 1000:9.1f} {size / 1e6:6.2f}")
        print(f"\nstorage, LZW, {frames} writes spread over N threads (does the codec release the GIL?)")
        print(f"  {'threads':8s} {'images/s':>9s} {'speedup':>8s}")
        base_rate = None
        for n in workers_list:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                t = time.perf_counter()
                list(pool.map(lambda i, n=n: _write(f"{d}/t{n}_{i}.tiff", img, "LZW"), range(frames)))
                rate = frames / (time.perf_counter() - t)
            base_rate = base_rate or rate
            print(f"  {n:<8d} {rate:9.1f} {rate / base_rate:8.2f}x")


def bench_pipeline(
    frames: int, names: list[str], channels: list[str], workers_list: list[int], compression: str | None,
    rowsperstrip: int | None = None, codec_threads: int = 1,
) -> None:
    """one producer thread (the 'camera'), a store queue with N workers (like protocol.py's pool)"""
    print(f"pipeline, {frames} frames, compression={compression or 'none'}, rowsperstrip={rowsperstrip or 'default'}, "
          f"codec_threads={codec_threads}, switchinterval={sys.getswitchinterval()*1000:.1f}ms: synth on the main thread, store queue with N workers")
    print(f"  {'strategy':10s} {'workers':>7s} {'images/s':>9s} {'synth ms':>9s} {'queue max':>9s}")
    with tempfile.TemporaryDirectory() as d:
        for name in names:
            for n in workers_list:
                plan = channel_cycle(channels, frames)
                strat = make_strategies(plan, [name])[0]
                pool = concurrent.futures.ThreadPoolExecutor(max_workers=n)
                pending: list[concurrent.futures.Future[int]] = []
                qmax = 0
                synth_ms: list[float] = []
                t0 = time.perf_counter()
                for i, r in enumerate(plan):
                    t = time.perf_counter()
                    img = strat.frame(r).copy()  # the server copies the frame into its store entry
                    synth_ms.append((time.perf_counter() - t) * 1000)
                    pending.append(pool.submit(_write, f"{d}/{name}_{n}_{i}.tiff", img, compression, codec_threads, rowsperstrip))
                    pending = [f for f in pending if not f.done()]
                    qmax = max(qmax, len(pending))
                pool.shutdown(wait=True)
                total = time.perf_counter() - t0
                print(f"  {name:10s} {n:7d} {frames / total:9.1f} {statistics.mean(synth_ms):9.1f} {qmax:9d}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["synth", "store", "pipeline", "all"])
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--strategies", default="baseline,windows,bank,prefetch")
    p.add_argument("--channels", default="fluo405,fluo488,fluo561,fluo638,bfledfull", help="cycled like an acquisition")
    p.add_argument("--workers", default="1,2,4", help="store worker counts to try")
    p.add_argument("--compression", default="LZW", help="LZW, zlib or none (pipeline bench)")
    p.add_argument("--rowsperstrip", type=int, default=None, help="pipeline bench: TIFF strip height (default: tifffile's 256 KB strips)")
    p.add_argument("--codec-threads", type=int, default=1, help="pipeline bench: tifffile maxworkers (strips compressed concurrently)")
    p.add_argument("--switch-interval-ms", type=float, default=None, help="set sys.setswitchinterval (GIL handoff latency) for the run")
    a = p.parse_args()
    names = a.strategies.split(",")
    channels = a.channels.split(",")
    workers = [int(w) for w in a.workers.split(",")]
    comp = None if a.compression.lower() == "none" else a.compression
    if a.what in ("synth", "all"):
        bench_synth(a.frames, names, channels)
        print()
    if a.what in ("store", "all"):
        bench_store(max(10, a.frames // 3), workers)
        print()
    if a.switch_interval_ms is not None:
        sys.setswitchinterval(a.switch_interval_ms / 1000)
    if a.what in ("pipeline", "all"):
        bench_pipeline(a.frames, names, channels, workers, comp, a.rowsperstrip, a.codec_threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
