# SPDX-License-Identifier: Apache-2.0
"""Incremental audio encoders for the streaming speech endpoint.

Each encoder is fed float32 mono chunks (``feed``) and returns whatever encoded
bytes are ready; ``finish`` flushes the rest. ``pcm`` and ``wav`` are done in
process; compressed formats go through an ffmpeg subprocess so that the stream is
one continuous bitstream (separately encoded MP3/Opus files would glitch at the joins).
"""

from __future__ import annotations

import queue
import shutil
import struct
import subprocess
import threading

import numpy as np

FFMPEG_ARGS = {
    "mp3": ["-c:a", "libmp3lame", "-q:a", "4", "-write_xing", "0", "-f", "mp3"],
    "opus": ["-c:a", "libopus", "-b:a", "48k", "-f", "ogg"],
    "aac": ["-c:a", "aac", "-b:a", "64k", "-f", "adts"],
    "flac": ["-c:a", "flac", "-f", "flac"],
}
STREAMABLE = {"pcm", "wav", *FFMPEG_ARGS}


def to_s16(chunk: np.ndarray) -> bytes:
    return (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()


class PcmEncoder:
    def __init__(self, rate: int):
        self.rate = rate

    def feed(self, chunk: np.ndarray) -> bytes:
        return to_s16(chunk)

    def finish(self) -> bytes:
        return b""


class WavEncoder(PcmEncoder):
    """WAV with an unknown-length header (sizes 0xFFFFFFFF), which players accept for streams."""

    def __init__(self, rate: int):
        super().__init__(rate)
        self._header = (
            b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF)
        )

    def feed(self, chunk: np.ndarray) -> bytes:
        data = to_s16(chunk)
        if self._header:
            data, self._header = self._header + data, b""
        return data


class FfmpegEncoder:
    def __init__(self, fmt: str, rate: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(f"streaming {fmt} needs ffmpeg on PATH")
        self.proc = subprocess.Popen(
            [ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(rate), "-ac", "1", "-i", "-",
             *FFMPEG_ARGS[fmt], "-flush_packets", "1", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self._q: queue.Queue[bytes | None] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            data = self.proc.stdout.read1(65536)
            if not data:
                self._q.put(None)
                return
            self._q.put(data)

    def _drain(self, wait: float = 0.0) -> bytes:
        """Collect ready output; wait up to ``wait`` s for the first piece."""
        out = []
        while True:
            try:
                item = self._q.get(timeout=wait) if wait else self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._q.put(None)
                break
            out.append(item)
            wait = 0.0
        return b"".join(out)

    def feed(self, chunk: np.ndarray) -> bytes:
        self.proc.stdin.write(to_s16(chunk))
        self.proc.stdin.flush()
        # ffmpeg answers within tens of ms; waiting briefly sends this chunk's audio now
        # instead of one chunk (a whole sentence) later.
        return self._drain(wait=0.15)

    def finish(self) -> bytes:
        self.proc.stdin.close()
        out = []
        while True:
            item = self._q.get()
            if item is None:
                break
            out.append(item)
        rc = self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}: {err}")
        return b"".join(out)

    def close(self):
        """Abort (client went away): kill ffmpeg without waiting for the stream to finish."""
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def make_encoder(fmt: str, rate: int):
    if fmt == "pcm":
        return PcmEncoder(rate)
    if fmt == "wav":
        return WavEncoder(rate)
    return FfmpegEncoder(fmt, rate)
