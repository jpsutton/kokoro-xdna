#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible speech endpoint for Kokoro TTS with AMD XDNA NPU offload.

    POST /v1/audio/speech   {"input": "...", "voice": "af_heart", "response_format": "wav", "speed": 1.0}
    GET  /v1/audio/voices
    GET  /v1/models
    GET  /health
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from kokoro_npu.engine import DEFAULT_VOICE, SAMPLE_RATE, KokoroNPU
from kokoro_npu.stream_encode import make_encoder

log = logging.getLogger("server")

FORMATS = {
    "wav": ("audio/wav", "WAV", "PCM_16"),
    "flac": ("audio/flac", "FLAC", "PCM_16"),
    "mp3": ("audio/mpeg", "MP3", None),
    "opus": ("audio/ogg", "OGG", "OPUS"),
    "aac": ("audio/aac", None, None),  # via ffmpeg (ADTS)
    "pcm": ("audio/pcm", None, None),  # raw signed 16-bit little-endian mono @ 24 kHz
}

# OpenAI's stock voice names -> Kokoro voices, so unmodified OpenAI clients work.
OPENAI_VOICES = {
    "alloy": "af_heart",
    "ash": "am_michael",
    "ballad": "bm_lewis",
    "coral": "af_bella",
    "echo": "am_echo",
    "fable": "bm_george",
    "nova": "af_nova",
    "onyx": "am_onyx",
    "sage": "af_sarah",
    "shimmer": "af_sky",
    "verse": "am_adam",
}


class SpeechRequest(BaseModel):
    """Lenient on purpose: OpenAI clients send null / extra fields (instructions, stream_format, ...)."""

    model: str | None = "kokoro"
    input: str = Field(..., min_length=1, max_length=20000)
    voice: str | None = None
    response_format: str | None = "wav"
    speed: float | None = None
    lang: str | None = None
    # Streaming: audio is sent sentence by sentence as it is generated.
    # `stream: true` or OpenAI's `stream_format: "audio"` turn it on, `stream: false` turns it off;
    # unset follows STREAM_DEFAULT (env, default off). `stream_format: "sse"` is not supported.
    stream: bool | None = None
    stream_format: str | None = None

    @field_validator("response_format")
    @classmethod
    def _fmt(cls, v):
        v = (v or "wav").lower()
        if v not in FORMATS:
            raise ValueError(f"response_format must be one of {sorted(FORMATS)}")
        return v


engine: KokoroNPU | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    use_npu = os.environ.get("KOKORO_NPU", "1") != "0"
    threads = int(os.environ.get("ORT_THREADS", "2"))
    t0 = time.time()
    engine = KokoroNPU(use_npu=use_npu, intra_op_threads=threads)
    engine.synthesize("Warming up.", DEFAULT_VOICE)
    log.info("engine ready (npu=%s, threads=%d, %.1fs)", use_npu, threads, time.time() - t0)
    yield


app = FastAPI(title="Kokoro NPU TTS", lifespan=lifespan)

BODY_LOG_CHARS = int(os.environ.get("LOG_BODY_CHARS", "0"))  # 0 (default) disables request-body logging


@app.middleware("http")
async def access_log(request: Request, call_next):
    """One line per request: client, request headers that matter, body preview, status, timing."""
    t0 = time.perf_counter()
    body = await request.body() if request.method in ("POST", "PUT", "PATCH") else b""
    log.debug("headers: %s", dict(request.headers))
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error: %s %s body=%r", request.method, request.url.path, body[:BODY_LOG_CHARS])
        raise
    dt = time.perf_counter() - t0
    preview = body[:BODY_LOG_CHARS].decode("utf-8", "replace") if BODY_LOG_CHARS and body else ""
    log.info(
        "%s %s %s -> %d %s %.3fs | ua=%r ctype=%r body=%s%s",
        request.client.host if request.client else "-",
        request.method,
        request.url.path,
        response.status_code,
        response.headers.get("content-type", "-"),
        dt,
        request.headers.get("user-agent", "-"),
        request.headers.get("content-type", "-"),
        preview,
        "..." if len(body) > BODY_LOG_CHARS > 0 else "",
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    body = (await request.body())[:500]
    log.warning("422 for %s %s: %s | body=%r", request.method, request.url.path, exc.errors(), body)
    return JSONResponse(status_code=422, content={"detail": jsonable_errors(exc)})


def jsonable_errors(exc: RequestValidationError):
    return [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]


def encode(audio: np.ndarray, fmt: str) -> bytes:
    media, sf_format, subtype = FORMATS[fmt]
    if fmt == "aac":
        if shutil.which("ffmpeg") is None:
            raise HTTPException(400, "aac needs ffmpeg on PATH")
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        return subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "-", "-f", "adts", "-"],
            input=pcm, capture_output=True, check=True,
        ).stdout
    if sf_format is None:
        return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format=sf_format, subtype=subtype)
    return buf.getvalue()


STREAM_DEFAULT = os.environ.get("STREAM_DEFAULT", "0") == "1"


async def stream_speech(req: SpeechRequest, voice: str, speed: float):
    """Chunked response body: encode and send each sentence as soon as it is synthesized."""
    t0 = time.perf_counter()
    first = None
    nbytes = 0
    seconds = 0.0
    enc = make_encoder(req.response_format, SAMPLE_RATE)
    gen = engine.synthesize_stream(req.input, voice, speed, req.lang or "en-us")
    ok = False
    try:
        while True:
            chunk = await run_in_threadpool(next, gen, None)
            if chunk is None:
                break
            data = await run_in_threadpool(enc.feed, chunk)
            seconds += len(chunk) / SAMPLE_RATE
            if data:
                if first is None:
                    first = time.perf_counter() - t0
                nbytes += len(data)
                yield data
        tail = await run_in_threadpool(enc.finish)
        if tail:
            nbytes += len(tail)
            yield tail
        ok = True
    except Exception:
        log.exception("streaming synthesis failed after %d bytes", nbytes)  # the body just ends
    finally:
        if not ok and hasattr(enc, "close"):
            enc.close()
        total = time.perf_counter() - t0
        log.info("stream %s voice=%s chars=%d audio=%.2fs bytes=%d first_byte=%s total=%.2fs",
                 "done" if ok else "ABORTED", voice, len(req.input), seconds, nbytes,
                 f"{first:.2f}s" if first is not None else "-", total)


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    assert engine is not None
    voice = req.voice or DEFAULT_VOICE
    voice = OPENAI_VOICES.get(voice, voice) if voice not in engine.voices else voice
    if voice not in engine.voices:
        raise HTTPException(400, f"unknown voice {req.voice!r}; see /v1/audio/voices")
    speed = min(max(req.speed or 1.0, 0.5), 2.0)  # kokoro-onnx asserts 0.5..2.0
    if req.stream_format and req.stream_format.lower() not in ("audio", "sse"):
        raise HTTPException(400, "stream_format must be 'audio'")
    if req.stream_format and req.stream_format.lower() == "sse":
        raise HTTPException(400, "stream_format 'sse' is not supported; use 'audio' (raw chunked audio)")
    stream = req.stream if req.stream is not None else (bool(req.stream_format) or STREAM_DEFAULT)
    if stream:
        return StreamingResponse(
            stream_speech(req, voice, speed),
            media_type=FORMATS[req.response_format][0],
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
        )
    t0 = time.perf_counter()
    try:
        audio, sr = await run_in_threadpool(engine.synthesize, req.input, voice, speed, req.lang or "en-us")
    except Exception as e:
        log.exception("synthesis failed")
        raise HTTPException(500, f"synthesis failed: {e}") from e
    dt = time.perf_counter() - t0
    log.debug("synthesized voice=%s speed=%s fmt=%s chars=%d audio=%.2fs in %.2fs (RTF %.3f)",
             voice, req.speed, req.response_format, len(req.input), len(audio) / sr, dt, dt / max(len(audio) / sr, 1e-6))
    data = encode(audio, req.response_format)
    return Response(
        content=data,
        media_type=FORMATS[req.response_format][0],
        headers={
            "X-Audio-Seconds": f"{len(audio) / sr:.3f}",
            "X-Synthesis-Seconds": f"{dt:.3f}",
            "X-RTF": f"{dt / max(len(audio) / sr, 1e-6):.3f}",
        },
    )


@app.get("/v1/audio/voices")
async def voices():
    assert engine is not None
    return {"voices": engine.voices, "default": DEFAULT_VOICE}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "kokoro", "object": "model", "owned_by": "local"}]}


@app.get("/health")
async def health():
    assert engine is not None
    return {"status": "ok", "npu": engine.use_npu, "stats": engine.npu_stats()}


if __name__ == "__main__":
    level = os.environ.get("LOG_LEVEL", "INFO").upper()  # DEBUG also logs all request headers
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # uvicorn's own access line duplicates ours
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8880")),
                log_level=level.lower(), access_log=False)
