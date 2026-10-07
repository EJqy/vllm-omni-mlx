"""
MOSS-TTS-Nano voice cloning through mlx-audio (#73).

Nano has its own GPT2/codebook pipeline and 48 kHz stereo codec.
The public audio contract is explicitly downmixed
48 kHz mono PCM16. Reference waveforms retain their channels and are
resampled to the codec rate before being passed to mlx-audio.
"""

from __future__ import annotations

import base64
import binascii
import io
import sys
import threading
import wave
from dataclasses import dataclass
from typing import Any, Iterator

import mlx.core as mx

DEFAULT_MODEL = "mlx-community/MOSS-TTS-Nano-100M"
DEFAULT_CODEC_MODEL = "mlx-community/MOSS-Audio-Tokenizer-Nano"
SAMPLE_RATE = 48000
MAX_REF_SECONDS = 30.0

# PCM16 wire format is little-endian; mx.array exposes native-endian buffers.
if sys.byteorder != "little":
    raise ImportError("MOSS Nano PCM16 output requires a little-endian host")


@dataclass(frozen=True)
class MossNanoConfig:
    """Nano defaults match mlx-audio 0.5.7, including both samplers."""

    model_ref: str = DEFAULT_MODEL
    codec_model_ref: str | None = None
    max_new_frames: int = 375
    max_text_tokens: int = 75
    do_sample: bool = True
    text_temperature: float = 1.0
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 0.8
    audio_top_p: float = 0.95
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.2
    streaming_interval: float = 0.5
    streaming_initial_interval: float = 0.08

    def __post_init__(self) -> None:
        for name in ("max_new_frames", "max_text_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


def _all_finite(x: mx.array) -> bool:
    """True when x has no NaN or ±Inf values."""
    bad = mx.logical_or(mx.isnan(x), mx.isinf(x))
    return not mx.any(bad).item()


def _validate_model(model: Any) -> None:
    variant = getattr(getattr(model, "config", None), "model_type", None)
    if variant != "moss_tts_nano":
        raise ValueError(f"MOSS Nano needs model_type='moss_tts_nano', got {variant!r}")
    if model.sample_rate != SAMPLE_RATE:
        raise ValueError(f"MOSS Nano needs a 48000 Hz codec, got {model.sample_rate!r}")


def load_moss_nano_model(config: MossNanoConfig) -> Any:
    """Load the language model, tokenizer and MLX codec before serving.

    ``codec_model_ref`` may name a local MLX codec snapshot for offline
    deployment. Otherwise use the MLX-converted codec: the language-model
    config currently points to the original OpenMOSS weights.
    """
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:
        raise RuntimeError(
            "MOSS Nano needs mlx-audio 0.5.7; install 'vllm-omni-mlx[tts]'"
        ) from exc
    model = load_model(config.model_ref)
    _validate_model(model)
    model._ensure_audio_tokenizer(source=config.codec_model_ref or DEFAULT_CODEC_MODEL)
    return model


def decode_ref_audio(data: str, sample_rate: int = SAMPLE_RATE) -> mx.array:
    """Decode base64 audio at the codec rate, preserving mono/stereo layout.

    This is deliberately independent from Qwen's 24 kHz reference helper.
    Duration counts frames, not interleaved channel values.
    """
    if not isinstance(data, str) or not data:
        raise ValueError("voice.ref_audio must be non-empty base64-encoded audio")
    try:
        payload = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"ref_audio must be base64-encoded audio: {exc}") from None
    try:
        from mlx_audio.audio_io import read as audio_read

        raw, actual_rate = audio_read(
            io.BytesIO(payload), dtype="float32", sample_rate=sample_rate
        )
        samples = mx.array(raw, dtype=mx.float32)
    except Exception as exc:
        raise ValueError(f"ref_audio could not be decoded as audio: {exc}") from None
    if actual_rate != sample_rate:
        raise ValueError(
            f"ref_audio decoder returned {actual_rate} Hz, expected {sample_rate} Hz"
        )
    if samples.ndim not in (1, 2) or (
        samples.ndim == 2 and samples.shape[1] not in (1, 2)
    ):
        raise ValueError("ref_audio must contain mono or stereo audio")
    if samples.size == 0:
        raise ValueError("ref_audio decoded to zero samples")
    if not _all_finite(samples):
        raise ValueError("ref_audio contains non-finite samples")
    duration = samples.shape[0] / sample_rate
    if duration > MAX_REF_SECONDS:
        raise ValueError(
            f"ref_audio is {duration:.1f}s; the cap is {MAX_REF_SECONDS:.0f}s"
        )
    return samples


def _pcm16(audio: Any) -> bytes:
    """Sample-major float audio → explicitly downmixed little-endian PCM16."""
    samples = (
        audio.astype(mx.float32)
        if isinstance(audio, mx.array)
        else mx.array(audio, dtype=mx.float32)
    )
    if samples.ndim == 2 and samples.shape[1] in (1, 2):
        samples = mx.mean(samples, axis=1)
    elif samples.ndim != 1:
        raise RuntimeError(
            f"MOSS Nano returned an unsupported audio shape {tuple(samples.shape)}"
        )
    if not _all_finite(samples):
        raise RuntimeError("MOSS Nano returned non-finite audio samples")
    pcm = (mx.clip(samples, -1.0, 1.0) * 32767.0).astype(mx.int16)
    mx.eval(pcm)
    return bytes(memoryview(pcm))


class MossNanoService:
    """Batch-1, cloning-only speech service; request validation is eager."""

    sample_rate = SAMPLE_RATE
    channels = 1
    model_type = "moss_tts_nano"

    def __init__(self, model: Any, config: MossNanoConfig | None = None):
        _validate_model(model)
        self._model = model
        self.config = config or MossNanoConfig()
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.config.model_ref

    @property
    def voices(self) -> list[str]:
        return []

    def _clone_inputs(
        self,
        input: str,
        voice: Any,
        speed: float,
        instructions: str | None,
        language: str | None,
    ) -> mx.array:
        if not isinstance(input, str) or not input.strip():
            raise ValueError("input must be a non-empty string")
        if speed != 1.0:
            raise ValueError("speed != 1.0 is not supported yet")
        if instructions:
            raise ValueError(
                "instructions are not supported by MOSS Nano voice cloning"
            )
        if language not in (None, "", "auto"):
            raise ValueError(
                "language is selected from the input text by MOSS Nano; use 'auto'"
            )
        if not isinstance(voice, dict):
            raise ValueError(
                "MOSS Nano has no preset voices; voice must be a cloning object "
                "with ref_audio (base64 audio); ref_text is optional and ignored"
            )
        unknown = set(voice) - {"ref_audio", "ref_text"}
        if unknown:
            raise ValueError(
                f"voice object supports ref_audio and ref_text, got {sorted(unknown)}"
            )
        if voice.get("ref_text") is not None and not isinstance(voice["ref_text"], str):
            raise ValueError(
                "voice.ref_text must be a string when provided (MOSS Nano ignores it)"
            )
        if "ref_audio" not in voice:
            raise ValueError(
                "voice.ref_audio (base64 audio) is required for MOSS Nano cloning"
            )
        # Parsing happens outside the generation lock and before HTTP headers.
        return decode_ref_audio(voice["ref_audio"], sample_rate=self.sample_rate)

    def _require_open(self) -> Any:
        if self._model is None:
            raise RuntimeError("MOSS Nano service is closed")
        return self._model

    def _buffered_chunks(self, text: str, ref_audio: mx.array) -> Iterator[bytes]:
        cfg = self.config
        results = self._require_open().generate(
            text=text,
            ref_audio=ref_audio,
            ref_audio_sample_rate=self.sample_rate,
            mode="voice_clone",
            stream=False,
            max_tokens=cfg.max_new_frames,
            voice_clone_max_text_tokens=cfg.max_text_tokens,
            do_sample=cfg.do_sample,
            text_temperature=cfg.text_temperature,
            text_top_p=cfg.text_top_p,
            text_top_k=cfg.text_top_k,
            audio_temperature=cfg.audio_temperature,
            audio_top_p=cfg.audio_top_p,
            audio_top_k=cfg.audio_top_k,
            audio_repetition_penalty=cfg.audio_repetition_penalty,
            audio_tokenizer_source=cfg.codec_model_ref or DEFAULT_CODEC_MODEL,
        )
        try:
            for result in results:
                if result.sample_rate != self.sample_rate:
                    raise RuntimeError(
                        f"MOSS Nano returned {result.sample_rate} Hz audio, expected {self.sample_rate} Hz"
                    )
                if result.audio is not None and result.audio.size:
                    yield _pcm16(result.audio)
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

    def speech_bytes(
        self,
        input: str,
        voice: str | dict | None = None,
        response_format: str = "wav",
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
    ) -> tuple[bytes, str]:
        if response_format not in ("wav", "pcm"):
            raise ValueError("response_format must be 'wav' or 'pcm'")
        ref_audio = self._clone_inputs(input, voice, speed, instructions, language)
        with self._lock:
            pcm = b"".join(self._buffered_chunks(input, ref_audio))
        if not pcm:
            raise RuntimeError("MOSS Nano generated no audio")
        if response_format == "pcm":
            return pcm, "audio/pcm"
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm)
        return buffer.getvalue(), "audio/wav"

    def speech_stream(
        self,
        input: str,
        voice: str | dict | None = None,
        speed: float = 1.0,
        instructions: str | None = None,
        language: str | None = None,
        streaming_interval: float | None = None,
        streaming_initial_interval: float | None = None,
    ) -> Iterator[bytes]:
        self._clone_inputs(input, voice, speed, instructions, language)
        interval = (
            self.config.streaming_interval
            if streaming_interval is None
            else streaming_interval
        )
        initial = (
            self.config.streaming_initial_interval
            if streaming_initial_interval is None
            else streaming_initial_interval
        )
        for name, value in (
            ("streaming_interval", interval),
            ("streaming_initial_interval", initial),
        ):
            if not 0.0 < value <= 10.0:
                raise ValueError(f"{name} must be in (0, 10] seconds")
        raise ValueError("MOSS Nano streaming is not available yet; tracked in #73")

    def close(self) -> None:
        """Release this service's model reference once active generation ends."""
        with self._lock:
            self._model = None
