"""Nano service seams: cloning, sample rate and lock lifetime."""

import base64
import importlib.util
import io
import struct
import unittest
import wave
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx

from vllm_omni_mlx.tts import moss_nano
from vllm_omni_mlx.tts.moss_nano import (
    MossNanoConfig,
    MossNanoService,
    decode_ref_audio,
)

HAS_MLX_AUDIO = importlib.util.find_spec("mlx_audio") is not None


def reference_wav(sample_rate=48000, channels=1, seconds=0.5):
    """A real small PCM clip so resampling and channel handling are exercised."""
    # Build the wire-format fixture independently of the MLX conversion under test.
    samples = struct.pack("<h", 8192) * (int(sample_rate * seconds) * channels)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeModel:
    sample_rate = 48000
    config = SimpleNamespace(model_type="moss_tts_nano")

    def __init__(self):
        self.kwargs = None
        # Stereo must be averaged, never flattened into twice the duration.
        self.audio = mx.array(
            [[1.0, -1.0], [0.25, 0.75], [-2.0, -2.0]], dtype=mx.float32
        )
        self.finished = False

    def generate(self, **kwargs):
        self.kwargs = kwargs
        try:
            yield SimpleNamespace(audio=self.audio, sample_rate=self.sample_rate)
        finally:
            self.finished = True


@unittest.skipUnless(
    HAS_MLX_AUDIO, "reference decoding needs the optional mlx-audio extra"
)
class NanoReferenceTest(unittest.TestCase):
    def test_resamples_24k_reference_to_48k_without_losing_stereo(self):
        reference = decode_ref_audio(reference_wav(sample_rate=24000, channels=2))
        self.assertEqual(reference.shape, (24000, 2))
        self.assertIsInstance(reference, mx.array)
        self.assertEqual(reference.dtype, mx.float32)
        self.assertTrue(
            mx.allclose(reference[100:-100], mx.array(0.25), atol=0.001).item()
        )

    def test_mono_remains_mono(self):
        reference = decode_ref_audio(reference_wav(channels=1))
        self.assertEqual(reference.shape, (24000,))

    def test_invalid_reference_payloads(self):
        for data in (
            None,
            b"raw audio",
            "",
            "not base64",
            base64.b64encode(b"not audio").decode(),
        ):
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(ValueError, "ref_audio"),
            ):
                decode_ref_audio(data)

    def test_duration_uses_frames_not_channel_count(self):
        # Both limits are inclusive, including when each frame has two channels.
        for frames in (24000, 30 * 48000):
            for channels in (1, 2):
                shape = (frames,) if channels == 1 else (frames, channels)
                samples = mx.zeros(shape, dtype=mx.float32)
                with (
                    self.subTest(frames=frames, channels=channels),
                    mock.patch(
                        "mlx_audio.audio_io.read", return_value=(samples, 48000)
                    ),
                ):
                    self.assertEqual(decode_ref_audio("YQ==").shape, shape)

    def test_duration_rejects_one_frame_outside_either_bound(self):
        for frames in (24000 - 1, 30 * 48000 + 1):
            for channels in (1, 2):
                shape = (frames,) if channels == 1 else (frames, channels)
                samples = mx.zeros(shape, dtype=mx.float32)
                with (
                    self.subTest(frames=frames, channels=channels),
                    mock.patch(
                        "mlx_audio.audio_io.read", return_value=(samples, 48000)
                    ),
                    self.assertRaisesRegex(ValueError, r"must be 0\.5-30s"),
                ):
                    decode_ref_audio("YQ==")

    def test_bad_decoded_audio_is_a_request_error(self):
        for samples, rate in (
            (mx.array([], dtype=mx.float32), 48000),
            (mx.array([float("nan")]), 48000),
            (mx.array([float("inf")]), 48000),
            (mx.array([float("-inf")]), 48000),
            (mx.zeros((10, 3)), 48000),
            (mx.zeros(10), 24000),
        ):
            with self.subTest(shape=samples.shape, rate=rate):
                with mock.patch(
                    "mlx_audio.audio_io.read", return_value=(samples, rate)
                ):
                    with self.assertRaisesRegex(ValueError, "ref_audio"):
                        decode_ref_audio("YQ==")


class NanoPCMTest(unittest.TestCase):
    def test_pcm16_wire_format_for_mlx_arrays(self):
        # Conversion rounds before casting: +/-0.5 maps to +/-16384.
        expected = struct.pack("<6h", -32767, -16384, 0, 16384, 32767, 32767)
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dtype):
                audio = mx.array([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=dtype)
                self.assertEqual(moss_nano._pcm16(audio), expected)

    def test_pcm16_preserves_strided_sample_order(self):
        audio = mx.array(
            [-0.75, 9.0, -0.5, 9.0, -0.25, 9.0, 0.0, 9.0, 0.25, 9.0, 0.5, 9.0]
        )[::2]
        expected = struct.pack("<6h", -24575, -16384, -8192, 0, 8192, 16384)
        self.assertEqual(moss_nano._pcm16(audio), expected)

    def test_pcm16_rejects_nonfinite_samples(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(RuntimeError, "non-finite"),
            ):
                moss_nano._pcm16(mx.array([value]))


class NanoServiceTest(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.service = MossNanoService(self.model)
        self.voice = {"ref_audio": reference_wav()}
        references = {
            self.voice["ref_audio"]: mx.zeros(24000, dtype=mx.float32),
            reference_wav(sample_rate=24000, channels=2): mx.zeros(
                (24000, 2), dtype=mx.float32
            ),
        }

        def fake_decode(data, sample_rate=48000):
            if data not in references:
                raise ValueError("ref_audio is invalid")
            self.assertEqual(sample_rate, 48000)
            return references[data]

        patcher = mock.patch.object(
            moss_nano, "decode_ref_audio", side_effect=fake_decode
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cloning_metadata(self):
        self.assertEqual(self.service.voices, [])
        self.assertEqual(self.service.model_type, "moss_tts_nano")
        self.assertEqual((self.service.sample_rate, self.service.channels), (48000, 1))

    def test_buffered_output_is_downmixed_pcm16_and_48k_wav(self):
        pcm, media_type = self.service.speech_bytes(
            "hello", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(media_type, "audio/pcm")
        self.assertEqual(struct.unpack("<3h", pcm), (0, 16384, -32767))
        payload, media_type = self.service.speech_bytes("hello", voice=self.voice)
        self.assertEqual(media_type, "audio/wav")
        with wave.open(io.BytesIO(payload)) as wav:
            self.assertEqual(
                (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()),
                (48000, 1, 2),
            )
            self.assertEqual(wav.getnframes(), 3)
            self.assertEqual(wav.readframes(3), pcm)
        self.assertTrue(self.model.finished)

    def test_reference_rate_and_nano_sampling_are_forwarded(self):
        config = MossNanoConfig(
            max_new_frames=42,
            max_text_tokens=30,
            do_sample=False,
            audio_temperature=0.2,
            codec_model_ref="/cached/codec",
        )
        service = MossNanoService(self.model, config)
        voice = {
            "ref_audio": reference_wav(sample_rate=24000, channels=2),
            "ref_text": "ignored transcript",
        }
        service.speech_bytes("hello", voice=voice)
        kwargs = self.model.kwargs
        self.assertIsInstance(kwargs["ref_audio"], mx.array)
        self.assertEqual(kwargs["ref_audio"].shape, (24000, 2))
        self.assertEqual(kwargs["ref_audio_sample_rate"], 48000)
        self.assertEqual(kwargs["mode"], "voice_clone")
        self.assertFalse(kwargs["stream"])
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["max_tokens"], 42)
        self.assertEqual(kwargs["voice_clone_max_text_tokens"], 30)
        self.assertEqual(kwargs["audio_temperature"], 0.2)
        self.assertEqual(kwargs["audio_tokenizer_source"], "/cached/codec")
        self.assertNotIn("ref_text", kwargs)

    def test_optional_reference_text_is_ignored(self):
        for ref_text in ("", "transcript", "a completely different transcript"):
            self.service.speech_bytes(
                "hello", voice={**self.voice, "ref_text": ref_text}
            )
            self.assertNotIn("ref_text", self.model.kwargs)

    def test_validation_is_eager_for_buffered_and_streaming(self):
        cases = [
            ({"input": " "}, "input"),
            ({"voice": None}, "no preset voices"),
            ({"voice": "Vivian"}, "no preset voices"),
            ({"voice": {}}, "ref_audio"),
            ({"voice": {**self.voice, "unknown": 1}}, "voice object"),
            ({"voice": {**self.voice, "ref_text": 123}}, "ref_text"),
            ({"voice": {"ref_audio": "invalid"}}, "ref_audio"),
            ({"speed": 2.0}, "speed"),
            ({"instructions": "whisper"}, "instructions"),
            ({"language": "English"}, "language"),
        ]
        # Fail immediately if validation moves inside the lock; holding a real
        # non-reentrant lock here would hang the test on that regression.
        with mock.patch.object(self.service, "_lock") as lock:
            lock.__enter__.side_effect = AssertionError(
                "validation entered the generation lock"
            )
            for method in (self.service.speech_bytes, self.service.speech_stream):
                for overrides, hint in cases:
                    with self.subTest(method=method.__name__, overrides=overrides):
                        with self.assertRaisesRegex(ValueError, hint):
                            method(
                                **{"input": "hello", "voice": self.voice, **overrides}
                            )
            lock.__enter__.assert_not_called()
        self.assertIsNone(self.model.kwargs)

    def test_format_and_intervals_are_eager_errors(self):
        with self.assertRaisesRegex(ValueError, "response_format"):
            self.service.speech_bytes("hello", voice=self.voice, response_format="mp3")
        for field in ("streaming_interval", "streaming_initial_interval"):
            for value in (0, -1, 11, float("nan"), float("inf")):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaisesRegex(ValueError, field),
                ):
                    self.service.speech_stream(
                        "hello", voice=self.voice, **{field: value}
                    )

    def test_streaming_yields_finished_pcm_in_sample_aligned_chunks(self):
        stream = self.service.speech_stream(
            "hello", voice=self.voice, streaming_interval=2 / 48000
        )
        self.addCleanup(stream.close)
        self.assertIsNone(self.model.kwargs)
        first = next(stream)
        self.assertEqual(first, struct.pack("<2h", 0, 16384))
        self.assertTrue(self.model.finished)
        self.assertFalse(self.model.kwargs["stream"])
        remaining = list(stream)
        self.assertEqual(remaining, [struct.pack("<h", -32767)])
        pcm, _ = self.service.speech_bytes(
            "hello", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(first + b"".join(remaining), pcm)

    def test_partial_stream_close_leaves_service_usable(self):
        stream = self.service.speech_stream(
            "hello", voice=self.voice, streaming_interval=2 / 48000
        )
        self.addCleanup(stream.close)
        next(stream)
        # A paused client must not hold the generation lock.
        self.assertTrue(self.service._lock.acquire(blocking=False))
        self.service._lock.release()
        stream.close()
        pcm, _ = self.service.speech_bytes(
            "hello again", voice=self.voice, response_format="pcm"
        )
        self.assertEqual(pcm, struct.pack("<3h", 0, 16384, -32767))
        self.assertEqual(self.model.kwargs["text"], "hello again")

    def test_generation_failure_releases_lock(self):
        self.model.audio = mx.zeros((1, 2, 3))
        with self.assertRaisesRegex(RuntimeError, "audio shape"):
            self.service.speech_bytes("hello", voice=self.voice)
        self.assertTrue(self.model.finished)
        self.assertTrue(self.service._lock.acquire(blocking=False))
        self.service._lock.release()

    def test_empty_generation_is_a_model_failure(self):
        self.model.audio = mx.zeros(0)
        with self.assertRaisesRegex(RuntimeError, "generated no audio"):
            self.service.speech_bytes("hello", voice=self.voice)

    def test_invalid_budget_fails_before_loading(self):
        for name in ("max_new_frames", "max_text_tokens"):
            for value in (0, -1, 1.5):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    MossNanoConfig(**{name: value})


class NanoLoaderTest(unittest.TestCase):
    @unittest.skipUnless(
        HAS_MLX_AUDIO, "loader integration needs the optional mlx-audio extra"
    )
    def test_loads_codec_at_boot_and_accepts_local_codec_source(self):
        model = FakeModel()
        model._ensure_audio_tokenizer = mock.Mock()
        config = MossNanoConfig(
            model_ref="/models/nano", codec_model_ref="/models/codec"
        )
        with mock.patch("mlx_audio.tts.utils.load_model", return_value=model) as load:
            self.assertIs(moss_nano.load_moss_nano_model(config), model)
        load.assert_called_once_with("/models/nano")
        model._ensure_audio_tokenizer.assert_called_once_with(source="/models/codec")

    def test_rejects_wrong_family_or_rate_before_serving(self):
        for model in (
            SimpleNamespace(config=SimpleNamespace(model_type="qwen3_tts")),
            SimpleNamespace(
                config=SimpleNamespace(model_type="moss_tts_nano"), sample_rate=24000
            ),
        ):
            with self.assertRaisesRegex(ValueError, "MOSS Nano"):
                MossNanoService(model)


if __name__ == "__main__":
    unittest.main()
