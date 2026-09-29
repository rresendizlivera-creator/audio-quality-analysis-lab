"""Tests use generated, synthetic audio only."""

import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_quality import analyze_wav, decode_pcm, make_html_report  # noqa: E402


class TestAudioQuality(unittest.TestCase):
    def make_wav(self, samples, channels=1, sample_rate=16000):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        target = Path(folder.name) / "test.wav"
        with wave.open(str(target), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
        return target

    def test_pcm_decoding(self):
        self.assertEqual(decode_pcm(bytes([0, 128, 255]), 1), [-1.0, 0.0, 127 / 128])
        self.assertEqual(decode_pcm(bytes([0, 0, 255, 127]), 2), [0, 32767 / 32768])
        self.assertAlmostEqual(decode_pcm(bytes([0, 0, 128]), 3)[0], -1.0)
        with self.assertRaises(ValueError):
            decode_pcm(bytes([0]), 2)

    def test_sine_has_expected_characteristics(self):
        samples = [int(0.4 * 32767 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)]
        report = analyze_wav(self.make_wav(samples))
        self.assertAlmostEqual(report["duration_seconds"], 1.0)
        self.assertEqual(report["channels"], 1)
        self.assertEqual(report["bit_depth"], 16)
        self.assertEqual(report["clipped_samples_percent"], 0)
        self.assertLess(report["peak_dbfs"], -7)
        self.assertEqual(len(report["waveform_peaks"]), 50)
        self.assertIn("<html", make_html_report(report))

    def test_clip_and_silence_detection(self):
        samples = [0] * 16000 + [32767] * 3200
        report = analyze_wav(self.make_wav(samples))
        self.assertGreater(report["clipped_samples_percent"], 10)
        self.assertGreaterEqual(report["longest_near_silent_interval_seconds"], 1.0)
        self.assertTrue(any("clipping" in flag["message"].lower() for flag in report["technical_flags"]))

    def test_empty_file_rejected(self):
        path = self.make_wav([])
        with self.assertRaisesRegex(ValueError, "no audio"):
            analyze_wav(path)

    def test_report_escapes_filename(self):
        samples = [100] * 1600
        path = self.make_wav(samples)
        result = analyze_wav(path)
        result["file"] = '<script>alert("x")</script>.wav'
        generated = make_html_report(result)
        self.assertNotIn('<script>alert', generated)
        self.assertIn('&lt;script&gt;', generated)


if __name__ == "__main__":
    unittest.main()
