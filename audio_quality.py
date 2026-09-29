"""Offline WAV recording diagnostics using only the Python standard library.

This demonstration reports objective signal measurements. It does not grade a
speaker, decide intelligibility, or replace professional listening.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import wave
from pathlib import Path

SILENCE_DBFS = -50.0
SILENCE_LINEAR = 10 ** (SILENCE_DBFS / 20)
CLIPPING_LINEAR = 0.998
FRAME_SECONDS = 0.02


def decode_pcm(data: bytes, sample_width: int) -> list[float]:
    """Decode little-endian PCM data to approximately [-1, 1)."""
    if sample_width not in (1, 2, 3, 4):
        raise ValueError("Supported PCM sample widths: 8, 16, 24, and 32 bit")
    if len(data) % sample_width:
        raise ValueError("PCM buffer is not aligned with sample width")
    if sample_width == 1:  # 8-bit PCM is unsigned by WAV convention.
        return [(value - 128) / 128.0 for value in data]
    scale = float(1 << (8 * sample_width - 1))
    return [
        int.from_bytes(data[offset:offset + sample_width], "little", signed=True) / scale
        for offset in range(0, len(data), sample_width)
    ]


def dbfs(value: float) -> float | None:
    return round(20 * math.log10(value), 2) if value > 0 else None


def analyze_wav(path: str | Path) -> dict:
    """Analyze an uncompressed PCM WAV file in small, fixed-duration frames."""
    file_path = Path(path)
    with wave.open(str(file_path), "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("Only uncompressed PCM WAV files are supported")
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        total_frames = wav.getnframes()
        if channels < 1 or sample_rate < 1 or sample_width not in (1, 2, 3, 4):
            raise ValueError("Unsupported WAV parameters")
        if total_frames == 0:
            raise ValueError("The WAV file contains no audio frames")

        frame_size = max(1, round(sample_rate * FRAME_SECONDS))
        count = 0
        sum_squares = 0.0
        sample_sum = 0.0
        peak = 0.0
        clipped = 0
        near_silent = 0
        frame_peaks: list[float] = []
        silent_streak = 0
        longest_silent_frames = 0
        channel_squares = [0.0] * channels
        channel_counts = [0] * channels

        while True:
            chunk = wav.readframes(frame_size)
            if not chunk:
                break
            samples = decode_pcm(chunk, sample_width)
            frame_squared = 0.0
            frame_peak = 0.0
            for idx, sample in enumerate(samples):
                magnitude = abs(sample)
                squared = sample * sample
                count += 1
                sample_sum += sample
                sum_squares += squared
                frame_squared += squared
                peak = max(peak, magnitude)
                frame_peak = max(frame_peak, magnitude)
                clipped += magnitude >= CLIPPING_LINEAR
                near_silent += magnitude < SILENCE_LINEAR
                channel = idx % channels
                channel_squares[channel] += squared
                channel_counts[channel] += 1
            frame_peaks.append(round(frame_peak, 5))
            if math.sqrt(frame_squared / len(samples)) < SILENCE_LINEAR:
                silent_streak += len(samples) // channels
                longest_silent_frames = max(longest_silent_frames, silent_streak)
            else:
                silent_streak = 0

    rms = math.sqrt(sum_squares / count)
    dc_offset = sample_sum / count
    channel_rms = [
        math.sqrt(squares / amount) if amount else 0.0
        for squares, amount in zip(channel_squares, channel_counts)
    ]
    clipping_percent = clipped * 100.0 / count
    duration = total_frames / sample_rate
    longest_silence = longest_silent_frames / sample_rate
    flags: list[dict[str, str]] = []

    if clipping_percent >= 0.1:
        flags.append({"level": "review", "message": "Possible clipping: at least 0.1% of samples are near full scale."})
    if peak < 10 ** (-25 / 20):
        flags.append({"level": "review", "message": "Low peak level (below -25 dBFS)."})
    if longest_silence >= 1.0:
        flags.append({"level": "notice", "message": "At least one continuous near-silent interval is 1 second or longer."})
    if abs(dc_offset) > 0.03:
        flags.append({"level": "review", "message": "Potential DC offset (absolute mean exceeds 0.03)."})
    if channels == 2 and min(channel_rms) > 0 and max(channel_rms) > 2 * min(channel_rms):
        flags.append({"level": "notice", "message": "Stereo channel RMS imbalance exceeds approximately 6 dB."})
    if not flags:
        flags.append({"level": "info", "message": "No threshold-based technical flags were detected; listening review may still be needed."})

    return {
        "file": file_path.name,
        "format": "PCM WAV",
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "bit_depth": sample_width * 8,
        "duration_seconds": round(duration, 3),
        "peak_dbfs": dbfs(peak),
        "rms_dbfs": dbfs(rms),
        "dc_offset": round(dc_offset, 5),
        "clipped_samples_percent": round(clipping_percent, 3),
        "near_silent_samples_percent": round(near_silent * 100.0 / count, 2),
        "longest_near_silent_interval_seconds": round(longest_silence, 3),
        "channel_rms_dbfs": [dbfs(item) for item in channel_rms],
        "waveform_peaks": frame_peaks,
        "technical_flags": flags,
        "method": "Offline 20-ms frame analysis; silence < -50 dBFS RMS; near-full-scale samples >= 0.998.",
        "limitations": "Heuristic technical diagnostics only; no assessment of intelligibility, pronunciation, persona, or speech naturalness.",
    }


def make_html_report(result: dict) -> str:
    """Create a standalone report; no external libraries, scripts, or uploads."""
    waveform = result["waveform_peaks"]
    bucket_count = min(160, len(waveform))
    if bucket_count:
        summarized = [
            max(waveform[start:end])
            for start, end in [
                (i * len(waveform) // bucket_count, (i + 1) * len(waveform) // bucket_count)
                for i in range(bucket_count)
            ]
        ]
    else:
        summarized = []
    bars = []
    for index, amp in enumerate(summarized):
        height = max(2.0, amp * 110)
        bars.append(
            f'<rect x="{index * (960 / bucket_count):.2f}" y="{60-height/2:.2f}" '
            f'width="{max(1, 960 / bucket_count - 1):.2f}" height="{height:.2f}" rx="1" />'
        )
    metrics = [
        ("Duration", f'{result["duration_seconds"]} s'),
        ("Sample rate", f'{result["sample_rate_hz"]:,} Hz'),
        ("Channels", str(result["channels"])),
        ("Bit depth", f'{result["bit_depth"]}-bit'),
        ("Peak", f'{result["peak_dbfs"]} dBFS'),
        ("RMS", f'{result["rms_dbfs"]} dBFS'),
        ("Near-clipping", f'{result["clipped_samples_percent"]}%'),
        ("Longest silence", f'{result["longest_near_silent_interval_seconds"]} s'),
    ]
    cards = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in metrics
    )
    flags = "".join(
        f'<li class="{html.escape(flag["level"])}">{html.escape(flag["message"])}</li>'
        for flag in result["technical_flags"]
    )
    file_name = html.escape(result["file"])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Quality Analysis | {file_name}</title>
<style>
:root{{color-scheme:dark}} *{{box-sizing:border-box}} body{{margin:0;background:#0b1220;color:#eff6ff;font:16px/1.5 system-ui,-apple-system,Segoe UI,Arial,sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:56px 24px}} .eyebrow{{color:#38bdf8;letter-spacing:.17em;text-transform:uppercase;font-weight:750;font-size:12px}}
h1{{font-size:clamp(30px,5vw,48px);line-height:1.1;margin:12px 0}} .sub{{color:#9db2ce;margin:0 0 36px}} .panel{{background:#141f32;border:1px solid #2a3c56;border-radius:18px;padding:26px;margin-top:22px;box-shadow:0 8px 35px #0002}}
h2{{font-size:19px;margin:0 0 20px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}} .metric{{background:#101a2b;border:1px solid #293e59;border-radius:12px;padding:18px}}
.metric span{{display:block;color:#a7bad2;font-size:13px}} .metric strong{{font-size:23px;font-variant-numeric:tabular-nums}} svg{{width:100%;height:auto;background:#0b1525;border-radius:10px}}
.waveform{{fill:#38bdf8}} ul{{padding-left:24px}} li{{margin:10px 0}} .review::marker{{color:#fb923c}} .notice::marker{{color:#facc15}} .info::marker{{color:#4ade80}}
footer{{color:#91a5bf;font-size:13px;margin-top:28px}} code{{color:#7dd3fc}}
</style></head><body><main><div class="eyebrow">Independent demonstration · Local processing</div>
<h1>Audio Quality Analysis Lab</h1><p class="sub">Technical recording diagnostics for <strong>{file_name}</strong>. No audio is uploaded or transmitted.</p>
<section class="grid">{cards}</section>
<section class="panel"><h2>Waveform overview · per-frame peak</h2><svg viewBox="0 0 960 120" role="img" aria-label="Waveform peak summary"><path d="M0 60H960" stroke="#365472" stroke-width="1"/><g class="waveform">{''.join(bars)}</g></svg></section>
<section class="panel"><h2>Technical review flags</h2><ul>{flags}</ul></section>
<footer>Method: {html.escape(result['method'])}<br><strong>Scope:</strong> {html.escape(result['limitations'])}</footer>
</main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze uncompressed PCM WAV recording quality offline.")
    parser.add_argument("wav_file", type=Path, help="Path to a WAV input file")
    parser.add_argument("--json", type=Path, dest="json_file", help="Write metrics to a JSON file")
    parser.add_argument("--html", type=Path, dest="html_file", help="Write a standalone HTML report")
    args = parser.parse_args()
    try:
        result = analyze_wav(args.wav_file)
    except (FileNotFoundError, EOFError, wave.Error, ValueError) as error:
        parser.exit(2, f"Audio analysis error: {error}\n")
    summary = {key: value for key, value in result.items() if key != "waveform_peaks"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_file:
        args.json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.html_file:
        args.html_file.write_text(make_html_report(result), encoding="utf-8")


if __name__ == "__main__":
    main()
