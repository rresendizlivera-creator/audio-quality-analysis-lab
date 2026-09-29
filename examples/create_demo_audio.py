"""Generate an original synthetic WAV file; no client or external audio needed."""

import math
import struct
import wave
from pathlib import Path


def create_demo(output: Path) -> Path:
    sample_rate = 16_000
    seconds = 2.5
    samples = []
    for index in range(int(sample_rate * seconds)):
        t = index / sample_rate
        if t < 0.35 or t >= 2.25:
            amplitude = 0.0
        elif 1.4 <= t < 1.5:
            amplitude = 1.0  # Intentionally saturated sample interval for testing.
        else:
            amplitude = 0.35 * math.sin(2 * math.pi * 440 * t)
        samples.append(int(max(-1, min(1, amplitude)) * 32767))
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return output


if __name__ == "__main__":
    destination = Path(__file__).with_name("demo.wav")
    print(f"Generated synthetic demo: {create_demo(destination)}")
