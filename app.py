from __future__ import annotations

import io
import wave
from typing import Final

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from scipy import signal


app = Flask(__name__, template_folder=".")

SAMPLE_RATE: Final[int] = 22_050
MIN_DURATION: Final[int] = 5
MAX_DURATION: Final[int] = 180

MODE_RANGES: Final[dict[str, tuple[float, float]]] = {
    "normal": (13.0, 16.0),
    "adhd": (18.0, 25.0),
}


def normalize(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    samples -= np.mean(samples, dtype=np.float64)

    deviation = float(np.std(samples, dtype=np.float64))

    if deviation < 1e-8:
        return np.zeros_like(samples)

    return samples / deviation


def brown_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(length).astype(np.float32)
    brown = np.cumsum(white, dtype=np.float64)

    high_pass = signal.butter(
        2,
        20.0,
        btype="highpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    brown = signal.sosfiltfilt(high_pass, brown)

    return normalize(brown)


def pink_band_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    white = rng.standard_normal(length).astype(np.float32)

    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(
        length,
        d=1.0 / SAMPLE_RATE,
    )

    weighting = np.ones_like(frequencies)
    weighting[1:] = 1.0 / np.sqrt(frequencies[1:])

    spectrum *= weighting

    pink = np.fft.irfft(spectrum, n=length)

    band_pass = signal.butter(
        4,
        [80.0, 9_000.0],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    pink = signal.sosfiltfilt(band_pass, pink)

    return normalize(pink)


def noise_bed(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    brown = brown_noise(length, rng)
    pink = pink_band_noise(length, rng)

    return normalize(
        0.58 * brown + 0.42 * pink
    )


def drifting_phase(
    length: int,
    low_hz: float,
    high_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / SAMPLE_RATE

    center_frequency = rng.uniform(
        low_hz,
        high_hz,
    )

    control_count = max(
        4,
        int(np.ceil(time[-1] / 2.0)) + 1,
    )

    control_time = np.linspace(
        0.0,
        time[-1],
        control_count,
    )

    drift = rng.normal(
        0.0,
        0.22,
        control_count,
    )

    drift = np.interp(
        time,
        control_time,
        drift,
    )

    instantaneous_frequency = np.clip(
        center_frequency + drift,
        low_hz,
        high_hz,
    )

    return (
        2.0
        * np.pi
        * np.cumsum(instantaneous_frequency)
        / SAMPLE_RATE
    )


def synthesize_focus_audio(
    duration: int,
    mode: str,
) -> bytes:
    low_hz, high_hz = MODE_RANGES[mode]

    length = int(duration * SAMPLE_RATE)
    rng = np.random.default_rng()

    stable_layer = noise_bed(length, rng)
    pulse_carrier = noise_bed(length, rng)

    phase = drifting_phase(
        length,
        low_hz,
        high_hz,
        rng,
    )

    # La modulación se aplica al ruido.
    # No se agregan tonos sinusoidales audibles.
    pulse = 0.5 + 0.5 * np.sin(phase)

    # Modulación suave para evitar efecto helicóptero.
    modulated_layer = pulse_carrier * (
        0.70 + 0.30 * pulse
    )

    # 80 % de base constante y 20 % de capa modulada.
    audio = (
        0.80 * stable_layer
        + 0.20 * modulated_layer
    )

    # Fade in/out para evitar clics.
    fade_samples = min(
        int(0.75 * SAMPLE_RATE),
        length // 2,
    )

    fade = np.ones(
        length,
        dtype=np.float32,
    )

    ramp = np.linspace(
        0.0,
        1.0,
        fade_samples,
        dtype=np.float32,
    )

    fade[:fade_samples] = ramp
    fade[-fade_samples:] = ramp[::-1]

    audio = normalize(audio * fade)

    # Margen de seguridad contra clipping.
    pcm = np.int16(
        np.clip(
            audio * 0.89,
            -1.0,
            1.0,
        )
        * 32_767
    )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

    output.seek(0)

    return output.getvalue()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}

    mode = str(
        payload.get("mode", "normal")
    ).lower().strip()

    try:
        duration = int(
            float(
                payload.get(
                    "duration",
                    60,
                )
            )
        )
    except (TypeError, ValueError):
        return jsonify(
            error="La duración debe ser un número entero."
        ), 400

    if mode not in MODE_RANGES:
        return jsonify(
            error="El modo debe ser 'normal' o 'adhd'."
        ), 400

    if not MIN_DURATION <= duration <= MAX_DURATION:
        return jsonify(
            error=(
                f"La duración debe estar entre "
                f"{MIN_DURATION} y {MAX_DURATION} segundos."
            )
        ), 400

    wav_bytes = synthesize_focus_audio(
        duration,
        mode,
    )

    return send_file(
        io.BytesIO(wav_bytes),
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"focus-{mode}-{duration}s.wav",
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
