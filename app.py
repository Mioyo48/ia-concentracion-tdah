from __future__ import annotations

import io
import wave
from typing import Final

import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file
from scipy import signal


app = Flask(__name__)

SAMPLE_RATE: Final[int] = 22_050
MIN_DURATION: Final[int] = 5
MAX_DURATION: Final[int] = 180

FOCUS_RANGES = {
    "normal": (13.0, 16.0),
    "adhd": (18.0, 25.0),
}


# ---------------------------------------------------------------------
# MOTOR DE AUDIO
# ---------------------------------------------------------------------

def normalize(audio: np.ndarray) -> np.ndarray:
    """Centra y normaliza la señal evitando divisiones por cero."""

    audio = np.asarray(audio, dtype=np.float32)
    audio -= np.mean(audio)

    deviation = np.std(audio)

    if deviation < 1e-8:
        return np.zeros_like(audio)

    return audio / deviation


def generate_brown_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera ruido marrón mediante integración de ruido blanco."""

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


def generate_pink_band_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera ruido rosa y lo filtra en banda."""

    white = rng.standard_normal(length).astype(np.float32)

    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(
        length,
        d=1.0 / SAMPLE_RATE,
    )

    weighting = np.ones_like(frequencies)
    weighting[1:] = 1.0 / np.sqrt(frequencies[1:])

    spectrum *= weighting

    pink = np.fft.irfft(
        spectrum,
        n=length,
    )

    band_pass = signal.butter(
        4,
        [80.0, 9_000.0],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    pink = signal.sosfiltfilt(
        band_pass,
        pink,
    )

    return normalize(pink)


def generate_noise_bed(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mezcla ruido marrón y ruido rosa filtrado."""

    brown = generate_brown_noise(
        length,
        rng,
    )

    pink = generate_pink_band_noise(
        length,
        rng,
    )

    return normalize(
        0.58 * brown + 0.42 * pink
    )


def generate_drifting_phase(
    length: int,
    low_frequency: float,
    high_frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Genera una fase con ligera deriva aleatoria.

    Esto evita que la modulación sea completamente mecánica
    o repetitiva.
    """

    time = np.arange(length) / SAMPLE_RATE

    center_frequency = rng.uniform(
        low_frequency,
        high_frequency,
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
        low_frequency,
        high_frequency,
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
    """
    Genera audio WAV para concentración.

    Arquitectura:

    - 80 % ruido base constante.
    - 20 % capa de ruido modulada.
    - Modulación de amplitud aplicada sobre ruido.
    - No se suman tonos sinusoidales audibles.
    - Frecuencia beta:
        Normal: 13-16 Hz
        ADHD: 18-25 Hz
    - Deriva aleatoria para evitar habituación.
    """

    low_frequency, high_frequency = FOCUS_RANGES[mode]

    length = int(duration * SAMPLE_RATE)
    rng = np.random.default_rng()

    stable_layer = generate_noise_bed(
        length,
        rng,
    )

    modulated_carrier = generate_noise_bed(
        length,
        rng,
    )

    phase = generate_drifting_phase(
        length,
        low_frequency,
        high_frequency,
        rng,
    )

    # Pulso modulador.
    pulse = 0.5 + 0.5 * np.sin(phase)

    # AM suave aplicada al ruido.
    # La señal sinusoidal solo controla amplitud;
    # no se añade como tono audible.
    modulated_layer = modulated_carrier * (
        0.70 + 0.30 * pulse
    )

    # Arquitectura anti-helicóptero:
    # 80 % estable + 20 % modulado.
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

    # Conversión a PCM 16-bit con margen de seguridad.
    pcm = np.int16(
        np.clip(
            audio * 0.89,
            -1.0,
            1.0,
        )
        * 32767
    )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    output.seek(0)

    return output.getvalue()


# ---------------------------------------------------------------------
# FRONTEND COMPLETO EMBEBIDO
# ---------------------------------------------------------------------

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>NeuroFocus</title>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>

    <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
        rel="stylesheet"
    >

    <style>
        :root {
            --bg: #020617;
            --panel: rgba(15, 23, 42, 0.72);
            --panel-light: rgba(30, 41, 59, 0.58);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e2e8f0;
            --muted: #94a3b8;
            --sky: #38bdf8;
            --sky-light: #7dd3fc;
            --rose: #f43f5e;
        }

        * {
            box-sizing: border-box;
        }

        body {
            min-height: 100vh;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px;

            color: var(--text);
            font-family: "Inter", sans-serif;

            background:
                radial-gradient(
                    circle at 15% 15%,
                    rgba(56, 189, 248, 0.20),
                    transparent 32%
                ),
                radial-gradient(
                    circle at 85% 80%,
                    rgba(244, 63, 94, 0.16),
                    transparent 30%
                ),
                radial-gradient(
                    circle at center,
                    #1e293b 0%,
                    #0f172a 52%,
                    #020617 100%
                );
        }

        .app {
            width: min(100%, 760px);
            padding: 48px;
            border: 1px solid var(--border);
            border-radius: 30px;

            background: var(--panel);
            box-shadow:
                0 30px 90px rgba(0, 0, 0, 0.45),
                inset 0 1px rgba(255, 255, 255, 0.08);

            backdrop-filter: blur(28px);
        }

        .top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
        }

        .eyebrow {
            margin: 0;
            color: var(--sky-light);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
        }

        .live {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--rose);
            box-shadow: 0 0 16px var(--rose);
        }

        h1 {
            margin: 28px 0 16px;
            font-size: clamp(42px, 8vw, 78px);
            line-height: 0.98;
            letter-spacing: -0.07em;
        }

        h1 span {
            color: var(--sky-light);
            text-shadow: 0 0 30px rgba(56, 189, 248, 0.25);
        }

        .description {
            max-width: 560px;
            margin: 0 0 38px;
            color: var(--muted);
            line-height: 1.7;
        }

        .label {
            display: block;
            margin-bottom: 12px;
            color: #cbd5e1;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .modes {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            padding: 6px;

            border: 1px solid var(--border);
            border-radius: 18px;
            background: rgba(2, 6, 23, 0.35);
        }

        .mode {
            padding: 17px;
            border: 1px solid transparent;
            border-radius: 13px;
            color: var(--muted);
            text-align: left;
            background: transparent;
            cursor: pointer;
            transition: 0.2s ease;
        }

        .mode:hover {
            color: var(--text);
            background: rgba(148, 163, 184, 0.08);
        }

        .mode.active {
            color: var(--text);
            border-color: rgba(56, 189, 248, 0.35);
            background: rgba(56, 189, 248, 0.13);
        }

        .mode-title {
            display: block;
            font-weight: 700;
        }

        .mode-subtitle {
            display: block;
            margin-top: 6px;
            color: var(--muted);
            font-size: 12px;
        }

        .duration {
            margin-top: 30px;
        }

        .duration-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .duration-value {
            color: var(--sky-light);
            font-weight: 700;
        }

        input[type="range"] {
            width: 100%;
            margin: 18px 0 0;
            accent-color: var(--sky);
            cursor: pointer;
        }

        .button {
            width: 100%;
            margin: 30px 0 18px;
            padding: 17px;
            border: 0;
            border-radius: 15px;

            color: #082f49;
            font-weight: 800;
            background: linear-gradient(
                100deg,
                var(--sky-light),
                var(--sky)
            );

            box-shadow: 0 14px 30px rgba(14, 165, 233, 0.25);
            cursor: pointer;
            transition: 0.2s ease;
        }

        .button:hover {
            transform: translateY(-2px);
            filter: brightness(1.08);
        }

        .button:disabled {
            opacity: 0.6;
            cursor: wait;
        }

        .player {
            padding: 22px;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: rgba(2, 6, 23, 0.38);
        }

        .player-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 18px;
        }

        .player-title {
            margin: 0;
            font-size: 14px;
            font-weight: 700;
        }

        .status {
            display: block;
            margin-top: 6px;
            color: var(--muted);
            font-size: 12px;
        }

        .player-buttons {
            display: flex;
            gap: 10px;
        }

        .player-button {
            width: 42px;
            height: 42px;
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--text);
            background: var(--panel-light);
            cursor: pointer;
            transition: 0.2s ease;
        }

        .player-button:hover {
            color: var(--sky-light);
            border-color: var(--sky);
            transform: translateY(-2px);
        }

        .player-button:disabled {
            opacity: 0.35;
            cursor: default;
        }

        .wave {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 4px;
            height: 40px;
            margin: 20px 0;
        }

        .wave span {
            width: 3px;
            height: 16px;
            border-radius: 5px;
            background: linear-gradient(
                to top,
                var(--rose),
                var(--sky)
            );
            animation: pulse 1.3s ease-in-out infinite alternate;
            animation-play-state: paused;
        }

        .wave.playing span {
            animation-play-state: running;
        }

        .wave span:nth-child(2) { height: 28px; }
        .wave span:nth-child(3) { height: 20px; }
        .wave span:nth-child(4) { height: 34px; }
        .wave span:nth-child(5) { height: 24px; }
        .wave span:nth-child(6) { height: 31px; }
        .wave span:nth-child(7) { height: 18px; }
        .wave span:nth-child(8) { height: 26px; }

        @keyframes pulse {
            from {
                transform: scaleY(1);
            }

            to {
                transform: scaleY(0.45);
                opacity: 0.6;
            }
        }

        .progress {
            display: grid;
            grid-template-columns: 42px 1fr 42px;
            align-items: center;
            gap: 12px;
        }

        .time {
            color: var(--muted);
            font-size: 11px;
        }

        .time:last-child {
            text-align: right;
        }

        .footer {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-top: 26px;
            color: var(--muted);
            font-size: 11px;
        }

        @media (max-width: 600px) {
            body {
                padding: 10px;
            }

            .app {
                padding: 26px 20px;
                border-radius: 24px;
            }

            .modes {
                grid-template-columns: 1fr;
            }

            .footer {
                flex-direction: column;
            }
        }
    </style>
</head>

<body>
    <main class="app">
        <div class="top">
            <p class="eyebrow">NeuroFocus / Focus Engine</p>

            <span class="live">
                <i class="dot"></i>
                Audio activo
            </span>
        </div>

        <h1>
            Entra en tu<br>
            <span>zona profunda.</span>
        </h1>

        <p class="description">
            Audio generativo para concentración sostenida, inspirado
            en técnicas de modulación acústica y texturas de ruido.
            No utiliza melodías ni tonos agudos distractores.
        </p>

        <span class="label">Perfil de concentración</span>

        <div class="modes">
            <button class="mode active" data-mode="normal">
                <span class="mode-title">Normal Focus</span>
                <span class="mode-subtitle">
                    Modulación beta suave · 13–16 Hz
                </span>
            </button>

            <button class="mode" data-mode="adhd">
                <span class="mode-title">ADHD Focus</span>
                <span class="mode-subtitle">
                    Modulación beta activa · 18–25 Hz
                </span>
            </button>
        </div>

        <div class="duration">
            <div class="duration-header">
                <span class="label">Duración de la sesión</span>
                <span class="duration-value" id="durationValue">
                    01:00
                </span>
            </div>

            <input
                id="duration"
                type="range"
                min="5"
                max="180"
                value="60"
                step="5"
            >
        </div>

        <button class="button" id="generateButton">
            Generar música focus
        </button>

        <section class="player">
            <div class="player-header">
                <div>
                    <p class="player-title">NeuroFocus Session</p>
                    <span class="status" id="status">
                        Lista para generar
                    </span>
                </div>

                <div class="player-buttons">
                    <button
                        class="player-button"
                        id="playButton"
                        disabled
                    >
                        ▶
                    </button>

                    <a
                        class="player-button"
                        id="downloadButton"
                        href="#"
                        download="neurofocus.wav"
                        style="display: grid; place-items: center; text-decoration: none;"
                    >
                        ↓
                    </a>
                </div>
            </div>

            <div class="wave" id="wave">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
                <span></span>
            </div>

            <div class="progress">
                <span class="time" id="currentTime">00:00</span>

                <input
                    id="progress"
                    type="range"
                    min="0"
                    max="100"
                    value="0"
                    step="0.1"
                    disabled
                >

                <span class="time" id="totalTime">00:00</span>
            </div>

            <audio id="audio"></audio>
        </section>

        <div class="footer">
            <span>Ruido marrón + ruido rosa filtrado</span>
            <span>AM no tonal · Deriva neural</span>
        </div>
    </main>

    <script>
        const state = {
            mode: "normal",
            objectUrl: null
        };

        const audio = document.getElementById("audio");
        const duration = document.getElementById("duration");
        const durationValue = document.getElementById("durationValue");
        const generateButton = document.getElementById("generateButton");
        const playButton = document.getElementById("playButton");
        const downloadButton = document.getElementById("downloadButton");
        const progress = document.getElementById("progress");
        const currentTime = document.getElementById("currentTime");
        const totalTime = document.getElementById("totalTime");
        const status = document.getElementById("status");
        const wave = document.getElementById("wave");

        function formatTime(seconds) {
            if (!Number.isFinite(seconds)) {
                return "00:00";
            }

            const minutes = Math.floor(seconds / 60)
                .toString()
                .padStart(2, "0");

            const remainingSeconds = Math.floor(seconds % 60)
                .toString()
                .padStart(2, "0");

            return minutes + ":" + remainingSeconds;
        }

        duration.addEventListener("input", () => {
            durationValue.textContent =
                formatTime(Number(duration.value));
        });

        document.querySelectorAll(".mode").forEach((button) => {
            button.addEventListener("click", () => {
                state.mode = button.dataset.mode;

                document.querySelectorAll(".mode")
                    .forEach((item) => {
                        item.classList.toggle(
                            "active",
                            item === button
                        );
                    });

                status.textContent =
                    state.mode === "adhd"
                        ? "Perfil ADHD seleccionado"
                        : "Perfil normal seleccionado";
            });
        });

        playButton.addEventListener("click", () => {
            if (!audio.src) {
                return;
            }

            if (audio.paused) {
                audio.play();
            } else {
                audio.pause();
            }
        });

        progress.addEventListener("input", () => {
            if (audio.duration) {
                audio.currentTime =
                    Number(progress.value) /
                    100 *
                    audio.duration;
            }
        });

        audio.addEventListener("loadedmetadata", () => {
            totalTime.textContent =
                formatTime(audio.duration);

            progress.disabled = false;
        });

        audio.addEventListener("timeupdate", () => {
            if (!audio.duration) {
                return;
            }

            progress.value =
                audio.currentTime /
                audio.duration *
                100;

            currentTime.textContent =
                formatTime(audio.currentTime);
        });

        audio.addEventListener("play", () => {
            playButton.textContent = "Ⅱ";
            wave.classList.add("playing");
        });

        audio.addEventListener("pause", () => {
            playButton.textContent = "▶";
            wave.classList.remove("playing");
        });

        generateButton.addEventListener("click", async () => {
            generateButton.disabled = true;
            generateButton.textContent = "Sintetizando audio...";
            status.textContent =
                "Generando textura focus para concentración...";

            try {
                const response = await fetch("/generate", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        mode: state.mode,
                        duration: Number(duration.value)
                    })
                });

                if (!response.ok) {
                    const errorData =
                        await response.json();

                    throw new Error(
                        errorData.error ||
                        "No se pudo generar el audio."
                    );
                }

                const blob = await response.blob();

                if (state.objectUrl) {
                    URL.revokeObjectURL(
                        state.objectUrl
                    );
                }

                state.objectUrl =
                    URL.createObjectURL(blob);

                audio.src = state.objectUrl;
                audio.load();

                downloadButton.href =
                    state.objectUrl;

                downloadButton.download =
                    "neurofocus-" +
                    state.mode +
                    "-" +
                    duration.value +
                    "s.wav";

                playButton.disabled = false;

                status.textContent =
                    "Audio listo. Reproduciendo...";

                await audio.play();
            } catch (error) {
                status.textContent =
                    error.message ||
                    "Error generando el audio.";
            } finally {
                generateButton.disabled = false;
                generateButton.textContent =
                    "Generar música focus";
            }
        });
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------
# RUTAS FLASK
# ---------------------------------------------------------------------

@app.get("/")
def home():
    return render_template_string(INDEX_HTML)


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}

    mode = str(
        payload.get("mode", "normal")
    ).lower().strip()

    try:
        duration = int(
            float(
                payload.get("duration", 60)
            )
        )
    except (TypeError, ValueError):
        return jsonify(
            error="La duración debe ser numérica."
        ), 400

    if mode not in FOCUS_RANGES:
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

    wav_data = synthesize_focus_audio(
        duration,
        mode,
    )

    return send_file(
        io.BytesIO(wav_data),
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"neurofocus-{mode}-{duration}s.wav",
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
