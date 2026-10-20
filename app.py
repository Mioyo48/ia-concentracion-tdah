from __future__ import annotations

import io
import wave

import numpy as np
import streamlit as st
from scipy import signal


SAMPLE_RATE = 22_050
MIN_DURATION = 5
MAX_DURATION = 180

FOCUS_RANGES = {
    "normal": (13.0, 16.0),
    "adhd": (18.0, 25.0),
}


st.set_page_config(
    page_title="NeuroFocus Audio",
    page_icon="🎧",
    layout="centered",
)


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(
                    circle at 15% 10%,
                    rgba(56, 189, 248, .20),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 85% 85%,
                    rgba(244, 63, 94, .16),
                    transparent 30%
                ),
                linear-gradient(
                    135deg,
                    #020617,
                    #0f172a,
                    #1e293b
                );
        }

        .block-container {
            max-width: 900px;
            padding-top: 3rem;
        }

        .hero {
            padding: 3rem;
            border: 1px solid rgba(148, 163, 184, .2);
            border-radius: 28px;
            background: rgba(15, 23, 42, .72);
            box-shadow: 0 30px 80px rgba(0, 0, 0, .4);
        }

        .eyebrow {
            color: #7dd3fc;
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        h1 {
            font-size: clamp(2.8rem, 8vw, 5.8rem);
            line-height: .95;
            letter-spacing: -.07em;
        }

        .accent {
            color: #7dd3fc;
        }

        .description {
            color: #94a3b8;
            line-height: 1.7;
        }

        .notice {
            padding: 1rem;
            border-left: 3px solid #38bdf8;
            border-radius: 8px;
            color: #cbd5e1;
            background: rgba(56, 189, 248, .08);
            font-size: .85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize(samples: np.ndarray) -> np.ndarray:
    """Normaliza una señal de audio."""

    samples = np.asarray(samples, dtype=np.float32)
    samples = samples - np.mean(samples)

    deviation = np.std(samples)

    if deviation < 1e-8:
        return np.zeros_like(samples)

    return samples / deviation


def generate_brown_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera ruido marrón."""

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
    """Genera ruido rosa filtrado en banda."""

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
    """Mezcla ruido marrón y rosa."""

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
    """Genera una frecuencia beta con deriva aleatoria."""

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
        0,
        time[-1],
        control_count,
    )

    drift = rng.normal(
        0,
        0.22,
        control_count,
    )

    drift = np.interp(
        time,
        control_time,
        drift,
    )

    frequency = np.clip(
        center_frequency + drift,
        low_frequency,
        high_frequency,
    )

    return (
        2
        * np.pi
        * np.cumsum(frequency)
        / SAMPLE_RATE
    )


@st.cache_data(show_spinner=False)
def synthesize_focus_audio(
    duration: int,
    mode: str,
    seed: int,
) -> bytes:
    """Genera un WAV procedural para deep focus."""

    low_frequency, high_frequency = FOCUS_RANGES[mode]

    length = int(
        duration * SAMPLE_RATE
    )

    rng = np.random.default_rng(seed)

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

    # El seno únicamente controla la amplitud.
    # No se agrega como tono audible.
    pulse = 0.5 + 0.5 * np.sin(phase)

    modulated_layer = modulated_carrier * (
        0.70 + 0.30 * pulse
    )

    # 80 % base estable + 20 % capa modulada.
    audio = (
        0.80 * stable_layer
        + 0.20 * modulated_layer
    )

    # Fade para evitar clics.
    fade_samples = min(
        int(0.75 * SAMPLE_RATE),
        length // 2,
    )

    fade = np.ones(
        length,
        dtype=np.float32,
    )

    ramp = np.linspace(
        0,
        1,
        fade_samples,
        dtype=np.float32,
    )

    fade[:fade_samples] = ramp
    fade[-fade_samples:] = ramp[::-1]

    audio = normalize(audio * fade)

    # Margen de volumen para evitar clipping.
    audio = np.clip(
        audio * 0.89,
        -1,
        1,
    )

    pcm = np.int16(
        audio * 32_767
    )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    output.seek(0)

    return output.getvalue()


def format_time(seconds: int) -> str:
    minutes = seconds // 60
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:02d}"


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">
            NeuroFocus / Audio Generator
        </div>

        <h1>
            Deep<br>
            <span class="accent">Focus.</span>
        </h1>

        <p class="description">
            Generador de audio procedural para sesiones de concentración.
            Utiliza ruido marrón, ruido rosa filtrado y modulación de amplitud
            suave sin melodías ni samples externos.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

col1, col2 = st.columns(2)

with col1:
    mode_label = st.selectbox(
        "Perfil de concentración",
        options=[
            "ADHD Focus",
            "Normal Focus",
        ],
    )

with col2:
    duration = st.slider(
        "Duración del audio",
        min_value=MIN_DURATION,
        max_value=MAX_DURATION,
        value=60,
        step=5,
        format="%d segundos",
    )

mode = (
    "adhd"
    if mode_label == "ADHD Focus"
    else "normal"
)

seed = st.number_input(
    "Semilla de generación",
    min_value=1,
    max_value=999_999,
    value=7421,
    step=1,
    help="Usa otra semilla para crear una textura diferente.",
)

st.markdown(
    """
    <div class="notice">
        El audio se genera desde cero mediante algoritmos matemáticos.
        No se utilizan canciones, samples ni melodías de terceros.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

generate = st.button(
    "🎧 Crear audio Deep Focus",
    type="primary",
    use_container_width=True,
)

if generate:
    with st.spinner("Generando textura acústica..."):
        audio_bytes = synthesize_focus_audio(
            duration=duration,
            mode=mode,
            seed=int(seed),
        )

        st.session_state["audio_bytes"] = audio_bytes
        st.session_state["audio_mode"] = mode
        st.session_state["audio_duration"] = duration

if "audio_bytes" in st.session_state:
    audio_bytes = st.session_state["audio_bytes"]
    audio_mode = st.session_state["audio_mode"]
    audio_duration = st.session_state["audio_duration"]

    st.success(
        f"Audio creado: {audio_mode.upper()} · "
        f"{format_time(audio_duration)}"
    )

    st.audio(
        audio_bytes,
        format="audio/wav",
    )

    st.download_button(
        label="⬇️ Descargar audio WAV",
        data=audio_bytes,
        file_name=(
            f"neurofocus-{audio_mode}-"
            f"{audio_duration}s.wav"
        ),
        mime="audio/wav",
        use_container_width=True,
    )

    st.caption(
        "Guarda también la semilla, duración y modo para conservar "
        "la trazabilidad de esta generación."
    )

st.divider()

st.caption(
    "Audio procedural original. No es un tratamiento médico para TDAH."
)
