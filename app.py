from __future__ import annotations

import io
import wave

import numpy as np
import streamlit as st
from scipy import signal


SAMPLE_RATE = 22_050
MIN_DURATION = 5
MAX_DURATION = 180


st.set_page_config(
    page_title="NeuroFocus",
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
            background: rgba(15, 23, 42, .75);
            box-shadow: 0 30px 80px rgba(0, 0, 0, .45);
        }

        .eyebrow {
            color: #7dd3fc;
            font-size: .75rem;
            font-weight: 800;
            letter-spacing: .18em;
            text-transform: uppercase;
        }

        .hero h1 {
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


def normalize_signal(audio: np.ndarray) -> np.ndarray:
    """Normaliza el audio sin llevarlo a saturación."""

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = audio - np.mean(audio)

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if rms < 1e-8:
        return np.zeros_like(audio)

    return audio / rms


def brown_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera ruido marrón controlado."""

    white = rng.standard_normal(
        length
    ).astype(np.float32)

    brown = np.cumsum(
        white,
        dtype=np.float32,
    )

    band_filter = signal.butter(
        3,
        [30, 350],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    brown = signal.sosfiltfilt(
        band_filter,
        brown,
    )

    return normalize_signal(brown)


def pink_band_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera ruido rosa filtrado para reducir el efecto de viento."""

    white = rng.standard_normal(
        length
    ).astype(np.float32)

    spectrum = np.fft.rfft(white)

    frequencies = np.fft.rfftfreq(
        length,
        d=1 / SAMPLE_RATE,
    )

    pink_weight = np.zeros_like(
        frequencies
    )

    valid = frequencies > 0

    pink_weight[valid] = (
        1 / np.sqrt(frequencies[valid])
    )

    band_mask = (
        (frequencies >= 100)
        & (frequencies <= 5_500)
    )

    spectrum *= pink_weight
    spectrum *= band_mask

    pink = np.fft.irfft(
        spectrum,
        n=length,
    )

    return normalize_signal(pink)


def texture_noise(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una textura media más definida."""

    white = rng.standard_normal(
        length
    ).astype(np.float32)

    texture_filter = signal.butter(
        3,
        [250, 3_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    texture = signal.sosfiltfilt(
        texture_filter,
        white,
    )

    return normalize_signal(texture)


def focus_noise_bed(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Crea la capa base de concentración."""

    brown = brown_noise(
        length,
        rng,
    )

    pink = pink_band_noise(
        length,
        rng,
    )

    texture = texture_noise(
        length,
        rng,
    )

    return normalize_signal(
        0.42 * brown
        + 0.40 * pink
        + 0.18 * texture
    )


def drifting_phase(
    length: int,
    low_frequency: float,
    high_frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera modulación beta con deriva suave."""

    time = np.arange(length) / SAMPLE_RATE

    points = max(
        4,
        int(time[-1] / 3) + 2,
    )

    point_times = np.linspace(
        0,
        time[-1],
        points,
    )

    frequencies = rng.uniform(
        low_frequency,
        high_frequency,
        points,
    )

    frequencies = np.interp(
        time,
        point_times,
        frequencies,
    )

    return (
        2
        * np.pi
        * np.cumsum(frequencies)
        / SAMPLE_RATE
    )


@st.cache_data(show_spinner=False)
def synthesize_focus_audio(
    duration: int,
    mode: str,
    seed: int,
) -> bytes:
    """
    Genera audio focus procedural.

    La modulación se aplica sobre ruido.
    No se agregan tonos musicales ni samples externos.
    """

    rng = np.random.default_rng(seed)

    if mode == "adhd":
        low_frequency = 18.0
        high_frequency = 25.0
    else:
        low_frequency = 13.0
        high_frequency = 16.0

    length = int(
        duration * SAMPLE_RATE
    )

    # 80 % de capa estable.
    base_layer = focus_noise_bed(
        length,
        rng,
    )

    # 20 % de capa rítmica.
    pulse_carrier = texture_noise(
        length,
        rng,
    )

    phase_a = drifting_phase(
        length,
        low_frequency,
        high_frequency,
        rng,
    )

    phase_b = drifting_phase(
        length,
        low_frequency * 0.98,
        high_frequency * 0.98,
        rng,
    )

    # Dos modulaciones ligeramente diferentes.
    # Esto genera movimiento sin crear un pitido.
    pulse_a = (
        0.5
        + 0.5 * np.sin(phase_a)
    )

    pulse_b = (
        0.5
        + 0.5 * np.sin(phase_b)
    )

    pulse_a = pulse_a ** 2.2
    pulse_b = pulse_b ** 1.8

    pulse = (
        0.68 * pulse_a
        + 0.32 * pulse_b
    )

    # Modulación perceptible pero suave.
    envelope = 0.22 + 0.78 * pulse

    rhythmic_layer = (
        pulse_carrier * envelope
    )

    # Arquitectura anti-helicóptero.
    audio = (
        0.80 * base_layer
        + 0.20 * rhythmic_layer
    )

    # Fade in y fade out.
    fade_length = min(
        int(0.8 * SAMPLE_RATE),
        length // 2,
    )

    fade = np.ones(
        length,
        dtype=np.float32,
    )

    fade_in = np.linspace(
        0,
        1,
        fade_length,
    )

    fade[:fade_length] = fade_in
    fade[-fade_length:] = fade_in[::-1]

    audio *= fade

    # Suaviza picos agresivos.
    audio = np.tanh(
        audio * 0.65
    )

    # Nivel final moderado.
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = audio / peak
        audio *= 0.28

    # Estéreo ligero.
    stereo_side = texture_noise(
        length,
        rng,
    )

    left = audio + stereo_side * 0.018
    right = audio - stereo_side * 0.018

    stereo = np.column_stack(
        (left, right)
    )

    stereo = np.clip(
        stereo,
        -1,
        1,
    )

    pcm = np.int16(
        stereo * 32_767
    )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(
            pcm.tobytes()
        )

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
            NeuroFocus / Deep Focus Audio
        </div>

        <h1>
            Deep<br>
            <span class="accent">Focus.</span>
        </h1>

        <p class="description">
            Generador de audio procedural para concentración.
            Utiliza ruido marrón, ruido rosa filtrado y modulación
            rítmica de amplitud sin melodías ni samples externos.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

col1, col2 = st.columns(2)

with col1:
    selected_mode = st.selectbox(
        "Perfil de audio",
        [
            "ADHD Focus",
            "Normal Focus",
        ],
    )

with col2:
    duration = st.slider(
        "Duración",
        min_value=MIN_DURATION,
        max_value=MAX_DURATION,
        value=60,
        step=5,
        format="%d segundos",
    )

mode = (
    "adhd"
    if selected_mode == "ADHD Focus"
    else "normal"
)

seed = st.number_input(
    "Semilla de generación",
    min_value=1,
    max_value=999_999,
    value=7421,
    step=1,
    help="Cambia este número para obtener otra textura.",
)

st.markdown(
    """
    <div class="notice">
        El audio se crea desde cero con algoritmos matemáticos.
        No utiliza canciones, samples ni melodías de terceros.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if st.button(
    "🎧 Crear audio Deep Focus",
    type="primary",
    use_container_width=True,
):
    with st.spinner(
        "Generando textura de concentración..."
    ):
        st.session_state["audio_bytes"] = (
            synthesize_focus_audio(
                duration=duration,
                mode=mode,
                seed=int(seed),
            )
        )

        st.session_state["audio_mode"] = mode
        st.session_state["audio_duration"] = duration

if "audio_bytes" in st.session_state:
    audio_bytes = st.session_state["audio_bytes"]
    audio_mode = st.session_state["audio_mode"]
    audio_duration = st.session_state["audio_duration"]

    st.success(
        "Audio generado: "
        f"{audio_mode.upper()} · "
        f"{format_time(audio_duration)}"
    )

    st.audio(
        audio_bytes,
        format="audio/wav",
    )

    st.download_button(
        "⬇️ Descargar audio WAV",
        data=audio_bytes,
        file_name=(
            f"deep-focus-{audio_mode}-"
            f"{audio_duration}s.wav"
        ),
        mime="audio/wav",
        use_container_width=True,
    )

st.divider()

st.caption(
    "Audio procedural original para concentración. "
    "No es un tratamiento médico para TDAH."
)
