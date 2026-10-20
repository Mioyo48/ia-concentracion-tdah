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
    page_title="NeuroFocus Deep Focus",
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
    """Normaliza el audio mediante RMS."""

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio -= np.mean(audio)

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if rms < 1e-8:
        return np.zeros_like(audio)

    return audio / rms


def midi_to_hz(note: float) -> float:
    """Convierte una nota MIDI a frecuencia."""

    return 440.0 * (
        2.0 ** ((note - 69.0) / 12.0)
    )


def generate_beta_phase(
    length: int,
    low_frequency: float,
    high_frequency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una fase beta con deriva de frecuencia."""

    time = np.arange(
        length,
        dtype=np.float64,
    ) / SAMPLE_RATE

    control_count = max(
        4,
        int(time[-1] / 4) + 2,
    )

    control_time = np.linspace(
        0,
        time[-1],
        control_count,
    )

    frequencies = rng.uniform(
        low_frequency,
        high_frequency,
        control_count,
    )

    frequencies = np.interp(
        time,
        control_time,
        frequencies,
    )

    return (
        2.0
        * np.pi
        * np.cumsum(frequencies)
        / SAMPLE_RATE
    )


def generate_soft_texture(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genera una textura ambiental muy baja."""

    white = rng.standard_normal(
        length
    ).astype(np.float32)

    texture_filter = signal.butter(
        3,
        [180, 5_000],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    texture = signal.sosfiltfilt(
        texture_filter,
        white,
    )

    return normalize_signal(texture)


@st.cache_data(show_spinner=False)
def synthesize_focus_audio(
    duration: int,
    mode: str,
    seed: int,
) -> bytes:
    """
    Genera una pieza original de deep focus.

    Incluye:

    - Pads armónicos.
    - Motivos melódicos.
    - Pulsos rítmicos.
    - Modulación beta.
    - Estéreo suave.
    - Sin canciones ni samples externos.
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

    pad = np.zeros(
        length,
        dtype=np.float32,
    )

    rhythm = np.zeros(
        length,
        dtype=np.float32,
    )

    melody = np.zeros(
        length,
        dtype=np.float32,
    )

    # Progresión original en una escala atmosférica.
    progression = [
        [50, 57, 62, 65],
        [48, 55, 60, 64],
        [53, 60, 65, 69],
        [45, 52, 57, 60],
        [50, 57, 62, 67],
        [48, 55, 60, 65],
        [53, 60, 64, 69],
        [45, 52, 57, 62],
    ]

    segment_length = int(
        8 * SAMPLE_RATE
    )

    segment_count = int(
        np.ceil(length / segment_length)
    )

    for segment_index in range(segment_count):
        start = (
            segment_index
            * segment_length
        )

        if start >= length:
            break

        end = min(
            start + segment_length,
            length,
        )

        chord = progression[
            segment_index
            % len(progression)
        ]

        local_length = end - start

        local_time = np.arange(
            local_length,
            dtype=np.float64,
        ) / SAMPLE_RATE

        # Pad armónico.
        amplitudes = [
            0.34,
            0.20,
            0.13,
            0.08,
        ]

        for note_index, note in enumerate(
            chord
        ):
            frequency = midi_to_hz(note)

            phase_offset = rng.uniform(
                0,
                2 * np.pi,
            )

            slow_movement = (
                0.92
                + 0.08
                * np.sin(
                    2
                    * np.pi
                    * 0.08
                    * local_time
                    + phase_offset
                )
            )

            pad[start:end] += (
                amplitudes[note_index]
                * slow_movement
                * np.sin(
                    2
                    * np.pi
                    * frequency
                    * local_time
                    + phase_offset
                )
            )

            # Segundo armónico suave.
            pad[start:end] += (
                amplitudes[note_index]
                * 0.12
                * np.sin(
                    2
                    * np.pi
                    * frequency
                    * 2
                    * local_time
                    + phase_offset
                )
            )

        # Pulso musical de baja frecuencia.
        root_frequency = midi_to_hz(
            chord[0] + 12
        )

        rhythm[start:end] += (
            0.34
            * np.sin(
                2
                * np.pi
                * root_frequency
                * local_time
            )
        )

        rhythm[start:end] += (
            0.13
            * np.sin(
                2
                * np.pi
                * root_frequency
                * 1.5
                * local_time
            )
        )

        # Motivo melódico original.
        note_length = int(
            1.6 * SAMPLE_RATE
        )

        for note_index in range(5):
            note_start = (
                start
                + note_index
                * note_length
            )

            note_end = min(
                note_start + note_length,
                end,
            )

            if note_start >= length:
                break

            note_time = np.arange(
                note_end - note_start,
                dtype=np.float64,
            ) / SAMPLE_RATE

            selected_note = chord[
                (
                    note_index
                    + segment_index
                )
                % len(chord)
            ]

            melody_frequency = midi_to_hz(
                selected_note + 12
            )

            note_duration = max(
                note_length / SAMPLE_RATE,
                0.01,
            )

            envelope = np.sin(
                np.pi
                * note_time
                / note_duration
            ) ** 2

            melody[
                note_start:note_end
            ] += (
                0.18
                * envelope
                * np.sin(
                    2
                    * np.pi
                    * melody_frequency
                    * note_time
                )
            )

    # Modulación beta aplicada principalmente al ritmo.
    beta_phase = generate_beta_phase(
        length,
        low_frequency,
        high_frequency,
        rng,
    )

    beta_pulse = (
        0.5
        + 0.5
        * np.sin(beta_phase)
    )

    beta_pulse = beta_pulse ** 1.7

    rhythm *= (
        0.30
        + 0.70 * beta_pulse
    )

    pad = normalize_signal(pad)
    rhythm = normalize_signal(rhythm)
    melody = normalize_signal(melody)

    texture = generate_soft_texture(
        length,
        rng,
    )

    # Mezcla musical.
    # La textura ambiental queda muy baja para evitar sonido de viento.
    audio = (
        0.48 * pad
        + 0.30 * rhythm
        + 0.19 * melody
        + 0.03 * texture
    )

    # Fade de entrada y salida.
    fade_length = min(
        int(1.2 * SAMPLE_RATE),
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

    # Suavizado de picos.
    audio = np.tanh(
        audio * 0.65
    )

    # Volumen final seguro.
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = (
            audio
            / peak
            * 0.38
        )

    # Estéreo sutil.
    stereo_width = (
        0.025
        * normalize_signal(
            pad * 0.7
            + melody * 0.3
        )
    )

    left = audio + stereo_width
    right = audio - stereo_width

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


def extract_preview(
    audio_bytes: bytes,
) -> np.ndarray:
    """Extrae una vista previa de la onda para Streamlit."""

    raw = np.frombuffer(
        audio_bytes[44:],
        dtype=np.int16,
    )

    if len(raw) < 2:
        return np.zeros(100)

    stereo = raw.reshape(
        -1,
        2,
    )

    left_channel = (
        stereo[:, 0].astype(np.float32)
        / 32_767
    )

    preview_size = min(
        1_500,
        len(left_channel),
    )

    indexes = np.linspace(
        0,
        len(left_channel) - 1,
        preview_size,
    ).astype(int)

    return left_channel[indexes]


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
            Música generativa original con pads armónicos,
            ondas rítmicas y modulación beta para sesiones
            prolongadas de concentración.
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
    "Semilla musical",
    min_value=1,
    max_value=999_999,
    value=7421,
    step=1,
    help="Cambia este número para crear otra composición.",
)

st.markdown(
    """
    <div class="notice">
        Cada audio se crea desde cero con síntesis matemática.
        No se utilizan canciones, samples ni melodías protegidas.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if st.button(
    "🎧 Crear música Deep Focus",
    type="primary",
    use_container_width=True,
):
    with st.spinner(
        "Generando composición focus..."
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
        "Composición creada: "
        f"{audio_mode.upper()} · "
        f"{format_time(audio_duration)}"
    )

    st.audio(
        audio_bytes,
        format="audio/wav",
    )

    st.download_button(
        "⬇️ Descargar música WAV",
        data=audio_bytes,
        file_name=(
            f"deep-focus-{audio_mode}-"
            f"{audio_duration}s.wav"
        ),
        mime="audio/wav",
        use_container_width=True,
    )

    st.subheader("Visualización de la onda")

    st.line_chart(
        extract_preview(audio_bytes),
        height=180,
        use_container_width=True,
    )

st.divider()

st.caption(
    "Composición procedural original. "
    "No es un tratamiento médico para TDAH."
)
