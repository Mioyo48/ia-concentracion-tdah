from __future__ import annotations

import io
import secrets
import wave

import numpy as np
import streamlit as st
from scipy import signal


SAMPLE_RATE = 22_050
MAX_DURATION = 20 * 60
CHUNK_SECONDS = 8
NOTE_SECONDS = 0.5
SEGMENT_SECONDS = 8

CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
NOTE_FRAMES = int(SAMPLE_RATE * NOTE_SECONDS)
SEGMENT_FRAMES = int(SAMPLE_RATE * SEGMENT_SECONDS)


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


def normalize(audio: np.ndarray) -> np.ndarray:
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
    return 440.0 * (
        2.0 ** ((note - 69.0) / 12.0)
    )


def build_music_plan(
    total_frames: int,
    mode: str,
    rng: np.random.Generator,
) -> dict:
    """Crea una planificación musical aleatoria para toda la canción."""

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

    segment_count = int(
        np.ceil(
            total_frames / SEGMENT_FRAMES
        )
    )

    notes_per_segment = int(
        SEGMENT_SECONDS / NOTE_SECONDS
    )

    chord_ids = rng.integers(
        0,
        len(progression),
        segment_count,
    )

    note_choices = rng.integers(
        0,
        4,
        (
            segment_count,
            notes_per_segment,
        ),
    )

    pad_phases = rng.uniform(
        0,
        2 * np.pi,
        (
            segment_count,
            4,
        ),
    )

    melody_phases = rng.uniform(
        0,
        2 * np.pi,
        (
            segment_count,
            notes_per_segment,
        ),
    )

    if mode == "adhd":
        low_frequency = 14.5
        high_frequency = 17.5
    else:
        low_frequency = 11.5
        high_frequency = 14.5

    total_seconds = total_frames / SAMPLE_RATE

    control_count = max(
        4,
        int(total_seconds / 4) + 2,
    )

    beta_control_times = np.linspace(
        0,
        total_seconds,
        control_count,
    )

    beta_frequencies = rng.uniform(
        low_frequency,
        high_frequency,
        control_count,
    )

    return {
        "progression": progression,
        "chord_ids": chord_ids,
        "note_choices": note_choices,
        "pad_phases": pad_phases,
        "melody_phases": melody_phases,
        "beta_control_times": beta_control_times,
        "beta_frequencies": beta_frequencies,
    }


def generate_texture(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Textura ambiental reducida para evitar sonido de viento."""

    white = rng.standard_normal(
        length
    ).astype(np.float32)

    band_filter = signal.butter(
        2,
        [180, 4_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    texture = signal.sosfiltfilt(
        band_filter,
        white,
    )

    return normalize(texture)


def render_chunk(
    start_frame: int,
    frames: int,
    total_frames: int,
    plan: dict,
    rng: np.random.Generator,
    beta_phase: float,
) -> tuple[np.ndarray, float]:
    """Renderiza un bloque de ocho segundos."""

    progression = plan["progression"]
    segment_index = start_frame // SEGMENT_FRAMES

    chord = progression[
        plan["chord_ids"][segment_index]
    ]

    pad = np.zeros(
        frames,
        dtype=np.float32,
    )

    rhythm = np.zeros(
        frames,
        dtype=np.float32,
    )

    melody = np.zeros(
        frames,
        dtype=np.float32,
    )

    local_time = np.arange(
        frames,
        dtype=np.float64,
    ) / SAMPLE_RATE

    # Pads armónicos.
    amplitudes = [
        0.34,
        0.20,
        0.13,
        0.08,
    ]

    for note_index, note in enumerate(chord):
        frequency = midi_to_hz(
            note + 12
        )

        phase_offset = plan[
            "pad_phases"
        ][segment_index][note_index]

        movement = (
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

        pad += (
            amplitudes[note_index]
            * movement
            * np.sin(
                2
                * np.pi
                * frequency
                * local_time
                + phase_offset
            )
        )

        pad += (
            amplitudes[note_index]
            * 0.11
            * np.sin(
                2
                * np.pi
                * frequency
                * 2
                * local_time
                + phase_offset
            )
        )

    # Pulso musical base a 120 BPM.
    root_frequency = midi_to_hz(
        chord[0] + 24
    )

    rhythm += (
        0.30
        * np.sin(
            2
            * np.pi
            * root_frequency
            * local_time
        )
    )

    rhythm += (
        0.12
        * np.sin(
            2
            * np.pi
            * root_frequency
            * 1.5
            * local_time
        )
    )

    # Motivo generativo con cambios cada medio segundo.
    notes_per_segment = int(
        SEGMENT_SECONDS / NOTE_SECONDS
    )

    for note_index in range(notes_per_segment):
        note_start = (
            note_index
            * NOTE_FRAMES
        )

        note_end = min(
            note_start + NOTE_FRAMES,
            frames,
        )

        if note_start >= frames:
            break

        note_time = np.arange(
            note_end - note_start,
            dtype=np.float64,
        ) / SAMPLE_RATE

        selected_note = chord[
            plan["note_choices"][
                segment_index
            ][note_index]
        ]

        frequency = midi_to_hz(
            selected_note + 24
        )

        phase_offset = plan[
            "melody_phases"
        ][segment_index][note_index]

        envelope = np.sin(
            np.pi
            * note_time
            / NOTE_SECONDS
        ) ** 2

        melody[
            note_start:note_end
        ] += (
            0.17
            * envelope
            * np.sin(
                2
                * np.pi
                * frequency
                * note_time
                + phase_offset
            )
        )

    # Modulación beta continua.
    global_time = (
        start_frame
        + np.arange(frames)
    ) / SAMPLE_RATE

    frequencies = np.interp(
        global_time,
        plan["beta_control_times"],
        plan["beta_frequencies"],
    )

    phase = (
        beta_phase
        + 2
        * np.pi
        * np.cumsum(frequencies)
        / SAMPLE_RATE
    )

    beta_phase = float(phase[-1])

    beta_wave = (
        0.5
        + 0.5
        * np.sin(phase)
    )

    # Profundidad media: suficiente para estimular,
    # sin producir un bombeo agresivo.
    beta_envelope = (
        0.60
        + 0.40
        * beta_wave
    )

    music = (
        0.52 * pad
        + 0.27 * rhythm
        + 0.19 * melody
    )

    # AM aplicada sobre música, no sobre ruido puro.
    music *= beta_envelope

    texture = generate_texture(
        frames,
        rng,
    )

    audio = (
        music
        + 0.018 * texture
    )

    # Suavizado analógico.
   audio = np.tanh(
       audio * 0.45
    )

    # Volumen maestro más bajo y limpio.
    audio *= 0.55

    # Fade inicial y final.
    fade_length = min(
        int(1.5 * SAMPLE_RATE),
        total_frames // 2,
    )

    fade = np.ones(
        frames,
        dtype=np.float32,
    )

    if start_frame < fade_length:
        fade_in_length = min(
            frames,
            fade_length - start_frame,
        )

        fade[:fade_in_length] *= np.linspace(
            start_frame / fade_length,
            1,
            fade_in_length,
        )

    end_frame = start_frame + frames

    if end_frame > total_frames - fade_length:
        fade_out_start = max(
            0,
            total_frames - fade_length - start_frame,
        )

        fade_out_length = frames - fade_out_start

        fade[fade_out_start:] *= np.linspace(
            1,
            0,
            fade_out_length,
        )

    audio *= fade

    # Imagen estéreo moderada.
    side = (
        0.018
        * np.tanh(
            0.5 * melody
            + 0.2 * rhythm
        )
    )

    left = audio + side
    right = audio - side

    stereo = np.column_stack(
        (left, right)
    )

    peak = np.max(
        np.abs(stereo)
    )

    target_peak = 0.24

    if peak > target_peak:
        stereo *= target_peak / peak

    stereo = np.clip(
        stereo,
        -0.24,
        0.24,
    )

    return stereo, beta_phase


def generate_focus_wav(
    duration: int,
    mode: str,
    seed: int,
    progress_callback=None,
) -> bytes:
    """Genera WAV por bloques, permitiendo hasta 20 minutos."""

    total_frames = int(
        duration * SAMPLE_RATE
    )

    rng = np.random.default_rng(
        seed
    )

    plan = build_music_plan(
        total_frames,
        mode,
        rng,
    )

    output = io.BytesIO()

    beta_phase = 0.0

    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)

        for start_frame in range(
            0,
            total_frames,
            CHUNK_FRAMES,
        ):
            frames = min(
                CHUNK_FRAMES,
                total_frames - start_frame,
            )

            stereo, beta_phase = render_chunk(
                start_frame=start_frame,
                frames=frames,
                total_frames=total_frames,
                plan=plan,
                rng=rng,
                beta_phase=beta_phase,
            )

            pcm = np.int16(
                stereo * 32_767
            )

            wav.writeframes(
                pcm.tobytes()
            )

            if progress_callback:
                progress_callback(
                    min(
                        1.0,
                        (
                            start_frame + frames
                        ) / total_frames,
                    )
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
    """Extrae una vista previa de la forma de onda."""

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

    left = (
        stereo[:, 0].astype(np.float32)
        / 32_767
    )

    preview_samples = min(
        1_500,
        len(left),
    )

    indexes = np.linspace(
        0,
        len(left) - 1,
        preview_samples,
    ).astype(int)

    return left[indexes]


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">
            NeuroFocus / Deep Focus Generator
        </div>

        <h1>
            Deep<br>
            <span class="accent">Focus.</span>
        </h1>

        <p class="description">
            Música generativa original con pads, motivos melódicos,
            ritmo estable a 120 BPM y modulación beta suave.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

col1, col2 = st.columns(2)

with col1:
    selected_mode = st.selectbox(
        "Perfil de estimulación",
        [
            "ADHD Focus",
            "Normal Focus",
        ],
    )

with col2:
    duration = st.slider(
        "Duración de la canción",
        min_value=60,
        max_value=MAX_DURATION,
        value=300,
        step=60,
        format="%d segundos",
    )

mode = (
    "adhd"
    if selected_mode == "ADHD Focus"
    else "normal"
)

st.markdown(
    """
    <div class="notice">
        Cada generación usa una semilla aleatoria nueva.
        La misma configuración produce una composición diferente
        cada vez.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if st.button(
    "🎧 Generar nueva composición",
    type="primary",
    use_container_width=True,
):
    seed = secrets.randbits(32)

    st.session_state["generation_seed"] = seed

    progress_bar = st.progress(
        0,
        text="Preparando composición...",
    )

    with st.spinner(
        "Generando audio por bloques..."
    ):
        audio_bytes = generate_focus_wav(
            duration=duration,
            mode=mode,
            seed=seed,
            progress_callback=lambda value: progress_bar.progress(
                int(value * 100),
                text=f"Generando: {int(value * 100)}%",
            ),
        )

    progress_bar.empty()

    st.session_state["audio_bytes"] = audio_bytes
    st.session_state["audio_mode"] = mode
    st.session_state["audio_duration"] = duration

if "audio_bytes" in st.session_state:
    audio_bytes = st.session_state["audio_bytes"]
    audio_mode = st.session_state["audio_mode"]
    audio_duration = st.session_state["audio_duration"]
    generation_seed = st.session_state["generation_seed"]

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
        "⬇️ Descargar composición WAV",
        data=audio_bytes,
        file_name=(
            f"deep-focus-{audio_mode}-"
            f"{audio_duration}s-"
            f"{generation_seed}.wav"
        ),
        mime="audio/wav",
        use_container_width=True,
    )

    st.caption(
        f"Semilla de esta composición: {generation_seed}"
    )

    st.subheader("Forma de onda")

    st.line_chart(
        extract_preview(audio_bytes),
        height=180,
        use_container_width=True,
    )

st.divider()

st.caption(
    "Audio procedural original. "
    "No es un tratamiento médico para TDAH."
)
