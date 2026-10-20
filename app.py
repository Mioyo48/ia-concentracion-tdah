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
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    audio = audio - np.mean(audio)

    rms = np.sqrt(np.mean(audio ** 2))

    if rms < 1e-8:
        return np.zeros_like(audio)

    return audio / rms


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def build_music_plan(
    total_frames: int,
    mode: str,
    rng: np.random.Generator,
) -> dict:
    """
    Crea una planificación musical diferente en cada generación.

    Incluye:
    - Cambios de acordes sin repetir inmediatamente.
    - Microvariaciones de tiempo.
    - Silencios aleatorios.
    - Velocidades diferentes.
    - Modulación beta variable.
    """

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

    segment_count = max(
        1,
        int(
            np.ceil(
                total_frames / SEGMENT_FRAMES
            )
        ),
    )

    notes_per_segment = int(
        SEGMENT_SECONDS / NOTE_SECONDS
    )

    chord_ids = np.empty(
        segment_count,
        dtype=np.int32,
    )

    chord_ids[0] = rng.integers(
        0,
        len(progression),
    )

    for index in range(1, segment_count):
        available = np.arange(
            len(progression)
        )

        available = available[
            available != chord_ids[index - 1]
        ]

        chord_ids[index] = rng.choice(
            available
        )

    # Esta variable corrige el KeyError:
    # render_chunk() la utiliza para escoger las notas.
    note_choices = rng.integers(
        0,
        4,
        size=(
            segment_count,
            notes_per_segment,
        ),
    )

    # Algunos eventos descansan para evitar
    # que todo suene mecánico y repetitivo.
    note_active = (
        rng.random(
            (
                segment_count,
                notes_per_segment,
            )
        )
        > 0.08
    )

    # Pequeñas variaciones de tiempo.
    note_starts = np.clip(
        rng.normal(
            0.0,
            0.018,
            (
                segment_count,
                notes_per_segment,
            ),
        ),
        -0.04,
        0.04,
    )

    # Duraciones variables para evitar notas idénticas.
    note_lengths = np.clip(
        rng.normal(
            0.39,
            0.05,
            (
                segment_count,
                notes_per_segment,
            ),
        ),
        0.26,
        0.49,
    )

    # Variaciones de intensidad.
    note_velocities = np.clip(
        rng.normal(
            0.96,
            0.11,
            (
                segment_count,
                notes_per_segment,
            ),
        ),
        0.70,
        1.18,
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
        beta_low = 14.5
        beta_high = 17.5
    else:
        beta_low = 11.5
        beta_high = 14.5

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
        beta_low,
        beta_high,
        control_count,
    )

    macro_phase = rng.uniform(
        0,
        2 * np.pi,
    )

    return {
        "progression": progression,
        "chord_ids": chord_ids,
        "note_choices": note_choices,
        "note_active": note_active,
        "note_starts": note_starts,
        "note_lengths": note_lengths,
        "note_velocities": note_velocities,
        "pad_phases": pad_phases,
        "melody_phases": melody_phases,
        "beta_control_times": beta_control_times,
        "beta_frequencies": beta_frequencies,
        "macro_phase": macro_phase,
    }


def generate_texture(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Textura muy suave y filtrada.
    Se mantiene baja para evitar sonido de viento saturado.
    """

    white = rng.standard_normal(
        length
    ).astype(np.float32)

    texture_filter = signal.butter(
        2,
        [220, 3_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    texture = signal.sosfiltfilt(
        texture_filter,
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
    """
    Renderiza un bloque de audio con:
    - Pads armónicos.
    - Pulsos musicales.
    - Motivos melódicos.
    - Modulación beta suave.
    - Variaciones aleatorias.
    """

    progression = plan["progression"]

    segment_index = min(
        start_frame // SEGMENT_FRAMES,
        len(plan["chord_ids"]) - 1,
    )

    chord = progression[
        plan["chord_ids"][segment_index]
    ]

    segment_offset_frames = (
        start_frame % SEGMENT_FRAMES
    )

    segment_offset_seconds = (
        segment_offset_frames / SAMPLE_RATE
    )

    local_time = (
        np.arange(
            frames,
            dtype=np.float64,
        )
        / SAMPLE_RATE
    )

    segment_time = (
        segment_offset_seconds
        + local_time
    )

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

    def render_pad(
        pad_chord: list[int],
        phases: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        output = np.zeros(
            len(times),
            dtype=np.float32,
        )

        amplitudes = [
            0.28,
            0.17,
            0.10,
            0.06,
        ]

        for note_index, note in enumerate(
            pad_chord
        ):
            frequency = midi_to_hz(
                note + 12
            )

            phase_offset = phases[
                note_index
            ]

            movement = (
                0.92
                + 0.08
                * np.sin(
                    2
                    * np.pi
                    * 0.07
                    * times
                    + phase_offset
                )
            )

            output += (
                amplitudes[note_index]
                * movement
                * np.sin(
                    2
                    * np.pi
                    * frequency
                    * times
                    + phase_offset
                )
            )

            output += (
                amplitudes[note_index]
                * 0.08
                * np.sin(
                    2
                    * np.pi
                    * frequency
                    * 2
                    * times
                    + phase_offset
                )
            )

        return output

    current_pad = render_pad(
        chord,
        plan["pad_phases"][segment_index],
        segment_time,
    )

    # Crossfade entre acordes para que los cambios
    # no suenen cortados ni robóticos.
    if segment_index > 0:
        previous_chord = progression[
            plan["chord_ids"][segment_index - 1]
        ]

        previous_pad = render_pad(
            previous_chord,
            plan["pad_phases"][segment_index - 1],
            segment_time + SEGMENT_SECONDS,
        )

        crossfade_length = 1.2

        crossfade = np.clip(
            segment_time / crossfade_length,
            0.0,
            1.0,
        )

        pad = (
            previous_pad * (1.0 - crossfade)
            + current_pad * crossfade
        )
    else:
        pad = current_pad

    notes_per_segment = int(
        SEGMENT_SECONDS / NOTE_SECONDS
    )

    selected_choices = plan[
        "note_choices"
    ][segment_index]

    selected_active = plan[
        "note_active"
    ][segment_index]

    selected_starts = plan[
        "note_starts"
    ][segment_index]

    selected_lengths = plan[
        "note_lengths"
    ][segment_index]

    selected_velocities = plan[
        "note_velocities"
    ][segment_index]

    selected_phases = plan[
        "melody_phases"
    ][segment_index]

    root_frequency = midi_to_hz(
        chord[0] + 12
    )

    for note_index in range(
        notes_per_segment
    ):
        if not selected_active[note_index]:
            continue

        note_start = (
            note_index * NOTE_SECONDS
            + float(
                selected_starts[note_index]
            )
        )

        note_duration = float(
            selected_lengths[note_index]
        )

        note_end = (
            note_start + note_duration
        )

        relative_start = int(
            np.floor(
                (
                    note_start
                    - segment_offset_seconds
                )
                * SAMPLE_RATE
            )
        )

        relative_end = int(
            np.ceil(
                (
                    note_end
                    - segment_offset_seconds
                )
                * SAMPLE_RATE
            )
        )

        clip_start = max(
            0,
            relative_start,
        )

        clip_end = min(
            frames,
            relative_end,
        )

        if clip_end <= clip_start:
            continue

        event_time = (
            segment_offset_seconds
            + np.arange(
                clip_start,
                clip_end,
                dtype=np.float64,
            )
            / SAMPLE_RATE
        )

        elapsed = (
            event_time - note_start
        )

        normalized_elapsed = np.clip(
            elapsed / note_duration,
            0.0,
            1.0,
        )

        envelope = (
            np.sin(
                np.pi
                * normalized_elapsed
            )
            ** 2
        )

        velocity = float(
            selected_velocities[note_index]
        )

        phase_offset = float(
            selected_phases[note_index]
        )

        selected_note = chord[
            int(
                selected_choices[note_index]
            )
        ]

        melody_frequency = midi_to_hz(
            selected_note + 24
        )

        melody_slice = (
            0.12
            * velocity
            * envelope
            * np.sin(
                2
                * np.pi
                * melody_frequency
                * elapsed
                + phase_offset
            )
        )

        melody_slice += (
            0.018
            * velocity
            * envelope
            * np.sin(
                2
                * np.pi
                * melody_frequency
                * 2
                * elapsed
                + phase_offset
            )
        )

        melody[
            clip_start:clip_end
        ] += melody_slice.astype(
            np.float32
        )

        # Pulso musical breve, con acentos
        # diferentes para mayor energía.
        pulse_duration = min(
            0.32,
            max(
                0.16,
                note_duration * 0.80,
            ),
        )

        pulse_position = np.clip(
            elapsed / pulse_duration,
            0.0,
            1.0,
        )

        pulse_envelope = (
            np.sin(
                np.pi
                * pulse_position
            )
            ** 2
        )

        pulse_envelope[
            elapsed > pulse_duration
        ] = 0.0

        accent = (
            1.10
            if note_index % 4 == 0
            else 0.82
        )

        pulse_slice = (
            0.15
            * velocity
            * accent
            * pulse_envelope
            * np.sin(
                2
                * np.pi
                * root_frequency
                * elapsed
                + phase_offset * 0.35
            )
        )

        pulse_slice += (
            0.035
            * velocity
            * pulse_envelope
            * np.sin(
                2
                * np.pi
                * root_frequency
                * 1.5
                * elapsed
                + phase_offset
            )
        )

        rhythm[
            clip_start:clip_end
        ] += pulse_slice.astype(
            np.float32
        )

    # Modulación beta continua con ligera deriva.
    global_time = (
        start_frame
        + np.arange(frames)
    ) / SAMPLE_RATE

    beta_frequencies = np.interp(
        global_time,
        plan["beta_control_times"],
        plan["beta_frequencies"],
    )

    phase = (
        beta_phase
        + 2
        * np.pi
        * np.cumsum(beta_frequencies)
        / SAMPLE_RATE
    )

    beta_phase = float(
        phase[-1]
    )

    beta_wave = (
        0.5
        + 0.5
        * np.sin(phase)
    )

    # Profundidad moderada para evitar
    # bombeo o efecto helicóptero.
    beta_envelope = (
        0.78
        + 0.22
        * beta_wave
    )

    music = (
        0.50 * pad
        + 0.28 * rhythm * beta_envelope
        + 0.22 * melody * beta_envelope
    )

    texture = generate_texture(
        frames,
        rng,
    )

    audio = (
        music
        + 0.006 * texture
    )

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
            1.0,
            fade_in_length,
        ).astype(np.float32)

    end_frame = start_frame + frames

    if end_frame > total_frames - fade_length:
        fade_out_start = max(
            0,
            total_frames
            - fade_length
            - start_frame,
        )

        fade_out_length = (
            frames - fade_out_start
        )

        fade[fade_out_start:] *= np.linspace(
            1.0,
            0.0,
            fade_out_length,
        ).astype(np.float32)

    audio *= fade

    # Saturación muy ligera para suavizar
    # sin destruir el rango dinámico.
    audio = np.tanh(
        audio * 0.60
    )

    audio *= 0.58

    # Estéreo sutil.
    side = (
        0.018
        * np.tanh(
            0.5 * melody
            + 0.15 * pad
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
        stereo *= (
            target_peak / peak
        )

    stereo = np.clip(
        stereo,
        -0.24,
        0.24,
    )

    return stereo.astype(
        np.float32
    ), beta_phase


def generate_focus_wav(
    duration: int,
    mode: str,
    seed: int,
    progress_callback=None,
) -> bytes:
    """Genera un archivo WAV estéreo de hasta 20 minutos."""

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
    """Extrae muestras para mostrar la forma de onda."""

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
            Música generativa original con pads armónicos,
            pulsos musicales, variaciones melódicas y modulación
            beta suave para sesiones de concentración.
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
        Cada generación utiliza una semilla aleatoria nueva.
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
                text=(
                    f"Generando: "
                    f"{int(value * 100)}%"
                ),
            ),
        )

    progress_bar.empty()

    st.session_state["audio_bytes"] = audio_bytes
    st.session_state["audio_mode"] = mode
    st.session_state["audio_duration"] = duration


if "audio_bytes" in st.session_state:
    audio_bytes = bytes(
        st.session_state["audio_bytes"]
    )

    audio_mode = st.session_state[
        "audio_mode"
    ]

    audio_duration = st.session_state[
        "audio_duration"
    ]

    generation_seed = st.session_state[
        "generation_seed"
    ]

    st.success(
        "Composición creada: "
        f"{audio_mode.upper()} · "
        f"{format_time(audio_duration)}"
    )

    st.audio(
        audio_bytes,
        format="audio/wav",
    )

    file_name = (
        f"deep-focus-{audio_mode}-"
        f"{audio_duration}s-"
        f"{generation_seed}.wav"
    )

    st.download_button(
        label="⬇️ Descargar composición WAV",
        data=audio_bytes,
        file_name=file_name,
        mime="audio/wav",
        key="download_focus_audio",
        on_click="ignore",
        use_container_width=True,
    )

    st.caption(
        "Tamaño del archivo: "
        f"{len(audio_bytes) / 1_000_000:.1f} MB"
    )

    st.caption(
        f"Semilla de esta composición: "
        f"{generation_seed}"
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
    "No es un tratamiento médico para el TDAH."
)
