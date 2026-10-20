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
    """Crea una composición con variaciones humanas y aleatorias."""

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

    notes_per_segment = 16

    chord_ids = np.zeros(
        segment_count,
        dtype=np.int32,
    )

    previous_chord = -1

    # Evita repetir el mismo acorde consecutivamente.
    for segment_index in range(segment_count):
        candidates = [
            index
            for index in range(len(progression))
            if index != previous_chord
        ]

        selected_chord = int(
            rng.choice(candidates)
        )

        chord_ids[segment_index] = selected_chord
        previous_chord = selected_chord

    # El estudio usado como referencia trabajó con música
    # alrededor de 120 BPM. Aquí se humaniza ligeramente.
    tempo_bpm = rng.uniform(
        116.0,
        124.0,
    )

    beat_seconds = 60.0 / tempo_bpm

    note_starts = np.zeros(
        (
            segment_count,
            notes_per_segment,
        ),
        dtype=np.float64,
    )

    note_lengths = rng.uniform(
        0.28,
        0.48,
        (
            segment_count,
            notes_per_segment,
        ),
    )

    note_velocities = rng.uniform(
        0.72,
        1.12,
        (
            segment_count,
            notes_per_segment,
        ),
    )

    note_active = rng.random(
        (
            segment_count,
            notes_per_segment,
        )
    ) > 0.10

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

    for segment_index in range(segment_count):
        microtiming = rng.normal(
            0.0,
            0.018,
            notes_per_segment,
        )

        starts = (
            np.arange(notes_per_segment)
            * beat_seconds
            + microtiming
        )

        starts[0] = 0.0

        note_starts[segment_index] = np.clip(
            starts,
            0.0,
            SEGMENT_SECONDS - 0.05,
        )

        # Siempre comienza cada segmento con una nota.
        note_active[
            segment_index,
            0,
        ] = True

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
        "note_starts": note_starts,
        "note_lengths": note_lengths,
        "note_velocities": note_velocities,
        "note_active": note_active,
        "pad_phases": pad_phases,
        "melody_phases": melody_phases,
        "beta_control_times": beta_control_times,
        "beta_frequencies": beta_frequencies,
        "macro_phase": rng.uniform(
            0,
            2 * np.pi,
        ),
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
    """Renderiza un bloque musical menos robótico."""

    progression = plan["progression"]

    segment_index = (
        start_frame // SEGMENT_FRAMES
    )

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

    # Pads armónicos con movimiento lento.
    amplitudes = [
        0.31,
        0.19,
        0.14,
        0.09,
    ]

    macro_phase = plan["macro_phase"]

    macro_movement = (
        0.94
        + 0.06
        * np.sin(
            2
            * np.pi
            * 0.07
            * (
                start_frame
                / SAMPLE_RATE
                + local_time
            )
            + macro_phase
        )
    )

    for note_index, note in enumerate(chord):
        frequency = midi_to_hz(
            note + 12
        )

        phase_offset = plan[
            "pad_phases"
        ][segment_index][note_index]

        pad_voice = np.sin(
            2
            * np.pi
            * frequency
            * local_time
            + phase_offset
        )

        harmonic_voice = np.sin(
            2
            * np.pi
            * frequency
            * 2
            * local_time
            + phase_offset
        )

        pad += (
            amplitudes[note_index]
            * macro_movement
            * pad_voice
        )

        pad += (
            amplitudes[note_index]
            * 0.09
            * harmonic_voice
        )

    notes_per_segment = 16

    # Ritmo y melodía humanizados.
    for note_index in range(notes_per_segment):
        if not plan["note_active"][
            segment_index
        ][note_index]:
            continue

        note_start_seconds = plan[
            "note_starts"
        ][segment_index][note_index]

        note_duration = plan[
            "note_lengths"
        ][segment_index][note_index]

        note_velocity = plan[
            "note_velocities"
        ][segment_index][note_index]

        note_start = int(
            note_start_seconds
            * SAMPLE_RATE
        )

        note_end = int(
            (
                note_start_seconds
                + note_duration
            )
            * SAMPLE_RATE
        )

        note_start = max(
            0,
            note_start,
        )

        note_end = min(
            frames,
            note_end,
        )

        if note_end <= note_start:
            continue

        note_time = np.arange(
            note_start,
            note_end,
            dtype=np.float64,
        ) / SAMPLE_RATE

        relative_time = (
            note_time
            - note_start_seconds
        )

        envelope = np.sin(
            np.pi
            * np.clip(
                relative_time
                / note_duration,
                0,
                1,
            )
        ) ** 1.6

        selected_note = chord[
            plan["note_choices"][
                segment_index
            ][note_index]
        ]

        rhythm_note = chord[
            (note_index + 1) % len(chord)
        ]

        rhythm_frequency = midi_to_hz(
            rhythm_note + 24
        )

        melody_frequency = midi_to_hz(
            selected_note + 24
        )

        phase_offset = plan[
            "melody_phases"
        ][segment_index][note_index]

        # Acentos irregulares para evitar un patrón mecánico.
        accent = (
            1.18
            if note_index % 4 == 0
            else 0.88
        )

        rhythm_slice = (
            0.13
            * note_velocity
            * accent
            * envelope
            * np.sin(
                2
                * np.pi
                * rhythm_frequency
                * relative_time
                + phase_offset
            )
        )

        rhythm_slice += (
            0.035
            * envelope
            * np.sin(
                2
                * np.pi
                * rhythm_frequency
                * 2
                * relative_time
                + phase_offset
            )
        )

        melody_slice = (
            0.10
            * note_velocity
            * envelope
            * np.sin(
                2
                * np.pi
                * melody_frequency
                * relative_time
                + phase_offset
            )
        )

        rhythm[
            note_start:note_end
        ] += rhythm_slice

        melody[
            note_start:note_end
        ] += melody_slice

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

    beta_phase = float(
        phase[-1]
    )

    beta_wave = (
        0.5
        + 0.5
        * np.sin(phase)
    )

    # Menor profundidad para que no suene robótico.
    beta_envelope = (
        0.76
        + 0.24
        * beta_wave
    )

    # El pulso beta afecta sobre todo al ritmo,
    # no a toda la canción por igual.
    rhythm *= beta_envelope

    music = (
        0.54 * pad
        + 0.27 * rhythm
        + 0.19 * melody
    )

    texture = generate_texture(
        frames,
        rng,
    )

    audio = (
        music
        + 0.012 * texture
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
            1,
            fade_in_length,
        )

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
            1,
            0,
            fade_out_length,
        )

    audio *= fade

    # Saturación muy suave.
    audio = np.tanh(
        audio * 0.48
    )

    audio *= 0.58

    # Estéreo con movimiento mínimo.
    side = (
        0.022
        * np.tanh(
            0.65 * melody
            + 0.25 * rhythm
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

    target_peak = 0.22

    if peak > target_peak:
        stereo *= target_peak / peak

    stereo = np.clip(
        stereo,
        -target_peak,
        target_peak,
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
        f"Tamaño del archivo: "
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
    "No es un tratamiento médico para TDAH."
)
