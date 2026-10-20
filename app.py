from __future__ import annotations

import io
import secrets
import wave

import numpy as np
import streamlit as st
from scipy import signal


# 16 kHz reduce bastante el tamaño del WAV
# y mantiene buena calidad para música de concentración.
SAMPLE_RATE = 16_000
MAX_DURATION = 20 * 60

CHUNK_SECONDS = 8
SEGMENT_SECONDS = 8
STEP_SECONDS = 1.0
STEPS_PER_SEGMENT = 8

CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
SEGMENT_FRAMES = SAMPLE_RATE * SEGMENT_SECONDS


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

    if audio.size == 0:
        return audio

    audio = audio - np.mean(audio)

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
    """
    Progresión musical en Re mayor.
    Todos los acordes utilizan notas compatibles
    para evitar desafinación y disonancias aleatorias.
    """

    progression = [
        [50, 57, 62, 66],  # D mayor
        [47, 54, 59, 62],  # B menor
        [43, 50, 55, 59],  # G mayor
        [45, 52, 57, 61],  # A mayor
        [50, 57, 62, 66],  # D mayor
        [45, 52, 57, 61],  # A mayor
        [47, 54, 59, 62],  # B menor
        [43, 50, 55, 62],  # G mayor
    ]

    segment_count = max(
        1,
        int(
            np.ceil(
                total_frames / SEGMENT_FRAMES
            )
        ),
    )

    # Secuencia armónica musical.
    base_sequence = np.array(
        [0, 1, 2, 3, 0, 3, 1, 2],
        dtype=np.int32,
    )

    start_offset = int(
        rng.integers(
            0,
            len(base_sequence),
        )
    )

    chord_ids = np.array(
        [
            base_sequence[
                (index + start_offset)
                % len(base_sequence)
            ]
            for index in range(
                segment_count
            )
        ],
        dtype=np.int32,
    )

    note_choices = rng.integers(
        0,
        4,
        size=(
            segment_count,
            STEPS_PER_SEGMENT,
        ),
    )

    note_active = (
        rng.random(
            (
                segment_count,
                STEPS_PER_SEGMENT,
            )
        )
        > 0.43
    )

    note_lengths = np.clip(
        rng.normal(
            0.72,
            0.10,
            (
                segment_count,
                STEPS_PER_SEGMENT,
            ),
        ),
        0.48,
        0.94,
    )

    note_velocities = np.clip(
        rng.normal(
            0.72,
            0.10,
            (
                segment_count,
                STEPS_PER_SEGMENT,
            ),
        ),
        0.45,
        0.95,
    )

    note_offsets = np.clip(
        rng.normal(
            0.0,
            0.012,
            (
                segment_count,
                STEPS_PER_SEGMENT,
            ),
        ),
        -0.025,
        0.025,
    )

    # Una sola octava para evitar saltos desafinados.
    lead_octaves = np.full(
        (
            segment_count,
            STEPS_PER_SEGMENT,
        ),
        24,
        dtype=np.int16,
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
            STEPS_PER_SEGMENT,
        ),
    )

    if mode == "adhd":
        beta_low = 15.0
        beta_high = 18.0
    else:
        beta_low = 12.0
        beta_high = 15.0

    total_seconds = (
        total_frames / SAMPLE_RATE
    )

    control_count = max(
        4,
        int(total_seconds / 8) + 2,
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

    return {
        "progression": progression,
        "chord_ids": chord_ids,
        "note_choices": note_choices,
        "note_active": note_active,
        "note_lengths": note_lengths,
        "note_velocities": note_velocities,
        "note_offsets": note_offsets,
        "lead_octaves": lead_octaves,
        "pad_phases": pad_phases,
        "melody_phases": melody_phases,
        "beta_control_times": beta_control_times,
        "beta_frequencies": beta_frequencies,
    }


def generate_texture(
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Textura ambiental mínima.
    Se mantiene muy baja para evitar sonido de viento.
    """

    noise = rng.standard_normal(
        length
    ).astype(np.float32)

    texture_filter = signal.butter(
        2,
        [280, 3_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    texture = signal.sosfiltfilt(
        texture_filter,
        noise,
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
    Renderiza música suave con acordes, bajo ligero
    y una melodía afinada.
    """

    segment_index = min(
        start_frame // SEGMENT_FRAMES,
        len(plan["chord_ids"]) - 1,
    )

    progression = plan["progression"]

    chord = progression[
        plan["chord_ids"][segment_index]
    ]

    segment_offset = (
        start_frame % SEGMENT_FRAMES
    ) / SAMPLE_RATE

    local_time = (
        np.arange(
            frames,
            dtype=np.float64,
        )
        / SAMPLE_RATE
    )

    segment_time = (
        segment_offset + local_time
    )

    pad = np.zeros(
        frames,
        dtype=np.float32,
    )

    bass = np.zeros(
        frames,
        dtype=np.float32,
    )

    melody = np.zeros(
        frames,
        dtype=np.float32,
    )

    def soft_tone(
        frequency: float,
        times: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        """
        Timbre suave sin desafinación artificial.
        """

        base_phase = (
            2
            * np.pi
            * frequency
            * times
            + phase
        )

        tone = (
            0.80
            * np.sin(base_phase)
        )

        tone += (
            0.14
            * np.sin(
                base_phase * 2.0
            )
        )

        tone += (
            0.04
            * np.sin(
                base_phase * 3.0
            )
        )

        return tone.astype(
            np.float32
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
            0.23,
            0.15,
            0.09,
            0.05,
        ]

        for note_index, note in enumerate(
            pad_chord
        ):
            frequency = midi_to_hz(
                note + 12
            )

            phase = phases[
                note_index
            ]

            movement = (
                0.95
                + 0.05
                * np.sin(
                    2
                    * np.pi
                    * 0.045
                    * times
                    + phase
                )
            )

            output += (
                amplitudes[note_index]
                * movement
                * soft_tone(
                    frequency,
                    times,
                    phase,
                )
            )

        return output

    current_pad = render_pad(
        chord,
        plan["pad_phases"][segment_index],
        segment_time,
    )

    # Cambio progresivo de acorde.
    if segment_index > 0:
        previous_chord = progression[
            plan["chord_ids"][segment_index - 1]
        ]

        previous_pad = render_pad(
            previous_chord,
            plan["pad_phases"][segment_index - 1],
            segment_time + SEGMENT_SECONDS,
        )

        crossfade = np.clip(
            segment_time / 1.8,
            0.0,
            1.0,
        )

        pad = (
            previous_pad * (1.0 - crossfade)
            + current_pad * crossfade
        )
    else:
        pad = current_pad

    def make_envelope(
        duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_count = max(
            1,
            int(
                duration * SAMPLE_RATE
            ),
        )

        times = (
            np.arange(
                sample_count,
                dtype=np.float64,
            )
            / SAMPLE_RATE
        )

        position = np.clip(
            times / duration,
            0.0,
            1.0,
        )

        envelope = (
            np.sin(
                np.pi * position
            )
            ** 2
        )

        return times, envelope

    def add_event(
        target: np.ndarray,
        start_seconds: float,
        values: np.ndarray,
    ) -> None:
        start_index = int(
            np.floor(
                (
                    start_seconds
                    - segment_offset
                )
                * SAMPLE_RATE
            )
        )

        end_index = (
            start_index + len(values)
        )

        clipped_start = max(
            0,
            start_index,
        )

        clipped_end = min(
            frames,
            end_index,
        )

        if clipped_end <= clipped_start:
            return

        source_start = (
            clipped_start - start_index
        )

        source_end = (
            source_start
            + clipped_end
            - clipped_start
        )

        target[
            clipped_start:clipped_end
        ] += values[
            source_start:source_end
        ]

    selected_choices = plan[
        "note_choices"
    ][segment_index]

    selected_active = plan[
        "note_active"
    ][segment_index]

    selected_lengths = plan[
        "note_lengths"
    ][segment_index]

    selected_velocities = plan[
        "note_velocities"
    ][segment_index]

    selected_offsets = plan[
        "note_offsets"
    ][segment_index]

    selected_phases = plan[
        "melody_phases"
    ][segment_index]

    # Melodía lenta, afinada y con silencios.
    for step in range(
        STEPS_PER_SEGMENT
    ):
        if not selected_active[step]:
            continue

        start_seconds = (
            step * STEP_SECONDS
            + float(
                selected_offsets[step]
            )
        )

        duration = float(
            selected_lengths[step]
        )

        velocity = float(
            selected_velocities[step]
        )

        note_index = int(
            selected_choices[step]
        )

        frequency = midi_to_hz(
            chord[note_index] + 24
        )

        phase = float(
            selected_phases[step]
        )

        times, envelope = make_envelope(
            duration
        )

        tone = soft_tone(
            frequency,
            times,
            phase,
        )

        melody_event = (
            0.10
            * velocity
            * envelope
            * tone
        )

        add_event(
            melody,
            start_seconds,
            melody_event,
        )

    # Bajo muy espaciado: solo dos notas por segmento.
    for step in (0, 4):
        start_seconds = (
            step * STEP_SECONDS
        )

        duration = 1.35

        frequency = midi_to_hz(
            chord[0]
        )

        times, envelope = make_envelope(
            duration
        )

        bass_tone = (
            0.72
            * np.sin(
                2
                * np.pi
                * frequency
                * times
            )
        )

        bass_tone += (
            0.12
            * np.sin(
                2
                * np.pi
                * frequency
                * 2
                * times
            )
        )

        bass_event = (
            0.055
            * envelope
            * bass_tone
        )

        add_event(
            bass,
            start_seconds,
            bass_event.astype(
                np.float32
            ),
        )

    # Modulación muy ligera.
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

    modulation = (
        0.97
        + 0.03
        * beta_wave
    )

    music = (
        0.72 * pad
        + 0.07 * bass
        + 0.21 * melody
    )

    music *= modulation

    texture = generate_texture(
        frames,
        rng,
    )

    audio = (
        music
        + 0.0015 * texture
    )

    # Reverberación corta y suave.
    reverb = np.zeros_like(
        audio
    )

    for delay_seconds, gain in [
        (0.17, 0.10),
        (0.31, 0.06),
        (0.47, 0.035),
    ]:
        delay = int(
            delay_seconds
            * SAMPLE_RATE
        )

        if delay < frames:
            reverb[delay:] += (
                audio[:-delay]
                * gain
            )

    audio = (
        0.94 * audio
        + 0.06 * reverb
    )

    # Filtro final para sonido más cálido.
    master_filter = signal.butter(
        2,
        5_500,
        btype="lowpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    audio = signal.sosfiltfilt(
        master_filter,
        audio,
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

    end_frame = (
        start_frame + frames
    )

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

    # Volumen limpio sin saturación fuerte.
    audio = np.tanh(
        audio * 0.42
    )

    audio *= 0.56

    # Estéreo muy ligero.
    side = (
        0.010
        * np.tanh(
            0.35 * melody
            + 0.12 * pad
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
        stereo *= (
            target_peak / peak
        )

    stereo = np.clip(
        stereo,
        -0.22,
        0.22,
    )

    return (
        stereo.astype(np.float32),
        beta_phase,
    )


def generate_focus_wav(
    duration: int,
    mode: str,
    seed: int,
    progress_callback=None,
) -> bytes:
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


def extract_preview(
    audio_bytes: bytes,
) -> np.ndarray:
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


def format_time(seconds: int) -> str:
    minutes = seconds // 60
    remaining = seconds % 60

    return f"{minutes:02d}:{remaining:02d}"


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
            Música generativa suave con acordes afinados,
            melodías lentas, bajo ambiental y modulación
            ligera para concentración.
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
        La misma configuración genera una composición
        diferente en cada ejecución.
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

    progress_bar = st.progress(
        0,
        text="Preparando música...",
    )

    with st.spinner(
        "Generando composición..."
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

    st.session_state["audio_bytes"] = bytes(
        audio_bytes
    )

    st.session_state["audio_mode"] = mode
    st.session_state["audio_duration"] = duration
    st.session_state["generation_seed"] = seed


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

    # Descarga directa en bytes.
    # No utiliza base64 ni vuelve a generar el audio.
    st.download_button(
        label="⬇️ Descargar composición WAV",
        data=audio_bytes,
        file_name=file_name,
        mime="audio/wav",
        key=f"download_{generation_seed}",
        use_container_width=True,
    )

    st.caption(
        "Tamaño del archivo: "
        f"{len(audio_bytes) / 1_000_000:.1f} MB"
    )

    st.caption(
        f"Semilla: {generation_seed}"
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
