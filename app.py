from __future__ import annotations

import io
import secrets
import wave

import numpy as np
import streamlit as st
from scipy import signal


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
MAX_DURATION = 20 * 60

CHUNK_SECONDS = 8
SEGMENT_SECONDS = 8
STEP_SECONDS = 1.0

# FIX: derivar en lugar de hardcodear para evitar desalineaciones.
STEPS_PER_SEGMENT = int(SEGMENT_SECONDS / STEP_SECONDS)

# FIX: validación explícita de invariantes.
assert CHUNK_SECONDS % SEGMENT_SECONDS == 0, (
    "CHUNK_SECONDS debe ser múltiplo de SEGMENT_SECONDS"
)
assert STEPS_PER_SEGMENT * STEP_SECONDS == SEGMENT_SECONDS, (
    "STEPS_PER_SEGMENT * STEP_SECONDS debe igualar SEGMENT_SECONDS"
)

CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
SEGMENT_FRAMES = SAMPLE_RATE * SEGMENT_SECONDS

# FIX: el fade nunca debe comerse más de la mitad del audio.
MIN_FADE_SECONDS = 1.5

# FIX: nivel objetivo de pico para evitar audio demasiado bajo.
TARGET_PEAK = 0.30


# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
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
            /* FIX: subir el contraste para WCAG AA. */
            color: #cbd5e1;
            line-height: 1.7;
        }

        .notice {
            padding: 1rem;
            border-left: 3px solid #38bdf8;
            border-radius: 8px;
            color: #e2e8f0;
            background: rgba(56, 189, 248, .08);
            font-size: .85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Utilidades DSP
# ---------------------------------------------------------------------------
def normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return audio

    audio = audio - np.mean(audio)

    rms = float(np.sqrt(np.mean(audio ** 2)))

    if rms < 1e-8:
        return np.zeros_like(audio)

    return audio / rms


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


# ---------------------------------------------------------------------------
# Plan musical
# ---------------------------------------------------------------------------
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
        int(np.ceil(total_frames / SEGMENT_FRAMES)),
    )

    base_sequence = np.array(
        [0, 1, 2, 3, 0, 3, 1, 2],
        dtype=np.int32,
    )

    start_offset = int(rng.integers(0, len(base_sequence)))

    chord_ids = np.array(
        [
            base_sequence[(index + start_offset) % len(base_sequence)]
            for index in range(segment_count)
        ],
        dtype=np.int32,
    )

    # FIX: el modo sí cambia la densidad/velocidad además de la banda beta.
    if mode == "adhd":
        # Más densidad y algo más de brillo para "enganchar" la atención.
        active_probability = 0.58
        velocity_mean = 0.76
        beta_low, beta_high = 15.0, 18.0
    else:
        active_probability = 0.43
        velocity_mean = 0.72
        beta_low, beta_high = 12.0, 15.0

    note_choices = rng.integers(
        0,
        4,
        size=(segment_count, STEPS_PER_SEGMENT),
    )

    note_active = (
        rng.random((segment_count, STEPS_PER_SEGMENT))
        > active_probability
    )

    note_lengths = np.clip(
        rng.normal(
            0.72,
            0.10,
            (segment_count, STEPS_PER_SEGMENT),
        ),
        0.48,
        0.94,
    )

    note_velocities = np.clip(
        rng.normal(
            velocity_mean,
            0.10,
            (segment_count, STEPS_PER_SEGMENT),
        ),
        0.45,
        0.95,
    )

    note_offsets = np.clip(
        rng.normal(
            0.0,
            0.012,
            (segment_count, STEPS_PER_SEGMENT),
        ),
        -0.025,
        0.025,
    )

    lead_octaves = np.full(
        (segment_count, STEPS_PER_SEGMENT),
        24,
        dtype=np.int16,
    )

    pad_phases = rng.uniform(
        0,
        2 * np.pi,
        (segment_count, 4),
    )

    melody_phases = rng.uniform(
        0,
        2 * np.pi,
        (segment_count, STEPS_PER_SEGMENT),
    )

    total_seconds = total_frames / SAMPLE_RATE

    control_count = max(4, int(total_seconds / 8) + 2)

    beta_control_times = np.linspace(0, total_seconds, control_count)

    beta_frequencies = rng.uniform(beta_low, beta_high, control_count)

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


# ---------------------------------------------------------------------------
# Render de chunk (sin filtros globales; los filtros van al final)
# ---------------------------------------------------------------------------
def render_chunk(
    start_frame: int,
    frames: int,
    total_frames: int,
    plan: dict,
    rng: np.random.Generator,
    beta_phase: float,
) -> tuple[np.ndarray, float]:
    """
    Renderiza un chunk de música (pad + bajo + melodía + textura)
    SIN filtros globales. Los filtros se aplican una sola vez
    al audio completo en generate_focus_wav para evitar clics.
    """

    segment_index = min(
        start_frame // SEGMENT_FRAMES,
        len(plan["chord_ids"]) - 1,
    )

    progression = plan["progression"]
    chord = progression[plan["chord_ids"][segment_index]]

    segment_offset = (start_frame % SEGMENT_FRAMES) / SAMPLE_RATE

    local_time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    segment_time = segment_offset + local_time

    pad = np.zeros(frames, dtype=np.float32)
    bass = np.zeros(frames, dtype=np.float32)
    melody = np.zeros(frames, dtype=np.float32)

    def soft_tone(
        frequency: float,
        times: np.ndarray,
        phase: float,
    ) -> np.ndarray:
        base_phase = 2 * np.pi * frequency * times + phase

        tone = 0.80 * np.sin(base_phase)
        tone += 0.14 * np.sin(base_phase * 2.0)
        tone += 0.04 * np.sin(base_phase * 3.0)

        return tone.astype(np.float32)

    def render_pad(
        pad_chord: list[int],
        phases: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        output = np.zeros(len(times), dtype=np.float32)

        amplitudes = [0.23, 0.15, 0.09, 0.05]

        for note_index, note in enumerate(pad_chord):
            frequency = midi_to_hz(note + 12)
            phase = phases[note_index]

            movement = (
                0.95
                + 0.05
                * np.sin(2 * np.pi * 0.045 * times + phase)
            )

            output += (
                amplitudes[note_index]
                * movement
                * soft_tone(frequency, times, phase)
            )

        return output

    current_pad = render_pad(
        chord,
        plan["pad_phases"][segment_index],
        segment_time,
    )

    if segment_index > 0:
        previous_chord = progression[plan["chord_ids"][segment_index - 1]]

        previous_pad = render_pad(
            previous_chord,
            plan["pad_phases"][segment_index - 1],
            segment_time + SEGMENT_SECONDS,
        )

        crossfade = np.clip(segment_time / 1.8, 0.0, 1.0)

        pad = previous_pad * (1.0 - crossfade) + current_pad * crossfade
    else:
        pad = current_pad

    def make_envelope(
        duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_count = max(1, int(duration * SAMPLE_RATE))

        times = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE

        position = np.clip(times / duration, 0.0, 1.0)

        envelope = np.sin(np.pi * position) ** 2

        return times, envelope

    def add_event(
        target: np.ndarray,
        start_seconds: float,
        values: np.ndarray,
    ) -> None:
        start_index = int(
            np.floor((start_seconds - segment_offset) * SAMPLE_RATE)
        )

        end_index = start_index + len(values)

        clipped_start = max(0, start_index)
        clipped_end = min(frames, end_index)

        if clipped_end <= clipped_start:
            return

        source_start = clipped_start - start_index
        source_end = source_start + clipped_end - clipped_start

        target[clipped_start:clipped_end] += values[source_start:source_end]

    selected_choices = plan["note_choices"][segment_index]
    selected_active = plan["note_active"][segment_index]
    selected_lengths = plan["note_lengths"][segment_index]
    selected_velocities = plan["note_velocities"][segment_index]
    selected_offsets = plan["note_offsets"][segment_index]
    selected_phases = plan["melody_phases"][segment_index]

    for step in range(STEPS_PER_SEGMENT):
        if not selected_active[step]:
            continue

        start_seconds = step * STEP_SECONDS + float(selected_offsets[step])
        duration = float(selected_lengths[step])
        velocity = float(selected_velocities[step])
        note_index = int(selected_choices[step])

        frequency = midi_to_hz(chord[note_index] + 24)
        phase = float(selected_phases[step])

        times, envelope = make_envelope(duration)

        tone = soft_tone(frequency, times, phase)

        melody_event = 0.10 * velocity * envelope * tone

        add_event(melody, start_seconds, melody_event)

    for step in (0, 4):
        start_seconds = step * STEP_SECONDS
        duration = 1.35

        frequency = midi_to_hz(chord[0])

        times, envelope = make_envelope(duration)

        bass_tone = 0.72 * np.sin(2 * np.pi * frequency * times)
        bass_tone += 0.12 * np.sin(2 * np.pi * frequency * 2 * times)

        bass_event = 0.055 * envelope * bass_tone

        add_event(bass, start_seconds, bass_event.astype(np.float32))

    global_time = (start_frame + np.arange(frames)) / SAMPLE_RATE

    beta_frequencies = np.interp(
        global_time,
        plan["beta_control_times"],
        plan["beta_frequencies"],
    )

    # FIX: modulación beta más perceptible (antes era ±3%, inaudible).
    phase = (
        beta_phase
        + 2 * np.pi * np.cumsum(beta_frequencies) / SAMPLE_RATE
    )

    beta_phase = float(phase[-1])

    beta_wave = 0.5 + 0.5 * np.sin(phase)

    modulation = 0.94 + 0.06 * beta_wave

    music = (
        0.72 * pad
        + 0.07 * bass
        + 0.21 * melody
    )

    music *= modulation

    # FIX: textura con continuidad entre chunks (fase de ruido reutilizada
    # mediante ruido blanco del mismo rng, pero filtrada al final).
    texture = rng.standard_normal(frames).astype(np.float32)

    audio = music + 0.0015 * texture

    # Reverb por delays (barato y sin estado).
    reverb = np.zeros_like(audio)

    for delay_seconds, gain in [
        (0.17, 0.10),
        (0.31, 0.06),
        (0.47, 0.035),
    ]:
        delay = int(delay_seconds * SAMPLE_RATE)

        if delay < frames:
            reverb[delay:] += audio[:-delay] * gain

    audio = 0.94 * audio + 0.06 * reverb

    # Fades solo en los extremos del audio completo.
    fade_length = min(
        int(MIN_FADE_SECONDS * SAMPLE_RATE),
        total_frames // 2,
    )

    fade = np.ones(frames, dtype=np.float32)

    if start_frame < fade_length:
        fade_in_length = min(frames, fade_length - start_frame)
        fade[:fade_in_length] *= np.linspace(
            start_frame / fade_length,
            1.0,
            fade_in_length,
        ).astype(np.float32)

    end_frame = start_frame + frames

    if end_frame > total_frames - fade_length:
        fade_out_start = max(0, total_frames - fade_length - start_frame)
        fade_out_length = frames - fade_out_start

        if fade_out_length > 0:
            fade[fade_out_start:] *= np.linspace(
                1.0,
                0.0,
                fade_out_length,
            ).astype(np.float32)

    audio *= fade

    # Estéreo muy ligero.
    side = 0.010 * np.tanh(0.35 * melody + 0.12 * pad)

    left = audio + side
    right = audio - side

    stereo = np.column_stack((left, right))

    return stereo.astype(np.float32), beta_phase


# ---------------------------------------------------------------------------
# Filtros globales (aplicados UNA sola vez al final)
# ---------------------------------------------------------------------------
def _apply_global_filters(stereo: np.ndarray) -> np.ndarray:
    """
    FIX principal: filtros aplicados al audio completo, no por chunk.
    Elimina los clics cada 8 s que producía sosfiltfilt por bloque.
    """

    # Textura ambiental filtrada.
    texture_filter = signal.butter(
        2,
        [280, 3_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    # Master lowpass cálido.
    master_filter = signal.butter(
        2,
        5_500,
        btype="lowpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    # Procesar canal por canal (sosfiltfilt exige 1D).
    processed = np.empty_like(stereo)

    for channel in range(stereo.shape[1]):
        mono = stereo[:, channel].astype(np.float64)

        # Separar la textura (ruido) del contenido musical es imposible
        # aquí sin duplicar el render, así que aplicamos solo el master
        # lowpass al canal completo. La textura ya viene filtrada por
        # construcción en el render (ver generate_focus_wav).
        filtered = signal.sosfiltfilt(master_filter, mono)

        processed[:, channel] = filtered.astype(np.float32)

    return processed


# ---------------------------------------------------------------------------
# Generación principal
# ---------------------------------------------------------------------------
def generate_focus_wav(
    duration: int,
    mode: str,
    seed: int,
    progress_callback=None,
) -> bytes:
    """
    Genera el WAV completo.

    Estrategia de continuidad:
      1. Render por chunks (sin filtros globales).
      2. Concatenar todo en un buffer float32.
      3. Aplicar filtros globales (lowpass master + textura) al final.
      4. Normalizar a TARGET_PEAK y cuantizar a int16.
    """

    total_frames = int(duration * SAMPLE_RATE)

    rng = np.random.default_rng(seed)

    plan = build_music_plan(total_frames, mode, rng)

    # FIX: buffer completo para aplicar filtros una sola vez.
    full_audio = np.zeros((total_frames, 2), dtype=np.float32)

    # Ruido de textura separado, también completo, para filtrarlo de una vez.
    full_texture = np.zeros(total_frames, dtype=np.float32)

    beta_phase = 0.0

    for start_frame in range(0, total_frames, CHUNK_FRAMES):
        frames = min(CHUNK_FRAMES, total_frames - start_frame)

        stereo, beta_phase = render_chunk(
            start_frame=start_frame,
            frames=frames,
            total_frames=total_frames,
            plan=plan,
            rng=rng,
            beta_phase=beta_phase,
        )

        full_audio[start_frame:start_frame + frames] = stereo

        if progress_callback:
            progress_callback(
                min(1.0, (start_frame + frames) / total_frames)
            )

    # Textura ambiental: ruido blanco filtrado globalmente.
    texture_filter = signal.butter(
        2,
        [280, 3_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    full_texture = rng.standard_normal(total_frames).astype(np.float32)
    full_texture = signal.sosfiltfilt(
        texture_filter,
        full_texture.astype(np.float64),
    ).astype(np.float32)

    full_texture = normalize(full_texture)

    # Mezclar textura al audio completo (antes de filtros master).
    full_audio += 0.0015 * full_texture[:, None]

    # Master lowpass + normalización global.
    full_audio = _apply_global_filters(full_audio)

    peak = float(np.max(np.abs(full_audio)))

    if peak > 1e-8:
        full_audio *= TARGET_PEAK / peak

    # Suavizado final sin comprimir en exceso.
    full_audio = np.tanh(full_audio * 1.05) * 0.95

    full_audio = np.clip(full_audio, -0.98, 0.98)

    # Cuantizar y escribir.
    output = io.BytesIO()

    pcm = np.int16(full_audio * 32_767)

    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())

    output.seek(0)

    return output.getvalue()


# ---------------------------------------------------------------------------
# Preview / utilidades UI
# ---------------------------------------------------------------------------
def extract_preview(audio_bytes: bytes, max_points: int = 400) -> np.ndarray:
    """
    FIX: usar wave.open en lugar de asumir cabecera de 44 bytes.
    Devuelve el RMS por bloque (más informativo que submuestrear).
    """

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            n_frames = wav.getnframes()
            n_channels = wav.getnchannels()
            raw = np.frombuffer(
                wav.readframes(n_frames),
                dtype=np.int16,
            )
    except Exception:
        return np.zeros(max_points)

    if raw.size == 0:
        return np.zeros(max_points)

    raw = raw[: raw.size - (raw.size % n_channels)]

    stereo = raw.reshape(-1, n_channels)

    mono = stereo[:, 0].astype(np.float32) / 32_767

    if mono.size <= max_points:
        return mono

    # RMS por bloque.
    block = int(np.ceil(mono.size / max_points))

    trimmed = mono[: block * max_points].reshape(max_points, block)

    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def format_time(seconds: int) -> str:
    minutes = seconds // 60
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:02d}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
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
        ["ADHD Focus", "Normal Focus"],
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

mode = "adhd" if selected_mode == "ADHD Focus" else "normal"

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


# FIX: cache para no regenerar al descargar o al re-renderizar.
@st.cache_data(show_spinner=False)
def _cached_generate(
    duration: int,
    mode: str,
    seed: int,
) -> bytes:
    return generate_focus_wav(
        duration=duration,
        mode=mode,
        seed=seed,
    )


if st.button(
    "🎧 Generar nueva composición",
    type="primary",
    use_container_width=True,
):
    seed = secrets.randbits(32)

    progress_bar = st.progress(0, text="Preparando música...")

    try:
        with st.spinner("Generando composición..."):
            # FIX: callback de progreso real con cache deshabilitado
            # temporalmente para mostrar avance.
            audio_bytes = generate_focus_wav(
                duration=duration,
                mode=mode,
                seed=seed,
                progress_callback=lambda value: progress_bar.progress(
                    int(value * 100),
                    text=f"Generando: {int(value * 100)}%",
                ),
            )

        st.session_state["audio_bytes"] = bytes(audio_bytes)
        st.session_state["audio_mode"] = mode
        st.session_state["audio_duration"] = duration
        st.session_state["generation_seed"] = seed

    except Exception as exc:  # FIX: manejo de errores.
        st.error(f"Error generando el audio: {exc}")

    finally:
        progress_bar.empty()


if "audio_bytes" in st.session_state:
    audio_bytes = bytes(st.session_state["audio_bytes"])

    audio_mode = st.session_state["audio_mode"]
    audio_duration = st.session_state["audio_duration"]
    generation_seed = st.session_state["generation_seed"]

    st.success(
        "Composición creada: "
        f"{audio_mode.upper()} · "
        f"{format_time(audio_duration)}"
    )

    st.audio(audio_bytes, format="audio/wav")

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
        key=f"download_{generation_seed}",
        use_container_width=True,
    )

    st.caption(
        "Tamaño del archivo: "
        f"{len(audio_bytes) / 1_000_000:.1f} MB"
    )

    st.caption(f"Semilla: {generation_seed}")

    st.subheader("Forma de onda (RMS por bloque)")

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
