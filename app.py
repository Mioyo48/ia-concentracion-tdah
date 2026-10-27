from __future__ import annotations

import io
import secrets
import wave

import numpy as np
import streamlit as st
from scipy import signal


# ===========================================================================
# Constantes
# ===========================================================================
SAMPLE_RATE = 16_000
MAX_DURATION = 20 * 60

CHUNK_SECONDS = 8
SEGMENT_SECONDS = 8
STEP_SECONDS = 1.0
STEPS_PER_SEGMENT = int(SEGMENT_SECONDS / STEP_SECONDS)

assert CHUNK_SECONDS % SEGMENT_SECONDS == 0
assert STEPS_PER_SEGMENT * STEP_SECONDS == SEGMENT_SECONDS

CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
SEGMENT_FRAMES = SAMPLE_RATE * SEGMENT_SECONDS

MIN_FADE_SECONDS = 1.5
TARGET_PEAK = 0.30

# Ganancia fija del filtro Kellett para llevar el ruido rosa
# a un nivel comparable al de la música. Ajustable.
PINK_GAIN = 0.45


# ===========================================================================
# Configuración de página
# ===========================================================================
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
                radial-gradient(circle at 15% 10%, rgba(56, 189, 248, .20), transparent 30%),
                radial-gradient(circle at 85% 85%, rgba(244, 63, 94, .16), transparent 30%),
                linear-gradient(135deg, #020617, #0f172a, #1e293b);
        }
        .block-container { max-width: 900px; padding-top: 3rem; }
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
        .accent { color: #7dd3fc; }
        .description { color: #cbd5e1; line-height: 1.7; }
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


# ===========================================================================
# Helpers
# ===========================================================================
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


# ===========================================================================
# Ruido rosa (filtro Kellett con estado continuo entre chunks)
# ===========================================================================
class PinkNoiseGenerator:
    """
    Ruido rosa (1/f) con estado de filtro continuo entre chunks.

    Usa el filtro de Paul Kellett (aproximación clásica de -3 dB/octava).
    Cada instancia mantiene su propio estado para que el ruido sea
    continuo y decorrelacionado por canal.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.b0_zi = np.zeros(1)
        self.b1_zi = np.zeros(1)
        self.b2_zi = np.zeros(1)
        self.b3_zi = np.zeros(1)
        self.b4_zi = np.zeros(1)
        self.b5_zi = np.zeros(1)
        self.b6_state = 0.0

    def generate(self, n: int) -> np.ndarray:
        white = self.rng.standard_normal(n).astype(np.float64)

        b0, self.b0_zi = signal.lfilter(
            [0.0555179], [1, -0.99886], white, zi=self.b0_zi
        )
        b1, self.b1_zi = signal.lfilter(
            [0.0750759], [1, -0.99332], white, zi=self.b1_zi
        )
        b2, self.b2_zi = signal.lfilter(
            [0.1538520], [1, -0.96900], white, zi=self.b2_zi
        )
        b3, self.b3_zi = signal.lfilter(
            [0.3104856], [1, -0.86650], white, zi=self.b3_zi
        )
        b4, self.b4_zi = signal.lfilter(
            [0.5329522], [1, -0.55000], white, zi=self.b4_zi
        )
        b5, self.b5_zi = signal.lfilter(
            [-0.0168980], [1, 0.7616], white, zi=self.b5_zi
        )

        b6_prev = np.empty_like(white)
        b6_prev[0] = self.b6_state
        b6_prev[1:] = white[:-1] * 0.115926
        self.b6_state = float(white[-1] * 0.115926)

        pink = (
            b0 + b1 + b2 + b3 + b4 + b5 + b6_prev + white * 0.5362
        ) * PINK_GAIN

        return pink.astype(np.float32)


# ===========================================================================
# Plan musical
# ===========================================================================
def build_music_plan(
    total_frames: int,
    energia: float,
    rng: np.random.Generator,
) -> dict:
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

    segment_count = max(1, int(np.ceil(total_frames / SEGMENT_FRAMES)))

    base_sequence = np.array([0, 1, 2, 3, 0, 3, 1, 2], dtype=np.int32)
    start_offset = int(rng.integers(0, len(base_sequence)))

    chord_ids = np.array(
        [
            base_sequence[(i + start_offset) % len(base_sequence)]
            for i in range(segment_count)
        ],
        dtype=np.int32,
    )

    # Energía controla densidad y velocidad media de la melodía.
    active_probability = 0.30 + energia * 0.40
    velocity_mean = 0.62 + energia * 0.16

    note_choices = rng.integers(0, 4, size=(segment_count, STEPS_PER_SEGMENT))
    note_active = rng.random((segment_count, STEPS_PER_SEGMENT)) < active_probability
    note_lengths = np.clip(
        rng.normal(0.72, 0.10, (segment_count, STEPS_PER_SEGMENT)), 0.48, 0.94
    )
    note_velocities = np.clip(
        rng.normal(velocity_mean, 0.10, (segment_count, STEPS_PER_SEGMENT)),
        0.45,
        0.95,
    )
    note_offsets = np.clip(
        rng.normal(0.0, 0.008, (segment_count, STEPS_PER_SEGMENT)), -0.02, 0.02
    )

    pad_phases = rng.uniform(0, 2 * np.pi, (segment_count, 4))
    melody_phases = rng.uniform(0, 2 * np.pi, (segment_count, STEPS_PER_SEGMENT))

    # Tempo en el rango útil para atención (60–75 BPM).
    tempo_bpm = 60.0 + energia * 15.0

    return {
        "progression": progression,
        "chord_ids": chord_ids,
        "note_choices": note_choices,
        "note_active": note_active,
        "note_lengths": note_lengths,
        "note_velocities": note_velocities,
        "note_offsets": note_offsets,
        "pad_phases": pad_phases,
        "melody_phases": melody_phases,
        "tempo_bpm": tempo_bpm,
    }


# ===========================================================================
# Pulso rítmico (ancla temporal sobre rejilla global de beats)
# ===========================================================================
def add_rhythmic_pulse(
    target: np.ndarray,
    start_frame: int,
    frames: int,
    tempo_bpm: float,
    pulse_gain: float,
) -> None:
    if pulse_gain <= 0 or tempo_bpm <= 0:
        return

    beat_duration = 60.0 / tempo_bpm
    chunk_start_time = start_frame / SAMPLE_RATE
    chunk_end_time = (start_frame + frames) / SAMPLE_RATE

    pulse_duration = 0.20
    pulse_samples = int(pulse_duration * SAMPLE_RATE)
    t = np.arange(pulse_samples) / SAMPLE_RATE

    attack_samples = max(1, int(0.005 * SAMPLE_RATE))
    envelope = np.ones(pulse_samples, dtype=np.float32)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples, dtype=np.float32)
    envelope *= np.exp(-t / 0.09).astype(np.float32)

    body = (
        0.7 * np.sin(2 * np.pi * 90 * t)
        + 0.15 * np.sin(2 * np.pi * 180 * t)
    ).astype(np.float32)
    click = (0.15 * np.sin(2 * np.pi * 400 * t) * np.exp(-t / 0.015)).astype(
        np.float32
    )
    pulse_wave = (body + click) * envelope

    accent_pattern = [1.0, 0.6, 0.8, 0.6]

    first_beat = max(
        0, int(np.floor((chunk_start_time - pulse_duration) / beat_duration))
    )
    last_beat = int(np.ceil(chunk_end_time / beat_duration))

    for k in range(first_beat, last_beat + 1):
        beat_time = k * beat_duration
        accent = accent_pattern[k % len(accent_pattern)]

        start_sample = int((beat_time - chunk_start_time) * SAMPLE_RATE)
        end_sample = start_sample + pulse_samples

        src_start = max(0, -start_sample)
        dst_start = max(0, start_sample)
        dst_end = min(frames, end_sample)

        if dst_end <= dst_start:
            continue

        src_end = src_start + (dst_end - dst_start)
        target[dst_start:dst_end] += (
            pulse_wave[src_start:src_end] * pulse_gain * accent
        )


# ===========================================================================
# Render por chunk
# ===========================================================================
def render_chunk(
    start_frame: int,
    frames: int,
    total_frames: int,
    plan: dict,
    pink_L: PinkNoiseGenerator,
    pink_R: PinkNoiseGenerator,
    noise_level: float,
    pulse_gain: float,
) -> np.ndarray:
    segment_index = min(start_frame // SEGMENT_FRAMES, len(plan["chord_ids"]) - 1)
    progression = plan["progression"]
    chord = progression[plan["chord_ids"][segment_index]]

    segment_offset = (start_frame % SEGMENT_FRAMES) / SAMPLE_RATE
    local_time = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    segment_time = segment_offset + local_time

    pad = np.zeros(frames, dtype=np.float32)
    bass = np.zeros(frames, dtype=np.float32)
    melody = np.zeros(frames, dtype=np.float32)

    def soft_tone(frequency, times, phase):
        base_phase = 2 * np.pi * frequency * times + phase
        tone = (
            0.80 * np.sin(base_phase)
            + 0.14 * np.sin(base_phase * 2.0)
            + 0.04 * np.sin(base_phase * 3.0)
        )
        return tone.astype(np.float32)

    def render_pad(pad_chord, phases, times):
        out = np.zeros(len(times), dtype=np.float32)
        amplitudes = [0.23, 0.15, 0.09, 0.05]
        for i, note in enumerate(pad_chord):
            freq = midi_to_hz(note + 12)
            phase = phases[i]
            movement = 0.95 + 0.05 * np.sin(2 * np.pi * 0.045 * times + phase)
            out += amplitudes[i] * movement * soft_tone(freq, times, phase)
        return out

    current_pad = render_pad(chord, plan["pad_phases"][segment_index], segment_time)

    if segment_index > 0:
        prev_chord = progression[plan["chord_ids"][segment_index - 1]]
        prev_pad = render_pad(
            prev_chord,
            plan["pad_phases"][segment_index - 1],
            segment_time + SEGMENT_SECONDS,
        )
        xf = np.clip(segment_time / 1.8, 0.0, 1.0)
        pad = prev_pad * (1.0 - xf) + current_pad * xf
    else:
        pad = current_pad

    def make_envelope(duration):
        n = max(1, int(duration * SAMPLE_RATE))
        times = np.arange(n, dtype=np.float64) / SAMPLE_RATE
        pos = np.clip(times / duration, 0.0, 1.0)
        return times, np.sin(np.pi * pos) ** 2

    def add_event(target, start_seconds, values):
        si = int(np.floor((start_seconds - segment_offset) * SAMPLE_RATE))
        ei = si + len(values)
        cs = max(0, si)
        ce = min(frames, ei)
        if ce <= cs:
            return
        ss = cs - si
        se = ss + ce - cs
        target[cs:ce] += values[ss:se]

    choices = plan["note_choices"][segment_index]
    active = plan["note_active"][segment_index]
    lengths = plan["note_lengths"][segment_index]
    velocities = plan["note_velocities"][segment_index]
    offsets = plan["note_offsets"][segment_index]
    phases = plan["melody_phases"][segment_index]

    for step in range(STEPS_PER_SEGMENT):
        if not active[step]:
            continue
        start_s = step * STEP_SECONDS + float(offsets[step])
        duration = float(lengths[step])
        velocity = float(velocities[step])
        note_index = int(choices[step])
        freq = midi_to_hz(chord[note_index] + 24)
        phase = float(phases[step])
        times, env = make_envelope(duration)
        tone = soft_tone(freq, times, phase)
        add_event(melody, start_s, 0.10 * velocity * env * tone)

    for step in (0, 4):
        start_s = step * STEP_SECONDS
        duration = 1.35
        freq = midi_to_hz(chord[0])
        times, env = make_envelope(duration)
        bass_tone = (
            0.72 * np.sin(2 * np.pi * freq * times)
            + 0.12 * np.sin(2 * np.pi * freq * 2 * times)
        )
        add_event(bass, start_s, (0.055 * env * bass_tone).astype(np.float32))

    music = 0.72 * pad + 0.07 * bass + 0.21 * melody

    # Ruido rosa estéreo decorrelacionado (una instancia por canal).
    n_l = pink_L.generate(frames)
    n_r = pink_R.generate(frames)

    music_stereo = np.column_stack((music, music))
    noise_stereo = np.column_stack((n_l, n_r))

    mixed = music_stereo * (1.0 - noise_level) + noise_stereo * noise_level

    # Pulso rítmico centrado.
    pulse = np.zeros(frames, dtype=np.float32)
    add_rhythmic_pulse(pulse, start_frame, frames, plan["tempo_bpm"], pulse_gain)
    mixed[:, 0] += pulse
    mixed[:, 1] += pulse

    # Fades solo en los extremos del audio completo.
    fade_length = min(int(MIN_FADE_SECONDS * SAMPLE_RATE), total_frames // 2)
    fade = np.ones(frames, dtype=np.float32)

    if start_frame < fade_length:
        fi = min(frames, fade_length - start_frame)
        fade[:fi] *= np.linspace(start_frame / fade_length, 1.0, fi).astype(np.float32)

    end_frame = start_frame + frames
    if end_frame > total_frames - fade_length:
        fo_start = max(0, total_frames - fade_length - start_frame)
        fo_len = frames - fo_start
        if fo_len > 0:
            fade[fo_start:] *= np.linspace(1.0, 0.0, fo_len).astype(np.float32)

    mixed *= fade[:, None]
    return mixed.astype(np.float32)


# ===========================================================================
# Generación completa
# ===========================================================================
def generate_focus_wav(
    duration: int,
    energia: float,
    noise_level: float,
    pulse_gain: float,
    seed: int,
    progress_callback=None,
) -> bytes:
    total_frames = int(duration * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    plan = build_music_plan(total_frames, energia, rng)

    pink_L = PinkNoiseGenerator(rng)
    pink_R = PinkNoiseGenerator(rng)

    full = np.zeros((total_frames, 2), dtype=np.float32)

    for start_frame in range(0, total_frames, CHUNK_FRAMES):
        frames = min(CHUNK_FRAMES, total_frames - start_frame)
        chunk = render_chunk(
            start_frame=start_frame,
            frames=frames,
            total_frames=total_frames,
            plan=plan,
            pink_L=pink_L,
            pink_R=pink_R,
            noise_level=noise_level,
            pulse_gain=pulse_gain,
        )
        full[start_frame:start_frame + frames] = chunk
        if progress_callback:
            progress_callback(min(1.0, (start_frame + frames) / total_frames))

    # Filtro maestro global (una sola vez, sin clics).
    master = signal.butter(2, 5500, btype="lowpass", fs=SAMPLE_RATE, output="sos")
    for ch in range(2):
        full[:, ch] = signal.sosfiltfilt(
            master, full[:, ch].astype(np.float64)
        ).astype(np.float32)

    peak = float(np.max(np.abs(full)))
    if peak > 1e-8:
        full *= TARGET_PEAK / peak

    full = np.tanh(full * 1.05) * 0.95
    full = np.clip(full, -0.98, 0.98)

    pcm = np.int16(full * 32767)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    output.seek(0)
    return output.getvalue()


# ===========================================================================
# Preview y utilidades
# ===========================================================================
def extract_preview(audio_bytes: bytes, max_points: int = 400) -> np.ndarray:
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
            n = wav.getnframes()
            ch = wav.getnchannels()
            raw = np.frombuffer(wav.readframes(n), dtype=np.int16)
    except Exception:
        return np.zeros(max_points)

    if raw.size == 0:
        return np.zeros(max_points)

    raw = raw[: raw.size - (raw.size % ch)]
    stereo = raw.reshape(-1, ch)
    mono = stereo[:, 0].astype(np.float32) / 32767

    if mono.size <= max_points:
        return mono

    block = int(np.ceil(mono.size / max_points))
    trimmed = mono[: block * max_points].reshape(max_points, block)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def format_time(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ===========================================================================
# UI
# ===========================================================================
st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">NeuroFocus · Herramienta de apoyo</div>
        <h1>Neuro<br><span class="accent">Focus.</span></h1>
        <p class="description">
            Música generativa con ruido rosa para enmascarar distracciones
            y un pulso estable que sirve de ancla temporal. Diseñado
            como apoyo al entorno de trabajo de personas con TDAH.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

duration = st.slider(
    "Duración",
    min_value=60,
    max_value=MAX_DURATION,
    value=600,
    step=60,
    format="%d segundos",
)

st.write("")

col_e, col_r, col_p = st.columns(3)

with col_e:
    energia_pct = st.slider(
        "Energía",
        0, 100, 50,
        help=(
            "Tempo, densidad de la melodía y brillo. "
            "Más energía para empezar tareas; menos para trabajo tranquilo."
        ),
    )
    st.caption(f"Tempo: ~{60 + energia_pct * 0.15:.0f} BPM")

with col_r:
    ruido_pct = st.slider(
        "Ruido de fondo",
        0, 100, 45,
        help=(
            "Ruido rosa (1/f) que enmascara distracciones del entorno. "
            "La evidencia sugiere mejoras pequeñas pero reales en "
            "atención para TDAH (g ≈ 0.25)."
        ),
    )

with col_p:
    pulso_pct = st.slider(
        "Ancla rítmica",
        0, 100, 35,
        help=(
            "Pulso suave y estable a tempo fijo. Da una referencia "
            "temporal para sostener la atención."
        ),
    )

energia = energia_pct / 100.0
noise_level = (ruido_pct / 100.0) * 0.35
pulse_gain = (pulso_pct / 100.0) * 0.06

st.markdown(
    """
    <div class="notice">
        NeuroFocus no es un tratamiento médico. Es una herramienta
        de apoyo: el ruido rosa y el pulso estable pueden ayudar a
        enmascarar distracciones y sostener la atención, pero no
        sustituyen el diagnóstico ni el tratamiento del TDAH.
        Baja el volumen si notas molestias.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if st.button("🎧 Generar composición", type="primary", use_container_width=True):
    seed = secrets.randbits(32)
    bar = st.progress(0, text="Preparando...")
    try:
        with st.spinner("Generando..."):
            audio_bytes = generate_focus_wav(
                duration=duration,
                energia=energia,
                noise_level=noise_level,
                pulse_gain=pulse_gain,
                seed=seed,
                progress_callback=lambda v: bar.progress(
                    int(v * 100), text=f"Generando: {int(v * 100)}%"
                ),
            )
        st.session_state["audio_bytes"] = bytes(audio_bytes)
        st.session_state["settings"] = {
            "duration": duration,
            "energia": energia_pct,
            "ruido": ruido_pct,
            "pulso": pulso_pct,
            "seed": seed,
        }
    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        bar.empty()


if "audio_bytes" in st.session_state:
    audio_bytes = bytes(st.session_state["audio_bytes"])
    s = st.session_state["settings"]

    st.success(
        f"Listo · {format_time(s['duration'])} · "
        f"Energía {s['energia']}% · Ruido {s['ruido']}% · Ancla {s['pulso']}%"
    )

    st.audio(audio_bytes, format="audio/wav")

    file_name = (
        f"neurofocus-{s['duration']}s-"
        f"e{s['energia']}-r{s['ruido']}-p{s['pulso']}-"
        f"{s['seed']}.wav"
    )

    st.download_button(
        label="⬇️ Descargar WAV",
        data=audio_bytes,
        file_name=file_name,
        mime="audio/wav",
        key=f"dl_{s['seed']}",
        use_container_width=True,
    )

    st.caption(
        f"Tamaño: {len(audio_bytes) / 1_000_000:.1f} MB · Semilla: {s['seed']}"
    )

    st.subheader("Forma de onda (RMS por bloque)")
    st.line_chart(extract_preview(audio_bytes), height=180, use_container_width=True)

st.divider()
st.caption(
    "Audio procedural original. No es un tratamiento médico para el TDAH. "
    "Los efectos del ruido rosa son pequeños y variables entre personas. "
    "Si tienes TDAH, consulta con un profesional para un plan de tratamiento integral."
)
