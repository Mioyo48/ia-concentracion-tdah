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
PREVIEW_SECONDS = 30

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
PINK_GAIN = 0.45


# ===========================================================================
# Configuración de página
# ===========================================================================
st.set_page_config(page_title="NeuroFocus", page_icon="🎧", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(56,189,248,.20), transparent 30%),
                radial-gradient(circle at 85% 85%, rgba(244,63,94,.16), transparent 30%),
                linear-gradient(135deg, #020617, #0f172a, #1e293b);
        }
        .block-container { max-width: 900px; padding-top: 2.5rem; }
        .hero {
            padding: 2.4rem;
            border: 1px solid rgba(148,163,184,.2);
            border-radius: 24px;
            background: rgba(15,23,42,.75);
            box-shadow: 0 30px 80px rgba(0,0,0,.45);
        }
        .eyebrow {
            color: #7dd3fc; font-size: .72rem; font-weight: 800;
            letter-spacing: .18em; text-transform: uppercase;
        }
        .hero h1 {
            font-size: clamp(2.4rem, 7vw, 4.8rem);
            line-height: .95; letter-spacing: -.06em; margin: .3rem 0 0 0;
        }
        .accent { color: #7dd3fc; }
        .description { color: #cbd5e1; line-height: 1.65; margin-top: .8rem; }
        .notice {
            padding: 1rem; border-left: 3px solid #38bdf8; border-radius: 8px;
            color: #e2e8f0; background: rgba(56,189,248,.08); font-size: .85rem;
        }
        .preview-tag {
            display: inline-block; padding: .2rem .6rem; border-radius: 6px;
            background: rgba(56,189,248,.15); color: #7dd3fc;
            font-size: .72rem; font-weight: 700; letter-spacing: .1em;
            text-transform: uppercase;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# Helpers DSP
# ===========================================================================
def normalize(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    audio = audio - float(np.mean(audio))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-8:
        return np.zeros_like(audio)
    return audio / rms


def midi_to_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69.0) / 12.0))


def soft_tone(freq, times, phase):
    bp = 2 * np.pi * freq * times + phase
    tone = (
        0.78 * np.sin(bp)
        + 0.16 * np.sin(bp * 2.0)
        + 0.06 * np.sin(bp * 3.0)
    )
    return tone.astype(np.float32)


def adsr_envelope(n_samples: int, sr: int,
                  a: float = 0.02, d: float = 0.12,
                  s: float = 0.55, r: float = 0.35) -> np.ndarray:
    """ADSR con escalado proporcional si no cabe en n_samples."""
    total = a + d + r
    if total > n_samples / sr and total > 0:
        scale = (n_samples / sr) / total
        a, d, r = a * scale, d * scale, r * scale

    na, nd, nr = int(a * sr), int(d * sr), int(r * sr)
    ns = max(0, n_samples - na - nd - nr)

    env = np.zeros(n_samples, dtype=np.float32)
    idx = 0
    if na > 0:
        env[idx:idx + na] = np.linspace(0.0, 1.0, na, dtype=np.float32); idx += na
    if nd > 0:
        env[idx:idx + nd] = np.linspace(1.0, s, nd, dtype=np.float32); idx += nd
    if ns > 0:
        env[idx:idx + ns] = s; idx += ns
    if nr > 0 and idx < n_samples:
        rr = min(nr, n_samples - idx)
        env[idx:idx + rr] = np.linspace(s, 0.0, rr, dtype=np.float32)
    return env


# ===========================================================================
# Ruido rosa (filtro Kellett, estado continuo entre chunks)
# ===========================================================================
class PinkNoiseGenerator:
    """Ruido rosa 1/f con estado persistente. Instancia por canal."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.b0_zi = np.zeros(1); self.b1_zi = np.zeros(1)
        self.b2_zi = np.zeros(1); self.b3_zi = np.zeros(1)
        self.b4_zi = np.zeros(1); self.b5_zi = np.zeros(1)
        self.b6_state = 0.0

    def generate(self, n: int) -> np.ndarray:
        w = self.rng.standard_normal(n).astype(np.float64)
        b0, self.b0_zi = signal.lfilter([0.0555179], [1, -0.99886], w, zi=self.b0_zi)
        b1, self.b1_zi = signal.lfilter([0.0750759], [1, -0.99332], w, zi=self.b1_zi)
        b2, self.b2_zi = signal.lfilter([0.1538520], [1, -0.96900], w, zi=self.b2_zi)
        b3, self.b3_zi = signal.lfilter([0.3104856], [1, -0.86650], w, zi=self.b3_zi)
        b4, self.b4_zi = signal.lfilter([0.5329522], [1, -0.55000], w, zi=self.b4_zi)
        b5, self.b5_zi = signal.lfilter([-0.0168980], [1, 0.7616], w, zi=self.b5_zi)

        b6_prev = np.empty_like(w)
        b6_prev[0] = self.b6_state
        b6_prev[1:] = w[:-1] * 0.115926
        self.b6_state = float(w[-1] * 0.115926)

        pink = (b0 + b1 + b2 + b3 + b4 + b5 + b6_prev + w * 0.5362) * PINK_GAIN
        return pink.astype(np.float32)


# ===========================================================================
# Reverb por convolución
# ===========================================================================
def make_ir(seed: int, sr: int, duration: float = 1.8,
            decay: float = 3.5) -> np.ndarray:
    n = int(duration * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    ir = (noise * np.exp(-t * decay)).astype(np.float32)
    sos = signal.butter(2, 5000, btype="lowpass", fs=sr, output="sos")
    ir = signal.sosfilt(sos, ir).astype(np.float32)
    ir /= (np.sqrt(np.sum(ir ** 2)) + 1e-8)
    return ir


def apply_reverb(stereo: np.ndarray, sr: int, wet: float = 0.18) -> np.ndarray:
    if wet <= 0:
        return stereo
    ir_L = make_ir(seed=0, sr=sr)
    ir_R = make_ir(seed=1, sr=sr)
    wL = signal.oaconvolve(stereo[:, 0], ir_L, mode="same").astype(np.float32)
    wR = signal.oaconvolve(stereo[:, 1], ir_R, mode="same").astype(np.float32)
    out = np.empty_like(stereo)
    out[:, 0] = stereo[:, 0] * (1 - wet) + wL * wet * 2.0
    out[:, 1] = stereo[:, 1] * (1 - wet) + wR * wet * 2.0
    return out


# ===========================================================================
# Cadena master
# ===========================================================================
def master_chain(stereo: np.ndarray, sr: int) -> np.ndarray:
    hp = signal.butter(2, 30, btype="highpass", fs=sr, output="sos")
    lp = signal.butter(2, 5500, btype="lowpass", fs=sr, output="sos")
    out = np.empty_like(stereo)
    for ch in range(2):
        x = stereo[:, ch].astype(np.float64)
        x = signal.sosfiltfilt(hp, x)
        x = signal.sosfiltfilt(lp, x)
        out[:, ch] = x.astype(np.float32)

    peak = float(np.max(np.abs(out)))
    if peak > 1e-8:
        out *= TARGET_PEAK / peak

    out = np.tanh(out * 1.05) * 0.95
    return np.clip(out, -0.98, 0.98).astype(np.float32)


# ===========================================================================
# Plan musical
# ===========================================================================
def build_music_plan(total_frames: int, energia: float,
                     rng: np.random.Generator) -> dict:
    progression = [
        [50, 57, 62, 66], [47, 54, 59, 62],
        [43, 50, 55, 59], [45, 52, 57, 61],
        [50, 57, 62, 66], [45, 52, 57, 61],
        [47, 54, 59, 62], [43, 50, 55, 62],
    ]
    segment_count = max(1, int(np.ceil(total_frames / SEGMENT_FRAMES)))
    base_seq = np.array([0, 1, 2, 3, 0, 3, 1, 2], dtype=np.int32)
    start = int(rng.integers(0, len(base_seq)))
    chord_ids = np.array(
        [base_seq[(i + start) % len(base_seq)] for i in range(segment_count)],
        dtype=np.int32,
    )

    active_p = 0.30 + energia * 0.40
    vel_mean = 0.62 + energia * 0.16

    return {
        "progression": progression,
        "chord_ids": chord_ids,
        "note_choices": rng.integers(0, 4, size=(segment_count, STEPS_PER_SEGMENT)),
        "note_active": rng.random((segment_count, STEPS_PER_SEGMENT)) < active_p,
        "note_lengths": np.clip(
            rng.normal(0.75, 0.10, (segment_count, STEPS_PER_SEGMENT)), 0.5, 1.0
        ),
        "note_velocities": np.clip(
            rng.normal(vel_mean, 0.10, (segment_count, STEPS_PER_SEGMENT)),
            0.45, 0.95,
        ),
        "note_offsets": np.clip(
            rng.normal(0.0, 0.008, (segment_count, STEPS_PER_SEGMENT)), -0.02, 0.02
        ),
        "pad_phases": rng.uniform(0, 2 * np.pi, (segment_count, 4)),
        "melody_phases": rng.uniform(0, 2 * np.pi, (segment_count, STEPS_PER_SEGMENT)),
        "tempo_bpm": 60.0 + energia * 15.0,
    }


# ===========================================================================
# Pulso rítmico
# ===========================================================================
def add_rhythmic_pulse(target, start_frame, frames, tempo_bpm, gain):
    if gain <= 0 or tempo_bpm <= 0:
        return
    beat = 60.0 / tempo_bpm
    t0 = start_frame / SAMPLE_RATE
    t1 = (start_frame + frames) / SAMPLE_RATE

    pulse_dur = 0.20
    pulse_n = int(pulse_dur * SAMPLE_RATE)
    t = np.arange(pulse_n) / SAMPLE_RATE

    attack = max(1, int(0.006 * SAMPLE_RATE))
    env = np.ones(pulse_n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    env *= np.exp(-t / 0.08).astype(np.float32)

    body = (
        0.55 * np.sin(2 * np.pi * 90 * t)
        + 0.20 * np.sin(2 * np.pi * 180 * t)
        + 0.10 * np.sin(2 * np.pi * 400 * t) * np.exp(-t / 0.02)
    ).astype(np.float32)
    wave = (body * env).astype(np.float32)

    accents = [1.0, 0.6, 0.8, 0.6]
    first = max(0, int(np.floor((t0 - pulse_dur) / beat)))
    last = int(np.ceil(t1 / beat))

    for k in range(first, last + 1):
        bt = k * beat
        acc = accents[k % len(accents)]
        s = int((bt - t0) * SAMPLE_RATE)
        e = s + pulse_n
        src_s = max(0, -s)
        dst_s = max(0, s)
        dst_e = min(frames, e)
        if dst_e <= dst_s:
            continue
        src_e = src_s + (dst_e - dst_s)
        target[dst_s:dst_e] += wave[src_s:src_e] * gain * acc


# ===========================================================================
# Render de un chunk
# ===========================================================================
def render_chunk(start_frame, frames, total_frames, plan,
                 pink_L, pink_R, noise_level, pulse_gain) -> np.ndarray:
    seg_i = min(start_frame // SEGMENT_FRAMES, len(plan["chord_ids"]) - 1)
    chord = plan["progression"][plan["chord_ids"][seg_i]]

    seg_off = (start_frame % SEGMENT_FRAMES) / SAMPLE_RATE
    t_local = np.arange(frames, dtype=np.float64) / SAMPLE_RATE
    t_seg = seg_off + t_local

    def render_pad_stereo(pad_chord, phases, times):
        L = np.zeros(len(times), dtype=np.float32)
        R = np.zeros(len(times), dtype=np.float32)
        amps = [0.23, 0.15, 0.09, 0.05]
        detune = 0.004  # ±4 cents
        for i, note in enumerate(pad_chord):
            freq = midi_to_hz(note + 12)
            ph = phases[i]
            mov = 0.95 + 0.05 * np.sin(2 * np.pi * 0.045 * times + ph)
            L += amps[i] * mov * soft_tone(freq * (1 + detune), times, ph)
            R += amps[i] * mov * soft_tone(freq * (1 - detune), times, ph + 0.3)
        return L, R

    pad_L, pad_R = render_pad_stereo(chord, plan["pad_phases"][seg_i], t_seg)

    if seg_i > 0:
        prev_chord = plan["progression"][plan["chord_ids"][seg_i - 1]]
        pL, pR = render_pad_stereo(
            prev_chord, plan["pad_phases"][seg_i - 1], t_seg + SEGMENT_SECONDS
        )
        xf = np.clip(t_seg / 1.8, 0.0, 1.0)
        pad_L = pL * (1 - xf) + pad_L * xf
        pad_R = pR * (1 - xf) + pad_R * xf

    bass = np.zeros(frames, dtype=np.float32)
    melody = np.zeros(frames, dtype=np.float32)

    def add_event(target, start_s, values):
        si = int(np.floor((start_s - seg_off) * SAMPLE_RATE))
        ei = si + len(values)
        cs, ce = max(0, si), min(frames, ei)
        if ce <= cs:
            return
        ss = cs - si
        se = ss + ce - cs
        target[cs:ce] += values[ss:se]

    choices = plan["note_choices"][seg_i]
    active = plan["note_active"][seg_i]
    lengths = plan["note_lengths"][seg_i]
    vels = plan["note_velocities"][seg_i]
    offs = plan["note_offsets"][seg_i]
    mph = plan["melody_phases"][seg_i]

    for step in range(STEPS_PER_SEGMENT):
        if not active[step]:
            continue
        st_s = step * STEP_SECONDS + float(offs[step])
        dur = float(lengths[step])
        vel = float(vels[step])
        note = chord[int(choices[step])] + 24
        freq = midi_to_hz(note)
        ph = float(mph[step])
        n = max(1, int(dur * SAMPLE_RATE))
        tt = np.arange(n) / SAMPLE_RATE
        env = adsr_envelope(n, SAMPLE_RATE, a=0.02, d=0.15, s=0.5, r=0.30)
        tone = soft_tone(freq, tt, ph)
        # Vibrato sutil
        vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.2 * tt + ph)
        tone *= vib.astype(np.float32)
        add_event(melody, st_s, 0.10 * vel * env * tone)

    for step in (0, 4):
        st_s = step * STEP_SECONDS
        dur = 1.35
        freq = midi_to_hz(chord[0])
        n = int(dur * SAMPLE_RATE)
        tt = np.arange(n) / SAMPLE_RATE
        env = adsr_envelope(n, SAMPLE_RATE, a=0.01, d=0.20, s=0.4, r=0.5)
        bass_tone = (
            0.72 * np.sin(2 * np.pi * freq * tt)
            + 0.12 * np.sin(2 * np.pi * freq * 2 * tt)
        ).astype(np.float32)
        add_event(bass, st_s, 0.055 * env * bass_tone)

    # Mezcla base (música)
    pad_mono = 0.5 * (pad_L + pad_R)
    pad_side = 0.5 * (pad_L - pad_R)
    music_L = 0.72 * pad_mono + 0.07 * bass + 0.21 * melody + pad_side
    music_R = 0.72 * pad_mono + 0.07 * bass + 0.21 * melody - pad_side

    # Ruido rosa estéreo decorrelacionado
    nL = pink_L.generate(frames)
    nR = pink_R.generate(frames)

    mix_L = music_L * (1.0 - noise_level) + nL * noise_level
    mix_R = music_R * (1.0 - noise_level) + nR * noise_level

    # Pulso
    pulse = np.zeros(frames, dtype=np.float32)
    add_rhythmic_pulse(pulse, start_frame, frames, plan["tempo_bpm"], pulse_gain)
    mix_L += pulse
    mix_R += pulse

    # Fades en los extremos del audio completo
    fade_len = min(int(MIN_FADE_SECONDS * SAMPLE_RATE), total_frames // 2)
    fade = np.ones(frames, dtype=np.float32)
    if start_frame < fade_len:
        fi = min(frames, fade_len - start_frame)
        fade[:fi] *= np.linspace(start_frame / fade_len, 1.0, fi).astype(np.float32)
    end = start_frame + frames
    if end > total_frames - fade_len:
        fo_start = max(0, total_frames - fade_len - start_frame)
        fo_len = frames - fo_start
        if fo_len > 0:
            fade[fo_start:] *= np.linspace(1.0, 0.0, fo_len).astype(np.float32)
    mix_L *= fade
    mix_R *= fade

    return np.column_stack((mix_L, mix_R)).astype(np.float32)


# ===========================================================================
# Generación completa
# ===========================================================================
def generate_focus_wav(duration, energia, noise_level, pulse_gain, reverb_wet,
                       seed, progress_callback=None) -> bytes:
    total_frames = int(duration * SAMPLE_RATE)
    rng = np.random.default_rng(seed)
    plan = build_music_plan(total_frames, energia, rng)
    pink_L = PinkNoiseGenerator(rng)
    pink_R = PinkNoiseGenerator(rng)

    full = np.zeros((total_frames, 2), dtype=np.float32)

    for start in range(0, total_frames, CHUNK_FRAMES):
        frames = min(CHUNK_FRAMES, total_frames - start)
        chunk = render_chunk(
            start, frames, total_frames, plan,
            pink_L, pink_R, noise_level, pulse_gain,
        )
        full[start:start + frames] = chunk
        if progress_callback:
            progress_callback(min(1.0, (start + frames) / total_frames))

    # Reverb global (una pasada)
    full = apply_reverb(full, SAMPLE_RATE, wet=reverb_wet)

    # Master (DC block, HPF, LPF, limiter)
    full = master_chain(full, SAMPLE_RATE)

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
    mono = raw.reshape(-1, ch)[:, 0].astype(np.float32) / 32767
    if mono.size <= max_points:
        return mono
    block = int(np.ceil(mono.size / max_points))
    trimmed = mono[: block * max_points].reshape(max_points, block)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def format_time(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


# ===========================================================================
# Cache de generación
# ===========================================================================
@st.cache_data(show_spinner=False)
def _cached_generate(duration, energia_pct, ruido_pct, pulso_pct,
                     reverb_pct, seed) -> bytes:
    return generate_focus_wav(
        duration=duration,
        energia=energia_pct / 100.0,
        noise_level=(ruido_pct / 100.0) * 0.35,
        pulse_gain=(pulso_pct / 100.0) * 0.06,
        reverb_wet=(reverb_pct / 100.0) * 0.35,
        seed=seed,
    )


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
            y un pulso estable que sirve de ancla temporal. Suena desde
            que abres la página.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------------------------
# Preview autoplay
# ---------------------------------------------------------------------------
if "preview_bytes" not in st.session_state:
    with st.spinner("Cargando preview..."):
        st.session_state["preview_bytes"] = _cached_generate(
            PREVIEW_SECONDS, 50, 45, 35, 40, 42
        )
    st.session_state["preview_settings"] = {
        "duration": PREVIEW_SECONDS, "energia": 50, "ruido": 45,
        "pulso": 35, "reverb": 40, "seed": 42,
    }

st.markdown('<span class="preview-tag">● Preview · suena al cargar</span>',
            unsafe_allow_html=True)

st.audio(
    st.session_state["preview_bytes"],
    format="audio/wav",
    autoplay=True,
)

st.caption(
    f"Preview de {PREVIEW_SECONDS} s · "
    f"Energía {st.session_state['preview_settings']['energia']}% · "
    f"Ruido {st.session_state['preview_settings']['ruido']}% · "
    f"Ancla {st.session_state['preview_settings']['pulso']}% · "
    f"Espacio {st.session_state['preview_settings']['reverb']}%"
)

st.write("")
st.divider()

# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------
st.subheader("Ajusta la composición")

r1c1, r1c2 = st.columns(2)
with r1c1:
    energia_pct = st.slider(
        "Energía", 0, 100, 50,
        help="Tempo, densidad y brillo. Súbelo para arrancar; bájalo para trabajo tranquilo.",
    )
    st.caption(f"Tempo: ~{60 + energia_pct * 0.15:.0f} BPM")

with r1c2:
    ruido_pct = st.slider(
        "Ruido de fondo", 0, 100, 45,
        help="Ruido rosa (1/f) que enmascara distracciones del entorno.",
    )

r2c1, r2c2 = st.columns(2)
with r2c1:
    pulso_pct = st.slider(
        "Ancla rítmica", 0, 100, 35,
        help="Pulso suave y estable a tempo fijo. Referencia temporal para la atención.",
    )

with r2c2:
    reverb_pct = st.slider(
        "Espacio (reverb)", 0, 100, 40,
        help="Cantidad de reverb por convolución. Más espacio = sonido más amplio y difuso.",
    )

st.write("")

col_prev, col_full = st.columns(2)

with col_prev:
    if st.button("🔄 Actualizar preview", use_container_width=True):
        with st.spinner("Regenerando preview..."):
            seed = secrets.randbits(32)
            st.session_state["preview_bytes"] = _cached_generate(
                PREVIEW_SECONDS, energia_pct, ruido_pct, pulso_pct,
                reverb_pct, seed,
            )
            st.session_state["preview_settings"] = {
                "duration": PREVIEW_SECONDS, "energia": energia_pct,
                "ruido": ruido_pct, "pulso": pulso_pct,
                "reverb": reverb_pct, "seed": seed,
            }
        st.rerun()

with col_full:
    generar_full = st.button(
        "🎧 Generar versión completa",
        type="primary",
        use_container_width=True,
    )

if generar_full:
    seed = secrets.randbits(32)
    duration_full = st.session_state.get("full_duration", 600)

    bar = st.progress(0, text="Generando versión completa...")
    try:
        with st.spinner("Generando..."):
            audio_bytes = generate_focus_wav(
                duration=duration_full,
                energia=energia_pct / 100.0,
                noise_level=(ruido_pct / 100.0) * 0.35,
                pulse_gain=(pulso_pct / 100.0) * 0.06,
                reverb_wet=(reverb_pct / 100.0) * 0.35,
                seed=seed,
                progress_callback=lambda v: bar.progress(
                    int(v * 100), text=f"Generando: {int(v * 100)}%"
                ),
            )
        st.session_state["full_audio"] = bytes(audio_bytes)
        st.session_state["full_settings"] = {
            "duration": duration_full, "energia": energia_pct,
            "ruido": ruido_pct, "pulso": pulso_pct,
            "reverb": reverb_pct, "seed": seed,
        }
        st.session_state["full_ready"] = True
    except Exception as exc:
        st.error(f"Error generando: {exc}")
    finally:
        bar.empty()

# ---------------------------------------------------------------------------
# Duración de la versión completa (solo aparece cuando el usuario va a generar)
# ---------------------------------------------------------------------------
st.write("")
duracion_full = st.slider(
    "Duración de la versión completa",
    min_value=60, max_value=MAX_DURATION, value=600, step=60,
    format="%d s",
    help="Solo se aplica a 'Generar versión completa'. El preview siempre dura 30 s.",
)
st.session_state["full_duration"] = duracion_full

# ---------------------------------------------------------------------------
# Reproductor de la versión completa
# ---------------------------------------------------------------------------
if st.session_state.get("full_ready") and "full_audio" in st.session_state:
    s = st.session_state["full_settings"]
    full_bytes = bytes(st.session_state["full_audio"])

    st.divider()
    st.subheader("🎵 Versión completa")

    st.success(
        f"{format_time(s['duration'])} · "
        f"Energía {s['energia']}% · Ruido {s['ruido']}% · "
        f"Ancla {s['pulso']}% · Espacio {s['reverb']}%"
    )

    st.audio(full_bytes, format="audio/wav")

    st.download_button(
        label="⬇️ Descargar WAV",
        data=full_bytes,
        file_name=(
            f"neurofocus-{s['duration']}s-"
            f"e{s['energia']}-r{s['ruido']}-p{s['pulso']}-v{s['reverb']}-"
            f"{s['seed']}.wav"
        ),
        mime="audio/wav",
        key=f"dl_{s['seed']}",
        use_container_width=True,
    )

    st.caption(
        f"Tamaño: {len(full_bytes) / 1_000_000:.1f} MB · Semilla: {s['seed']}"
    )

    st.subheader("Forma de onda (RMS por bloque)")
    st.line_chart(extract_preview(full_bytes), height=160, use_container_width=True)

st.divider()
st.caption(
    "Audio procedural original. No es un tratamiento médico para el TDAH. "
    "Los efectos del ruido rosa son pequeños y variables. "
    "Baja el volumen si notas molestias."
)
