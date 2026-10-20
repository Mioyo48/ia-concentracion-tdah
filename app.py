import streamlit as st
import numpy as np
from scipy.signal import butter, sosfilt
import io
import wave
import struct
import random

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Focus Audio",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080B0F !important;
    color: #C8D0DC !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: #080B0F !important;
}

[data-testid="stHeader"] { display: none; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
footer { display: none !important; }
.block-container {
    max-width: 560px !important;
    padding: 3.5rem 1.5rem 4rem !important;
    margin: 0 auto;
}

/* ── Typography ── */
h1 {
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.18em !important;
    color: #E8EDF4 !important;
    margin-bottom: 0.2rem !important;
}
.subtitle {
    font-size: 0.72rem;
    color: #4A5568;
    letter-spacing: 0.06em;
    margin-bottom: 3rem;
}

/* ── Mode cards ── */
.mode-row {
    display: flex;
    gap: 12px;
    margin-bottom: 2.5rem;
}
.mode-card {
    flex: 1;
    border: 1px solid #1A2030;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: #0D1117;
}
.mode-card.active {
    border-color: #3B82F6;
    background: #0F1929;
}
.mode-card h3 {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #E2E8F0;
    margin: 0 0 0.3rem;
}
.mode-card p {
    font-size: 0.65rem;
    color: #4A5568;
    line-height: 1.5;
    margin: 0;
}
.mode-card .freq-badge {
    display: inline-block;
    margin-top: 0.6rem;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: #3B82F6;
    background: #0A1628;
    border: 1px solid #1E3A5F;
    border-radius: 4px;
    padding: 2px 7px;
}

/* ── Slider labels ── */
.param-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin-bottom: 0.25rem;
    margin-top: 1.4rem;
    display: block;
}

/* ── Streamlit sliders ── */
[data-testid="stSlider"] > div > div > div {
    background: #1A2030 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #3B82F6 !important;
    border: 2px solid #1D4ED8 !important;
}

/* ── Radio buttons as cards ── */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 10px !important;
}
[data-testid="stRadio"] label {
    flex: 1;
    background: #0D1117 !important;
    border: 1px solid #1A2030 !important;
    border-radius: 10px !important;
    padding: 0.75rem 0.5rem !important;
    cursor: pointer !important;
    transition: border-color 0.2s !important;
    text-align: center;
    font-size: 0.72rem !important;
    color: #8892A4 !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    border-color: #3B82F6 !important;
    color: #E2E8F0 !important;
    background: #0F1929 !important;
}
[data-testid="stRadio"] input { display: none !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
}

/* ── Generate button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    margin-top: 2rem !important;
    background: #1D4ED8 !important;
    color: #EFF6FF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
}
[data-testid="stButton"] > button:hover {
    background: #2563EB !important;
}

/* ── Audio player ── */
audio {
    width: 100%;
    margin-top: 1.5rem;
    filter: invert(1) hue-rotate(180deg) brightness(0.85);
    border-radius: 6px;
}
[data-testid="stAudio"] {
    margin-top: 1rem;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #141A24;
    margin: 2rem 0;
}

/* ── Status pulse ── */
.status-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #22C55E;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
.status-line {
    font-size: 0.62rem;
    color: #374151;
    letter-spacing: 0.06em;
    margin-top: 1.8rem;
}

/* ── Noise type row ── */
.section-label {
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    color: #374151;
    margin-bottom: 0.5rem;
    margin-top: 1.6rem;
    display: block;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─── Audio engine ──────────────────────────────────────────────────────────────

def generate_brown_noise(n_samples: int) -> np.ndarray:
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    brown -= brown.mean()
    return brown / (np.abs(brown).max() + 1e-9)


def generate_pink_noise(n_samples: int) -> np.ndarray:
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1e-6
    power = 1.0 / np.sqrt(f)
    phase = np.random.uniform(0, 2 * np.pi, len(f))
    spectrum = power * np.exp(1j * phase)
    pink = np.fft.irfft(spectrum, n=n_samples)
    pink -= pink.mean()
    return pink / (np.abs(pink).max() + 1e-9)


def bandpass(signal: np.ndarray, lo: float, hi: float, sr: int) -> np.ndarray:
    sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return sosfilt(sos, signal)


def generate_focus_audio(
    mode: str,
    duration_min: int,
    noise_blend: float,       # 0.0 = all brown, 1.0 = all pink
    intensity: float,         # 0.0–1.0
    sr: int = 44100,
) -> bytes:
    """
    Architecture:
      Static Base Layer  (80%) — shaped broadband noise, constant
      Neural Mod Layer   (20%) — AM-modulated at beta frequency

    No pure tones. Modulation applied only to the smaller layer,
    eliminating the helicopter / pumping artifact.
    """

    n = duration_min * 60 * sr

    # ── Dynamic variability: slight randomness each call ──────────────────
    rng_seed = random.randint(0, 2**31)
    rng = np.random.default_rng(rng_seed)

    if mode == "TDAH":
        # Beta alta: estado de alerta máximo
        base_freq = rng.uniform(18.0, 25.0)
        freq_drift = rng.uniform(0.3, 1.2)     # Hz de deriva lenta
        lo_band, hi_band = 200.0, 8000.0        # espectro más brillante
    else:
        # Beta baja: concentración sostenida
        base_freq = rng.uniform(13.0, 16.0)
        freq_drift = rng.uniform(0.1, 0.6)
        lo_band, hi_band = 80.0, 6000.0         # espectro más cálido

    # ── Base noise (mezcla marrón + rosa) ────────────────────────────────
    brown = generate_brown_noise(n)
    pink  = generate_pink_noise(n)
    raw_noise = (1.0 - noise_blend) * brown + noise_blend * pink

    # Filtrar al rango de frecuencia elegido
    shaped = bandpass(raw_noise, lo_band, hi_band, sr)
    shaped /= (np.abs(shaped).max() + 1e-9)

    # ── Capa base estática (80% volumen) ──────────────────────────────────
    base_layer = shaped * 0.80

    # ── Capa de modulación neural (20% volumen) ──────────────────────────
    # Frecuencia de modulación con deriva lenta para evitar habituación
    t = np.linspace(0, duration_min * 60, n, endpoint=False)

    # Deriva sinusoidal lenta sobre la frecuencia beta
    freq_envelope = base_freq + freq_drift * np.sin(2 * np.pi * 0.05 * t)
    # Fase acumulada (modulación FM suave en la portadora AM)
    phase = 2 * np.pi * np.cumsum(freq_envelope) / sr

    # AM carrier con profundidad de modulación variable
    mod_depth = rng.uniform(0.35, 0.55)
    carrier = 0.5 * (1.0 + mod_depth * np.sin(phase))

    # Ruido independiente para la capa de modulación
    mod_noise_raw = (1.0 - noise_blend) * generate_brown_noise(n) + \
                    noise_blend * generate_pink_noise(n)
    mod_noise = bandpass(mod_noise_raw, lo_band, hi_band, sr)
    mod_noise /= (np.abs(mod_noise).max() + 1e-9)

    mod_layer = mod_noise * carrier * 0.20

    # ── Mezcla final ──────────────────────────────────────────────────────
    mixed = base_layer + mod_layer

    # Intensity scaling con curva logarítmica
    gain = 0.25 + 0.60 * (intensity ** 0.7)
    mixed *= gain

    # Fade in/out (2 s)
    fade_samples = min(2 * sr, n // 10)
    fade_in  = np.linspace(0, 1, fade_samples) ** 2
    fade_out = fade_in[::-1]
    mixed[:fade_samples]  *= fade_in
    mixed[-fade_samples:] *= fade_out

    # Limiter suave
    mixed = np.tanh(mixed * 1.5) / 1.5

    # ── Encode WAV 16-bit ──────────────────────────────────────────────────
    pcm = np.clip(mixed, -1.0, 1.0)
    pcm_int = (pcm * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int.tobytes())
    return buf.getvalue()


# ─── UI ───────────────────────────────────────────────────────────────────────

st.markdown('<h1>◉ FOCUS AUDIO</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Neural entrainment · Beta wave · Anti-fatigue</p>',
            unsafe_allow_html=True)

# ── Mode selector ─────────────────────────────────────────────────────────────
mode = st.radio(
    label="",
    options=["TDAH", "Normal"],
    horizontal=True,
    label_visibility="collapsed",
)

# Mode description
if mode == "TDAH":
    st.markdown("""
    <div style="font-size:0.65rem; color:#4A5568; line-height:1.7;
                border-left: 2px solid #1E3A5F; padding-left:12px;
                margin: 0.8rem 0 1.8rem;">
        Beta alta 18–25 Hz · Activación dopaminérgica · Banda espectral amplia<br>
        Diseñado para mantener el sistema de recompensa activo sin sobreestimular.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="font-size:0.65rem; color:#4A5568; line-height:1.7;
                border-left: 2px solid #1A2F1A; padding-left:12px;
                margin: 0.8rem 0 1.8rem;">
        Beta baja 13–16 Hz · Concentración sostenida · Flujo cognitivo estable<br>
        Ideal para tareas analíticas de larga duración.
    </div>
    """, unsafe_allow_html=True)

# ── Parameters ────────────────────────────────────────────────────────────────
st.markdown('<span class="param-label">DURACIÓN</span>', unsafe_allow_html=True)
duration = st.slider("dur", 5, 90, 25, step=5,
                     format="%d min", label_visibility="collapsed")

st.markdown('<span class="param-label">TEXTURA DE RUIDO</span>', unsafe_allow_html=True)
st.markdown("""<div style="display:flex; justify-content:space-between;
            font-size:0.58rem; color:#374151; margin-bottom:4px;">
            <span>Marrón (profundo)</span><span>Rosa (brillante)</span>
            </div>""", unsafe_allow_html=True)
noise_blend = st.slider("noise", 0.0, 1.0, 0.35, step=0.05,
                        label_visibility="collapsed")

st.markdown('<span class="param-label">INTENSIDAD</span>', unsafe_allow_html=True)
intensity = st.slider("intens", 0.1, 1.0, 0.65, step=0.05,
                      label_visibility="collapsed")

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("GENERAR AUDIO"):
    with st.spinner(""):
        st.markdown("""
        <div style="font-size:0.65rem; color:#374151;
                    letter-spacing:0.06em; margin:0.5rem 0;">
            Sintetizando capas neurales…
        </div>""", unsafe_allow_html=True)
        audio_bytes = generate_focus_audio(
            mode=mode,
            duration_min=duration,
            noise_blend=noise_blend,
            intensity=intensity,
        )

    st.audio(audio_bytes, format="audio/wav")

    freq_range = "18–25 Hz" if mode == "TDAH" else "13–16 Hz"
    blend_pct  = int(noise_blend * 100)
    st.markdown(f"""
    <div style="margin-top:1.2rem; padding: 1rem 1.1rem;
                background:#0D1117; border:1px solid #141A24;
                border-radius:8px; font-size:0.62rem;
                color:#4A5568; line-height:2.0; letter-spacing:0.04em;">
        Modo &nbsp;·&nbsp; <span style="color:#6B7280">{mode}</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Modulación &nbsp;·&nbsp; <span style="color:#6B7280">{freq_range}</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Textura &nbsp;·&nbsp;
        <span style="color:#6B7280">{100-blend_pct}% marrón · {blend_pct}% rosa</span>
        &nbsp;&nbsp;·&nbsp;&nbsp;
        Duración &nbsp;·&nbsp; <span style="color:#6B7280">{duration} min</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="status-line" style="margin-top:3rem;">
    Cada generación produce una sesión única · Sin habituación cerebral
</div>
""", unsafe_allow_html=True)
