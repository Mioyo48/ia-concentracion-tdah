import streamlit as st
import numpy as np
from scipy.signal import butter, sosfilt
import io
import wave
import struct
import random

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Focus Audio Pro",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #080B0F !important;
    color: #C8D0DC !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: #080B0F !important;
}

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { 
    display: none !important; 
}

.block-container {
    max-width: 560px !important;
    padding: 3.5rem 1.5rem 4rem !important;
    margin: 0 auto;
}

h1 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    color: #E8EDF4 !important;
    text-align: center;
    margin-bottom: 0.2rem !important;
}
.subtitle {
    font-size: 0.8rem;
    color: #4A5568;
    text-align: center;
    letter-spacing: 0.06em;
    margin-bottom: 3rem;
}

.param-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin-bottom: 0.25rem;
    margin-top: 1.4rem;
    display: block;
    text-transform: uppercase;
}

[data-testid="stButton"] > button {
    width: 100% !important;
    margin-top: 2rem !important;
    background: #1D4ED8 !important;
    color: #EFF6FF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    cursor: pointer !important;
}

audio {
    width: 100%;
    margin-top: 1.5rem;
    filter: invert(1) hue-rotate(180deg) brightness(0.85);
}
</style>
""", unsafe_allow_html=True)

# ─── Audio engine ──────────────────────────────────────────────────────────────

def generate_brown_noise(n_samples):
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    brown -= brown.mean()
    return brown / (np.abs(brown).max() + 1e-9)

def generate_pink_noise(n_samples):
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1e-6
    power = 1.0 / np.sqrt(f)
    phase = np.random.uniform(0, 2 * np.pi, len(f))
    spectrum = power * np.exp(1j * phase)
    pink = np.fft.irfft(spectrum, n=n_samples)
    pink -= pink.mean()
    return pink / (np.abs(pink).max() + 1e-9)

def bandpass(signal_data, lo, hi, sr):
    sos = butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band", output="sos")
    return sosfilt(sos, signal_data)

def generate_focus_audio(mode, duration_min, noise_blend, intensity, sr=44100):
    n = duration_min * 60 * sr
    rng = np.random.default_rng()

    if mode == "TDAH":
        base_freq = rng.uniform(18.0, 25.0)
        freq_drift = rng.uniform(0.3, 1.2)
        lo_band, hi_band = 200.0, 8000.0
    else:
        base_freq = rng.uniform(13.0, 16.0)
        freq_drift = rng.uniform(0.1, 0.6)
        lo_band, hi_band = 80.0, 6000.0

    # Base noise
    brown = generate_brown_noise(n)
    pink = generate_pink_noise(n)
    raw_noise = (1.0 - noise_blend) * brown + noise_blend * pink
    shaped = bandpass(raw_noise, lo_band, hi_band, sr)
    shaped /= (np.abs(shaped).max() + 1e-9)

    base_layer = shaped * 0.80

    # Neural Mod Layer
    t = np.linspace(0, duration_min * 60, n, endpoint=False)
    freq_envelope = base_freq + freq_drift * np.sin(2 * np.pi * 0.05 * t)
    phase = 2 * np.pi * np.cumsum(freq_envelope) / sr
    mod_depth = rng.uniform(0.35, 0.55)
    carrier = 0.5 * (1.0 + mod_depth * np.sin(phase))

    mod_noise_raw = (1.0 - noise_blend) * generate_brown_noise(n) + noise_blend * generate_pink_noise(n)
    mod_noise = bandpass(mod_noise_raw, lo_band, hi_band, sr)
    mod_noise /= (np.abs(mod_noise).max() + 1e-9)
    mod_layer = mod_noise * carrier * 0.20

    mixed = base_layer + mod_layer
    gain = 0.25 + 0.60 * (intensity ** 0.7)
    mixed *= gain
    
    # Fade and Limit
    fade_samples = min(2 * sr, n // 10)
    fade_in = np.linspace(0, 1, fade_samples) ** 2
    mixed[:fade_samples] *= fade_in
    mixed[-fade_samples:] *= fade_in[::-1]
    mixed = np.tanh(mixed * 1.5) / 1.5

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
st.markdown('<p class="subtitle">Neural entrainment · Beta wave · Anti-fatigue</p>', unsafe_allow_//html=True)

# CORRECCIÓN DEL ERROR: Se añade un label real y se oculta con label_visibility
mode = st.radio(
    label="Selecciona el modo de enfoque", 
    options=["TDAH", "Normal"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "TDAH":
    st.markdown('<div style="font-size:0.65rem; color:#4A5568; border-left: 2px solid #3B82F6; padding-left:12px; margin: 0.8rem 0 1.8rem;">Beta alta 18–25 Hz · Activación dopaminérgica</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="font-size:0.65rem; color:#4A5568; border-left: 2px solid #1E3A5F; padding-left:12px; margin: 0.8rem 0 1.8rem;">Beta baja 13–16 Hz · Flujo cognitivo estable</div>', unsafe_allow_html=True)

st.markdown('<span class="param-label">DURACIÓN (MINUTOS)</span>', unsafe_allow_html=True)
duration = st.slider("dur", 5, 90, 25, step=5, label_visibility="collapsed")

st.markdown('<span class="param-label">TEXTURA DE RUIDO</span>', unsafe_allow_//html=True)
noise_blend = st.slider("noise", 0.0, 1.0, 0.35, step=0.05, label_visibility="collapsed")

st.markdown('<span class="param-label">INTENSIDAD DE ESTÍMULO</span>', unsafe_allow_html=True)
intensity = st.slider("intens", 0.1, 1.0, 0.65, step=0.05, label_visibility="collapsed")

if st.button("GENERAR AUDIO"):
    with st.spinner("Sintetizando capas neurales..."):
        audio_bytes = generate_focus_audio(mode, duration, noise_blend, intensity)
        st.audio(audio_bytes, format="audio/wav")

st.markdown('<div style="text-align: center; color: #374151; font-size: 0.6rem; margin-top: 2rem;">Cada generación produce una sesión única · Sin habituación cerebral</div>', unsafe_allow_html=True)
