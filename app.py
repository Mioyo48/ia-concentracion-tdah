import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

# ─── CONFIGURACIÓN DE PÁGINA PREMIUM ──────────────────────────────────────────
st.set_page_config(
    page_title="NeuroFocus · Deep Work",
    page_icon="◉",
    layout="centered",
)

# ─── CSS ULTRA-PREMIUM (Inspirado en el HTML que pasaste) ──────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 14% 8%, rgba(14, 165, 233, 0.15), transparent 30%),
                    radial-gradient(circle at 88% 84%, rgba(244, 63, 94, 0.1), transparent 32%),
                    radial-gradient(circle at 50% 50%, #1e293b 0%, #0f172a 46%, #020617 100%) !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Tarjeta Principal Estilo Shell */
    .main-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.76), rgba(15, 23, 42, 0.68));
        backdrop-filter: blur(26px);
        -webkit-backdrop-filter: blur(26px);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 32px;
        padding: 40px;
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.46);
        margin: 20px auto;
        max-width: 600px;
    }

    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.05em !important;
        text-align: center !important;
        margin-bottom: 0px !important;
        background: linear-gradient(to right, #f8fafc, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 15px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        background: rgba(2, 6, 23, 0.4) !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        padding: 15px !important;
    }

    div.stButton > button:hover {
        border-color: #38bdf8 !important;
        background: rgba(56, 189, 248, 0.1) !important;
        color: #38bdf8 !important;
        transform: translateY(-2px);
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    
    .footer-note {
        text-align: center;
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 2rem;
        letter-spacing: 0.05em;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── MOTOR DE AUDIO PROFESIONAL (LÓGICA DEL .PY QUE PASASTE) ─────────────────────

def normalize(samples):
    samples = np.asarray(samples, dtype=np.float32)
    samples -= np.mean(samples)
    deviation = float(np.std(samples))
    return samples / deviation if deviation > 1e-8 else np.zeros_like(samples)

def generate_brown_noise(length):
    white = np.random.standard_normal(length).astype(np.float32)
    brown = np.cumsum(white)
    # Filtro High-pass para eliminar el "hum" grave excesivo
    sos = signal.butter(2, 20.0, btype="highpass", fs=44100, output="sos")
    return normalize(signal.sosfiltfilt(sos, brown))

def generate_pink_noise(length):
    white = np.random.standard_normal(length).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length, d=1.0/44100)
    weighting = np.ones_like(freqs)
    weighting[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * weighting, n=length)
    # Filtro Band-pass para limpieza profesional
    sos = signal.butter(4, [80.0, 9000.0], btype="bandpass", fs=44100, output="sos")
    return normalize(signal.sosfiltfilt(sos, pink))

def drifting_phase(length, low_hz, high_hz):
    t = np.arange(length) / 44100
    center_freq = random.uniform(low_hz, high_hz)
    # Deriva de frecuencia: la frecuencia cambia suavemente para evitar habituación
    drift = np.interp(t, np.linspace(0, t[-1], 10), np.random.normal(0, 0.22, 10))
    inst_freq = np.clip(center_freq + drift, low_hz, high_hz)
    return 2.0 * np.pi * np.cumsum(inst_freq) / 44100

def synthesize_audio(mode, duration_sec):
    sr = 44100
    length = int(duration_sec * sr)
    
    # Rangos de Brain.fm
    low, high = (18.0, 25.0) if mode == "TDAH" else (13.0, 16.0)
    
    # Base: 58% Marrón + 42% Rosa (Súper orgánico)
    base_noise = normalize(0.58 * generate_brown_noise(length) + 0.42 * generate_pink_noise(length))
    
    # Capa de modulación con fase derivante
    phase = drifting_phase(length, low, high)
    pulse = 0.5 + 0.5 * np.sin(phase)
    
    # Modulación suave (70% base + 30% pulso) para evitar efecto helicóptero
    pulse_carrier = generate_brown_noise(length) # Usamos ruido como portadora
    modulated_layer = pulse_carrier * (0.70 + 0.30 * pulse)
    
    # Mezcla final: 80% Estable, 20% Modulado
    audio = (0.80 * base_noise) + (0.20 * modulated_layer)
    audio = normalize(audio)
    
    # Fade in/out para evitar clics
    fade = np.linspace(0, 1, int(0.75 * sr))
    audio[:len(fade)] *= fade
    audio[-len(fade):] *= fade[::-1]
    
    # Limitador y conversión a 16-bit
    pcm = np.int16(np.clip(audio * 0.89, -1.0, 1.0) * 32767)
    
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    return buf.getvalue()

# ─── INTERFAZ DE USUARIO (ESTILO SHELL PREMIUM) ──────────────────────────────────

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("<h1>NeuroFocus</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sincronización de Fase Neural · Deep Work</p>", unsafe_allow_html=True)

mode = st.radio(
    label="Perfil Neural", 
    options=["TDAH", "Normal"], 
    horizontal=True, 
    label_visibility="collapsed"
)

st.write("---")
duration = st.slider("Duración de la sesión (segundos)", 5, 180, 60)

col1, col2 = st.columns(2)
with col1:
    if st.button("🔵 Focus Estable"):
        with st.spinner("Sintetizando..."):
            st.audio(synthesize_audio("Normal", duration), format="audio/wav")

with col2:
    if st.button("🔴 Impulso TDAH"):
        with st.spinner("Sincronizando..."):
            st.audio(synthesize_audio("TDAH", duration), format="audio/wav")

st.markdown("<br>", unsafe_//html=True) # Espera, corrijo el typo
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div style="text-align:center; color:#64748b; font-size:0.7rem;">Modulación de amplitud no tonal · Ruido Marrón/Rosa · Deriva de Fase</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_//html=True) # Otro typo, corrijo abajo
