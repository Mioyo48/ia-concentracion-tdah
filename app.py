import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

# ─── CONFIGURACIÓN DE ALTA GAMA ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuralFocus Core",
    page_icon="🧠",
    layout="centered",
)

# ─── DISEÑO INTERFAZ PREMIUM (GLASSMORPHISM) ──────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1e293b, #020617) !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    .main-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 40px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        margin: 20px auto;
        max-width: 600px;
    }

    h1 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center !important;
        font-size: 2.5rem !important;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-weight: 300;
        margin-bottom: 2rem;
        font-size: 1rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        padding: 15px !important;
    }

    div.stButton > button:hover {
        background: rgba(56, 189, 248, 0.2) !important;
        border-color: #38bdf8 !important;
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(56, 189, 248, 0.5);
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    
    .science-box {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 15px;
        padding: 20px;
        font-size: 0.85rem;
        color: #64748b;
        border: 1px solid rgba(255, 255, 255, 0.05);
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── MOTOR DE AUDIO NEURONAL ──────────────────────────────────────────────────

def generate_neural_audio(mode):
    duration = 60 
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Generación de Ruido Orgánico
    white = np.random.normal(0, 1, len(t))
    b_m, a_m = signal.butter(1, 0.005, 'low')
    brown = signal.filtfilt(b_m, a_m, white)
    b_p, a_p = signal.butter(1, 0.05, 'low')
    pink = signal.filtfilt(b_p, a_p, white)
    
    base_noise = (brown * 0.7) + (pink * 0.3)
    base_noise /= np.max(np.abs(base_noise))

    # 2. Parámetros de Modulación
    if mode == "TDAH":
        f_mod = random.uniform(18.0, 24.0)
        depth = 0.15 
        drift = random.uniform(0.1, 0.4)
    else:
        f_mod = random.uniform(12.0, 16.0)
        depth = 0.08
        drift = random.uniform(0.05, 0.2)

    # 3. Algoritmo de Modulación de Fase Orgánica
    freq_env = f_mod + drift * np.sin(2 * np.pi * 0.02 * t)
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    modulator = 1 + depth * np.sin(phase)
    
    # Capas Asíncronas para evitar efecto helicóptero
    layer_static = base_noise * 0.7
    layer_neural = (base_noise * modulator) * 0.3
    
    audio_final = layer_static + layer_neural
    audio_final /= np.max(np.abs(audio_final))
    
    audio_int16 = (audio_final * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, audio_int16)
    return buf.getvalue()

# ─── INTERFAZ DE USUARIO (UI) ──────────────────────────────────────────────────

st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown("<h1>NeuralFocus Core</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sincronización de Fase Neural para el Alto Rendimiento Cognitivo</p>", unsafe_allow_html=True)

mode = st.radio(
    label="Seleccione el perfil de sincronización", 
    options=["TDAH", "Normal"], 
    horizontal=True, 
    label_visibility="collapsed"
)

st.write("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔵 Focus Estable"):
        with st.spinner("Sincronizando..."):
            audio = generate_neural_audio("Normal")
            st.audio(audio, format="audio/wav")

with col2:
    if st.button("🔴 Impulso TDAH"):
        with st.spinner("Activando..."):
            audio = generate_neural_audio("TDAH")
            st.audio(audio, format="audio/wav")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
    <div class="science-box">
        <strong>Anatomía del Sonido:</strong><br>
        Este sistema implementa <i>Amplitude Modulation (AM)</i> sobre una base de ruido rosa y marrón. <br>
        Al modular el ruido a frecuencias Beta, inducimos un estado de <b>entrainment</b> en la corteza prefrontal.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
