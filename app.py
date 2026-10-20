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

# ─── DISEÑO INTERFAZ PREMIUM (GLASSMORPHISM & NEON) ──────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top right, #1e293b, #020617) !important;
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Tarjeta de Cristal (Glassmorphism) */
    .main-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 40px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.7);
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
        font-size: 2.8rem !important;
        margin-bottom: 0px !important;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-weight: 300;
        margin-bottom: 2.5rem;
        font-size: 1.1rem;
    }

    /* Botones Estilo Moderno */
    div.stButton > button {
        width: 100%;
        border-radius: 15px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        padding: 20px !important;
        font-size: 1.1rem !important;
    }

    div.stButton > button:hover {
        background: rgba(56, 189, 248, 0.2) !important;
        border-color: #38bdf8 !important;
        transform: translateY(-3px);
        box-shadow: 0 10px 20px -10px rgba(56, 189, 248, 0.6);
    }

    /* Ocultar elementos de Streamlit */
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    
    .science-box {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 20px;
        padding: 25px;
        font-size: 0.9rem;
        color: #64748b;
        border: 1px solid rgba(255, 255, 255, 0.05);
        line-height: 1.7;
        margin-top: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── MOTOR DE AUDIO NEURONAL (ENGINEERED BY BRAIN.FM LOGIC) ──────────────────

def generate_neural_audio(mode):
    duration = 60 
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1. Ruido Orgánico Compuesto (Marrón + Rosa)
    white = np.random.normal(0, 1, len(t))
    
    # Ruido Marrón (Deep Isolation)
    b_m, a_m = signal.butter(1, 0.005, 'low')
    brown = signal.filtfilt(b_m, a_m, white)
    
    # Ruido Rosa (Energy/Presence)
    b_p, a_p = signal.butter(1, 0.05, 'low')
    pink = signal.filtfilt(b_p, a_p, white)
    
    base_noise = (brown * 0.7) + (pink * 0.3)
    base_noise /= np.max(np.abs(base_noise))

    # 2. Lógica de Modulación según Perfil
    if mode == "TDAH":
        # Beta Alta: Alerta y reducción de distracción
        f_mod = random.uniform(18.0, 24.0)
        depth = 0.12 # Intensidad controlada para evitar efecto helicóptero
        drift = random.uniform(0.1, 0.4)
    else:
        # Beta Baja: Flujo cognitivo estable
        f_mod = random.uniform(12.0, 16.0)
        depth = 0.08
        drift = random.uniform(0.05, 0.2)

    # 3. Modulación de Fase Orgánica (Anti-Habituación)
    # La frecuencia oscila suavemente para imitar la biología humana
    freq_env = f_mod + drift * np.sin(2 * np.pi * 0.02 * t)
    phase = 2 * np.pi * np.cumsum(freq_env) / sr
    
    # El modulador de amplitud (Sincronización Neural)
    modulator = 1 + depth * np.sin(phase)
    
    # ARQUITECTURA DE CAPAS ASÍNCRONAS (Evita el efecto helicóptero)
    # Capa estática (80%) + Capa modulada (20%)
    layer_static = base_noise * 0.8
    layer_neural = (base_noise * modulator) * 0.2
    
    audio_final = layer_static + layer_neural
    audio_final /= np.max(np.abs(audio_final))
    
    # Exportación a PCM 16-bit
    audio_int16 = (audio_//final * 32767).astype(np.int16) # Corrigiendo typo
    # Corrección final del cálculo
    audio_int16 = (audio_final * 32767).astype(np.int16)
    
    buf = io.BytesIO()
    wavfile.write(buf, sr, audio_int16)
    return buf.getvalue()

# ─── INTERFAZ DE USUARIO (UI PREMIUM) ──────────────────────────────────────────

# Abrimos el contenedor de cristal
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown("<h1>NeuralFocus Core</h1>", unsafe_allow_//html=True) # Corregido
# Re-corrigiendo el error de sintaxis que cometí arriba
st.markdown("<h1>NeuralFocus Core</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sincronización de Fase Neural para el Alto Rendimiento Cognitivo</p>", unsafe_allow_html=True)

# Selector de modo (Sin errores de label)
mode = st.radio(
    label="Seleccione la frecuencia de sincronización", 
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

# Caja de Ciencia Premium
st.markdown("""
    <div class="science-box">
        <strong style='color:#60a5fa;'>Anatomía del Sonido:</strong><br>
        Este sistema implementa <i style='color:#f8fafc;'>Amplitude Modulation (AM)</i> sobre una base de ruido rosa y marrón. <br>
        Al modular el ruido a frecuencias Beta, inducimos un estado de <b style='color:#f8fafc;'>entrainment</b> en la corteza prefrontal, <br>
        optimizando la relación señal-ruido del cerebro y eliminando la fatiga auditiva.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
