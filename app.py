import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

# 1. Configuración de la página
st.set_page_config(page_title="NeuralFocus Pro", page_icon="🧠", layout="centered")

# 2. CSS Profesional
st.markdown("""
    <style>
    .main { background-color: #020617 !important; color: white !important; }
    h1 { color: #60a5fa !important; text-align: center !important; font-family: 'Inter', sans-serif; }
    .stButton>button { 
        width: 100%; border-radius: 20px; font-weight: bold; height: 3em; 
        background-color: #1e293b !important; color: white !important;
        border: 1px solid #334155 !important;
    }
    .stButton>button:hover { border-color: #60a5fa !important; background-color: #334155 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NeuralFocus Pro")
st.markdown("<p style='text-align:center; color:#94a3b8;'>Sincronización Neural Orgánica (Estilo Brain.fm)</p>", unsafe_allow_html=True)

# 3. Motor de Audio (Modulación de Amplitud Neural)
def generar_audio_brainfm(modo):
    duracion = 60 
    tasa_muestreo = 44100
    t = np.linspace(0, duracion, int(tasa_muestreo * duracion), endpoint=False)

    # Capa de Ruido Base (Marrón)
    blanco = np.random.normal(0, 1, len(t))
    b, a = signal.butter(1, 0.005, 'low')
    base_marron = signal.filtfilt(b, a, blanco)
    base_marron /= np.max(np.abs(base_marron))

    # Lógica de Frecuencias Beta
    if modo == "TDAH":
        f_mod = random.uniform(18.0, 22.0) 
        intensidad = 0.12 
    else:
        f_mod = random.uniform(12.0, 15.0)
        intensidad = 0.08

    # Modulación de Amplitud (El efecto Brain.fm)
    modulador = 1 + intensidad * np.sin(2 * np.pi * f_mod * t)
    pulso_neural = base_marron * modulador
    pulso_neural /= np.max(np.abs(pulso_neural))

    # Mezcla Final: 80% Constante + 20% Pulso (Cero efecto helicóptero)
    audio_final = (base_marron * 0.8) + (pulso_neural * 0.2)
    audio_final /= np.max(np.abs(audio_final))
    
    audio_int16 = (audio_final * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wavfile.write(byte_io, tasa_muestreo, audio_int16)
    return byte_io.getvalue()

# 4. Interfaz de Usuario
st.write("---")
modo = st.radio(
    label="Seleccione el modo de enfoque", 
    options=["TDAH", "Normal"], 
    horizontal=True, 
    label_visibility="collapsed"
)

col1, col2 = st.columns(2)

with col1:
    if st.button("🔵 Enfoque Estable"):
        with st.spinner("Sincronizando..."):
            audio = generar_audio_brainfm("Normal")
            st.audio(audio, format="audio/wav")

with col2:
    if st.button("🔴 Impulso TDAH"):
        with st.spinner("Activando..."):
            audio = generar_audio_brainfm("TDAH")
            st.audio(audio, format="audio/wav")

st.write("---")
st.markdown("<p style='text-align:center; color:#64748b; font-size:0.8rem;'>Sincronización de fase neural basada en ondas Beta.</p>", unsafe_allow_html=True)
