import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

# Configuración de la página
st.set_page_config(page_title="NeuralFocus Pro", page_icon="🧠")

# Estilo CSS Minimalista y Profesional
st.markdown("""
    <style>
    .main { background-color: #020617; color: white; }
    h1 { color: #60a5fa; text-align: center; font-size: 2.5rem !important; }
    .subtitle { text-align: center; color: #94a3b8; font-size: 1rem; margin-bottom: 30px; }
    .stButton>button { 
        width: 100%; border-radius: 20px; font-weight: bold; height: 3.5em; 
        transition: 0.3s; border: 1px solid #334155; background-color: #1e293b !important; color: white !important;
    }
    button:hover { background-color: #334155 !important; border-color: #60a5fa !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NeuralFocus Pro")
st.markdown("<p class='subtitle'>Modulación Neural Orgánica para Enfoque Profundo</p>", unsafe_allow_html=True)

def generar_audio_organico(modo):
    duracion = 60 
    tasa_muestreo = 44100
    t = np.linspace(0, duracion, int(tasa_muestreo * duracion), endpoint=False)

    # --- 1. CREACIÓN DE LA BASE ESTÁTICA (La manta de sonido) ---
    # Generamos ruido blanco y lo filtramos para que sea Ruido Marrón profundo
    blanco = np.random.normal(0, 1, len(t))
    b, a = signal.butter(1, 0.005, 'low')
    base_marron = signal.filtfilt(b, a, blanco)
    base_marron /= np.max(np.abs(base_marron))

    # --- 2. EL PULSO SUTIL (La parte que concentra) ---
    if modo == "TDAH":
        f_mod = random.uniform(18.0, 22.0) # Beta alta
        intensidad = 0.12 # Intensidad BAJA para evitar el efecto helicóptero
    else:
        f_mod = random.uniform(12.0, 15.0) # Beta baja
        intensidad = 0.08 # Muy sutil

    # Creamos el modulador (Seno suave)
    modulador = 1 + intensidad * np.sin(2 * np.pi * f_mod * t)
    
    # Aplicamos la modulación SOLO a una capa de ruido, no a todo
    pulso_estímulo = base_marron * modulador
    pulso_estímulo /= np.max(np.abs(pulso_estímulo))

    # --- 3. LA MEZCLA MAESTRA (El secreto del sonido orgánico) ---
    # Mezclamos una capa que NUNCA cambia con la capa que PULSA.
    # 80% Base Constante + 20% Pulso Neural
    # Esto elimina el efecto de "helicóptero" porque el sonido nunca cae a cero.
    audio_final = (base_marron * 0.8) + (pulso_estímulo * 0.2)
    
    # Normalización final
    audio_final /= np.max(np.abs(audio_final))
    
    # Convertir a 16-bit PCM
    audio_int16 = (audio_final * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wavfile.write(byte_io, tasa_muestreo, audio_int16)
    return byte_io.getvalue()

# Interfaz de Usuario
st.write("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔵 Enfoque Estable", key="btn_n"):
        with st.spinner("Sincronizando..."):
            audio = generar_audio_organico("Normal")
            st.audio(audio, format="audio/wav")

with col2:
    if st.button("🔴 Impulso TDAH", key="btn_t"):
        with st.spinner("Activando..."):
            audio = generar_audio_organico("TDAH")
            st.audio(audio, format="audio/wav")

st.write("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.8rem;'>
    Este sistema utiliza <b>Mezcla de Capas Asíncronas</b>. <br>
    Al mantener una base constante y un pulso sutil, inducimos el estado de flujo <br>
    sin generar fatiga auditiva ni ruidos mecánicos.
</div>
""", unsafe_allow_html=True)
