import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

# Configuración de la página
st.set_page_config(page_title="NeuralFocus Pro", page_icon="🧠")

# Estilo CSS Profesional y Oscuro
st.markdown("""
    <style>
    .main { background-color: #020617; color: white; }
    h1 { color: #60a5fa; text-align: center; font-size: 3rem !important; margin-bottom: 0px; }
    .subtitle { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 30px; }
    .stButton>button { 
        width: 100%; border-radius: 20px; font-weight: bold; height: 4em; font-size: 1.1rem;
        transition: 0.3s; border: 1px solid #38bdf8;
    }
    .btn-normal { background-color: #1e293b !important; color: #60a5fa !important; }
    .btn-tdah { background-color: #1e293b !important; color: #f43f5e !important; border: 2px solid #f43f5e !important; }
    button:hover { transform: scale(1.02); background-color: #334155 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 NeuralFocus Pro")
st.markdown("<p class='subtitle'>Sincronización de Fase Neural para Enfoque Profundo</p>", unsafe_allow_html=True)

def generar_audio_brainfm(modo):
    duracion = 60 
    tasa_muestreo = 44100
    t = np.linspace(0, duracion, int(tasa_muestreo * duracion), endpoint=False)

    # --- 1. CREACIÓN DEL "SOPLIDO" (RUIDO COMPLEJO) ---
    # Mezclamos Ruido Marrón (profundo) y Rosa (energía) para crear una base orgánica
    blanco = np.random.normal(0, 1, len(t))
    
    # Filtro para Ruido Marrón
    b_m, a_m = signal.butter(1, 0.005, 'low')
    marron = signal.filtfilt(b_m, a_m, blanco)
    
    # Filtro para Ruido Rosa
    b_p, a_p = signal.butter(1, 0.05, 'low')
    rosa = signal.filtfilt(b_p, a_p, blanco)
    
    # Base: 70% Marrón + 30% Rosa
    base_noise = (marron * 0.7) + (rosa * 0.3)
    base_noise /= np.max(np.abs(base_noise))

    # --- 2. EL SECRETO DE BRAIN.FM: MODULACIÓN DE AMPLITUD (AM) ---
    # En lugar de sumar un tono, hacemos que el ruido PULSE a la frecuencia Beta
    if modo == "TDAH":
        # Frecuencia Beta Alta (18-22Hz) para cerebro TDAH
        f_mod = random.uniform(18.0, 22.0) 
        intensidad = 0.4  # Pulso más fuerte para mantener el cerebro despierto
    else:
        # Frecuencia Beta Baja (12-15Hz) para concentración estable
        f_mod = random.uniform(12.0, 15.0)
        intensidad = 0.2  # Pulso más sutil

    # Creamos la onda moduladora (el "latido" de la concentración)
    # Usamos una onda senoidal que varía entre (1 - intensidad) y (1 + intensidad)
    modulador = 1 + intensidad * np.sin(2 * np.pi * f_mod * t)
    
    # APLICAMOS LA MODULACIÓN: El ruido ahora pulsa rítmicamente
    audio_modulado = base_noise * modulador
    
    # --- 3. CAPA DE ESTABILIZACIÓN ---
    # Añadimos una capa muy suave de ruido blanco filtrado para evitar el "vacío"
    estabilizador = np.random.normal(0, 0.02, len(t))
    audio_final = audio_modulado + estabilizador
    
    # Normalización final para evitar distorsión
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
    if st.button("🔵 Enfoque Estándar", key="btn_n"):
        with st.spinner("Sincronizando fase..."):
            audio = generar_audio_brainfm("Normal")
            st.audio(audio, format="audio/wav")
            st.success("Sincronizado a Beta Baja")

with col2:
    if st.button("🔴 Impulso TDAH", key="btn_t"):
        with st.spinner("Activando drive neural..."):
            audio = generar_audio_brainfm("TDAH")
            st.audio(audio, format="audio/wav")
            st.success("Sincronizado a Beta Alta")

st.write("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>
    <b>Tecnología de Modulación de Fase:</b><br>
    A diferencia de los beats binaurales, este sistema modula la amplitud del ruido <br>
    en tiempo real para inducir un estado de flujo (flow state) en la corteza prefrontal.
</div>
""", unsafe_allow_html=True)
