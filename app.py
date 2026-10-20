import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

# Configuración de la página
st.set_page_config(page_title="NeuralFocus IA Pro", page_icon="⚡")

# Estilo CSS futurista y energético
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    h1 { color: #38bdf8; text-align: center; text-shadow: 0 0 15px #38bdf8; font-size: 3rem !important; }
    p { text-align: center; color: #94a3b8; font-size: 1.2rem; }
    .stButton>button { 
        width: 100%; 
        border-radius: 15px; 
        font-weight: bold; 
        height: 4em; 
        font-size: 1.2rem;
        transition: 0.3s;
    }
    .btn-normal { background-color: #38bdf8 !important; color: #0f172a !important; }
    .btn-tdah { background-color: #f43f5e !important; color: white !important; box-shadow: 0 0 20px rgba(244, 63, 94, 0.6); }
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ NeuralFocus IA Pro")
st.write("Generador Dinámico de Estímulos Neuronales")
st.write("Cada vez que presiones el botón, la IA calculará una combinación de frecuencias única para tu cerebro.")

def generar_audio_dinamico(modo):
    duracion = 60 
    tasa_muestreo = 44100
    t = np.linspace(0, duracion, int(tasa_muestreo * duracion), endpoint=False)

    # 1. MEZCLA DE RUIDO (Marrón + Rosa para más energía)
    # El ruido rosa suena como lluvia fuerte, es más estimulante que el marrón puro
    blanco = np.random.normal(0, 1, len(t))
    b_marron, a_marron = signal.butter(1, 0.005, 'low')
    marron = signal.filtfilt(b_marron, a_marron, blanco)
    
    b_rosa, a_rosa = signal.butter(1, 0.05, 'low')
    rosa = signal.filtfilt(b_rosa, a_rosa, blanco)
    
    # Mezclamos los ruidos para una textura más "viva"
    fondo = (marron * 0.6) + (rosa * 0.4)
    fondo /= np.max(np.abs(fondo))

    # 2. LÓGICA ALEATORIA Y ENERGÉTICA
    if modo == "TDAH":
        # Rango de Beta Alta: 18Hz a 25Hz (Muy estimulante)
        f_mod = random.uniform(18.0, 25.0) 
        f_port = random.uniform(160, 200)
        intensidad_base = random.uniform(0.4, 0.6)
    else:
        # Rango de Beta Baja: 12Hz a 16Hz (Concentración estable)
        f_mod = random.uniform(12.0, 16.0)
        f_port = random.uniform(130, 160)
        intensidad_base = random.uniform(0.1, 0.3)

    # 3. PULSO DINÁMICO (Para que no sea la misma canción)
    # Creamos una onda que hace que la intensidad suba y baje cada 10 segundos
    variacion = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t) 
    
    onda_base = np.sin(2 * np.pi * f_port * t)
    modulador = 1 + (intensidad_base * variacion) * np.sin(2 * np.pi * f_mod * t)
    pulso_neural = onda_base * modulador
    pulso_neural /= np.max(np.abs(pulso_neural))

    # 4. MEZCLA FINAL ENERGÉTICA
    audio_final = (fondo * 0.6) + (pulso_neural * 0.4)
    audio_final /= np.max(np.abs(audio_final))
    
    audio_int16 = (audio_final * 32767).astype(np.int16)
    
    byte_io = io.BytesIO()
    wavfile.write(byte_io, tasa_muestreo, audio_int16)
    return byte_io.getvalue()

# Interfaz de Usuario
st.write("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔵 Enfoque Estable", key="btn_normal"):
        with st.spinner("Calculando frecuencias..."):
            audio = generar_audio_dinamico("Normal")
            st.audio(audio, format="audio/wav")
            st.success("✅ Combinación Única Generada")

with col2:
    if st.button("🔴 Máxima Energía TDAH", key="btn_tdah"):
        with st.spinner("Sincronizando neuronas..."):
            audio = generar_audio_dinamico("TDAH")
            st.audio(audio, format="audio/wav")
            st.success("⚡ Estímulo TDAH Activado")

st.write("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>
    <b>¿Por qué es diferente cada vez?</b><br>
    La IA selecciona frecuencias aleatorias dentro del rango Beta y aplica una modulación dinámica <br>
    para evitar que el cerebro se habitúe al sonido, manteniendo la alerta máxima.
</div>
""", unsafe_allow_html=True)
