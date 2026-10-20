import streamlit as st
import numpy as np
from scipy import signal
import io

# Configuración de la página
st.set_page_config(page_title="NeuralFocus IA", page_icon="🎧")

# Estilo CSS para que se vea futurista
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    h1 { color: #38bdf8; text-align: center; text-shadow: 0 0 10px #38bdf8; }
    p { text-align: center; color: #94a3b8; }
    .stButton>button { 
        width: 100%; 
        border-radius: 12px; 
        font-weight: bold; 
        height: 3em;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎧 NeuralFocus IA")
st.write("Sincronización cerebral para el máximo enfoque")

# Función para generar el audio
def generar_audio(modo):
    duracion = 60 
    tasa_muestreo = 44100
    t = np.linspace(0, duracion, int(tasa_muestreo * duracion), endpoint=False)

    # 1. Ruido Marrón
    blanco = np.random.normal(0, 1, len(t))
    b, a = signal.butter(1, 0.005, 'low')
    marron = signal.filtfilt(b, a, blanco)
    marron /= np.max(np.abs(marron))

    # 2. Lógica Neuronal
    if modo == "TDAH":
        f_port, f_mod, intensidad = 180, 20, 0.5
    else:
        f_port, f_mod, intensidad = 150, 13, 0.2

    onda_base = np.sin(2 * np.pi * f_port * t)
    modulador = 1 + intensidad * np.sin(2 * np.pi * f_mod * t)
    pulso_neural = onda_base * modulador
    pulso_neural /= np.max(np.abs(pulso_neural))

    # 3. Mezcla Final
    audio_final = (marron * 0.7) + (pulso_neural * 0.3)
    audio_final /= np.max(np.abs(audio_final))
    
    # Convertir a 16-bit PCM
    audio_int16 = (audio_final * 32767).astype(np.int16)
    
    # Guardar en un buffer de memoria (para no crear archivos físicos)
    byte_io = io.BytesIO()
    from scipy.io import wavfile
    wavfile.write(byte_io, tasa_muestreo, audio_int16)
    return byte_io.getvalue()

# Interfaz de usuario
st.write("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔵 Enfoque Normal"):
        audio = generar_audio("Normal")
        st.audio(audio, format="audio/wav")
        st.success("Modo Normal activado")

with col2:
    if st.button("🔴 Estímulo TDAH"):
        audio = generar_audio("TDAH")
        st.audio(audio, format="audio/wav")
        st.success("Modo TDAH activado")

st.write("---")
st.info("Este audio utiliza modulación de frecuencias Beta para estimular la corteza prefrontal y reducir las distracciones.")
