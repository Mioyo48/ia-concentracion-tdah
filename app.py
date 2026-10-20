import streamlit as st
import numpy as np
from scipy import signal
import io
from scipy.io import wavfile
import random

st.set_page_config(page_title="NeuralFocus", page_icon="🧠")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #020617 !important;
        color: white !important;
    }
    .main-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 30px;
        padding: 40px;
        margin: 20px auto;
        max-width: 600px;
    }
    h1 { text-align: center !important; color: #60a5fa !important; }
    div.stButton > button {
        width: 100%; border-radius: 15px !important;
        background: #1e293b !important; color: white !important;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"], footer { display: none !important; }
    </style>
    """, unsafe_allow_html=True)

def generate_audio(mode):
    duration, sr = 60, 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    white = np.random.normal(0, 1, len(t))
    b, a = signal.butter(1, 0.005, 'low')
    brown = signal.filtfilt(b, a, white)
    base = brown / np.max(np.abs(brown))
    if mode == "TDAH":
        f_mod, depth = random.uniform(18, 24), 0.12
    else:
        f_mod, depth = random.uniform(12, 16), 0.08
    mod = 1 + depth * np.sin(2 * np.pi * f_mod * t)
    final = (base * 0.8) + (base * mod * 0.2)
    final /= np.max(np.abs(final))
    pcm = np.int16(np.clip(final * 0.89, -1.0, 1.0) * 32767)
    buf = io.BytesIO()
    wavfile.write(buf, sr, pcm)
    return buf.getvalue()

st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown("<h1>NeuralFocus Core</h1>", unsafe_allow_html=True)
mode = st.radio("Modo", ["TDAH", "Normal"], horizontal=True, label_visibility="collapsed")
col1, col2 = st.columns(2)
with col1:
    if st.button("Enfoque"):
        st.audio(generate_audio("Normal"), format="audio/wav")
with col2:
    if st.button("Impulso"):
        st.audio(generate_audio("TDAH"), format="audio/wav")
st.markdown('</div>', unsafe_allow_html=True)
