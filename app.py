import io
import wave

import numpy as np
from scipy import signal


SAMPLE_RATE = 22_050


def normalize_signal(audio):
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - np.mean(audio)

    rms = np.sqrt(np.mean(audio ** 2))

    if rms < 1e-8:
        return np.zeros_like(audio)

    return audio / rms


def brown_noise(length, rng):
    white = rng.standard_normal(length).astype(np.float32)

    brown = np.cumsum(
        white,
        dtype=np.float32,
    )

    band_filter = signal.butter(
        3,
        [30, 350],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    brown = signal.sosfiltfilt(
        band_filter,
        brown,
    )

    return normalize_signal(brown)


def pink_band_noise(length, rng):
    white = rng.standard_normal(length).astype(np.float32)

    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(
        length,
        d=1 / SAMPLE_RATE,
    )

    pink_weight = np.zeros_like(frequencies)

    valid = frequencies > 0
    pink_weight[valid] = 1 / np.sqrt(
        frequencies[valid]
    )

    band_mask = (
        (frequencies >= 100)
        & (frequencies <= 5_500)
    )

    spectrum *= pink_weight
    spectrum *= band_mask

    pink = np.fft.irfft(
        spectrum,
        n=length,
    )

    return normalize_signal(pink)


def texture_noise(length, rng):
    white = rng.standard_normal(length).astype(np.float32)

    texture_filter = signal.butter(
        3,
        [250, 3_500],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos",
    )

    texture = signal.sosfiltfilt(
        texture_filter,
        white,
    )

    return normalize_signal(texture)


def focus_noise_bed(length, rng):
    """
    Capa constante:

    - Graves controlados tipo brown noise.
    - Ruido rosa filtrado.
    - Textura media para evitar sonido de viento.
    """

    brown = brown_noise(
        length,
        rng,
    )

    pink = pink_band_noise(
        length,
        rng,
    )

    texture = texture_noise(
        length,
        rng,
    )

    return normalize_signal(
        0.42 * brown
        + 0.40 * pink
        + 0.18 * texture
    )


def drifting_phase(
    length,
    low_frequency,
    high_frequency,
    rng,
):
    """
    Genera una modulación beta con pequeñas variaciones
    de frecuencia para evitar un pulso robótico.
    """

    time = np.arange(length) / SAMPLE_RATE

    points = max(
        4,
        int(time[-1] / 3) + 2,
    )

    point_times = np.linspace(
        0,
        time[-1],
        points,
    )

    random_frequencies = rng.uniform(
        low_frequency,
        high_frequency,
        points,
    )

    frequencies = np.interp(
        time,
        point_times,
        random_frequencies,
    )

    return (
        2
        * np.pi
        * np.cumsum(frequencies)
        / SAMPLE_RATE
    )


def synthesize_focus_audio(
    duration,
    mode="adhd",
    seed=None,
):
    """
    Genera audio focus procedural.

    No se utilizan canciones, samples ni melodías.
    La modulación controla ruido, no agrega tonos.
    """

    if seed is None:
        seed = np.random.randint(
            0,
            999_999,
        )

    rng = np.random.default_rng(seed)

    if mode == "adhd":
        low_frequency = 18.0
        high_frequency = 25.0
    else:
        low_frequency = 13.0
        high_frequency = 16.0

    length = int(
        duration * SAMPLE_RATE
    )

    # 80 % de capa estable.
    base_layer = focus_noise_bed(
        length,
        rng,
    )

    # 20 % de capa rítmica.
    pulse_carrier = texture_noise(
        length,
        rng,
    )

    phase_a = drifting_phase(
        length,
        low_frequency,
        high_frequency,
        rng,
    )

    phase_b = drifting_phase(
        length,
        low_frequency * 0.98,
        high_frequency * 0.98,
        rng,
    )

    # Dos pulsos ligeramente desfasados.
    # Esto crea sensación rítmica sin sonar a helicóptero.
    pulse_a = (
        0.5
        + 0.5 * np.sin(phase_a)
    )

    pulse_b = (
        0.5
        + 0.5 * np.sin(phase_b)
    )

    # Pulsos más suaves y definidos.
    pulse_a = pulse_a ** 2.2
    pulse_b = pulse_b ** 1.8

    pulse = (
        0.68 * pulse_a
        + 0.32 * pulse_b
    )

    # Modulación perceptible, pero contenida.
    envelope = 0.22 + 0.78 * pulse

    rhythmic_layer = (
        pulse_carrier * envelope
    )

    # Arquitectura anti-helicóptero:
    # 80 % constante + 20 % modulado.
    audio = (
        0.80 * base_layer
        + 0.20 * rhythmic_layer
    )

    # Fade in/out.
    fade_length = min(
        int(0.8 * SAMPLE_RATE),
        length // 2,
    )

    fade = np.ones(
        length,
        dtype=np.float32,
    )

    fade_in = np.linspace(
        0,
        1,
        fade_length,
    )

    fade[:fade_length] = fade_in
    fade[-fade_length:] = fade_in[::-1]

    audio *= fade

    # Saturación suave para suavizar picos.
    audio = np.tanh(
        audio * 0.65
    )

    # Nivel final moderado.
    # Evita que el audio suene saturado o agresivo.
    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:
        audio = audio / peak
        audio *= 0.28

    # Imagen estéreo muy ligera.
    stereo_side = texture_noise(
        length,
        rng,
    )

    left = audio + stereo_side * 0.018
    right = audio - stereo_side * 0.018

    stereo = np.column_stack(
        (left, right)
    )

    stereo = np.clip(
        stereo,
        -1,
        1,
    )

    pcm = np.int16(
        stereo * 32_767
    )

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(
            pcm.tobytes()
        )

    output.seek(0)

    return output.getvalue()
