"""
Feature extraction for a single raw vibration waveform packet.

Given `samples` (raw acceleration in g) and `sampling_rate` (Hz), this module
computes every time-domain, frequency-domain, and FFT-peak feature in your
target schema:

mean, std, variance, rms, peak, peak_to_peak, skewness, kurtosis,
crest_factor, shape_factor, impulse_factor, clearance_factor,
energy, entropy, dominant_frequency, max_fft, spectral_energy,
spectral_centroid, spectral_spread, spectral_entropy, spectral_flatness,
rolloff_85, rolloff_95, band_0_500, band_500_1000, band_1000_2000,
band_2000_4000, band_4000_end, num_fft_peaks, mean_peak_height,
std_peak_height, mean_peak_prominence, std_peak_prominence,
peak_spacing_mean, peak_spacing_std, all_peak_frequencies,
all_peak_amplitudes, all_peak_prominences
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.signal import find_peaks

import config


def _shannon_entropy(x: np.ndarray, bins: int = 100) -> float:
    """Shannon entropy of the amplitude distribution of a signal."""
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _spectral_entropy(mag: np.ndarray) -> float:
    """Shannon entropy of the normalized FFT magnitude spectrum (power distribution)."""
    power = mag ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    p = power / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _band_energy(freqs: np.ndarray, power: np.ndarray, lo: float, hi):
    if hi is None:
        mask = freqs >= lo
    else:
        mask = (freqs >= lo) & (freqs < hi)
    return float(power[mask].sum())


def extract_time_domain_features(samples: np.ndarray) -> dict:
    x = samples.astype(np.float64)
    n = x.size
    if n == 0:
        return {k: 0.0 for k in [
            "mean", "std", "variance", "rms", "peak", "peak_to_peak",
            "skewness", "kurtosis", "crest_factor", "shape_factor",
            "impulse_factor", "clearance_factor", "energy", "entropy",
        ]}

    mean = float(np.mean(x))
    std = float(np.std(x))
    variance = float(np.var(x))
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.max(x) - np.min(x))
    skewness = float(sp_stats.skew(x)) if n > 2 else 0.0
    kurtosis = float(sp_stats.kurtosis(x)) if n > 2 else 0.0

    mean_abs = float(np.mean(np.abs(x)))
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(x)))) ** 2

    crest_factor = peak / rms if rms > 1e-12 else 0.0
    shape_factor = rms / mean_abs if mean_abs > 1e-12 else 0.0
    impulse_factor = peak / mean_abs if mean_abs > 1e-12 else 0.0
    clearance_factor = peak / mean_sqrt_abs if mean_sqrt_abs > 1e-12 else 0.0

    energy = float(np.sum(x ** 2))
    entropy = _shannon_entropy(x)

    return {
        "mean": mean, "std": std, "variance": variance, "rms": rms,
        "peak": peak, "peak_to_peak": peak_to_peak,
        "skewness": skewness, "kurtosis": kurtosis,
        "crest_factor": crest_factor, "shape_factor": shape_factor,
        "impulse_factor": impulse_factor, "clearance_factor": clearance_factor,
        "energy": energy, "entropy": entropy,
    }


def extract_frequency_domain_features(samples: np.ndarray, sampling_rate: float) -> dict:
    x = samples.astype(np.float64)
    n = x.size
    empty = {k: 0.0 for k in [
        "dominant_frequency", "max_fft", "spectral_energy", "spectral_centroid",
        "spectral_spread", "spectral_entropy", "spectral_flatness",
        "rolloff_85", "rolloff_95",
    ]}
    for name, _, _ in config.FREQ_BANDS:
        empty[name] = 0.0
    peak_empty = {
        "num_fft_peaks": 0, "mean_peak_height": 0.0, "std_peak_height": 0.0,
        "mean_peak_prominence": 0.0, "std_peak_prominence": 0.0,
        "peak_spacing_mean": 0.0, "peak_spacing_std": 0.0,
        "all_peak_frequencies": [], "all_peak_amplitudes": [], "all_peak_prominences": [],
    }
    if n < 4 or sampling_rate is None or sampling_rate <= 0:
        empty.update(peak_empty)
        return empty

    # Remove DC component before FFT
    x = x - np.mean(x)
    windowed = x * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    mag = np.abs(fft_vals)
    power = mag ** 2

    dom_idx = int(np.argmax(mag))
    dominant_frequency = float(freqs[dom_idx])
    max_fft = float(mag[dom_idx])

    spectral_energy = float(power.sum())
    if mag.sum() > 1e-12:
        spectral_centroid = float((freqs * mag).sum() / mag.sum())
    else:
        spectral_centroid = 0.0
    spectral_spread = float(
        np.sqrt(((freqs - spectral_centroid) ** 2 * mag).sum() / max(mag.sum(), 1e-12))
    )
    spec_entropy = _spectral_entropy(mag)
    geo_mean = np.exp(np.mean(np.log(mag + 1e-12)))
    arith_mean = np.mean(mag) + 1e-12
    spectral_flatness = float(geo_mean / arith_mean)

    cumulative_power = np.cumsum(power)
    total_power = cumulative_power[-1] if cumulative_power.size else 0.0
    if total_power > 1e-12:
        rolloff_85 = float(freqs[np.searchsorted(cumulative_power, 0.85 * total_power)])
        rolloff_95 = float(freqs[np.searchsorted(cumulative_power, 0.95 * total_power)])
    else:
        rolloff_85 = 0.0
        rolloff_95 = 0.0

    band_feats = {}
    for name, lo, hi in config.FREQ_BANDS:
        band_feats[name] = _band_energy(freqs, power, lo, hi)

    feats = {
        "dominant_frequency": dominant_frequency,
        "max_fft": max_fft,
        "spectral_energy": spectral_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_entropy": spec_entropy,
        "spectral_flatness": spectral_flatness,
        "rolloff_85": rolloff_85,
        "rolloff_95": rolloff_95,
    }
    feats.update(band_feats)

    # --- FFT peak picking ---
    if mag.max() > 1e-12:
        min_prom = config.get_param("FFT_PEAK_MIN_PROMINENCE_FRAC") * mag.max()
        peak_idx, props = find_peaks(mag, prominence=min_prom)
    else:
        peak_idx, props = np.array([], dtype=int), {"prominences": np.array([])}

    if peak_idx.size > 0:
        peak_freqs = freqs[peak_idx]
        peak_amps = mag[peak_idx]
        peak_proms = props["prominences"]
        spacing = np.diff(np.sort(peak_freqs)) if peak_idx.size > 1 else np.array([])

        peak_feats = {
            "num_fft_peaks": int(peak_idx.size),
            "mean_peak_height": float(np.mean(peak_amps)),
            "std_peak_height": float(np.std(peak_amps)),
            "mean_peak_prominence": float(np.mean(peak_proms)),
            "std_peak_prominence": float(np.std(peak_proms)),
            "peak_spacing_mean": float(np.mean(spacing)) if spacing.size else 0.0,
            "peak_spacing_std": float(np.std(spacing)) if spacing.size else 0.0,
            "all_peak_frequencies": [round(float(f), 4) for f in peak_freqs],
            "all_peak_amplitudes": [round(float(a), 6) for a in peak_amps],
            "all_peak_prominences": [round(float(p), 6) for p in peak_proms],
        }
    else:
        peak_feats = peak_empty

    feats.update(peak_feats)
    return feats



# The live API returns camelCase packet keys (sr, nos, analyticsType, fMax,
# epochTime) while local JSON dumps use snake_case (sampling_rate, num_samples,
# analytics_type, f_max, epoch_time). Normalize so downstream code only ever
# has to handle one shape.
_PACKET_KEY_ALIASES = {
    "sr": "sampling_rate",
    "nos": "num_samples",
    "analyticsType": "analytics_type",
    "fMax": "f_max",
    "epochTime": "epoch_time",
}


def normalize_packet(pkt: dict) -> dict:
    out = dict(pkt)
    for api_key, local_key in _PACKET_KEY_ALIASES.items():
        if api_key in out and local_key not in out:
            out[local_key] = out.pop(api_key)
    return out


def extract_envelope_features(samples: np.ndarray, sampling_rate: float) -> dict:
    """
    Envelope-spectrum analysis (Hilbert-transform demodulation).

    Bearing fault impacts (BPFI/BPFO/BSF/FTF) often get buried in the raw
    spectrum under structural/gearbox noise. Rectifying the signal via its
    analytic-signal envelope, then taking the FFT of that envelope, surfaces
    the impact repetition frequency far more cleanly. This is the standard
    technique used in real bearing diagnostics (vs. just looking at the raw
    FFT peaks).
    """
    empty = {
        "envelope_dominant_frequency": 0.0,
        "envelope_max_amplitude": 0.0,
        "envelope_spectral_entropy": 0.0,
        "envelope_energy": 0.0,
        "envelope_all_peak_frequencies": [],
        "envelope_all_peak_amplitudes": [],
    }
    x = samples.astype(np.float64) if isinstance(samples, np.ndarray) else np.asarray(samples, dtype=np.float64)
    n = x.size
    if n < 8 or sampling_rate is None or sampling_rate <= 0:
        return empty

    from scipy.signal import hilbert

    x = x - np.mean(x)
    analytic = hilbert(x)
    envelope = np.abs(analytic)
    envelope = envelope - np.mean(envelope)

    windowed = envelope * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    mag = np.abs(fft_vals)

    if mag.max() < 1e-12:
        return empty

    dom_idx = int(np.argmax(mag))
    feats = {
        "envelope_dominant_frequency": float(freqs[dom_idx]),
        "envelope_max_amplitude": float(mag[dom_idx]),
        "envelope_spectral_entropy": _spectral_entropy(mag),
        "envelope_energy": float(np.sum(mag ** 2)),
    }

    min_prom = config.get_param("FFT_PEAK_MIN_PROMINENCE_FRAC") * mag.max()
    peak_idx, _ = find_peaks(mag, prominence=min_prom)
    if peak_idx.size > 0:
        feats["envelope_all_peak_frequencies"] = [round(float(f), 4) for f in freqs[peak_idx]]
        feats["envelope_all_peak_amplitudes"] = [round(float(a), 6) for a in mag[peak_idx]]
    else:
        feats["envelope_all_peak_frequencies"] = []
        feats["envelope_all_peak_amplitudes"] = []

    return feats


G_TO_MS2 = 9.80665  # standard gravity, g -> m/s^2


def integrate_accel_to_velocity(samples: np.ndarray, sampling_rate: float,
                                 highpass_hz: float = 2.0) -> np.ndarray:
    """
    Integrate a raw acceleration waveform (in g) to velocity (in mm/s), via
    frequency-domain integration: V(f) = A(f) / (j*omega).

    Frequency-domain integration (vs. time-domain cumulative sum) avoids the
    drift/blow-up that naive time integration produces. A high-pass cutoff
    (default 2 Hz) zeroes out near-DC content, which would otherwise dominate
    the integrated signal (dividing by omega->0 blows up low frequencies).
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n < 4 or sampling_rate is None or sampling_rate <= 0:
        return x

    x = (x - np.mean(x)) * G_TO_MS2  # g -> m/s^2
    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    omega = 2 * np.pi * freqs

    velocity_fft = np.zeros_like(fft_vals)
    mask = freqs > highpass_hz
    velocity_fft[mask] = fft_vals[mask] / (1j * omega[mask])

    velocity_ms = np.fft.irfft(velocity_fft, n=n)  # m/s
    return velocity_ms * 1000.0  # mm/s (standard vibration-velocity unit)


def extract_all_features(samples, sampling_rate: float) -> dict:
    """Full feature dict for one packet's samples array."""
    x = np.asarray([s for s in samples if isinstance(s, (int, float))], dtype=np.float64)
    feats = {}
    feats.update(extract_time_domain_features(x))
    feats.update(extract_frequency_domain_features(x, sampling_rate))
    feats.update(extract_envelope_features(x, sampling_rate))
    return feats
