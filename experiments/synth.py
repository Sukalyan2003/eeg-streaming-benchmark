"""
Synthetic EEG-like ground-truth signal generator.

Deterministic (seeded) multichannel signals with *known* spectral content, used as
ground truth for the overlap-add boundary-artifact experiment. No external data needed.

Content per channel:
  - band-limited oscillations (delta/theta/alpha/beta) with random phases
  - 1/f ("pink") background noise
  - 50 Hz mains line noise (to exercise the notch filter)
  - occasional alpha bursts (to create realistic non-stationarity near seams)

The signal is intentionally continuous and stationary-in-expectation so that any
discontinuity at window seams in a reconstruction is attributable to the windowing
strategy, not to the signal itself.
"""
from __future__ import annotations
import numpy as np

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f noise via FFT shaping."""
    white = rng.standard_normal(n)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, d=1.0)
    f[0] = f[1]
    X = X / np.sqrt(f)
    y = np.fft.irfft(X, n=n)
    return y / (np.std(y) + 1e-12)


def _band_osc(t: np.ndarray, lo: float, hi: float, rng: np.random.Generator,
              n_components: int = 6) -> np.ndarray:
    """Sum of sinusoids drawn uniformly from [lo, hi] with random phases."""
    out = np.zeros_like(t)
    for _ in range(n_components):
        fk = rng.uniform(lo, hi)
        ph = rng.uniform(0, 2 * np.pi)
        out += np.sin(2 * np.pi * fk * t + ph)
    return out / np.sqrt(n_components)


def generate(duration_s: float = 120.0, fs: int = 200, n_channels: int = 4,
             mains_hz: float = 50.0, seed: int = 0) -> tuple[int, np.ndarray]:
    """
    Returns (fs, signal) where signal has shape [n_channels, n_samples], units ~ microvolts.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    sig = np.empty((n_channels, n), dtype=np.float64)

    # Per-band amplitudes loosely resembling resting EEG (alpha-dominant).
    band_amp = {"delta": 12.0, "theta": 8.0, "alpha": 18.0, "beta": 5.0}

    for c in range(n_channels):
        x = np.zeros(n)
        for name, (lo, hi) in BANDS.items():
            x += band_amp[name] * _band_osc(t, lo, hi, rng)
        # alpha bursts (amplitude modulation) to add non-stationarity
        burst = 1.0 + 0.6 * (np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28)) > 0.5)
        x *= burst
        x += 6.0 * _pink_noise(n, rng)             # background
        x += 9.0 * np.sin(2 * np.pi * mains_hz * t + rng.uniform(0, 6.28))  # mains
        sig[c] = x

    return fs, sig


if __name__ == "__main__":
    fs, s = generate()
    print(f"generated {s.shape[0]} channels x {s.shape[1]} samples @ {fs} Hz "
          f"({s.shape[1]/fs:.0f} s); per-channel RMS ~ {s.std(axis=1).round(1)}")
