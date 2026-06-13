"""
Self-contained reference implementation of the streaming-EEG DSP primitives studied here.

These are the standard operations used by clinical EEG viewers: a linear-phase FIR bandpass
(designed with ``scipy.signal.firwin``, Hamming window) applied with zero-phase
forward--backward filtering (``scipy.signal.filtfilt``), and an IIR notch
(``scipy.signal.iirnotch``) applied the same way. The module depends only on NumPy and SciPy,
so every experiment in this directory is reproducible without any external package.

Two implementation facts drive the results in this study and are not specific to any product:

  1. ``scipy.signal.filtfilt`` requires the input to be longer than its default padding
     length (``3 * max(len(a), len(b))``); for an order-N FIR that is ``3 * (N+1)`` samples.
     Below that length filtfilt cannot run, so a streaming filter that must return *something*
     for a short window has to fall back to a causal filter.
  2. A causal linear-phase FIR (``lfilter``) has an uncompensated group delay of
     ``(numtaps - 1) / 2`` samples; the zero-phase path has none. The two regimes therefore
     differ in event timing by hundreds of milliseconds.

``apply_bandpass`` models exactly this: zero-phase when the window is long enough, causal
(reflect-padded ``lfilter``) when it is not. The fallback is made explicit so the
length-floor behaviour can be characterized, not so any particular system is audited.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy import signal


@lru_cache(maxsize=64)
def bandpass_coefficients(fs: float, lowcut: float, highcut: float,
                          order: int = 200) -> Tuple[np.ndarray, ...]:
    """FIR bandpass coefficients (Hamming-windowed firwin), cached by design parameters.

    Mirrors the conventional design: cutoffs are normalised to Nyquist and clipped to a valid
    range, the order is forced odd so the tap count is even (type-I linear phase), and firwin
    builds the band-pass taps. For order 200 this yields 202 taps.
    """
    nyquist = fs / 2
    low = max(0.001, min(lowcut / nyquist, 0.99)) if lowcut > 0 else 0.001
    high = max(0.001, min(highcut / nyquist, 0.99))
    if low >= high:
        low, high = high / 2, high
    if order % 2 == 0:
        order += 1
    fir_coefs = signal.firwin(order + 1, [low, high], pass_zero=False, window="hamming")
    return (fir_coefs,)


@lru_cache(maxsize=32)
def notch_coefficients(fs: float, notch_freq: float,
                       quality_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    """IIR notch coefficients (``iirnotch``), cached by design parameters."""
    nyquist = fs / 2
    freq = float(np.clip(float(notch_freq) / nyquist, 0.001, 0.99))
    b, a = signal.iirnotch(freq, quality_factor)
    return (b, a)


def apply_bandpass(data, fs, lowcut, highcut, order: int = 200) -> np.ndarray:
    """Zero-phase FIR bandpass, with the causal fallback used when the window is too short.

    Uses ``filtfilt`` when ``len(data) >= 3 * len(taps)`` (the zero-phase regime). For shorter
    inputs ``filtfilt`` cannot run, so the signal is reflect-padded and filtered causally with
    ``lfilter`` (the fallback regime), which leaves an uncompensated FIR group delay.
    """
    data = np.asarray(data, dtype=np.float64)
    (fir_coefs,) = bandpass_coefficients(float(fs), float(lowcut), float(highcut), int(order))
    min_length = 3 * len(fir_coefs)
    if len(data) < min_length:
        padlen = min_length - len(data)
        data_padded = np.pad(data, (padlen, padlen), mode="reflect")
        filtered_padded = signal.lfilter(fir_coefs, 1.0, data_padded)
        return filtered_padded[padlen:-padlen]
    return signal.filtfilt(fir_coefs, 1.0, data)


def apply_notch(data, fs, notch_freq) -> np.ndarray:
    """Zero-phase IIR notch. Quality factor follows the usual mains convention (40 at >=60 Hz)."""
    data = np.asarray(data, dtype=np.float64)
    quality_factor = 40.0 if notch_freq >= 60.0 else 30.0
    b, a = notch_coefficients(float(fs), float(notch_freq), quality_factor)
    return signal.filtfilt(b, a, data)
