"""
Windowing strategies for streaming EEG filtering, all driving the same reference DSP
primitives (`dsp.apply_bandpass` / `dsp.apply_notch`, NumPy/SciPy only).

Three strategies on a 1-D signal:
  - filter_whole   : filter the entire signal once (ground truth / ideal reference).
  - filter_naive   : filter each non-overlapping display window independently, concatenate.
                     This is the behaviour of a viewer that filters each fetched window on
                     its own.
  - filter_overlap : for each display window, filter an *extended* window that includes
                     `overlap_s` seconds of real neighboring samples on each side, then
                     keep only the center display window (overlap-add).

The bandpass uses zero-phase `filtfilt`, which pads each call with an artificial odd
reflection; at independent window seams that reflection does not match the true signal
continuation, which is the source of the boundary artifact. Supplying real context samples
(overlap) pushes the artificial reflection outside the kept region.
"""
from __future__ import annotations
import numpy as np
from dsp import apply_bandpass, apply_notch


def _apply(x: np.ndarray, fs: int, bp: tuple[float, float] | None,
           notch_hz: float | None, order: int = 200) -> np.ndarray:
    y = x
    if bp is not None:
        y = apply_bandpass(y, fs, bp[0], bp[1], order=order)
    if notch_hz is not None:
        y = apply_notch(y, fs, notch_hz)
    return y


def filter_whole(x, fs, bp=(0.5, 30.0), notch_hz=50.0, order=200):
    return _apply(np.asarray(x, float), fs, bp, notch_hz, order)


def filter_naive(x, fs, win_s=1.0, bp=(0.5, 30.0), notch_hz=50.0, order=200):
    x = np.asarray(x, float)
    n = x.size
    w = int(round(win_s * fs))
    out = np.empty(n, dtype=np.float64)
    for a in range(0, n, w):
        b = min(a + w, n)
        out[a:b] = _apply(x[a:b], fs, bp, notch_hz, order)
    return out


def filter_overlap(x, fs, win_s=1.0, overlap_s=0.5, bp=(0.5, 30.0), notch_hz=50.0, order=200):
    x = np.asarray(x, float)
    n = x.size
    w = int(round(win_s * fs))
    pad = int(round(overlap_s * fs))
    out = np.empty(n, dtype=np.float64)
    for a in range(0, n, w):
        b = min(a + w, n)
        ea = max(0, a - pad)          # extended window start (real samples only)
        eb = min(n, b + pad)          # extended window end
        ext = _apply(x[ea:eb], fs, bp, notch_hz, order)
        out[a:b] = ext[a - ea: a - ea + (b - a)]   # keep the clean center
    return out


def seam_indices(n: int, fs: int, win_s: float) -> np.ndarray:
    """Sample indices of internal window boundaries (seams)."""
    w = int(round(win_s * fs))
    return np.arange(w, n, w)
