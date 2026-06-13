"""
Filter fidelity and phase/temporal-distortion experiment.

Two questions about the reference bandpass (`dsp.apply_bandpass`, FIR order 200) and notch
(`dsp.apply_notch`, 50 Hz):

  A. Magnitude fidelity — does the zero-phase (filtfilt) path realize the intended
     passband/stopband, and does the notch reject 50 Hz?
  B. Phase / temporal distortion — the filtfilt path is zero-phase (no time shift). When the
     window is shorter than 3x(taps), filtfilt cannot run and a streaming filter must use a
     causal `lfilter`, whose linear-phase FIR group delay (~(taps-1)/2 samples) is NOT
     compensated, shifting events in time. We measure the temporal shift of a sharp transient
     (a proxy for an epileptiform spike).

Outputs under results/:
  fidelity_metrics.csv, fig_magnitude.png, fig_transient_shift.png
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy import signal as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dsp import apply_bandpass, apply_notch, bandpass_coefficients

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
FS = 200
BP = (0.5, 30.0)
NOTCH = 50.0
(TAPS,) = bandpass_coefficients(float(FS), BP[0], BP[1], 200)
FILTFILT_MIN = 3 * len(TAPS)


def measure_magnitude():
    """Multitone probe: amplitude gain per frequency through the whole-signal (zero-phase) path."""
    dur = 60.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    freqs = np.array([0.2, 0.5, 1, 2, 4, 8, 10, 13, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    gains = []
    for f in freqs:
        x = np.sin(2 * np.pi * f * t)
        y = apply_bandpass(x, FS, BP[0], BP[1])
        y = apply_notch(y, FS, NOTCH)
        # steady-state gain measured away from the edges
        s = slice(int(5 * FS), int(55 * FS))
        gains.append(np.sqrt(np.mean(y[s] ** 2)) / np.sqrt(np.mean(x[s] ** 2)))
    gains = np.array(gains)
    gdb = 20 * np.log10(np.maximum(gains, 1e-6))
    # designed zero-phase magnitude (|H|^2 for filtfilt) for reference overlay
    w, h = sps.freqz(TAPS, worN=4096, fs=FS)
    designed_db = 20 * np.log10(np.maximum(np.abs(h) ** 2, 1e-6))
    return freqs, gains, gdb, w, designed_db


def measure_transient_shift():
    """Embed a narrow Gaussian 'spike' and measure its peak time after each filtering path."""
    dur = 30.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    t0 = 15.0
    spike = np.exp(-0.5 * ((t - t0) / 0.02) ** 2)          # ~20 ms transient
    bg = 5.0 * np.sin(2 * np.pi * 10 * t)                  # alpha background
    x = bg + 40.0 * spike

    # Zero-phase path: filter the whole (long) signal -> filtfilt
    y_zero = apply_bandpass(x, FS, BP[0], BP[1])

    # Fallback path: filter a short window (< FILTFILT_MIN) around the spike -> lfilter
    half = 250                                             # 500-sample window (< 606) forces fallback
    c = int(t0 * FS)
    win = x[c - half:c + half]
    y_fb_win = apply_bandpass(win, FS, BP[0], BP[1])
    y_fb = np.full(n, np.nan)
    y_fb[c - half:c + half] = y_fb_win

    def peak_time(sig, lo, hi):
        seg = sig[lo:hi]
        return (lo + int(np.nanargmax(np.abs(seg)))) / FS

    lo, hi = c - half, c + half
    pt_in = peak_time(x, lo, hi)
    pt_zero = peak_time(y_zero, lo, hi)
    pt_fb = peak_time(y_fb, lo, hi)
    return t, x, y_zero, y_fb, t0, pt_in, pt_zero, pt_fb, (lo, hi)


def main():
    freqs, gains, gdb, w, designed_db = measure_magnitude()

    # passband (1-25 Hz) ripple, stopband, notch depth
    pb = (freqs >= 1) & (freqs <= 25)
    notch_idx = int(np.argmin(np.abs(freqs - 50)))
    rows = []
    for f, g in zip(freqs, gdb):
        rows.append(dict(freq_hz=f, gain_db=round(float(g), 3)))
    with open(RESULTS / "fidelity_metrics.csv", "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["freq_hz", "gain_db"]); wtr.writeheader()
        for r in rows:
            wtr.writerow(r)

    t, x, y_zero, y_fb, t0, pt_in, pt_zero, pt_fb, (lo, hi) = measure_transient_shift()
    shift_zero_ms = (pt_zero - pt_in) * 1e3
    shift_fb_ms = (pt_fb - pt_in) * 1e3

    summary = [
        f"Filter taps={len(TAPS)}, filtfilt floor={FILTFILT_MIN} samples ({FILTFILT_MIN/FS:.2f} s).",
        f"Passband (1-25 Hz) gain: {gdb[pb].mean():.2f} +/- {gdb[pb].std():.2f} dB "
        f"(ideal 0 dB for filtfilt).",
        f"50 Hz notch gain: {gdb[notch_idx]:.1f} dB (deep rejection expected).",
        f"Stopband @60 Hz: {gdb[-1]:.1f} dB.",
        "",
        "Temporal shift of a 20 ms spike (proxy for epileptiform event timing):",
        f"  zero-phase (filtfilt) shift = {shift_zero_ms:+.1f} ms",
        f"  fallback   (lfilter)  shift = {shift_fb_ms:+.1f} ms "
        f"(uncompensated FIR group delay ~ {(len(TAPS)-1)/2/FS*1e3:.0f} ms).",
    ]
    (RESULTS / "fidelity_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))

    # ---- fig: magnitude ----
    plt.figure(figsize=(8, 4))
    plt.plot(w, designed_db, color="0.6", lw=1, label="designed zero-phase |H|² (filtfilt)")
    plt.plot(freqs, gdb, "o-", color="tab:blue", label="measured (bandpass+notch)")
    plt.axvspan(BP[0], BP[1], color="tab:green", alpha=0.08, label="intended passband 0.5–30 Hz")
    plt.axvline(50, color="tab:red", ls="--", lw=1, label="50 Hz notch")
    plt.ylim(-80, 5); plt.xlim(0, 65)
    plt.xlabel("frequency (Hz)"); plt.ylabel("gain (dB)")
    plt.title("Filter magnitude response (zero-phase path)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS / "fig_magnitude.png", dpi=300); plt.close()

    # ---- fig: transient shift ----
    sl = slice(int((t0 - 0.8) * FS), int((t0 + 0.8) * FS))
    plt.figure(figsize=(8, 4))
    plt.plot(t[sl], x[sl] / np.max(np.abs(x[sl])), color="k", lw=1, alpha=0.5, label="input (spike)")
    plt.plot(t[sl], y_zero[sl] / np.nanmax(np.abs(y_zero[sl])), color="tab:green",
             label=f"zero-phase (filtfilt), shift {shift_zero_ms:+.0f} ms")
    yfb = y_fb[sl]
    plt.plot(t[sl], yfb / np.nanmax(np.abs(yfb)), color="tab:red",
             label=f"fallback (lfilter), shift {shift_fb_ms:+.0f} ms")
    plt.axvline(t0, color="gray", ls="--", lw=1, label="true spike time")
    plt.xlabel("time (s)"); plt.ylabel("normalized amplitude")
    plt.title("Event timing: zero-phase vs short-window fallback")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS / "fig_transient_shift.png", dpi=300); plt.close()
    print(f"\nWrote {RESULTS}/fidelity_metrics.csv, fidelity_summary.txt, "
          "fig_magnitude.png, fig_transient_shift.png")


if __name__ == "__main__":
    main()
