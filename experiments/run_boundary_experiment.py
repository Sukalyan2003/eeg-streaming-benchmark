"""
Boundary-artifact experiment (headline result).

Compares three windowed-filtering strategies against a whole-signal ground truth, using the
reference DSP primitives in `dsp.py` (NumPy/SciPy only):
  - naive per-window filtering (each fetched window filtered independently)
  - overlap-add / extended-window filtering (filter window +/- context, keep the center)

Outputs (under experiments/results/):
  - metrics.csv      : per (window_s, overlap_s) reconstruction + boundary error + latency
  - fig_seam.png     : waveform zoom at a window seam (GT vs naive vs overlap)
  - fig_profile.png  : mean |error| vs distance from seam (naive vs overlap)
  - fig_summary.png  : boundary RMSE and per-window latency vs window size

Run:  python run_boundary_experiment.py
"""
from __future__ import annotations
import time
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth import generate
from windowing import filter_whole, filter_naive, filter_overlap, seam_indices
from safe_window import fir_tap_count, filtfilt_min_valid_samples, filtfilt_padlen_samples

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

FS = 200
DURATION_S = 120.0
N_CH = 4
BP = (0.5, 30.0)
NOTCH = 50.0
WIN_SIZES = [1.0, 2.0, 5.0, 8.0]
OVERLAPS = [0.25, 0.5, 1.0, 2.0]   # production default = 0.5
SEAM_HALF = int(round(0.05 * FS))  # ±50 ms "seam" region
GUARD = int(round(0.6 * FS))       # interior excludes ±0.6 s around seams

# FIR order 200 -> 202 taps; scipy filtfilt requires > 3*taps samples, otherwise a streaming
# filter must fall back to a phase-distorting causal lfilter path for that window.
ORDER = 200
TAPS = fir_tap_count(ORDER)
FILTFILT_PADLEN = filtfilt_padlen_samples(TAPS)       # 606 samples = 3.03 s @ 200 Hz
FILTFILT_MIN = filtfilt_min_valid_samples(TAPS)       # shortest valid zero-phase input


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def seam_and_interior_error(err, seams, n):
    seam_mask = np.zeros(n, dtype=bool)
    guard_mask = np.zeros(n, dtype=bool)
    for s in seams:
        seam_mask[max(0, s - SEAM_HALF):min(n, s + SEAM_HALF)] = True
        guard_mask[max(0, s - GUARD):min(n, s + GUARD)] = True
    interior_mask = ~guard_mask
    return rms(err[seam_mask]), rms(err[interior_mask])


def time_per_window(fn, x, **kw):
    # warm up filter-coefficient cache
    fn(x[: 5 * FS], FS, **kw)
    t0 = time.perf_counter()
    fn(x, FS, **kw)
    dt = time.perf_counter() - t0
    n_windows = int(np.ceil(x.size / (kw["win_s"] * FS)))
    return 1e3 * dt / n_windows  # ms per window


def main():
    fs, sig = generate(DURATION_S, FS, N_CH, mains_hz=NOTCH, seed=0)
    n = sig.shape[1]
    gt = np.stack([filter_whole(sig[c], fs, BP, NOTCH) for c in range(N_CH)])

    rows = []
    for win_s in WIN_SIZES:
        seams = seam_indices(n, fs, win_s)

        # naive
        naive = np.stack([filter_naive(sig[c], fs, win_s, BP, NOTCH) for c in range(N_CH)])
        nb, ni = np.mean([seam_and_interior_error(naive[c] - gt[c], seams, n) for c in range(N_CH)], axis=0)
        lat_naive = time_per_window(filter_naive, sig[0], win_s=win_s, bp=BP, notch_hz=NOTCH)
        ext_naive = int(round(win_s * FS))
        rows.append(dict(window_s=win_s, overlap_s=0.0, method="naive",
                         ext_samples=ext_naive, filtfilt_active=int(ext_naive >= FILTFILT_MIN),
                         global_rmse=rms(naive - gt), boundary_rmse=nb,
                         interior_rmse=ni, boundary_to_interior_db=20 * np.log10(nb / (ni + 1e-12)),
                         ms_per_window=lat_naive))

        for ov in OVERLAPS:
            olap = np.stack([filter_overlap(sig[c], fs, win_s, ov, BP, NOTCH) for c in range(N_CH)])
            ob, oi = np.mean([seam_and_interior_error(olap[c] - gt[c], seams, n) for c in range(N_CH)], axis=0)
            lat = time_per_window(filter_overlap, sig[0], win_s=win_s, overlap_s=ov, bp=BP, notch_hz=NOTCH)
            ext = int(round((win_s + 2 * ov) * FS))
            rows.append(dict(window_s=win_s, overlap_s=ov, method="overlap",
                             ext_samples=ext, filtfilt_active=int(ext >= FILTFILT_MIN),
                             global_rmse=rms(olap - gt), boundary_rmse=ob,
                             interior_rmse=oi, boundary_to_interior_db=20 * np.log10(ob / (oi + 1e-12)),
                             ms_per_window=lat))

    # --- write metrics.csv ---
    with open(RESULTS / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    # --- headline number: dB reduction in boundary error at production default (win=1s, ov=0.5s) ---
    def get(win, ov, method):
        for r in rows:
            if r["window_s"] == win and r["overlap_s"] == ov and r["method"] == method:
                return r
        return None
    # Clean headline in the zero-phase (filtfilt-valid) regime: win=8 s.
    nrow = get(8.0, 0.0, "naive")
    orow = get(8.0, 0.5, "overlap")
    db = 20 * np.log10(nrow["boundary_rmse"] / (orow["boundary_rmse"] + 1e-12))
    # A small window (1 s chunk + 0.5 s overlap) falls below the zero-phase length floor.
    small = get(1.0, 0.5, "overlap")
    lines = [
        "RESULT 1 (zero-phase regime, win=8 s):",
        f"  boundary RMSE naive={nrow['boundary_rmse']:.3f} uV -> overlap(0.5 s)="
        f"{orow['boundary_rmse']:.3f} uV : reduction {db:.1f} dB at "
        f"+{100*(orow['ms_per_window']/nrow['ms_per_window']-1):.0f}% latency "
        f"({nrow['ms_per_window']:.3f} -> {orow['ms_per_window']:.3f} ms/window).",
        "",
        "RESULT 2 (zero-phase length floor):",
        f"  An order-{ORDER} FIR ({TAPS} taps) has filtfilt padlen={FILTFILT_PADLEN} samples; "
        f"zero-phase filtering needs at least {FILTFILT_MIN} samples "
        f"({FILTFILT_MIN/FS:.2f} s). Shorter windows must use a causal filter instead.",
        f"  A windowed filter processes a span of chunk + 2*overlap. Large windows clear the "
        f"floor and are zero-phase; small windows do not:",
        f"    e.g. chunk=1 s + overlap=0.5 s = {small['ext_samples']} samples (<{FILTFILT_MIN}) "
        f"=> filtfilt_active={small['filtfilt_active']}, interior RMSE="
        f"{small['interior_rmse']:.2f} uV (distorted everywhere, not just seams).",
        f"  Design criterion to stay zero-phase: keep chunk + 2*overlap >= "
        f"{FILTFILT_MIN/FS:.2f} s (>= {(FILTFILT_MIN/FS - 1.0)/2:.2f} s overlap each side at a "
        f"1 s chunk), i.e. (chunk + 2*overlap)*fs > 3*taps.",
    ]
    (RESULTS / "headline.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------- figures ----------
    _fig_seam(sig, gt, fs)
    _fig_profile(sig, gt, fs)
    _fig_summary(rows)
    print(f"\nWrote {RESULTS}/metrics.csv, headline.txt, fig_seam.png, fig_profile.png, fig_summary.png")


def _fig_seam(sig, gt, fs):
    win_s = 1.0
    c = 0
    naive = filter_naive(sig[c], fs, win_s, BP, NOTCH)
    olap = filter_overlap(sig[c], fs, win_s, 0.5, BP, NOTCH)
    seam = int(60 * fs)  # a seam at 60 s
    sl = slice(seam - int(0.5 * fs), seam + int(0.5 * fs))
    t = np.arange(sl.start, sl.stop) / fs
    plt.figure(figsize=(8, 4))
    plt.plot(t, gt[c][sl], label="ground truth (whole-signal)", lw=2, color="k")
    plt.plot(t, naive[sl], label="naive per-window", lw=1.2, color="tab:red", alpha=0.9)
    plt.plot(t, olap[sl], label="overlap-add (0.5 s)", lw=1.2, color="tab:green", alpha=0.9)
    plt.axvline(seam / fs, color="gray", ls="--", lw=1, label="window seam")
    plt.xlabel("time (s)"); plt.ylabel("amplitude (µV)")
    plt.title("Filtered EEG across a 1 s window seam")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS / "fig_seam.png", dpi=300); plt.close()


def _fig_profile(sig, gt, fs):
    win_s = 1.0
    n = sig.shape[1]
    seams = seam_indices(n, fs, win_s)
    K = int(0.25 * fs)
    naive = np.stack([filter_naive(sig[c], fs, win_s, BP, NOTCH) for c in range(N_CH)])
    olap = np.stack([filter_overlap(sig[c], fs, win_s, 0.5, BP, NOTCH) for c in range(N_CH)])
    prof_n = np.zeros(2 * K); prof_o = np.zeros(2 * K); cnt = 0
    for s in seams:
        if s - K < 0 or s + K > n:
            continue
        for c in range(N_CH):
            prof_n += np.abs(naive[c][s - K:s + K] - gt[c][s - K:s + K])
            prof_o += np.abs(olap[c][s - K:s + K] - gt[c][s - K:s + K])
        cnt += N_CH
    prof_n /= cnt; prof_o /= cnt
    lag = (np.arange(-K, K)) / fs * 1e3
    plt.figure(figsize=(8, 4))
    plt.plot(lag, prof_n, label="naive per-window", color="tab:red")
    plt.plot(lag, prof_o, label="overlap-add (0.5 s)", color="tab:green")
    plt.axvline(0, color="gray", ls="--", lw=1, label="seam")
    plt.xlabel("time relative to seam (ms)"); plt.ylabel("mean |error| vs ground truth (µV)")
    plt.title("Reconstruction error concentrated at window seams")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS / "fig_profile.png", dpi=300); plt.close()


def _fig_summary(rows):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    wins = WIN_SIZES
    naive_b = [next(r for r in rows if r["window_s"] == w and r["method"] == "naive")["boundary_rmse"] for w in wins]
    olap_b = [next(r for r in rows if r["window_s"] == w and r["overlap_s"] == 0.5)["boundary_rmse"] for w in wins]
    x = np.arange(len(wins)); bw = 0.35
    ax[0].bar(x - bw/2, naive_b, bw, label="naive", color="tab:red")
    ax[0].bar(x + bw/2, olap_b, bw, label="overlap-add 0.5 s", color="tab:green")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{w:g}s" for w in wins])
    ax[0].set_ylabel("boundary RMSE vs ground truth (µV)")
    ax[0].set_xlabel("display window size"); ax[0].set_title("Boundary error by window size")
    ax[0].legend(fontsize=8)

    naive_l = [next(r for r in rows if r["window_s"] == w and r["method"] == "naive")["ms_per_window"] for w in wins]
    olap_l = [next(r for r in rows if r["window_s"] == w and r["overlap_s"] == 0.5)["ms_per_window"] for w in wins]
    ax[1].plot(wins, naive_l, "o-", label="naive", color="tab:red")
    ax[1].plot(wins, olap_l, "o-", label="overlap-add 0.5 s", color="tab:green")
    ax[1].set_xlabel("display window size (s)"); ax[1].set_ylabel("latency (ms per window)")
    ax[1].set_title("Per-window filtering latency"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "fig_summary.png", dpi=300); plt.close()


if __name__ == "__main__":
    main()
