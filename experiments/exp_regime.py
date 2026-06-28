"""
Regime generalization: FIR-order sweep, the chunk-length transition, and seed robustness.

Generalizes Results 1-3 beyond the single order-200 / single-seed configuration:

  #1 FIR-order sweep — the zero-phase pad length is 3 x taps; verify the pad length and
     the minimum overlap to stay zero-phase scale with FIR order.
  #3 Transition curve — sweep total filtered length L = chunk + 2*overlap through the floor and
     show the abrupt jump in boundary error and event-timing shift at L > 3 x taps.
  #2 Seed robustness — repeat the overlap-add boundary result across seeds; report mean +/- SD.

Outputs under results/: regime_order_sweep.csv, regime_transition.csv,
fig_order_floor.png, fig_transition.png, regime_summary.txt
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth import generate
from windowing import filter_whole, filter_naive, filter_overlap, seam_indices
from dsp import bandpass_coefficients, apply_bandpass
from safe_window import filtfilt_min_valid_samples, filtfilt_padlen_samples, plan_window

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
FS = 200
BP = (0.5, 30.0)
NOTCH = 50.0
ORDERS = [50, 100, 150, 200, 250, 300, 400, 500]


def rms(a): return float(np.sqrt(np.mean(np.square(a))))


def event_shift_ms(order, fs=FS):
    """Group-delay shift of the fallback path for a given order (a short window forces lfilter)."""
    (taps,) = bandpass_coefficients(float(fs), BP[0], BP[1], order)
    n = int(20 * fs); t = np.arange(n) / fs; t0 = 10.0
    x = 5 * np.sin(2 * np.pi * 10 * t) + 40 * np.exp(-0.5 * ((t - t0) / 0.02) ** 2)
    half = (3 * len(taps)) // 2 - 5          # just below the floor -> fallback
    c = int(t0 * fs)
    yfb = apply_bandpass(x[c - half:c + half], fs, BP[0], BP[1], order=order)
    pk = (c - half + int(np.argmax(np.abs(yfb)))) / fs
    return (pk - t0) * 1e3, len(taps)


def order_sweep():
    rows = []
    for order in ORDERS:
        (taps,) = bandpass_coefficients(float(FS), BP[0], BP[1], order)
        padlen = filtfilt_padlen_samples(len(taps))
        min_valid = filtfilt_min_valid_samples(len(taps))
        shift, ntaps = event_shift_ms(order)
        one_s_plan = plan_window(FS, 1.0, 0.0, order)
        rows.append(dict(order=order, n_taps=ntaps, padlen_samples=padlen,
                         min_valid_samples=min_valid,
                         padlen_s_at_200=round(padlen / 200, 3),
                         padlen_s_at_256=round(padlen / 256, 3),
                         min_overlap_s_1s_window_200=round(one_s_plan.safe_overlap_s, 3),
                         fallback_shift_ms=round(shift, 1),
                         theo_shift_ms=round((ntaps - 1) / 2 / FS * 1e3, 1)))
    with open(RESULTS / "regime_order_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


def transition(order=200, seed=0):
    """Sweep extended length L=chunk+2*overlap through the floor; measure boundary error."""
    fs, sig = generate(120.0, FS, 1, mains_hz=NOTCH, seed=seed)
    x = sig[0]; n = x.size
    (taps,) = bandpass_coefficients(float(FS), BP[0], BP[1], order)
    min_valid_s = filtfilt_min_valid_samples(len(taps)) / FS
    gt = filter_whole(x, fs, BP, NOTCH, order=order)
    win_s = 1.0
    seams = seam_indices(n, fs, win_s)
    rows = []
    for total_s in np.arange(1.0, 5.01, 0.25):
        ov = max(0.0, (total_s - win_s) / 2)
        y = filter_overlap(x, fs, win_s, ov, BP, NOTCH, order=order)
        # boundary RMSE near seams
        sh = int(0.05 * fs); m = np.zeros(n, bool)
        for s in seams:
            m[max(0, s - sh):min(n, s + sh)] = True
        total_samples = int(round(total_s * FS))
        rows.append(dict(total_len_s=round(total_s, 2), overlap_s=round(ov, 3),
                         filtfilt_active=int(total_samples >= filtfilt_min_valid_samples(len(taps))),
                         boundary_rmse=round(rms((y - gt)[m]), 4)))
    with open(RESULTS / "regime_transition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, min_valid_s


def seed_robustness(order=200, n_seeds=24, win_s=8.0):
    nb_list, ob_list = [], []
    for seed in range(n_seeds):
        fs, sig = generate(120.0, FS, 1, mains_hz=NOTCH, seed=seed)
        x = sig[0]; n = x.size
        gt = filter_whole(x, fs, BP, NOTCH, order=order)
        seams = seam_indices(n, fs, win_s)
        sh = int(0.05 * fs); m = np.zeros(n, bool)
        for s in seams:
            m[max(0, s - sh):min(n, s + sh)] = True
        nb_list.append(rms((filter_naive(x, fs, win_s, BP, NOTCH, order=order) - gt)[m]))
        ob_list.append(rms((filter_overlap(x, fs, win_s, 0.5, BP, NOTCH, order=order) - gt)[m]))
    nb = np.array(nb_list); ob = np.array(ob_list)
    db = 20 * np.log10(nb / (ob + 1e-12))
    return nb, ob, db


def main():
    osweep = order_sweep()
    trows, min_valid_s = transition()
    nb, ob, db = seed_robustness()

    lines = ["#1 FIR-order sweep (zero-phase padlen = 3 x taps; min valid = padlen + 1):"]
    for r in osweep:
        lines.append(f"  order {r['order']:>3}: {r['n_taps']} taps, padlen {r['padlen_samples']} "
                     f"samples = {r['padlen_s_at_200']}s@200Hz / {r['padlen_s_at_256']}s@256Hz; "
                     f"min valid {r['min_valid_samples']} samples; "
                     f"min overlap @1s window = {r['min_overlap_s_1s_window_200']}s; "
                     f"fallback shift {r['fallback_shift_ms']}ms (theory {r['theo_shift_ms']}ms).")
    lines += ["",
              f"#3 Transition: boundary error drops sharply as L=chunk+2*overlap crosses the "
              f"minimum valid zero-phase length ({min_valid_s:.2f}s). See regime_transition.csv / fig_transition.png.",
              "",
              f"#2 Seed robustness (n={len(nb)} seeds, 8s windows):",
              f"  naive boundary RMSE   = {nb.mean():.3f} +/- {nb.std():.3f} uV",
              f"  overlap boundary RMSE = {ob.mean():.4f} +/- {ob.std():.4f} uV",
              f"  reduction             = {db.mean():.1f} +/- {db.std():.1f} dB"]
    (RESULTS / "regime_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # fig: floor vs order
    orders = [r["order"] for r in osweep]
    floors = [r["padlen_samples"] for r in osweep]
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(orders, floors, "o-", color="tab:blue")
    plt.margins(0.15)
    for r in osweep:
        plt.annotate(f"{r['fallback_shift_ms']:.0f} ms shift", (r["order"], r["padlen_samples"]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8)
    plt.xlabel("FIR order"); plt.ylabel("filtfilt pad length (samples)")
    plt.title("filtfilt length floor", fontsize=10)
    plt.tight_layout(pad=0.35); plt.savefig(RESULTS / "fig_order_floor.png", dpi=300); plt.close()

    # fig: transition
    L = [r["total_len_s"] for r in trows]
    be = [r["boundary_rmse"] for r in trows]
    plt.figure(figsize=(4.2, 2.6))
    plt.semilogy(L, be, "o-", color="tab:purple")
    plt.axvline(min_valid_s, color="tab:red", ls="--", lw=1,
                label=f"zero-phase floor = {min_valid_s:.2f} s")
    plt.xlabel("filtered length: chunk + 2·overlap (s)")
    plt.ylabel("boundary RMSE (µV, log)")
    plt.title("Boundary error vs length", fontsize=10)
    plt.legend(fontsize=8); plt.tight_layout(pad=0.35)
    plt.savefig(RESULTS / "fig_transition.png", dpi=300); plt.close()
    print(f"\nWrote regime_order_sweep.csv, regime_transition.csv, regime_summary.txt, "
          "fig_order_floor.png, fig_transition.png")


if __name__ == "__main__":
    main()
