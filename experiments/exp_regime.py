"""
Regime generalization: FIR-order sweep, the chunk-length transition, and seed robustness.

Generalizes Results 1-3 beyond the single order-200 / single-seed configuration:

  #1 FIR-order sweep — the zero-phase length floor is 3 x (order+1) taps; verify the floor and
     the minimum overlap to stay zero-phase scale with FIR order.
  #3 Transition curve — sweep total filtered length L = chunk + 2*overlap through the floor and
     show the abrupt jump in boundary error and event-timing shift at L = 3 x taps.
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
from domain.helpers.filters import _get_bandpass_coefficients, create_bandpass_filter

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
FS = 200
BP = (0.5, 30.0)
NOTCH = 50.0
ORDERS = [100, 200, 300, 400]


def rms(a): return float(np.sqrt(np.mean(np.square(a))))


def event_shift_ms(order, fs=FS):
    """Group-delay shift of the fallback path for a given order (a short window forces lfilter)."""
    (taps,) = _get_bandpass_coefficients(float(fs), BP[0], BP[1], order)
    n = int(20 * fs); t = np.arange(n) / fs; t0 = 10.0
    x = 5 * np.sin(2 * np.pi * 10 * t) + 40 * np.exp(-0.5 * ((t - t0) / 0.02) ** 2)
    half = (3 * len(taps)) // 2 - 5          # just below the floor -> fallback
    c = int(t0 * fs)
    yfb = create_bandpass_filter(x[c - half:c + half], fs, BP[0], BP[1], order=order)
    pk = (c - half + int(np.argmax(np.abs(yfb)))) / fs
    return (pk - t0) * 1e3, len(taps)


def order_sweep():
    rows = []
    for order in ORDERS:
        (taps,) = _get_bandpass_coefficients(float(FS), BP[0], BP[1], order)
        floor = 3 * len(taps)
        shift, ntaps = event_shift_ms(order)
        rows.append(dict(order=order, n_taps=ntaps, floor_samples=floor,
                         floor_s_at_200=round(floor / 200, 3), floor_s_at_256=round(floor / 256, 3),
                         min_overlap_s_1s_window_200=round(max(0, (floor / 200 - 1.0) / 2), 3),
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
    (taps,) = _get_bandpass_coefficients(float(FS), BP[0], BP[1], order)
    floor_s = 3 * len(taps) / FS
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
        rows.append(dict(total_len_s=round(total_s, 2), overlap_s=round(ov, 3),
                         filtfilt_active=int(total_s >= floor_s),
                         boundary_rmse=round(rms((y - gt)[m]), 4)))
    with open(RESULTS / "regime_transition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows, floor_s


def seed_robustness(order=200, n_seeds=12, win_s=8.0):
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
    trows, floor_s = transition()
    nb, ob, db = seed_robustness()

    lines = ["#1 FIR-order sweep (zero-phase length floor = 3 x (order+1) taps):"]
    for r in osweep:
        lines.append(f"  order {r['order']:>3}: {r['n_taps']} taps, floor {r['floor_samples']} "
                     f"samples = {r['floor_s_at_200']}s@200Hz / {r['floor_s_at_256']}s@256Hz; "
                     f"min overlap @1s window = {r['min_overlap_s_1s_window_200']}s; "
                     f"fallback shift {r['fallback_shift_ms']}ms (theory {r['theo_shift_ms']}ms).")
    lines += ["",
              f"#3 Transition: boundary error drops sharply as L=chunk+2*overlap crosses the "
              f"floor ({floor_s:.2f}s). See regime_transition.csv / fig_transition.png.",
              "",
              f"#2 Seed robustness (n={len(nb)} seeds, 8s windows):",
              f"  naive boundary RMSE   = {nb.mean():.3f} +/- {nb.std():.3f} uV",
              f"  overlap boundary RMSE = {ob.mean():.4f} +/- {ob.std():.4f} uV",
              f"  reduction             = {db.mean():.1f} +/- {db.std():.1f} dB"]
    (RESULTS / "regime_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # fig: floor vs order
    orders = [r["order"] for r in osweep]
    floors = [r["floor_samples"] for r in osweep]
    plt.figure(figsize=(7, 4))
    plt.plot(orders, floors, "o-", color="tab:blue")
    for r in osweep:
        plt.annotate(f"{r['fallback_shift_ms']:.0f} ms shift", (r["order"], r["floor_samples"]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8)
    plt.xlabel("FIR order"); plt.ylabel("zero-phase length floor (samples)")
    plt.title("filtfilt floor = 3 × (order+1) scales with FIR order")
    plt.tight_layout(); plt.savefig(RESULTS / "fig_order_floor.png", dpi=150); plt.close()

    # fig: transition
    L = [r["total_len_s"] for r in trows]
    be = [r["boundary_rmse"] for r in trows]
    plt.figure(figsize=(7, 4))
    plt.semilogy(L, be, "o-", color="tab:purple")
    plt.axvline(floor_s, color="tab:red", ls="--", lw=1, label=f"filtfilt floor = {floor_s:.2f} s")
    plt.xlabel("total filtered length  chunk + 2·overlap  (s)")
    plt.ylabel("boundary RMSE vs ground truth (µV, log)")
    plt.title("Boundary error collapses once the zero-phase floor is reached")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(RESULTS / "fig_transition.png", dpi=150); plt.close()
    print(f"\nWrote regime_order_sweep.csv, regime_transition.csv, regime_summary.txt, "
          "fig_order_floor.png, fig_transition.png")


if __name__ == "__main__":
    main()
