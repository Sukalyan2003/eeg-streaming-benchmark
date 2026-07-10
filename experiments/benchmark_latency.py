"""
True per-window latency benchmark for streaming EEG filtering.

The duration benchmark (`benchmark_edf_lengths.py`) reports *amortized* per-window cost: it filters
a whole segment and divides by the window count. That is a throughput-derived proxy. This script
instead measures the genuine request granularity of a streaming viewer: it filters the signal
**one display window at a time**, timing each window independently, and reports the distribution
(median / p95 / max) rather than an average. Each timed "request" filters all channels of one
window, exactly as a viewer would when the user scrolls to it.

For each method the per-window request is:
  - naive             : filter only the displayed chunk (no context);
  - fixed_overlap     : filter chunk + fixed 0.5 s real context per side, keep the center;
  - scheduled_overlap : filter chunk + the scheduler's safe context per side, keep the center.
We report filter-only latency and filter+feature latency (adding per-window Welch relative
band power), and compare both against a 60 Hz display-refresh budget (16.7 ms).

Runs over all local EDFs of both datasets (no downloads). Outputs under `results/`:
  - latency_benchmark.csv            (per dataset/file/method distribution stats)
  - latency_benchmark_summary.txt
  - fig_latency_benchmark.png        (median + p95 per method, per dataset)
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from benchmark_edf_lengths import DATASETS, DEFAULT_BANDS, discover_datasets, load_edf_segment
from dsp import apply_bandpass, apply_notch
from safe_window import plan_window

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

METHODS = ["naive", "fixed_overlap", "scheduled_overlap"]
DURATION_S = 120.0
CHUNK_S = 1.0
FIXED_OVERLAP_S = 0.5
ORDER = 200
BP = (0.5, 30.0)
REFRESH_BUDGET_MS = 1000.0 / 60.0  # one 60 Hz display refresh


def _bandpower_one(window: np.ndarray, fs: float, bands=DEFAULT_BANDS) -> float:
    nperseg = max(8, min(window.size, int(round(2.0 * fs))))
    freqs, psd = signal.welch(window, fs=fs, nperseg=min(nperseg, window.size))
    total_mask = (freqs >= 0.5) & (freqs <= 30.0)
    total = float(np.trapezoid(psd[total_mask], freqs[total_mask]))
    if total <= 0:
        return 0.0
    acc = 0.0
    for low, high in bands.values():
        m = (freqs >= low) & (freqs < high)
        acc += float(np.trapezoid(psd[m], freqs[m]) / total)
    return acc


def _overlap_for(method: str, fs: float):
    if method == "naive":
        return 0
    if method == "fixed_overlap":
        return int(round(FIXED_OVERLAP_S * fs))
    if method == "scheduled_overlap":
        return plan_window(fs, CHUNK_S, FIXED_OVERLAP_S, ORDER).safe_overlap_samples
    raise ValueError(method)


def measure_file(segment: dict, notch_hz: float) -> list[dict]:
    fs = segment["fs"]
    sig = segment["signals_uv"]
    n = sig.shape[1]
    w = int(round(CHUNK_S * fs))
    starts = list(range(0, n - w + 1, w))  # full windows only
    rows = []
    for method in METHODS:
        pad = _overlap_for(method, fs)
        # warm caches (not timed)
        ea, eb = 0, min(n, w + pad)
        for ch in sig:
            apply_notch(apply_bandpass(ch[ea:eb], fs, BP[0], BP[1], order=ORDER), fs, notch_hz)

        filt_ms, full_ms = [], []
        for a in starts:
            b = a + w
            ea = max(0, a - pad)
            eb = min(n, b + pad)
            # --- filter-only request: all channels of this window ---
            t0 = time.perf_counter()
            centers = []
            for ch in sig:
                ext = apply_notch(apply_bandpass(ch[ea:eb], fs, BP[0], BP[1], order=ORDER),
                                  fs, notch_hz)
                centers.append(ext[a - ea: a - ea + w])
            t1 = time.perf_counter()
            # --- add per-window band-power features ---
            for c in centers:
                _bandpower_one(c, fs)
            t2 = time.perf_counter()
            filt_ms.append(1e3 * (t1 - t0))
            full_ms.append(1e3 * (t2 - t0))

        fa = np.asarray(filt_ms)
        ua = np.asarray(full_ms)
        rows.append({
            "dataset": segment["_dataset"],
            "source_name": segment["source_name"],
            "fs": round(fs, 3),
            "channels": sig.shape[0],
            "method": method,
            "overlap_samples": pad,
            "n_windows": len(starts),
            "filt_median_ms": round(float(np.median(fa)), 4),
            "filt_p95_ms": round(float(np.percentile(fa, 95)), 4),
            "filt_max_ms": round(float(fa.max()), 4),
            "full_median_ms": round(float(np.median(ua)), 4),
            "full_p95_ms": round(float(np.percentile(ua, 95)), 4),
            "full_max_ms": round(float(ua.max()), 4),
        })
    return rows


def main() -> None:
    datasets = discover_datasets()
    if not datasets:
        print("No EDFs under data/; latency benchmark needs real recordings.")
        return

    rows = []
    for ds, files in datasets:
        for path in files:
            seg = load_edf_segment(path, DURATION_S, ds["primary_channels"])
            seg["_dataset"] = ds["key"]
            rows.extend(measure_file(seg, ds["notch_hz"]))
            print(f"  [{ds['label']}] {path.name}: timed {len(METHODS)} methods")

    with open(RESULTS / "latency_benchmark.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate across files: median of per-file medians, worst p95/max across files.
    datasets_seen = list(dict.fromkeys(r["dataset"] for r in rows))
    lines = [f"True per-window latency benchmark - {DURATION_S:g} s segments, {CHUNK_S:g} s windows,",
             f"FIR order {ORDER}; each request filters all channels of one window. "
             f"60 Hz refresh budget = {REFRESH_BUDGET_MS:.1f} ms."]
    agg = {}
    for ds in datasets_seen:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        fs = ds_rows[0]["fs"]; ch = ds_rows[0]["channels"]
        n_files = len(set(r["source_name"] for r in ds_rows))
        lines += ["", f"=== {ds}: {n_files} files, fs={fs} Hz, {ch} channels ==="]
        for method in METHODS:
            mr = [r for r in ds_rows if r["method"] == method]
            filt_med = float(np.median([r["filt_median_ms"] for r in mr]))
            filt_p95 = float(max(r["filt_p95_ms"] for r in mr))
            full_med = float(np.median([r["full_median_ms"] for r in mr]))
            full_p95 = float(max(r["full_p95_ms"] for r in mr))
            agg[(ds, method)] = (filt_med, filt_p95, full_med, full_p95)
            lines.append(
                f"  {method:17s}: filter median {filt_med:6.2f} ms (worst p95 {filt_p95:6.2f}); "
                f"filter+feature median {full_med:6.2f} ms (worst p95 {full_p95:6.2f}) "
                f"-> {'WITHIN' if full_p95 <= REFRESH_BUDGET_MS else 'OVER'} refresh budget")
    lines += ["", "Per-window latency is measured directly (one timed filter call per window), not "
                  "amortized from a whole-segment run."]
    (RESULTS / "latency_benchmark_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure: median + p95 (filter+feature) per method, grouped per dataset.
    fig, axes = plt.subplots(1, len(datasets_seen), figsize=(6.0 * len(datasets_seen), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets_seen):
        x = np.arange(len(METHODS))
        med = [agg[(ds, m)][2] for m in METHODS]
        p95 = [agg[(ds, m)][3] for m in METHODS]
        ax.bar(x - 0.2, med, width=0.4, label="median")
        ax.bar(x + 0.2, p95, width=0.4, label="p95")
        ax.axhline(REFRESH_BUDGET_MS, color="tab:red", ls="--", lw=1,
                   label=f"60 Hz refresh ({REFRESH_BUDGET_MS:.1f} ms)")
        ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15, fontsize=8)
        ax.set_ylabel("per-window latency, filter+feature (ms)")
        ax.text(0.03, 0.94, ds, transform=ax.transAxes, va="top", ha="left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_latency_benchmark.png", dpi=300)
    plt.close(fig)
    print("\nWrote latency_benchmark.csv, latency_benchmark_summary.txt, fig_latency_benchmark.png")


if __name__ == "__main__":
    main()
