"""
End-to-end EDF-duration benchmark for the windowed filtering pipeline.

This script supplies the systems evidence that a correctness-only paper lacks: how processing
time, real-time factor, per-window latency, and memory behave as the EDF segment grows, across
many real recordings at two sampling rates. It compares three streaming configurations plus an
offline reference:

  - naive              : filter each display chunk independently;
  - fixed_overlap      : overlap-add with the configured overlap, even if it is not safe;
  - scheduled_overlap  : overlap-add after the safe-window scheduler increases context as needed;
  - whole_signal       : offline whole-record filtering reference.

It runs three sweeps, all over real EDFs discovered under ``data/`` (no downloads here):

  1. PRIMARY  - every EDF in each dataset x several segment durations x all methods, at a fixed
     FIR order. Aggregated to mean +/- SD across files per (dataset, duration, method). This is
     the headline duration-scaling result with error bars across subjects.
  2. ORDER    - one representative file per dataset x one duration x several FIR orders x all
     methods. Shows how cost and the zero-phase guard scale with filter order.
  3. CHANNELS - one representative file x one duration x several channel counts x all methods.
     Shows how cost scales with the number of channels.

Datasets are auto-detected from filename prefixes:
  - ``chb*.edf``  -> CHB-MIT (256 Hz, 60 Hz mains);
  - ``SC*.edf``   -> Sleep-EDF Expanded (100 Hz, 50 Hz mains).
If no EDFs are present, the script falls back to the deterministic synthetic generator so the
benchmark remains runnable without any data.

Metrics per run: load time, filtering time, Welch band-power feature time, total processing time,
real-time factor (throughput), amortized per-window latency (the interactive-responsiveness
proxy, distinct from throughput), array memory, and Python peak memory.

Outputs under ``results/``:
  - edf_length_benchmark.csv            (primary, per file/duration/method)
  - edf_length_benchmark_summary.txt    (aggregated mean +/- SD + sub-sweep highlights)
  - edf_order_sweep.csv                 (order sub-sweep)
  - edf_channel_sweep.csv               (channel sub-sweep)
  - fig_edf_length_benchmark.png        (duration scaling, error bars across files, per dataset)
  - fig_edf_order_scaling.png           (processing time vs FIR order)
  - fig_edf_channel_scaling.png         (processing time vs channel count)

Example:
    python3 benchmark_edf_lengths.py
    python3 benchmark_edf_lengths.py --durations 30,60,120,300,600 --orders 100,200,300,400
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from safe_window import plan_window
from synth import generate
from windowing import filter_naive, filter_overlap, filter_whole


HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

METHODS = ["naive", "fixed_overlap", "scheduled_overlap", "whole_signal"]
METHOD_LABELS = {
    "naive": "naive",
    "fixed_overlap": "fixed",
    "scheduled_overlap": "scheduled",
    "whole_signal": "whole",
}
DEFAULT_DURATIONS = [30.0, 60.0, 120.0, 300.0, 600.0]
DEFAULT_ORDERS = [100, 200, 300, 400]
DEFAULT_CHANNEL_COUNTS = [1, 2, 4, 8, 16]
DEFAULT_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Dataset detection by filename prefix. Each entry sets the physically correct mains frequency
# and a default channel count for the primary sweep. fs is read from the file, not assumed.
DATASETS = [
    {"key": "chbmit", "label": "CHB-MIT", "glob": "chb*.edf",
     "notch_hz": 60.0, "primary_channels": 8},
    {"key": "sleepedf", "label": "Sleep-EDF", "glob": "SC*.edf",
     "notch_hz": 50.0, "primary_channels": 2},
]


def _parse_floats(text: str) -> list[float]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0:
            raise ValueError("values must be positive")
        out.append(value)
    if not out:
        raise ValueError("at least one value is required")
    return out


def _parse_ints(text: str) -> list[int]:
    return [int(round(v)) for v in _parse_floats(text)]


def _pick_eeg_channels(raw, max_channels: int):
    try:
        import mne
        picks = list(mne.pick_types(raw.info, eeg=True, exclude=[]))
    except Exception:
        picks = []
    if not picks:
        picks = list(range(len(raw.ch_names)))
    return picks[:max_channels]


def load_edf_segment(path: Path, duration_s: float, max_channels: int):
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    stop = min(raw.n_times, int(round(duration_s * fs)))
    actual_duration_s = stop / fs
    picks = _pick_eeg_channels(raw, max_channels)

    t0 = time.perf_counter()
    data = raw.get_data(picks=picks, start=0, stop=stop) * 1e6  # V -> uV
    load_s = time.perf_counter() - t0

    channel_names = [raw.ch_names[p] for p in picks]
    return {
        "source_kind": "edf",
        "source_name": path.name,
        "fs": fs,
        "signals_uv": np.asarray(data, dtype=np.float64),
        "load_s": load_s,
        "channel_names": channel_names,
        "actual_duration_s": actual_duration_s,
    }


def load_synthetic_segment(duration_s: float, fs: int, max_channels: int):
    t0 = time.perf_counter()
    fs_out, signals = generate(duration_s, fs, max_channels, mains_hz=50.0, seed=7)
    load_s = time.perf_counter() - t0
    return {
        "source_kind": "synthetic",
        "source_name": "synthetic",
        "fs": float(fs_out),
        "signals_uv": np.asarray(signals, dtype=np.float64),
        "load_s": load_s,
        "channel_names": [f"ch{i + 1}" for i in range(signals.shape[0])],
        "actual_duration_s": signals.shape[1] / float(fs_out),
    }


def relative_bandpower_features(filtered: np.ndarray, fs: float, chunk_s: float,
                                bands=DEFAULT_BANDS) -> tuple[int, float]:
    """Compute simple per-window relative band-power features and return count/checksum."""
    window = max(1, int(round(chunk_s * fs)))
    nperseg = max(8, min(window, int(round(2.0 * fs))))
    count = 0
    checksum = 0.0

    for channel in filtered:
        for start in range(0, channel.size, window):
            segment = channel[start:min(start + window, channel.size)]
            if segment.size < 8:
                continue
            freqs, psd = signal.welch(segment, fs=fs, nperseg=min(nperseg, segment.size))
            total_mask = (freqs >= 0.5) & (freqs <= 30.0)
            total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask]))
            if total_power <= 0:
                continue
            for low, high in bands.values():
                band_mask = (freqs >= low) & (freqs < high)
                checksum += float(np.trapezoid(psd[band_mask], freqs[band_mask]) / total_power)
            count += 1
    return count, checksum


def _filter_method(signals: np.ndarray, fs: float, method: str, chunk_s: float,
                   overlap_s: float, order: int, bp: tuple[float, float], notch_hz: float):
    if method == "naive":
        return np.stack([filter_naive(x, fs, chunk_s, bp, notch_hz, order=order) for x in signals])
    if method in {"fixed_overlap", "scheduled_overlap"}:
        return np.stack([filter_overlap(x, fs, chunk_s, overlap_s, bp, notch_hz, order=order)
                         for x in signals])
    if method == "whole_signal":
        return np.stack([filter_whole(x, fs, bp, notch_hz, order=order) for x in signals])
    raise ValueError(f"unknown method: {method}")


def benchmark_one(segment: dict, dataset_key: str, method: str, chunk_s: float,
                  requested_overlap_s: float, order: int, bp: tuple[float, float],
                  notch_hz: float) -> dict:
    fs = segment["fs"]
    signals = segment["signals_uv"]
    duration_s = segment["actual_duration_s"]

    if method == "naive":
        plan = plan_window(fs, chunk_s, 0.0, order)
        used_overlap_s = 0.0
    else:
        plan = plan_window(fs, chunk_s, requested_overlap_s, order)
        used_overlap_s = plan.safe_overlap_s if method == "scheduled_overlap" else requested_overlap_s

    exec_method = "fixed_overlap" if method == "scheduled_overlap" else method

    # Warm coefficient caches outside the measured region.
    warm_n = min(signals.shape[1], max(16, int(round(min(duration_s, 5.0) * fs))))
    _filter_method(signals[:1, :warm_n], fs, exec_method, chunk_s, used_overlap_s, order, bp, notch_hz)

    # Timing must not run under tracemalloc (instrumentation inflates wall time ~2x); we report
    # array memory from nbytes instead, which captures the dominant working set.
    t0 = time.perf_counter()
    filtered = _filter_method(signals, fs, exec_method, chunk_s, used_overlap_s, order, bp, notch_hz)
    filter_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    feature_windows, feature_checksum = relative_bandpower_features(filtered, fs, chunk_s)
    feature_s = time.perf_counter() - t1

    processing_s = filter_s + feature_s
    window_samples = max(1, int(round(chunk_s * fs)))
    num_windows = int(np.ceil(signals.shape[1] / window_samples))
    eeg_hours = duration_s / 3600.0
    seconds_per_eeg_hour = processing_s / max(eeg_hours, 1e-12)
    # Amortized per-display-window latency: the interactive-responsiveness proxy, distinct from
    # the whole-segment throughput captured by realtime_factor.
    per_window_latency_ms = 1e3 * processing_s / max(num_windows, 1)

    if method == "whole_signal":
        configured_safe = signals.shape[1] >= plan.min_valid_samples
        scheduled_safe = configured_safe
        configured_span_samples = signals.shape[1]
        scheduled_span_samples = signals.shape[1]
    else:
        configured_safe = plan.configured_filtfilt_safe
        scheduled_safe = plan_window(fs, chunk_s, used_overlap_s, order).configured_filtfilt_safe
        configured_span_samples = plan.configured_span_samples
        scheduled_span_samples = int(round(chunk_s * fs)) + 2 * int(round(used_overlap_s * fs))

    return {
        "dataset": dataset_key,
        "source_kind": segment["source_kind"],
        "source_name": segment["source_name"],
        "duration_s": round(duration_s, 3),
        "fs": round(fs, 3),
        "channels": signals.shape[0],
        "method": method,
        "order": order,
        "chunk_s": chunk_s,
        "requested_overlap_s": requested_overlap_s,
        "used_overlap_s": round(used_overlap_s, 6),
        "safe_overlap_s": round(plan.safe_overlap_s, 6),
        "padlen_samples": plan.padlen_samples,
        "min_valid_samples": plan.min_valid_samples,
        "configured_span_samples": configured_span_samples,
        "scheduled_span_samples": scheduled_span_samples,
        "configured_filtfilt_safe": int(configured_safe),
        "used_filtfilt_safe": int(scheduled_safe),
        "fallback_group_delay_ms": round(plan.fallback_group_delay_ms, 3),
        "num_windows": num_windows,
        "feature_windows": feature_windows,
        "load_s": round(float(segment["load_s"]), 6),
        "filter_s": round(filter_s, 6),
        "feature_s": round(feature_s, 6),
        "processing_s": round(processing_s, 6),
        "total_s": round(float(segment["load_s"]) + processing_s, 6),
        "realtime_factor": round(processing_s / max(duration_s, 1e-12), 8),
        "per_window_latency_ms": round(per_window_latency_ms, 4),
        "seconds_per_eeg_hour": round(seconds_per_eeg_hour, 3),
        "eeg_hours_per_wall_hour": round(duration_s / max(processing_s, 1e-12), 1),
        "array_memory_mb": round((signals.nbytes + filtered.nbytes) / 1e6, 3),
        "feature_checksum": round(feature_checksum, 6),
    }


# --------------------------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------------------------

def discover_datasets():
    """Return a list of (dataset_dict, [file_paths]) for datasets present under data/."""
    found = []
    for ds in DATASETS:
        files = sorted(DATA.glob(ds["glob"]))
        if files:
            found.append((ds, files))
    return found


def _agg(rows, key_fields, value_field):
    """Aggregate value_field by key tuple -> (mean, sd, n)."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in key_fields)].append(r[value_field])
    out = {}
    for k, vals in buckets.items():
        arr = np.asarray(vals, dtype=float)
        out[k] = (float(arr.mean()), float(arr.std()), len(arr))
    return out


def run_primary(datasets, durations, order, chunk_s, overlap_s, bp):
    rows = []
    for ds, files in datasets:
        for path in files:
            for duration_s in durations:
                seg = load_edf_segment(path, duration_s, ds["primary_channels"])
                for method in METHODS:
                    rows.append(benchmark_one(seg, ds["key"], method, chunk_s, overlap_s,
                                              order, bp, ds["notch_hz"]))
            print(f"  [{ds['label']}] {path.name}: done {len(durations)} durations x {len(METHODS)} methods")
    return rows


def run_order_sweep(datasets, duration_s, orders, chunk_s, overlap_s, bp):
    rows = []
    for ds, files in datasets:
        path = files[0]
        seg = load_edf_segment(path, duration_s, ds["primary_channels"])
        for order in orders:
            for method in METHODS:
                rows.append(benchmark_one(seg, ds["key"], method, chunk_s, overlap_s,
                                          order, bp, ds["notch_hz"]))
        print(f"  [order-sweep:{ds['label']}] {path.name}: {len(orders)} orders x {len(METHODS)} methods")
    return rows


def run_channel_sweep(datasets, duration_s, channel_counts, order, chunk_s, overlap_s, bp):
    # Prefer the highest-channel dataset (CHB-MIT) for the channel sweep.
    ds, files = max(datasets, key=lambda df: df[0]["primary_channels"])
    path = files[0]
    rows = []
    for n_ch in channel_counts:
        seg = load_edf_segment(path, duration_s, n_ch)
        if seg["signals_uv"].shape[0] < n_ch:
            continue  # file does not have that many channels
        for method in METHODS:
            rows.append(benchmark_one(seg, ds["key"], method, chunk_s, overlap_s,
                                      order, bp, ds["notch_hz"]))
    print(f"  [channel-sweep:{ds['label']}] {path.name}: "
          f"{len(channel_counts)} channel counts x {len(METHODS)} methods")
    return rows


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_primary_outputs(rows, durations, order) -> None:
    _write_csv(RESULTS / "edf_length_benchmark.csv", rows)

    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    proc = _agg(rows, ("dataset", "duration_s", "method"), "processing_s")
    rtf = _agg(rows, ("dataset", "duration_s", "method"), "realtime_factor")
    lat = _agg(rows, ("dataset", "duration_s", "method"), "per_window_latency_ms")
    sph = _agg(rows, ("dataset", "duration_s", "method"), "seconds_per_eeg_hour")
    safe = _agg(rows, ("dataset", "duration_s", "method"), "used_filtfilt_safe")

    def nearest(table, ds, duration_s, method):
        cands = [kk for kk in table if kk[0] == ds and kk[2] == method
                 and abs(kk[1] - duration_s) <= 1.0]
        return table[cands[0]] if cands else None

    lines = ["EDF-duration benchmark - end-to-end filtering + Welch band-power features.",
             f"FIR order {order}; aggregated as mean +/- SD across files per (dataset, duration, method)."]
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        n_files = len(set(r["source_name"] for r in ds_rows))
        fs = ds_rows[0]["fs"]
        chans = ds_rows[0]["channels"]
        sched = next((r for r in ds_rows if r["method"] == "scheduled_overlap"), ds_rows[0])
        lines += [
            "",
            f"=== {ds}: {n_files} files, fs={fs} Hz, {chans} channels ===",
            f"  padlen={sched['padlen_samples']} samples; shortest zero-phase input="
            f"{sched['min_valid_samples']} samples; requested overlap={sched['requested_overlap_s']}s "
            f"-> scheduled overlap={sched['used_overlap_s']}s.",
        ]
        for duration_s in durations:
            lines.append(f"  duration {duration_s:g}s:")
            for method in METHODS:
                p = nearest(proc, ds, duration_s, method)
                if p is None:
                    continue
                pm, ps, n = p
                rm, _, _ = nearest(rtf, ds, duration_s, method)
                lm, ls, _ = nearest(lat, ds, duration_s, method)
                hm, _, _ = nearest(sph, ds, duration_s, method)
                sm, _, _ = nearest(safe, ds, duration_s, method)
                lines.append(
                    f"    {method:17s}: processing={pm:.3f}+/-{ps:.3f}s, RTF={rm:.5f}, "
                    f"per-window latency={lm:.2f}+/-{ls:.2f}ms, {hm:.1f}s/EEG-hour, "
                    f"safe={'yes' if sm >= 0.5 else 'no '} (n={n})")
    lines += [
        "",
        "Notes:",
        "  - realtime_factor is whole-segment throughput (processing_s / duration_s).",
        "  - per_window_latency_ms is the amortized cost of one display window - the interactive",
        "    responsiveness proxy, which is the metric that matters for a streaming viewer.",
        "  - scheduled_overlap is the deployable guardrail: the only streaming path that stays in",
        "    the zero-phase regime; its extra cost over fixed_overlap is the price of correctness.",
    ]
    (RESULTS / "edf_length_benchmark_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Figure: duration scaling per dataset, error bars across files, one panel per dataset.
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.8, 2.8), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        for method in METHODS:
            xs, ys, es = [], [], []
            for duration_s in durations:
                p = nearest(proc, ds, duration_s, method)
                if p is None:
                    continue
                m, s, _ = p
                xs.append(duration_s); ys.append(m); es.append(s)
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                            label=METHOD_LABELS[method])
        ax.set_xlabel("duration (s)")
        ax.set_ylabel("processing time (s)")
        ax.set_title(ds, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout(pad=0.35)
    fig.savefig(RESULTS / "fig_edf_length_benchmark.png", dpi=300)
    plt.close(fig)


def write_order_outputs(rows) -> None:
    if not rows:
        return
    _write_csv(RESULTS / "edf_order_sweep.csv", rows)
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    plt.figure(figsize=(3.15, 2.0))
    for ds in datasets:
        for method in ["scheduled_overlap", "whole_signal"]:
            sub = sorted((r for r in rows if r["dataset"] == ds and r["method"] == method),
                         key=lambda r: r["order"])
            if sub:
                plt.plot([r["order"] for r in sub], [r["processing_s"] for r in sub],
                         "o-", label=f"{ds}:{METHOD_LABELS[method]}")
    plt.xlabel("FIR order")
    plt.ylabel("processing time (s)")
    plt.title("FIR order", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=6.5)
    plt.tight_layout(pad=0.25)
    plt.savefig(RESULTS / "fig_edf_order_scaling.png", dpi=300)
    plt.close()


def write_channel_outputs(rows) -> None:
    if not rows:
        return
    _write_csv(RESULTS / "edf_channel_sweep.csv", rows)
    plt.figure(figsize=(3.15, 2.0))
    for method in METHODS:
        sub = sorted((r for r in rows if r["method"] == method), key=lambda r: r["channels"])
        if sub:
            plt.plot([r["channels"] for r in sub], [r["processing_s"] for r in sub],
                     "o-", label=METHOD_LABELS[method])
    plt.xlabel("channels")
    plt.ylabel("processing time (s)")
    plt.title("Channel count", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=6.5)
    plt.tight_layout(pad=0.25)
    plt.savefig(RESULTS / "fig_edf_channel_scaling.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EDF processing across durations, "
                                                 "FIR orders, channel counts, and datasets.")
    parser.add_argument("--durations", default=",".join(str(d) for d in DEFAULT_DURATIONS),
                        help="Comma-separated segment durations in seconds (primary sweep).")
    parser.add_argument("--orders", default=",".join(str(o) for o in DEFAULT_ORDERS),
                        help="Comma-separated FIR orders (order sub-sweep).")
    parser.add_argument("--channel-counts", default=",".join(str(c) for c in DEFAULT_CHANNEL_COUNTS),
                        help="Comma-separated channel counts (channel sub-sweep).")
    parser.add_argument("--order", type=int, default=200, help="FIR order for the primary sweep.")
    parser.add_argument("--order-sweep-duration", type=float, default=120.0)
    parser.add_argument("--channel-sweep-duration", type=float, default=120.0)
    parser.add_argument("--synthetic-fs", type=int, default=200)
    parser.add_argument("--chunk-s", type=float, default=1.0)
    parser.add_argument("--overlap-s", type=float, default=0.5)
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=30.0)
    args = parser.parse_args()

    durations = _parse_floats(args.durations)
    orders = _parse_ints(args.orders)
    channel_counts = _parse_ints(args.channel_counts)
    bp = (args.lowcut, args.highcut)

    datasets = discover_datasets()
    if not datasets:
        print("No EDFs found under data/; running synthetic fallback.")
        rows = []
        for duration_s in durations:
            seg = load_synthetic_segment(duration_s, args.synthetic_fs, 4)
            for method in METHODS:
                rows.append(benchmark_one(seg, "synthetic", method, args.chunk_s, args.overlap_s,
                                          args.order, bp, 50.0))
        write_primary_outputs(rows, durations, args.order)
        return

    print(f"Datasets found: {', '.join(ds['label'] + f' (n={len(f)})' for ds, f in datasets)}")
    print("PRIMARY sweep (files x durations x methods)...")
    primary = run_primary(datasets, durations, args.order, args.chunk_s, args.overlap_s, bp)
    write_primary_outputs(primary, durations, args.order)

    print("ORDER sub-sweep...")
    order_rows = run_order_sweep(datasets, args.order_sweep_duration, orders,
                                 args.chunk_s, args.overlap_s, bp)
    write_order_outputs(order_rows)

    print("CHANNEL sub-sweep...")
    channel_rows = run_channel_sweep(datasets, args.channel_sweep_duration, channel_counts,
                                     args.order, args.chunk_s, args.overlap_s, bp)
    write_channel_outputs(channel_rows)

    print("\nWrote edf_length_benchmark.csv, edf_length_benchmark_summary.txt, "
          "edf_order_sweep.csv, edf_channel_sweep.csv, and 3 figures under results/.")


if __name__ == "__main__":
    main()
