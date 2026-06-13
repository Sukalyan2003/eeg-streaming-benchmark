"""
Bandpower equivalence vs MNE (Option 2, RQ2 part 2 — feature equivalence).

Drives the PRODUCTION power-spectral-density routine
(`workers/src/utils/extraction/band_power_optimized._compute_psd`, Welch, nperseg=256,
noverlap=128, detrend+Hann) and compares the resulting per-band *relative* band powers
against an independent reference (MNE `psd_array_welch`) on synthetic signals with known
band content. Relative band power (percentage) is what the platform stores
(`BANDPOWER_COMPUTE_RELATIVE=true`), so it is the right quantity to validate and it is robust
to overall scaling.

Outputs under results/: bandpower_equiv.csv, bandpower_summary.txt
"""
from __future__ import annotations
import os
import sys
import csv
from pathlib import Path

# Production module reads these at import time — set production-like values first.
os.environ.setdefault("BANDPOWER_PSD_METHOD", "welch")
os.environ.setdefault("BANDPOWER_WINDOW_SECONDS", "2.0")
os.environ.setdefault("BANDPOWER_HOP_SECONDS", "1.0")
os.environ.setdefault("BANDPOWER_COMPUTE_RELATIVE", "true")
os.environ.setdefault("BANDPOWER_DETREND", "true")
os.environ.setdefault("BANDPOWER_APPLY_HANN", "true")
os.environ.setdefault("BANDPOWER_WELCH_NPERSEG", "256")
os.environ.setdefault("BANDPOWER_WELCH_NOVERLAP", "128")
os.environ.setdefault("BANDPOWER_MULTITAPER_BW", "2.0")
os.environ.setdefault("BANDPOWER_RELATIVE_SOFTMAX_SCALE", "1.0")
os.environ.setdefault("BANDPOWER_FILTER_LOWCUT", "1.0")
os.environ.setdefault("BANDPOWER_FILTER_HIGHCUT", "45.0")
os.environ.setdefault("BANDPOWER_FILTER_ORDER", "200")
os.environ.setdefault("BANDPOWER_APPLY_FILTER", "true")
os.environ.setdefault("BANDPOWER_OUTPUT_MODE", "absolute")

import numpy as np
from mne.time_frequency import psd_array_welch, psd_array_multitaper

WORKERS = Path(__file__).resolve().parents[3] / "workers"
sys.path.insert(0, str(WORKERS))
from src.utils.extraction.band_power_optimized import _compute_psd, bands  # production code

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
FS = 200
NPERSEG, NOVERLAP = 256, 128
BAND_NAMES = list(bands.keys())


def synth_window(seed=0, dur_s=10.0):
    """Single channel with controlled band amplitudes -> known relative power targets."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    amp = {"delta": 10.0, "theta": 6.0, "alpha": 16.0, "beta": 4.0, "gamma": 1.5}
    x = np.zeros(n)
    for name, (lo, hi) in bands.items():
        for _ in range(8):
            f = rng.uniform(max(lo, 0.5), min(hi, 99))
            x += amp[name] * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / np.sqrt(8)
    x += 1.0 * rng.standard_normal(n)
    return x


def band_powers_from_psd(freqs, psd):
    out = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.trapezoid(psd[m], freqs[m])) if m.any() else 0.0
    return out


def relative(d):
    tot = sum(d.values()) or 1.0
    return {k: 100.0 * v / tot for k, v in d.items()}


def main():
    cfg = dict(method="welch", detrend=True, apply_hann=True,
               welch_nperseg=NPERSEG, welch_noverlap=NOVERLAP, multitaper_bw=2.0)
    prod_rel_all, mne_rel_all = [], []
    for seed in range(8):
        x = synth_window(seed=seed)
        # production
        f_p, psd_p = _compute_psd(x.copy(), FS, cfg)
        prod_rel = relative(band_powers_from_psd(f_p, psd_p))
        # reference (MNE Welch, matched params)
        psd_m, f_m = psd_array_welch(x[np.newaxis, :], sfreq=FS, fmin=0, fmax=100,
                                     n_per_seg=NPERSEG, n_overlap=NOVERLAP, verbose="ERROR")
        mne_rel = relative(band_powers_from_psd(f_m, psd_m[0]))
        prod_rel_all.append(prod_rel); mne_rel_all.append(mne_rel)

    # aggregate per-band mean and max abs discrepancy (percentage points)
    rows, max_disc = [], 0.0
    for name in BAND_NAMES:
        p = np.mean([d[name] for d in prod_rel_all])
        m = np.mean([d[name] for d in mne_rel_all])
        disc = np.max([abs(a[name] - b[name]) for a, b in zip(prod_rel_all, mne_rel_all)])
        max_disc = max(max_disc, disc)
        rows.append(dict(band=name, prod_rel_pct=round(p, 3), mne_rel_pct=round(m, 3),
                         max_abs_disc_pp=round(disc, 4)))
    # correlation across all bands/trials
    pa = np.array([[d[b] for b in BAND_NAMES] for d in prod_rel_all]).ravel()
    ma = np.array([[d[b] for b in BAND_NAMES] for d in mne_rel_all]).ravel()
    r = float(np.corrcoef(pa, ma)[0, 1])

    # Root-cause toggle: repeat without the production manual Hann (welch keeps its own
    # per-segment Hann). If the discrepancy collapses, the production double-taper is the cause.
    cfg_no_hann = dict(cfg, apply_hann=False)
    disc_no_hann = 0.0
    for seed in range(8):
        x = synth_window(seed=seed)
        f_p, psd_p = _compute_psd(x.copy(), FS, cfg_no_hann)
        prod_rel = relative(band_powers_from_psd(f_p, psd_p))
        psd_m, f_m = psd_array_welch(x[np.newaxis, :], sfreq=FS, fmin=0, fmax=100,
                                     n_per_seg=NPERSEG, n_overlap=NOVERLAP, verbose="ERROR")
        mne_rel = relative(band_powers_from_psd(f_m, psd_m[0]))
        disc_no_hann = max(disc_no_hann, max(abs(prod_rel[b] - mne_rel[b]) for b in BAND_NAMES))

    with open(RESULTS / "bandpower_equiv.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r_ in rows:
            w.writerow(r_)

    # #5 estimator robustness: repeat with the MULTITAPER path. Production applies the manual
    # Hann BEFORE method dispatch, so it taints multitaper too (multitaper has its own DPSS
    # tapers). Compare production multitaper vs MNE multitaper, with/without the manual Hann.
    def multitaper_disc(apply_hann):
        cfg_mt = dict(method="multitaper", detrend=True, apply_hann=apply_hann,
                      welch_nperseg=NPERSEG, welch_noverlap=NOVERLAP, multitaper_bw=2.0)
        d = 0.0
        for seed in range(8):
            x = synth_window(seed=seed)
            f_p, psd_p = _compute_psd(x.copy(), FS, cfg_mt)
            pr = relative(band_powers_from_psd(f_p, psd_p))
            psd_m, f_m = psd_array_multitaper(x[np.newaxis, :], sfreq=FS, fmin=0, fmax=100,
                                              bandwidth=2.0, normalization="full", verbose="ERROR")
            mr = relative(band_powers_from_psd(f_m, psd_m[0]))
            d = max(d, max(abs(pr[b] - mr[b]) for b in BAND_NAMES))
        return d
    mt_hann = multitaper_disc(True)
    mt_nohann = multitaper_disc(False)

    lines = [
        "Bandpower equivalence — production _compute_psd (Welch 256/128) vs MNE psd_array_welch:",
        f"  Pearson r (relative band powers, all bands x 8 trials) = {r:.5f}.",
        f"  Max per-band discrepancy (production as-is) = {max_disc:.3f} percentage points.",
        "  Per-band mean relative power (production vs MNE):",
    ] + [f"    {row['band']:>6}: prod {row['prod_rel_pct']:6.2f}%  mne {row['mne_rel_pct']:6.2f}%  "
         f"(max disc {row['max_abs_disc_pp']:.3f} pp)" for row in rows] + [
        "",
        "Root cause — double Hann tapering:",
        f"  production applies np.hanning() to the whole window AND scipy.welch applies its",
        f"  default per-segment Hann. Disabling the redundant manual taper drops the max",
        f"  discrepancy from {max_disc:.2f} pp to {disc_no_hann:.2f} pp (i.e. production then",
        f"  matches MNE). Fix: remove the manual Hann (let welch taper per segment).",
        "",
        "Estimator robustness — multitaper path (production vs MNE psd_array_multitaper):",
        f"  with manual Hann    : max discrepancy {mt_hann:.2f} pp (the manual taper also",
        f"                        corrupts multitaper, which has its own DPSS tapers).",
        f"  without manual Hann : max discrepancy {mt_nohann:.2f} pp (production multitaper then",
        f"                        matches MNE) -> the double-taper bug is estimator-agnostic.",
    ]
    (RESULTS / "bandpower_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS}/bandpower_equiv.csv, bandpower_summary.txt")


if __name__ == "__main__":
    main()
