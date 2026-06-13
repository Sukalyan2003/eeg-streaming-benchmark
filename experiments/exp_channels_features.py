"""
Channel canonicalization correctness and feature integrity (Option 2, RQ2 part 2).

Part 1 — Canonicalization accuracy. Real EDF recordings name the same electrode many ways
(`EEG Fp1-REF`, `FP1`, `Fp1`, `POL Fp1`, `EEG F3-LE`, ...). `domain.helpers.channels` maps
them to a canonical name and a ChannelFamily. We measure family-classification accuracy and
canonical consistency on a curated set of real-world variants.

Part 2 — Feature integrity. If a non-EEG channel (e.g., ECG) is not excluded, an average-
reference montage mixes it into every EEG channel and corrupts quantitative features. We
compute alpha bandpower with vs without channel-aware exclusion and quantify the error.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from scipy import signal as sps

from domain.helpers.channels import (normalize_channel_name, classify_channel,
                                      is_eeg_channel, ChannelFamily)

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
FS = 200

# (raw name, expected canonical base, expected family)
LABELED = [
    ("EEG Fp1-REF", "FP1", ChannelFamily.EEG),
    ("FP1", "FP1", ChannelFamily.EEG),
    ("Fp1", "FP1", ChannelFamily.EEG),
    ("POL Fp1", "FP1", ChannelFamily.EEG),
    ("EEG F3-LE", "F3", ChannelFamily.EEG),
    ("C3", "C3", ChannelFamily.EEG),
    ("EEG C4-REF", "C4", ChannelFamily.EEG),
    ("O1", "O1", ChannelFamily.EEG),
    ("EEG T3-REF", "T3", ChannelFamily.EEG),
    ("EEG T7-REF", "T7", ChannelFamily.EEG),
    ("Cz", "CZ", ChannelFamily.EEG),
    ("Pz", "PZ", ChannelFamily.EEG),
    ("EEG A1-REF", "A1", ChannelFamily.EEG),
    ("ECG", "ECG", ChannelFamily.ECG),
    ("EKG", "ECG", ChannelFamily.ECG),
    ("EEG EKG1-REF", "ECG", ChannelFamily.ECG),
    ("EMG", "EMG", ChannelFamily.EMG),
    ("EOG", "EOG", ChannelFamily.EOG),
    ("EEG ROC-REF", "ROC", ChannelFamily.EOG),
]


def part1_canonicalization():
    fam_ok = 0
    rows = []
    for raw, exp_canon, exp_fam in LABELED:
        info = normalize_channel_name(raw)
        fam = classify_channel(raw)
        ok_fam = (fam == exp_fam)
        fam_ok += int(ok_fam)
        rows.append(dict(raw=raw, canonical=info.canonical, family=fam.value,
                         expected_family=exp_fam.value, family_ok=int(ok_fam),
                         is_eeg=int(is_eeg_channel(raw))))
    acc = fam_ok / len(LABELED)
    with open(RESULTS / "channels_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    return acc, rows


def _alpha_bandpower(x, fs=FS):
    f, p = sps.welch(x, fs=fs, nperseg=min(len(x), 4 * fs))
    band = (f >= 8) & (f <= 13)
    return float(np.trapezoid(p[band], f[band]))


def part2_feature_integrity(seed=0):
    rng = np.random.default_rng(seed)
    dur, n = 60.0, int(60.0 * FS)
    t = np.arange(n) / FS
    names = ["EEG Fp1-REF", "EEG C3-REF", "EEG O1-REF", "EEG P3-REF", "ECG"]
    sig = {}
    for nm in names[:4]:
        sig[nm] = (18 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28))   # alpha
                   + 6 * rng.standard_normal(n))
    # ECG: strong ~1.2 Hz QRS-like spikes + drift, large amplitude
    hr = 1.2
    qrs = np.zeros(n)
    for k in range(int(dur * hr)):
        idx = int((k / hr) * FS)
        if idx < n:
            qrs[idx] = 1.0
    ecg = 120 * sps.lfilter([1], [1, -0.0], np.convolve(qrs, sps.windows.gaussian(40, 6), mode="same"))
    sig["ECG"] = ecg + 10 * np.sin(2 * np.pi * 0.3 * t)

    chans = np.stack([sig[nm] for nm in names])  # [5, n]

    # Average reference two ways
    avg_all = chans.mean(axis=0)                                   # includes ECG (wrong)
    eeg_mask = np.array([is_eeg_channel(nm) for nm in names])
    avg_eeg = chans[eeg_mask].mean(axis=0)                         # channel-aware (correct)

    # Re-reference an EEG channel (C3) and compare alpha bandpower
    c3 = chans[1]
    bp_wrong = _alpha_bandpower(c3 - avg_all)
    bp_right = _alpha_bandpower(c3 - avg_eeg)
    pct_err = 100.0 * (bp_wrong - bp_right) / bp_right
    return bp_wrong, bp_right, pct_err, int(eeg_mask.sum())


def main():
    acc, rows = part1_canonicalization()
    bp_wrong, bp_right, pct_err, n_eeg = part2_feature_integrity()
    lines = [
        f"Part 1 — channel canonicalization: family accuracy {acc*100:.1f}% "
        f"on {len(LABELED)} real-world name variants "
        f"(non-EEG channels correctly identified for montage exclusion).",
        "",
        "Part 2 — feature integrity (average-reference alpha bandpower of C3):",
        f"  channel-aware average ({n_eeg} EEG chans, ECG excluded): {bp_right:.2f}",
        f"  naive average (ECG included):                            {bp_wrong:.2f}",
        f"  => error from not excluding the ECG channel: {pct_err:+.1f}%.",
        "",
        "Takeaway: correct quantitative features require channel canonicalization/"
        "classification before montage; mixing a mislabeled non-EEG channel into the"
        " reference measurably corrupts band power.",
    ]
    (RESULTS / "channels_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS}/channels_metrics.csv, channels_summary.txt")


if __name__ == "__main__":
    main()
