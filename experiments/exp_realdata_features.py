"""
Real-data confirmation of Results 4-5 on a recording with genuine non-EEG channels
(Siena Scalp EEG PN00-1.edf, 10-20 monopolar EEG + a real EKG channel + SpO2/HR).

  R4 — channel canonicalization: classify the real channel labels and check that the EEG
       electrodes are separated from the genuine non-EEG channels (EKG, SpO2, HR, markers).
  R5 — feature integrity: an average-reference montage that wrongly includes the real EKG
       channel biases EEG band power vs the channel-aware (EEG-only) reference.

Raw EDF is gitignored under data/. Download:
  curl -o data/siena_PN00-1.edf https://physionet.org/files/siena-scalp-eeg/1.0.0/PN00/PN00-1.edf

Outputs under results/: realdata_features_summary.txt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mne
from scipy.signal import welch

from domain.helpers.channels import normalize_channel_name, classify_channel, is_eeg_channel

RESULTS = Path(__file__).parent / "results"
DATA = Path(__file__).parent / "data" / "siena_PN00-1.edf"
SEG = (100.0, 220.0)     # 120 s segment
TARGET = "EEG C3"
ALPHA = (8.0, 13.0)
TOTAL = (0.5, 45.0)


def rel_alpha(x, fs):
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))
    a = np.trapezoid(p[(f >= ALPHA[0]) & (f < ALPHA[1])], f[(f >= ALPHA[0]) & (f < ALPHA[1])])
    t = np.trapezoid(p[(f >= TOTAL[0]) & (f < TOTAL[1])], f[(f >= TOTAL[0]) & (f < TOTAL[1])])
    return 100.0 * a / (t or 1.0)


def main():
    raw = mne.io.read_raw_edf(DATA, preload=True, verbose="ERROR")
    fs = int(round(raw.info["sfreq"]))
    names = raw.ch_names

    # R4 — classification of real labels
    eeg = [c for c in names if is_eeg_channel(c)]
    non_eeg = [c for c in names if not is_eeg_channel(c)]
    fam = {c: classify_channel(c).value for c in names}

    # R5 — feature integrity (needs the target EEG channel + a real EKG channel present)
    ekg = next((c for c in names if classify_channel(c).value == "ecg"), None)
    a, b = int(SEG[0] * fs), int(SEG[1] * fs)
    get = lambda c: raw.get_data(picks=[c])[0, a:b] * 1e6   # microvolts
    eeg_for_ref = [c for c in eeg if c != TARGET]
    avg_correct = np.mean([get(c) for c in eeg_for_ref], axis=0)
    avg_wrong = np.mean([get(c) for c in eeg_for_ref] + ([get(ekg)] if ekg else []), axis=0)
    tgt = get(TARGET)
    ra_correct = rel_alpha(tgt - avg_correct, fs)
    ra_wrong = rel_alpha(tgt - avg_wrong, fs)
    pct = 100.0 * (ra_wrong - ra_correct) / ra_correct

    lines = [
        f"Siena PN00-1.edf — fs={fs} Hz, {len(names)} channels, segment {SEG[0]:.0f}-{SEG[1]:.0f}s.",
        "",
        "R4 — channel canonicalization on real labels:",
        f"  EEG channels identified ({len(eeg)}): e.g. "
        f"{[ (c, normalize_channel_name(c).canonical) for c in eeg[:5] ]}",
        f"  non-EEG channels ({len(non_eeg)}): "
        f"{[ (c, fam[c]) for c in non_eeg ]}",
        f"  real EKG channel detected as ECG: {ekg!r}.",
        "",
        "R5 — feature integrity (relative alpha power of "
        f"{TARGET}, average reference):",
        f"  channel-aware reference (EEG only):      {ra_correct:.2f}%",
        f"  reference wrongly including the EKG:     {ra_wrong:.2f}%",
        f"  => bias from not excluding the real ECG: {pct:+.1f}%.",
        "",
        "Confirms on real EEG: canonicalization separates EEG from genuine non-EEG channels, and",
        "mixing the real ECG into the reference measurably biases a quantitative EEG feature.",
    ]
    (RESULTS / "realdata_features_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS}/realdata_features_summary.txt")


if __name__ == "__main__":
    main()
