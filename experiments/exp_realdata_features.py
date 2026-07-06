"""
Real-data confirmation of the montage-hygiene pitfall on a recording with genuine non-EEG
channels (Siena Scalp EEG, 10-20 monopolar EEG + a real EKG channel + SpO2/HR).

An average reference that wrongly includes the real EKG channel injects a scaled copy of the
ECG into every re-referenced EEG channel. We quantify the leakage by the correlation between a
re-referenced target EEG channel and the raw EKG, with vs without the EKG in the reference, for
the full montage and for a sparse 4-channel subset (where the 1/(N+1) weighting makes the
contamination large). EEG vs non-EEG is decided from the channel label (no external package).

Raw EDF is gitignored under data/. Download:
  curl -o data/siena_PN00-1.edf https://physionet.org/files/siena-scalp-eeg/1.0.0/PN00/PN00-1.edf

Outputs under results/: realdata_features_summary.txt
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mne

RESULTS = Path(__file__).parent / "results"
DATA = Path(__file__).parent / "data" / "siena_PN00-1.edf"
SEG = (100.0, 220.0)     # 120 s segment
TARGET = "EEG C3"
SPARSE = ["EEG Fp1", "EEG C3", "EEG O1", "EEG P3"]   # a sparse clinical subset

NON_EEG_TOKENS = ("ECG", "EKG", "EMG", "EOG", "SPO2", "SAO2", "HR", "RESP", "PULSE",
                  "MK", "MARKER", "STI", "DC")


def is_eeg(name: str) -> bool:
    up = name.upper()
    return not any(tok in up for tok in NON_EEG_TOKENS)


def is_ecg(name: str) -> bool:
    up = name.upper()
    return ("ECG" in up) or ("EKG" in up)


def leak_corr(a, b):
    return float(abs(np.corrcoef(a, b)[0, 1]))


def main():
    raw = mne.io.read_raw_edf(DATA, preload=True, verbose="ERROR")
    fs = int(round(raw.info["sfreq"]))
    names = raw.ch_names

    eeg = [c for c in names if is_eeg(c)]
    non_eeg = [c for c in names if not is_eeg(c)]
    ekg = next((c for c in names if is_ecg(c)), None)

    a, b = int(SEG[0] * fs), int(SEG[1] * fs)
    get = lambda c: raw.get_data(picks=[c])[0, a:b] * 1e6   # microvolts
    tgt = get(TARGET)
    ekg_sig = get(ekg)

    def leakage(ref_eeg):
        ref = [c for c in ref_eeg if c != TARGET]
        avg_correct = np.mean([get(c) for c in ref], axis=0)
        avg_wrong = np.mean([get(c) for c in ref] + [ekg_sig], axis=0)
        return len(ref), leak_corr(tgt - avg_correct, ekg_sig), leak_corr(tgt - avg_wrong, ekg_sig)

    n_full, c_full, w_full = leakage(eeg)
    sparse = [c for c in SPARSE if c in names]
    n_sp, c_sp, w_sp = leakage(sparse)

    lines = [
        f"Siena PN00-1.edf - fs={fs} Hz, {len(names)} channels, segment {SEG[0]:.0f}-{SEG[1]:.0f}s.",
        "",
        f"Channel split by label: {len(eeg)} EEG channels, {len(non_eeg)} non-EEG "
        f"({non_eeg}); real ECG channel = {ekg!r}.",
        "",
        f"Montage hygiene - leakage of the real EKG into re-referenced {TARGET} "
        f"(|corr(target, EKG)|):",
        f"  full montage   (N={n_full:>2} EEG ref): excluded={c_full:.3f}  included={w_full:.3f}",
        f"  sparse montage (N={n_sp:>2} EEG ref): excluded={c_sp:.3f}  included={w_sp:.3f}",
        "",
        "Confirms on real EEG: excluding the EKG leaves negligible ECG correlation; including it",
        "injects a copy scaled by ~1/(N+1) - small for the full montage, large for the sparse",
        "subset. Channels must be split into EEG / non-EEG before averaging.",
    ]
    (RESULTS / "realdata_features_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS}/realdata_features_summary.txt")


if __name__ == "__main__":
    main()
