"""
Second-dataset confirmation of the windowed-filtering results on Sleep-EDF Expanded.

Sleep-EDF is sampled at 100 Hz (vs CHB-MIT's 256 Hz) and is a different population (sleep, not
epilepsy) with a different montage (Fpz-Cz / Pz-Oz). Confirming R1 (overlap-add seam-artifact
reduction) and R3 (causal-fallback group-delay shift) here shows the results are not specific to
one dataset, and makes the fs-dependence of the floor explicit: at 100 Hz the zero-phase padlen is
606 samples = 6.06 s, the shortest valid input is 607 samples, and the fallback group delay is
(taps-1)/2/fs ~ 1005 ms (vs 2.37 s / ~393 ms at 256 Hz).

For each subject's PSG recording this downloads the EDF to data/ if absent (kept; no redownload),
then measures the overlap-add boundary-RMSE reduction at 8 s windows and the fallback shift by
cross-correlation. Results are written incrementally so progress survives an interruption.

Outputs under results/: sleepedf_subjects.csv, sleepedf_summary.txt
"""
from __future__ import annotations
import csv
import urllib.request
from pathlib import Path
import numpy as np
import mne

from dsp import apply_bandpass, bandpass_coefficients
from windowing import filter_whole, filter_naive, filter_overlap, seam_indices

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
CSV = RESULTS / "sleepedf_subjects.csv"

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
SUBJECTS = ["SC4001E0", "SC4011E0", "SC4021E0", "SC4031E0",
            "SC4041E0", "SC4051E0", "SC4061E0", "SC4071E0"]
BP = (0.5, 30.0)
NOTCH = 50.0                 # Sleep-EDF is European data -> 50 Hz mains
SEG = (600.0, 720.0)         # 120 s segment
WIN_S = 8.0
PREF_CHANS = ["EEG Fpz-Cz", "EEG Pz-Oz"]


def rms(a): return float(np.sqrt(np.mean(np.square(a))))


def edf_path(sub): return DATA / f"{sub}-PSG.edf"


def ensure_download(sub) -> bool:
    p = edf_path(sub)
    if p.exists() and p.stat().st_size > 1_000_000:
        return True
    url = f"{BASE}/{sub}-PSG.edf"
    tmp = p.with_suffix(".part")
    try:
        print(f"[{sub}] downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(p)
        print(f"[{sub}] saved {p.stat().st_size/1e6:.0f} MB", flush=True)
        return True
    except Exception as e:                      # noqa: BLE001
        print(f"[{sub}] download failed: {e}", flush=True)
        if tmp.exists():
            tmp.unlink()
        return False


def seam_interior_rms(err, seams, n, fs):
    sh = int(0.05 * fs); guard = int(0.6 * fs)
    sm = np.zeros(n, bool); gm = np.zeros(n, bool)
    for s in seams:
        sm[max(0, s - sh):min(n, s + sh)] = True
        gm[max(0, s - guard):min(n, s + guard)] = True
    return rms(err[sm]), rms(err[~gm])


def process(sub) -> dict | None:
    raw = mne.io.read_raw_edf(edf_path(sub), preload=True, verbose="ERROR")
    fs = int(round(raw.info["sfreq"]))
    chans = [c for c in PREF_CHANS if c in raw.ch_names]
    if not chans:
        chans = [c for c in raw.ch_names if c.upper().startswith("EEG")][:2] or raw.ch_names[:2]
    a, b = int(SEG[0] * fs), int(SEG[1] * fs)
    if b > raw.n_times:
        return None
    sig = raw.get_data(picks=chans)[:, a:b] * 1e6     # microvolts
    n = sig.shape[1]
    (taps,) = bandpass_coefficients(float(fs), BP[0], BP[1], 200)
    floor = 3 * len(taps)

    # R1: overlap-add vs naive at 8 s windows (zero-phase regime: 8 s = 800 samples > floor)
    seams = seam_indices(n, fs, WIN_S)
    gt = np.stack([filter_whole(sig[c], fs, BP, NOTCH) for c in range(len(chans))])
    naive = np.stack([filter_naive(sig[c], fs, WIN_S, BP, NOTCH) for c in range(len(chans))])
    olap = np.stack([filter_overlap(sig[c], fs, WIN_S, 0.5, BP, NOTCH) for c in range(len(chans))])
    nb, _ = np.mean([seam_interior_rms(naive[c] - gt[c], seams, n, fs) for c in range(len(chans))], axis=0)
    ob, oi = np.mean([seam_interior_rms(olap[c] - gt[c], seams, n, fs) for c in range(len(chans))], axis=0)
    db = 20 * np.log10(nb / (ob + 1e-12))

    # R3: fallback group-delay shift. Use a sub-floor window long enough to contain the delay
    # (4 s = 400 samples < floor 606 at 100 Hz, and >> the ~100-sample group delay). Median over
    # channels so a low-in-band-energy channel cannot skew the estimate.
    short = int(4.0 * fs)
    mid = n // 2
    theo_ms = (len(taps) - 1) / 2 / fs * 1e3
    per_chan = []
    for c in range(len(chans)):
        win = sig[c][mid - short // 2: mid + short // 2]
        y_fb = apply_bandpass(win, fs, BP[0], BP[1])
        y_zero = gt[c][mid - short // 2: mid + short // 2]
        yfb = y_fb - y_fb.mean(); yz = y_zero - y_zero.mean()
        if rms(yz) < 1e-6:
            continue
        xcorr = np.correlate(yfb, yz, mode="full")
        zero_lag = len(yz) - 1
        hi = min(len(taps), len(xcorr) - 1 - zero_lag)
        pos = xcorr[zero_lag + 1: zero_lag + 1 + hi]
        per_chan.append((int(np.argmax(pos)) + 1) / fs * 1e3)
    shift_ms = float(np.median(per_chan)) if per_chan else float("nan")

    return dict(subject=sub, fs=fs, n_channels=len(chans),
                floor_samples=floor, floor_s=round(floor / fs, 3),
                naive_boundary_uv=round(nb, 4), overlap_boundary_uv=round(ob, 5),
                interior_uv=round(oi, 5), db_reduction=round(db, 2),
                fallback_shift_ms=round(shift_ms, 1), theory_shift_ms=round(theo_ms, 1))


def main():
    rows = []
    if CSV.exists():
        with open(CSV) as fh:
            rows = list(csv.DictReader(fh))
    done = {r["subject"] for r in rows}

    for sub in SUBJECTS:
        if sub in done:
            print(f"[{sub}] already processed, skipping", flush=True)
            continue
        if not ensure_download(sub):
            continue
        try:
            r = process(sub)
        except Exception as e:                  # noqa: BLE001
            print(f"[{sub}] processing failed: {e}", flush=True)
            continue
        if r is None:
            print(f"[{sub}] segment unavailable, skipping", flush=True)
            continue
        rows.append({k: str(v) for k, v in r.items()})
        with open(CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            w.writerows(rows)
        print(f"[{sub}] fs={r['fs']}  dB={r['db_reduction']}  "
              f"shift={r['fallback_shift_ms']}ms (theory {r['theory_shift_ms']})", flush=True)

    if not rows:
        print("no subjects processed", flush=True)
        return
    db = np.array([float(r["db_reduction"]) for r in rows])
    shift = np.array([float(r["fallback_shift_ms"]) for r in rows])
    theo = np.array([float(r["theory_shift_ms"]) for r in rows])
    err = np.abs(shift - theo)
    floor_s = float(rows[0]["floor_s"])
    lines = [
        f"Second-dataset confirmation - Sleep-EDF Expanded (n={len(rows)} subjects, 100 Hz, "
        f"8 s windows):",
        "",
        f"  zero-phase padlen = {rows[0]['floor_samples']} samples = {floor_s} s at 100 Hz; "
        f"shortest valid input = {int(float(rows[0]['floor_samples'])) + 1} samples "
        f"(vs 2.37 s at 256 Hz).",
        f"  R1 overlap-add boundary-RMSE reduction = {db.mean():.1f} +/- {db.std():.1f} dB",
        f"  R3 fallback group-delay shift = {shift.mean():.0f} +/- {shift.std():.0f} ms "
        f"(theory {theo.mean():.0f} ms; |measured-theory| = {err.mean():.1f} +/- {err.std():.1f} ms)",
        "",
        "Per-subject:",
    ] + [f"  {r['subject']}: dB={r['db_reduction']}, shift={r['fallback_shift_ms']} ms "
         f"(theory {r['theory_shift_ms']})" for r in rows]
    (RESULTS / "sleepedf_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
