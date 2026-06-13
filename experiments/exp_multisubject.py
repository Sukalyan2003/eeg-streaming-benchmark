"""
Multi-subject confirmation of the windowed-filtering results on CHB-MIT scalp EEG.

Addresses the single-recording limitation: instead of one file, this confirms R1 (overlap-add
seam-artifact reduction) and R3 (fallback group-delay event mis-timing) across many PhysioNet
CHB-MIT subjects and reports mean +/- SD. For each subject's first recording it:
  - downloads the EDF to data/ if not already present (no redownload; files are kept), then
  - measures the overlap-add boundary-RMSE reduction (dB) at 8 s windows, and
  - measures the causal-fallback group-delay shift by cross-correlation vs theory.

Results are written incrementally to results/multisubject_chbmit.csv so progress survives an
interruption. Run in the background:
  python exp_multisubject.py

Outputs under results/: multisubject_chbmit.csv, multisubject_summary.txt
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
CSV = RESULTS / "multisubject_chbmit.csv"

BASE = "https://physionet.org/files/chbmit/1.0.0"
SUBJECTS = [f"chb{n:02d}" for n in range(1, 13)]    # chb01..chb12, first recording each
BP = (0.5, 30.0)
NOTCH = 60.0                  # CHB-MIT is US data -> 60 Hz mains
SEG = (600.0, 720.0)         # 120 s segment away from the start
WIN_S = 8.0                  # zero-phase regime
PREF_CHANS = ["FP1-F7", "F7-T7", "F3-C3", "C3-P3", "P3-O1", "FP2-F8"]


def rms(a): return float(np.sqrt(np.mean(np.square(a))))


def edf_path(sub): return DATA / f"{sub}_01.edf"


def ensure_download(sub) -> bool:
    """Download <sub>_01.edf if absent. Keep the file. Returns True if available."""
    p = edf_path(sub)
    if p.exists() and p.stat().st_size > 1_000_000:
        return True
    url = f"{BASE}/{sub}/{sub}_01.edf"
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
    except Exception as e:                      # noqa: BLE001 - skip unreachable subjects
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
    if len(chans) < 4:
        chans = raw.ch_names[:4]
    chans = chans[:4]
    a, b = int(SEG[0] * fs), int(SEG[1] * fs)
    if b > raw.n_times:
        return None
    sig = raw.get_data(picks=chans)[:, a:b] * 1e6     # microvolts
    n = sig.shape[1]
    (taps,) = bandpass_coefficients(float(fs), BP[0], BP[1], 200)
    floor = 3 * len(taps)

    # R1: overlap-add vs naive at 8 s windows
    seams = seam_indices(n, fs, WIN_S)
    gt = np.stack([filter_whole(sig[c], fs, BP, NOTCH) for c in range(len(chans))])
    naive = np.stack([filter_naive(sig[c], fs, WIN_S, BP, NOTCH) for c in range(len(chans))])
    olap = np.stack([filter_overlap(sig[c], fs, WIN_S, 0.5, BP, NOTCH) for c in range(len(chans))])
    nb, _ = np.mean([seam_interior_rms(naive[c] - gt[c], seams, n, fs) for c in range(len(chans))], axis=0)
    ob, oi = np.mean([seam_interior_rms(olap[c] - gt[c], seams, n, fs) for c in range(len(chans))], axis=0)
    db = 20 * np.log10(nb / (ob + 1e-12))

    # R3: fallback group-delay shift (1 s window -> causal path). The causal FIR group delay is
    # positive and bounded by the filter length, so we recover the lag by cross-correlation
    # restricted to positive lags within one filter length [1, taps]. We estimate it on every
    # channel and take the median, so a single low-in-band-energy channel cannot skew the result.
    short = int(1.0 * fs)
    mid = n // 2
    theo_ms = (len(taps) - 1) / 2 / fs * 1e3
    per_chan = []
    for c in range(len(chans)):
        win = sig[c][mid - short // 2: mid + short // 2]
        y_fb = apply_bandpass(win, fs, BP[0], BP[1])
        y_zero = gt[c][mid - short // 2: mid + short // 2]
        yfb = y_fb - y_fb.mean(); yz = y_zero - y_zero.mean()
        if rms(yz) < 1e-6:                                 # flat segment -> unreliable
            continue
        xcorr = np.correlate(yfb, yz, mode="full")
        zero_lag = len(yz) - 1
        hi = min(len(taps), len(xcorr) - 1 - zero_lag)
        pos = xcorr[zero_lag + 1: zero_lag + 1 + hi]       # lags 1..hi (positive only)
        per_chan.append((int(np.argmax(pos)) + 1) / fs * 1e3)
    shift_ms = float(np.median(per_chan)) if per_chan else float("nan")

    return dict(subject=sub, fs=fs, n_channels=len(chans),
                floor_samples=floor, floor_s=round(floor / fs, 3),
                naive_boundary_uv=round(nb, 4), overlap_boundary_uv=round(ob, 5),
                interior_uv=round(oi, 5), db_reduction=round(db, 2),
                fallback_shift_ms=round(shift_ms, 1), theory_shift_ms=round(theo_ms, 1))


def main():
    rows = []
    # resume: keep any subjects already in the CSV
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
        with open(CSV, "w", newline="") as fh:    # rewrite incrementally
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            w.writerows(rows)
        print(f"[{sub}] fs={r['fs']}  dB={r['db_reduction']}  "
              f"shift={r['fallback_shift_ms']}ms (theory {r['theory_shift_ms']})", flush=True)

    # aggregate
    if not rows:
        print("no subjects processed", flush=True)
        return
    db = np.array([float(r["db_reduction"]) for r in rows])
    nb = np.array([float(r["naive_boundary_uv"]) for r in rows])
    ob = np.array([float(r["overlap_boundary_uv"]) for r in rows])
    shift = np.array([float(r["fallback_shift_ms"]) for r in rows])
    theo = np.array([float(r["theory_shift_ms"]) for r in rows])
    err = np.abs(shift - theo)
    lines = [
        f"Multi-subject CHB-MIT confirmation (n={len(rows)} subjects, first recording, 8 s windows):",
        "",
        f"  R1 overlap-add boundary-RMSE reduction = {db.mean():.1f} +/- {db.std():.1f} dB",
        f"     naive boundary RMSE   = {nb.mean():.2f} +/- {nb.std():.2f} uV",
        f"     overlap boundary RMSE = {ob.mean():.4f} +/- {ob.std():.4f} uV",
        f"  R3 fallback group-delay shift = {shift.mean():.0f} +/- {shift.std():.0f} ms "
        f"(theory {theo.mean():.0f} ms; |measured-theory| = {err.mean():.1f} +/- {err.std():.1f} ms)",
        "",
        "Per-subject:",
    ] + [f"  {r['subject']}: fs={r['fs']} Hz, dB={r['db_reduction']}, "
         f"shift={r['fallback_shift_ms']} ms (theory {r['theory_shift_ms']})" for r in rows]
    (RESULTS / "multisubject_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
