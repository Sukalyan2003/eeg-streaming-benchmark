# Option 2 (Non-AI / Classical-DSP) — Poster Plan

**Working title:** *Keeping Windowed EEG Filtering Zero-Phase and Artifact-Free: Overlap-Add and
the Zero-Phase Length Floor*

**SPMB track:** Poster (short abstract, poster presentation, plus a summary video for the
conference website).

This poster isolates the sharpest result of the Option 2 paper (see `../README.md`): the
time-domain filtering correctness story — overlap-add removes window-seam artifacts, and a
zero-phase length floor governs whether windowed filtering is zero-phase at all. It drops the
spectral-estimation arm (`R5`) of the paper.

---

## 1. Focused question

When EEG is filtered window-by-window for on-demand streaming, (a) does **overlap-add** remove the
boundary artifacts that naive per-window filtering introduces without adding latency, and (b) when
is the zero-phase filter actually zero-phase?

## 2. Method

- A standard FIR bandpass (order 200) + IIR notch (`experiments/dsp.py`, SciPy only), applied two
  ways: naive per-window versus overlap-add (filter an extended window, keep the center).
- Synthetic signals with known spectra for ground truth, plus two real datasets for external
  validity: PhysioNet CHB-MIT (256 Hz) and Sleep-EDF (100 Hz).
- Sweep window sizes; characterize the `filtfilt` pad length (`3 × taps`), shortest valid
  zero-phase input (`3 × taps + 1`), the safe-window scheduler, and the causal-fallback
  group-delay shift across FIR orders.

## 3. Metrics

- Boundary RMSE at window seams (dB reduction, overlap-add vs. naive).
- Per-window latency parity (to show the improvement is free).
- Event-timing shift (ms) of a transient on the zero-phase vs. causal-fallback path.

## 4. Result (obtained)

- Overlap-add reduces seam boundary RMSE by **~35 dB** (synthetic) and **more on real CHB-MIT
  EEG**, at negligible latency.
- The zero-phase pad length is `3 × taps = 606` samples; the shortest valid input is **607**
  samples. Below it the causal path mis-times events by up to **~500 ms** (`(taps−1)/2/fs`).
  The design rule **`(chunk + 2·overlap)·fs > 3·taps`** keeps windowed filtering zero-phase and
  artifact-free, and `safe_window.py` computes the needed overlap automatically.

## 5. Deliverables

- One-page abstract (`ABSTRACT.md`).
- Poster centered on the before/after seam figure (synthetic + real EEG) plus a compact
  floor/timing panel.
- A ~3-minute summary video.

## 6. Timeline (≈4 weeks)

| Week | Milestone |
|---|---|
| 1 | Measurement harness + synthetic ground-truth signals |
| 2–3 | Overlap-add vs. naive sweep; floor/timing characterization; multi-subject CHB-MIT |
| 4 | Figures, abstract, summary video |

## 7. Why this fits SPMB as a Poster

A focused, visual, self-contained signal-processing result that communicates instantly in a
poster and a short video. Expanded with the cascaded-tapering spectral-estimation result, it
becomes the Option 2 paper.
