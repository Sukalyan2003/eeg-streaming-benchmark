# Option 2 (Non-AI / Classical-DSP) — Poster Plan

**Working title:** *Overlap-Add Windowed Filtering Eliminates Boundary Artifacts in Streaming
EEG Re-Montaging*

**SPMB track:** Poster (short abstract, poster presentation, plus a summary video for the
conference website).

This poster isolates the single sharpest result of the Option 2 paper (see `../README.md`): the
effect of overlap-add on streaming filter continuity. It drops the feature-equivalence and
load/memory arms.

---

## 1. Focused question

When EEG is filtered window-by-window for on-demand streaming, does **overlap-add** remove the
boundary/edge artifacts that naïve per-window filtering introduces — without adding latency?

## 2. Method

- Identical FIR bandpass + IIR notch filters from `domain/helpers/filters.py`
  (`_get_bandpass_coefficients`, `_get_notch_coefficients`), applied two ways: naïve per-window
  versus overlap-add (the approach in `file-service/src/services/streaming/helpers.py`, with the
  fix recorded in `file-service/changelogs/OVERLAP_ADD_FIX_APPLIED.md`).
- Test on public EDF recordings plus synthetic signals with known spectra for ground truth.
- Sweep window sizes and filter cutoffs; hold end-to-end latency constant.

## 3. Metrics

- Edge-band energy at window boundaries (dB reduction, overlap-add vs. naïve).
- Latency parity (to show the improvement is free).
- A clear before/after waveform and spectrogram figure across a window boundary.

## 4. Expected result

Quantified elimination of boundary artifacts with no latency penalty — a crisp, visual,
practically relevant DSP finding.

## 5. Deliverables

- One-page abstract.
- Poster centered on the before/after boundary figure plus a compact methods panel.
- A ~3-minute summary video.

## 6. Timeline (≈4 weeks)

| Week | Milestone |
|---|---|
| 1 | Measurement harness + synthetic ground-truth signals |
| 2–3 | Overlap-add vs. naïve sweep across window sizes/filters |
| 4 | Figures, abstract, summary video |

## 7. Why this fits SPMB as a Poster

A focused, visual, self-contained signal-processing result that communicates instantly in a
poster and a short video. Expanded with the feature-equivalence and load/memory studies, it
becomes the Option 2 paper.
