# Option 2 (Non-AI / Classical-DSP) — Paper Plan

**Working title:** *Correctness Criteria for Windowed Real-Time EEG Signal Processing: Phase,
Windowing, and Spectral Estimation*

**SPMB track:** Paper (original, ~4–6 pages, oral presentation; downgrade to poster allowed).

This is a classical signal-processing study, with no machine learning. Browser- and cloud-based
EEG review applies standard DSP — bandpass/notch filtering and quantitative-EEG (QEEG) feature
extraction — to short windows on demand rather than to whole recordings. We characterize where
windowing silently breaks correctness and derive deterministic criteria that prevent it, using a
standard-library reference implementation (NumPy/SciPy/MNE) evaluated against synthetic ground
truth and multiple open clinical recordings.

The deliverables are `paper.tex` (+ `references.bib`), the experiment scripts and figures under
`experiments/`, and the poster materials under `Poster/`.

---

## 1. Problem and motivation

A whole-signal operation can become wrong once it is applied window-by-window: a zero-phase
filter may not be zero-phase on a short window, and a spectral estimator may be biased if the
window is tapered upstream. These failures are silent — the output still looks like a valid
signal or spectrum. We quantify three such failures and give one-line invariants that fix them.

## 2. Research questions

- **RQ1 (continuity).** Does overlap-add windowed filtering eliminate boundary artifacts versus
  naive per-window filtering, at equal latency?
- **RQ2 (phase/timing).** When is windowed zero-phase filtering actually zero-phase, and what is
  the event-timing cost of leaving that regime?
- **RQ3 (spectral estimation).** Does tapering a window before a Welch/multitaper estimator that
  already tapers bias relative band power, and by how much?

## 3. Background and related work

Position against standard EEG filtering (FIR vs. IIR, zero-phase vs. causal), block-convolution
methods (overlap-add / overlap-save), and quantitative-EEG spectral features (relative band power
via Welch / multitaper). The contribution is characterizing these **in a windowed, real-time
setting** and turning the findings into deterministic design criteria.

## 4. Methods

**Reference DSP (`experiments/dsp.py`).** FIR bandpass (order 200, 202 taps, Hamming `firwin`,
0.5–30 Hz) applied with zero-phase `filtfilt`; IIR notch (`iirnotch`); the documented `filtfilt`
length requirement (`3 × len(taps)`) with an explicit causal `lfilter` fallback for short
windows. NumPy/SciPy only — no external package.

**Windowing (`experiments/windowing.py`).** Whole-signal (zero-phase reference), naive
per-window, and overlap-add (filter an extended window, keep the center).

**Synthetic ground truth (`experiments/synth.py`).** Seeded multichannel EEG with known band
content, 1/f background, and a mains component, so any windowed discontinuity is attributable to
the windowing strategy.

**Spectral estimation.** `scipy.signal.welch` / MNE `psd_array_welch` / `psd_array_multitaper`
for the cascaded-tapering comparison.

## 5. Data

- **Synthetic** signals with known spectra (ground truth for filter fidelity and overlap-add).
- **PhysioNet CHB-MIT** (pediatric epilepsy, bipolar, 256 Hz) and **Sleep-EDF Expanded** (sleep,
  Fpz-Cz/Pz-Oz, 100 Hz) for two-dataset, two-sampling-rate external validity of the filtering
  results; EDFs are fetched at run time and gitignored.

## 6. Experimental design → results (already obtained — see `experiments/FINDINGS.md`)

1. **R1 — Overlap-add eliminates seam artifacts.** 8 s windows: boundary RMSE 14.93 µV → 0.275 µV
   (**34.7 dB**) at ~4% latency.
2. **R2 — Zero-phase length floor + design criterion.** `filtfilt` needs `3 × taps = 606` samples
   (3.03 s @ 200 Hz); below it the causal path distorts the whole window (interior RMSE 30.7 µV).
   Keep **`(chunk + 2·overlap)·fs ≥ 3·taps`**.
3. **R3 — Magnitude fidelity + event mis-timing.** Zero-phase: 0 ms shift; causal fallback:
   **+500 ms** = `(taps−1)/2/fs`. Passband −0.64 dB; notch/stopband below the −120 dB floor.
4. **R4 — Generalization across FIR order + robustness.** Floor `3(order+1)`, shift
   `(taps−1)/2/fs`; overlap-add reduction **35.1 ± 2.3 dB** over 12 seeds.
5. **R5 — Cascaded-tapering pitfall.** Double tapering biases relative band power up to **11.9 pp**
   (Welch) / **9.1 pp** (multitaper); tapering once matches the reference (0.64 / 0.0 pp). Taper
   once.
6. **R6 — Two-dataset real-data confirmation.** R1 and R3 reproduced across 10 CHB-MIT subjects
   (50.6 ± 2.2 dB; shift 389 ± 7 ms vs 393 predicted) **and** Sleep-EDF at 100 Hz, making the
   `fs`-dependent floor explicit (6.06 s / ~1 s shift at 100 Hz vs 2.37 s / ~0.4 s at 256 Hz).

## 7. Evaluation metrics

Boundary/interior RMSE vs. whole-signal ground truth (dB reduction); transient peak-time shift
(ms) vs. theoretical group delay; magnitude response vs. designed `|H|²`; per-band relative-power
discrepancy (pp) and Pearson r vs. a reference estimator.

## 8. Expected contributions

1. A reproducible, standard-library methodology that turns a windowed DSP chain into a
   correctness benchmark.
2. The zero-phase length-floor result and a deterministic design criterion, generalized across
   FIR order and confirmed on multiple real subjects.
3. Quantified overlap-add seam-artifact elimination at negligible latency.
4. A cascaded-tapering spectral-estimation result with a one-line remedy.

## 9. Reproducibility

Release the DSP reference implementation, synthetic-signal generators, metrics, and figure
scripts. Raw clinical EDFs are downloaded from PhysioNet at run time; no PHI is required.

## 10. Risks and mitigations

- **Reference-convention mismatch** (filter design, band edges) → fix shared coefficients and
  document band definitions before comparison.
- **Single-recording external validity** → confirm on multiple CHB-MIT subjects, report mean ± SD.
- **Estimator-definition differences vs. MNE** → align band edges and windowing; report Pearson r.

## 11. Why this fits SPMB as a Paper

A self-contained signal-processing study with deterministic, reproducible results (artifact
energy, event timing, spectral bias) and a strong reproducibility story — appropriate for a 4–6
page paper and an oral. The companion **Poster** (see `Poster/README.md`) isolates the single
sharpest result (overlap-add + the zero-phase length floor).
