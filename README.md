# Option 2 (Non-AI / Classical-DSP) — Paper Plan

**Working title:** *A Correctness and Runtime Benchmark for Windowed Real-Time EEG Signal
Processing: Phase, Windowing, and Spectral Estimation*

**SPMB track:** Paper (original, ~4–6 pages, oral presentation; downgrade to poster allowed).

This is a classical signal-processing study, with no machine learning. Browser- and cloud-based
EEG review applies standard DSP — bandpass/notch filtering and quantitative-EEG (QEEG) feature
extraction — to short windows on demand rather than to whole recordings. We characterize where
windowing silently breaks correctness and derive deterministic criteria that prevent it. The
updated contribution is a benchmark + guardrail package: a standard-library reference
implementation (NumPy/SciPy/MNE), an adaptive safe-window scheduler, and an EDF-duration runtime
benchmark evaluated against synthetic ground truth and open clinical recordings.

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
- **RQ4 (deployability).** Can the zero-phase guard be enforced automatically while preserving
  end-to-end processing time far below real time as EDF duration increases?

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

**Safe-window scheduler (`experiments/safe_window.py`).** Computes whether a configured
`chunk + 2·overlap` span is zero-phase safe, the minimum overlap needed to satisfy
`(chunk + 2·overlap)·fs > 3·taps`, and the fallback group delay that would be exposed if the
guard is ignored.

**Windowing (`experiments/windowing.py`).** Whole-signal (zero-phase reference), naive
per-window, and overlap-add (filter an extended window, keep the center).

**Synthetic ground truth (`experiments/synth.py`).** Seeded multichannel EEG with known band
content, 1/f background, and a mains component, so any windowed discontinuity is attributable to
the windowing strategy.

**Spectral estimation.** `scipy.signal.welch` / MNE `psd_array_welch` / `psd_array_multitaper`
for the cascaded-tapering comparison.

**EDF throughput benchmark (`experiments/benchmark_edf_lengths.py`).** Runs filtering plus Welch
relative-band-power extraction over all local recordings of both datasets, sweeping segment
duration (30–600 s), FIR order (100–400), and channel count (1–16). Reports load/filter/feature
time, array memory, and the real-time factor for the four paths, aggregated as mean ± SD across
the files of each dataset. Timing runs without memory instrumentation (which inflates wall time).

**Per-window latency benchmark (`experiments/benchmark_latency.py`).** Filters one display window
at a time and times each all-channel request independently, reporting the median/p95/max
distribution against a 60 Hz display-refresh budget — a direct latency measurement, not amortized.

**Boundary-policy experiment (`experiments/exp_edge_policy.py`).** Measures edge-window error of
the causal fallback, a reflection zero-phase policy, and real-context prefetching against the
whole-signal zero-phase reference, at the true record start and an interior-of-record view edge.

## 5. Data

- **Synthetic** signals with known spectra (ground truth for filter fidelity and overlap-add).
- **PhysioNet CHB-MIT** (pediatric epilepsy, bipolar, 256 Hz) and **Sleep-EDF Expanded** (sleep,
  Fpz-Cz/Pz-Oz, 100 Hz) for two-dataset, two-sampling-rate external validity of the filtering
  results; EDFs are fetched at run time and gitignored.

## 6. Experimental design → results (already obtained — see `experiments/FINDINGS.md`)

1. **R1 — Overlap-add eliminates seam artifacts.** 8 s windows: boundary RMSE 14.93 µV → 0.275 µV
   (**34.7 dB**) at only a few percent latency overhead.
2. **R2 — Zero-phase length floor + design criterion.** `filtfilt` has padlen `3 × taps = 606`
   samples for the order-200 filter, so the shortest valid input is **607 samples** (3.04 s @
   200 Hz); below it the causal path distorts the whole window (interior RMSE 30.7 µV). Keep
   **`(chunk + 2·overlap)·fs > 3·taps`**.
3. **R3 — Magnitude fidelity + event mis-timing.** Zero-phase: 0 ms shift; causal fallback:
   **+500 ms** = `(taps−1)/2/fs`. Passband −0.64 dB; notch/stopband below the −120 dB floor.
4. **R4 — Generalization across FIR order + robustness.** Padlen `3 × taps`, shortest valid
   zero-phase input `3 × taps + 1`, shift `(taps−1)/2/fs`, swept over **8 orders (50–500)**;
   overlap-add reduction **34.3 ± 2.4 dB** over 24 seeds.
5. **R5 — Cascaded-tapering pitfall.** Double tapering biases relative band power up to **11.9 pp**
   (Welch) / **9.1 pp** (multitaper); tapering once matches the reference (0.64 / 0.0 pp). Taper
   once.
6. **R6 — Two-dataset real-data confirmation.** R1 and R3 reproduced across 10 CHB-MIT subjects
   (50.6 ± 2.2 dB; shift 389 ± 7 ms vs 393 predicted) **and** Sleep-EDF at 100 Hz, making the
   `fs`-dependent floor explicit (6.06 s / ~1 s shift at 100 Hz vs 2.37 s / ~0.4 s at 256 Hz).
7. **R7 — EDF throughput benchmark + guardrail.** Over **all 18 local recordings** (10 CHB-MIT @
   256 Hz/8 ch, 8 Sleep-EDF @ 100 Hz/2 ch), sweeping duration (30–600 s), FIR order (100–400),
   and channels (1–16), mean ± SD across files (timing without memory instrumentation). The
   scheduler is **fs-aware** — a 1 s chunk / 0.5 s request becomes **0.6875 s** context at 256 Hz
   but **2.54 s** at 100 Hz to clear the 607-sample floor. At 600 s, scheduled-overlap keeps
   **interior** windows zero-phase safe at **3.05 ± 0.06 s** (CHB-MIT; RTF 5.1×10⁻³) / **0.74 ± 0.01 s**
   (Sleep-EDF), ~2.4× the faster-but-unsafe fixed overlap. Cost rises with FIR order (scheduler
   grows context) and is linear in channel count. Peak array memory ≈ 20 MB.
8. **R8 — Directly measured per-window latency.** Filtering one window at a time and timing each
   request: scheduled-overlap median **4.2 ms** (filter) / **5.7 ms** (with Welch features), p95
   **6.4 ms** on CHB-MIT; **1.1 / 1.5 ms** on Sleep-EDF. **Every method stays within one 60 Hz
   refresh (16.7 ms)**, and the direct measurement agrees with R7's amortized estimate.
9. **R9 — Record-edge boundary policies.** Edge-window RMSE vs the zero-phase reference:
   **prefetch** real context makes interior-of-record edges essentially exact (**93.6 dB** vs causal
   on CHB-MIT); reflection zero-phase avoids the causal fallback at true record boundaries (7–16 dB
   over causal) but leaves a residual error (CHB-MIT 17.9 µV). Prefetch when context exists,
   reflect-pad at true boundaries, never silently causal → operationally zero-phase at interior
   windows, with reduced error at the two true record boundaries.

## 7. Evaluation metrics

Boundary/interior RMSE vs. whole-signal ground truth (dB reduction); transient peak-time shift
(ms) vs. theoretical group delay; magnitude response vs. designed `|H|²`; per-band relative-power
discrepancy (pp) and Pearson r vs. a reference estimator; EDF processing time, real-time factor,
seconds per EEG-hour, and array memory footprint.

## 8. Expected contributions

1. A reproducible, standard-library methodology that turns a windowed EDF DSP chain into a
   correctness and runtime benchmark.
2. The zero-phase length-floor result and a deterministic safe-window scheduler, generalized
   across FIR order and confirmed on multiple real subjects.
3. Quantified overlap-add seam-artifact elimination at negligible latency.
4. A cascaded-tapering spectral-estimation result with a one-line remedy.
5. End-to-end EDF benchmarks showing the guardrail is deployable: throughput far below real time,
   directly measured per-window latency within a display refresh, and boundary policies (prefetch /
   reflection) that extend the zero-phase guarantee to record edges.

## 9. Reproducibility

Release the DSP reference implementation, synthetic-signal generators, safe-window scheduler,
runtime benchmark, metrics, and figure scripts. Raw clinical EDFs are downloaded from PhysioNet
at run time; no PHI is required.

## 10. Risks and mitigations

- **Reference-convention mismatch** (filter design, band edges) → fix shared coefficients and
  document band definitions before comparison.
- **Single-recording external validity** → confirm on multiple CHB-MIT subjects, report mean ± SD.
- **Estimator-definition differences vs. MNE** → align band edges and windowing; report Pearson r.
- **Runtime hardware dependence** → report real-time factor and repeat the EDF-duration benchmark
  on target deployment hardware before making product latency claims.

## 11. Why this fits SPMB as a Paper

A self-contained signal-processing study with deterministic, reproducible results (artifact
energy, event timing, spectral bias) and a strong reproducibility story — appropriate for a 4–6
page paper and an oral. The companion **Poster** (see `Poster/README.md`) isolates the single
sharpest result (overlap-add + the zero-phase length floor).
