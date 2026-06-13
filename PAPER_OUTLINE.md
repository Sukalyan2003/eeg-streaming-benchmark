# Option 2 — Paper Structure & Literature-Review Plan

Target: IEEE SPMB, Paper track (original, ~4–6 pages, IEEE two-column, oral). Working title:
*Correctness Criteria for Windowed Real-Time EEG Signal Processing: Phase, Windowing, and
Spectral Estimation.* The authoritative artifact is `paper.tex`; this file records the structure
and the literature-review plan.

> Citations are verified against IEEE Xplore / PubMed / publisher / dataset pages before
> inclusion (see `references.bib`). Do not cite anything unverified.

---

## 1. Paper structure (section by section, ~4–6 pages)

**Abstract (~200 words).** Web/clinical EEG viewers filter signals window-by-window. Using a
standard-library reference chain against synthetic ground truth and multiple open clinical
recordings, we establish three correctness criteria: (i) windowed filtering needs overlap-add to
avoid seam artifacts (~35 dB at negligible cost); (ii) zero-phase `filtfilt` has a length floor
(3×taps), below which a streaming filter must use a causal path that shifts events by ~500 ms;
(iii) a single taper is required in PSD estimation (cascaded tapering biases relative band power
by up to ~12 pp). Deterministic criteria, validated across FIR order and multiple subjects.

**I. Introduction (~0.75 p).** Streaming/web EEG review; the thesis that "standard" DSP becomes
subtly wrong under windowing; contributions list.

**II. Background & Related Work (~0.75–1 p).** Four threads (see §3): streaming EEG systems; EEG
digital filtering & phase; block convolution / boundary artifacts; spectral band-power
estimation. End each with the gap we fill.

**III. Methods (~1–1.25 p).**
- A. Reference DSP primitives (`dsp.py`): FIR order-200 bandpass via `filtfilt`, IIR notch, Welch
  PSD — standard libraries only, with the documented `filtfilt` length requirement and explicit
  causal fallback.
- B. Synthetic ground-truth signals with known spectra (reproducible generator).
- C. Windowing strategies: whole-signal (reference), naive per-window, overlap-add.
- D. Metrics: boundary/interior RMSE; magnitude response; event temporal shift; relative
  band-power agreement (vs MNE; Pearson r + max pp discrepancy).
- E. Reproducibility: fixed environment + seeds, multi-subject CHB-MIT, released code.

**IV. Results (~1.5 p)** — one subsection per result, each with a figure/table:
- R1 Overlap-add removes seam artifacts (`fig_seam`, `fig_profile`, `fig_summary`).
- R2 The `filtfilt` length floor and the design criterion `chunk + 2·overlap ≥ 3·taps`.
- R3 Magnitude fidelity + the ~500 ms event mis-timing on the causal fallback (`fig_magnitude`,
  `fig_transient_shift`).
- R4 Generalization: FIR-order sweep (floor = 3·(order+1)), transition curve, seed robustness
  with mean ± SD (`fig_order_floor`, `fig_transition`).
- R5 Cascaded-tapering pitfall vs MNE (Welch **and** multitaper).
- R6 Multi-subject real-data confirmation of R1/R3 on CHB-MIT (`fig_realdata_seam`).

**V. Discussion (~0.5 p).** Deterministic design criteria (`chunk + 2·overlap ≥ 3·taps`; taper
once); clinical impact (event timing); generality (properties of `filtfilt`, FIR group delay,
Welch/multitaper tapering); threats to validity (R1–R4 are deterministic code properties; R5
follows from cascaded-tapering algebra; confirmed on real data in R6).

**VI. Conclusion + future work (~0.25 p).** A small set of correctness invariants for streaming
clinical EEG DSP; pointers to larger cohorts and to montage/feature-level correctness.

**Figures/Tables:** Fig.1 seam waveform (`fig_seam`); Fig.2 seam error profile (`fig_profile`);
Fig.3 boundary-RMSE + latency vs window (`fig_summary`, double-column); Fig.4 magnitude response
(`fig_magnitude`); Fig.5 transient shift (`fig_transient_shift`); Fig.6 order floor
(`fig_order_floor`); Fig.7 transition curve (`fig_transition`); Fig.8 real-EEG seam
(`fig_realdata_seam`). Table I cascaded-tapering discrepancy (Welch & multitaper, with/without
double taper); Table II FIR-order sweep (floor and shift vs order). Headline numbers carry
mean ± SD (12 seeds; multiple CHB-MIT subjects).

## 2. Contributions (explicit, for the intro)
1. A reproducible, standard-library methodology that turns a windowed DSP chain into a
   correctness benchmark.
2. The `filtfilt` length-floor result with a measured ~500 ms event mis-timing and a deterministic
   guard, generalized across FIR order.
3. Quantified overlap-add benefit (~35 dB at negligible cost), confirmed on multiple subjects.
4. A cascaded-tapering spectral-estimation result with a one-line remedy.

## 3. Literature-review plan (by theme)

For each theme: **goal**, **databases/keywords**, **anchor references to verify**, **gap**.

### Theme A — Streaming / web-based & real-time EEG visualization
- Goal: position on-demand windowed serving of EEG; show prior systems rarely analyze DSP
  correctness under windowing.
- Where/keywords: IEEE Xplore, PubMed — "web-based EEG viewer", "real-time EEG streaming",
  "browser EEG visualization", "tele-EEG", "EEG cloud platform".
- Anchors to verify: EDF/EDF+ format (Kemp & Värri 1992; Kemp & Olivan 2003); browser
  physiological viewers.
- Gap: correctness of filtering/features *under streaming windowing* is unstudied.

### Theme B — Digital filtering of EEG: phase, zero-phase, and artifacts
- Goal: ground FIR/IIR, `filtfilt` (zero-phase) vs causal group delay, and the clinical risk of
  phase distortion / temporal smearing.
- Where/keywords: "EEG filtering artifacts", "zero-phase filtfilt EEG", "group delay
  electrophysiology", "filter distortion ERP".
- Anchors to verify: Widmann, Schröger & Maess (2015, J Neurosci Methods); de Cheveigné & Nelken
  (2019, Neuron); Oppenheim & Schafer (FIR/linear-phase/group delay).
- Gap: these warn about phase offline; none characterize the length-gated zero-phase→causal
  transition inside a streaming filter (our R2–R3).

### Theme C — Block convolution & boundary artifacts (overlap-add / overlap-save)
- Goal: formalize seam artifacts and the overlap-add fix.
- Where/keywords: "overlap-add", "overlap-save", "block convolution", "edge effects windowed
  filtering".
- Anchors to verify: Oppenheim & Schafer; Proakis & Manolakis; SciPy (Virtanen et al. 2020).
- Gap: overlap-add is textbook, but its *parameterization against a zero-phase length floor* in
  clinical EEG streaming (our R1–R2 link) is not characterized.

### Theme D — Spectral band power & quantitative EEG estimation
- Goal: ground Welch/multitaper relative band power; motivate the MNE equivalence and the
  cascaded-tapering finding.
- Where/keywords: "Welch power spectral density EEG", "multitaper spectral estimation",
  "relative band power QEEG", "spectral leakage taper".
- Anchors to verify: Welch (1967); Thomson (1982) multitaper; aEEG (Hellström-Westas et al.);
  MNE-Python PSD (Gramfort et al. 2013).
- Gap: implementation-level estimator errors (cascaded tapering) and their bias on clinical
  relative band power (our R5) are under-reported.

### Theme E (cross-cutting) — Reproducibility & software correctness in clinical neuro-DSP
- Goal: frame the benchmark stance and the medical-device angle.
- Where/keywords: "reproducibility EEG analysis", "software correctness biomedical signal
  processing", "IEC 62304 medical device software".
- Anchors to verify: reproducibility-in-neuroscience commentaries; IEC 62304 (standard).
- Gap: little work characterizes windowed clinical EEG DSP for these silent errors.

## 4. Review execution plan (process)
1. Seed from the anchors above; snowball forward (cited-by) and backward (references).
2. Databases: IEEE Xplore, PubMed/Medline, Google Scholar, plus SciPy/MNE documentation for tool
   semantics.
3. Maintain `references.bib`; record for each: claim it supports, and which result/gap it maps to.
   Target ~15–20 references for a 4–6 page paper.
4. Verify every citation (authors, year, venue) before inclusion — no unverified references.
