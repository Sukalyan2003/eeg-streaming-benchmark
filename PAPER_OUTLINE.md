# Option 2 — Paper Structure & Literature-Review Plan

Target: IEEE SPMB, Paper track (original, ~4–6 pages, IEEE two-column, oral). Working title:
*Correct-by-Construction Real-Time EEG Filtering and Quantitative Features: A Characterization
of Windowing, Phase, and Spectral Estimation in a Clinical Streaming Pipeline.*

> Citations below are **anchors to obtain and verify** (real, well-known works named by
> description); confirm exact bibliographic details on IEEE Xplore / PubMed / Google Scholar
> before inclusion. Do not cite anything unverified.

---

## 1. Paper structure (section by section, ~4–6 pages)

**Abstract (~150–200 words).** Web/clinical EEG viewers filter and re-reference signals on the
fly. We show, against a deployed pipeline, that three design choices silently determine
correctness: (i) windowed filtering needs overlap-add to avoid seam artifacts (≈35 dB at ~1%
cost); (ii) the zero-phase `filtfilt` path has a length floor (3×taps), below which the filter
falls back to a causal path that shifts events by ~500 ms; (iii) channel canonicalization and a
single-taper spectral estimate are required for correct quantitative features (a double-Hann
taper biases relative band power by up to ~12 pp; an unexcluded ECG channel by ~49%). We give
deterministic criteria and fixes.

**I. Introduction (~0.75 p).** Clinical context (EEG review, montages, QEEG), the move to
streaming/web viewers, the thesis that "standard" DSP becomes subtly wrong under windowing and
on-demand serving; contributions list; that findings are validated against a real platform.

**II. Background & Related Work (~0.75–1 p).** Five threads (see §3): streaming EEG systems;
EEG digital filtering & phase; block convolution / boundary artifacts; montage & channel
naming; spectral band-power estimation. End each with the gap we fill.

**III. Methods (~1–1.25 p).**
- A. System under test: the production filter/montage/bandpower primitives (FIR order-200
  bandpass via `filtfilt`, IIR notch, Welch PSD, channel canonicalization).
- B. Synthetic ground-truth signals with known spectra (reproducible generator).
- C. Windowing strategies: whole-signal (reference), naive per-window, overlap-add.
- D. Metrics: boundary RMSE & boundary-to-interior ratio; magnitude response; event temporal
  shift; channel family-classification accuracy; relative band-power agreement (vs MNE).
- E. Reproducibility: pinned source commits, fixed seeds, released code.

**IV. Results (~1.25–1.5 p)** — one subsection per result, each with a figure/table:
- R1 Overlap-add removes seam artifacts (`fig_seam`, `fig_profile`, `fig_summary`).
- R2 The `filtfilt` length floor and the small-chunk fallback regime (`metrics.csv`).
- R3 Magnitude fidelity + the ~500 ms event mis-timing in the fallback (`fig_magnitude`,
  `fig_transient_shift`).
- R4 Channel canonicalization accuracy + feature integrity (ECG contamination).
- R5 Bandpower equivalence vs MNE and the double-Hann taper bias.

**V. Discussion (~0.5 p).** Deterministic design criteria (e.g. `chunk + 2·overlap ≥ 3·taps`;
single taper; canonicalize-then-montage); clinical impact (event timing, biased QEEG);
generality beyond this platform; threats to validity (synthetic vs real data — note R1–R3 are
deterministic code properties; R4–R5 magnitudes are illustrative).

**VI. Conclusion + future work (~0.25 p).** A small set of correctness invariants for streaming
clinical EEG DSP; pointer to real-data confirmation (CHB-MIT/TUH) and the AI track.

**Figures/Tables (reuse the experiment outputs):** Fig.1 seam waveform; Fig.2 seam error
profile; Fig.3 boundary-RMSE + latency vs window; Fig.4 magnitude response; Fig.5 transient
shift; Table I canonicalization; Table II bandpower equivalence (prod vs MNE, with/without
double-Hann).

## 2. Contributions (explicit, for the intro)
1. A reproducible methodology that turns a deployed pipeline into a correctness benchmark.
2. The `filtfilt` length-floor / small-chunk fallback result with a measured ~500 ms clinical
   mis-timing and a deterministic guard.
3. Quantified overlap-add benefit (≈35 dB at ~1% cost).
4. Evidence that canonicalize-then-montage is necessary (ECG contamination ≈49%).
5. A spectral-estimation equivalence vs MNE that uncovers a double-Hann bias (≈12 pp).

## 3. Literature-review plan (by theme)

For each theme: **goal**, **databases/keywords**, **anchor references to verify**, **gap**.

### Theme A — Streaming / web-based & real-time EEG visualization
- Goal: position on-demand windowed serving of EEG; show prior systems rarely analyze DSP
  correctness under windowing.
- Where/keywords: IEEE Xplore, PubMed — "web-based EEG viewer", "real-time EEG streaming",
  "browser EEG visualization", "tele-EEG", "EEG cloud platform".
- Anchors to verify: EDF/EDF+ format (Kemp & Värri 1992; Kemp & Olivan 2003); browser
  physiological viewers; any QEEG/cloud platform papers.
- Gap: correctness of filtering/features *under streaming windowing* is unstudied.

### Theme B — Digital filtering of EEG: phase, zero-phase, and artifacts
- Goal: ground FIR/IIR, `filtfilt` (zero-phase) vs causal group delay, and the clinical risk of
  phase distortion / temporal smearing.
- Where/keywords: "EEG filtering artifacts", "zero-phase filtfilt EEG", "group delay
  electrophysiology", "filter distortion ERP".
- Anchors to verify: Widmann, Schröger & Maess (2015) *Digital filter design for
  electrophysiological data* (J Neurosci Methods); de Cheveigné & Nelken (2019) *Filters: when,
  why, and how (not) to use them* (Neuron); Tanner et al. (2015) on filter artifacts; Oppenheim
  & Schafer *Discrete-Time Signal Processing* (FIR/linear-phase/group delay).
- Gap: these warn about phase offline; none address the *silent* zero-phase→causal fallback in a
  length-gated streaming filter (our R3).

### Theme C — Block convolution & boundary artifacts (overlap-add / overlap-save)
- Goal: formalize seam artifacts and the overlap-add fix.
- Where/keywords: "overlap-add", "overlap-save", "block convolution", "edge effects windowed
  filtering".
- Anchors to verify: Oppenheim & Schafer (overlap-add/save); standard DSP texts (Proakis &
  Manolakis); SciPy signal-processing documentation (Virtanen et al. 2020, *SciPy 1.0*).
- Gap: overlap-add is textbook, but its *parameterization against a zero-phase length floor* in
  clinical EEG streaming (our R1–R2 link) is not characterized.

### Theme D — Montages, re-referencing, and channel-name standardization
- Goal: justify canonicalize-then-montage and the feature-integrity risk.
- Where/keywords: "EEG montage average reference", "Laplacian reference", "10–20 system channel
  naming", "EEG channel nomenclature", "BIDS EEG".
- Anchors to verify: 10–20 system (Jasper 1958; Klem et al. 1999); average/Laplacian referencing
  reviews; EEG-BIDS (Pernet et al. 2019, *Scientific Data*); MNE-Python montage handling
  (Gramfort et al. 2013).
- Gap: cross-vendor naming inconsistency as a *quantitative feature-correctness* hazard (our R4)
  is rarely measured.

### Theme E — Spectral band power & quantitative EEG estimation
- Goal: ground Welch/multitaper relative band power and aEEG; motivate the MNE equivalence and
  the taper finding.
- Where/keywords: "Welch power spectral density EEG", "multitaper spectral estimation",
  "relative band power QEEG", "amplitude-integrated EEG aEEG", "spectral leakage taper".
- Anchors to verify: Welch (1967); Thomson (1982) multitaper; Percival & Walden *Spectral
  Analysis for Physical Applications*; aEEG (Hellström-Westas et al.); MNE-Python PSD
  (Gramfort et al. 2013).
- Gap: implementation-level estimator errors (double-taper) and their bias on clinical relative
  band power (our R5) are under-reported.

### Theme F (cross-cutting) — Reproducibility & software correctness in clinical neuro-DSP
- Goal: frame the paper's "benchmark a deployed system" stance and the medical-device angle.
- Where/keywords: "reproducibility EEG analysis", "software correctness biomedical signal
  processing", "IEC 62304 medical device software".
- Anchors to verify: reproducibility-in-neuroscience commentaries; IEC 62304 (standard).
- Gap: little work *audits a production clinical EEG DSP path* for these silent errors.

## 4. Review execution plan (process)
1. Seed from the anchors above; snowball forward (cited-by) and backward (references).
2. Databases: IEEE Xplore (SPMB-adjacent), PubMed/Medline (clinical EEG), Google Scholar
   (coverage), plus the SciPy/MNE documentation for tool semantics.
3. Maintain a `references.bib`; record for each: claim it supports, and which result/gap it maps
   to. Target ~15–25 references for a 4–6 page paper.
4. Verify every citation (authors, year, venue) before inclusion — no unverified references.
