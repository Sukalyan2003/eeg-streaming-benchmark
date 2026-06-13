# Poster — Abstract, Contributions, Figures, Video

**SPMB Poster track** (short abstract + poster + summary video). This is the de-risked slice of
the paper (`../PAPER_DRAFT.md` / `../paper.tex`), centered on the overlap-add / zero-phase-floor
result, with the channel/feature findings as a secondary panel.

---

## Title
*Artifact-Free, Correctly-Timed Real-Time EEG Filtering: Overlap-Add and the Zero-Phase Length
Floor in a Clinical Streaming Pipeline*

## Short abstract (~230 words)
Networked clinical EEG viewers filter and re-reference signals on demand, one short window at a
time. We audit the Avinya Neurotech clinical EEG platform against synthetic ground truth and real
EEG (PhysioNet CHB-MIT) and show that windowed filtering is correct only under two conditions that
are easy to violate silently. First, filtering display windows independently produces visible
chunk-seam artifacts; an overlap-add scheme that filters an extended window and keeps the clean
center removes them by 34.7 dB on synthetic data and 48.5 dB on real EEG, at negligible
latency. Second, the platform's zero-phase forward–backward filter (`filtfilt`, FIR order 200)
only runs when the input has at least three filter lengths (606 samples); below that it silently
reverts to a causal filter whose uncompensated group delay shifts events by up to ~500 ms
(measured +391 ms on 256 Hz real EEG, matching theory). The default 10 s streaming
chunk is safe, but the configurable chunk size reaches into the unsafe regime. We show the floor
and the shift scale as `3×(order+1)` and `(taps−1)/2/fs` across FIR orders, and that the
overlap-add benefit is robust (35.1 ± 2.3 dB over 12 seeds). We distill a
deterministic design rule—`chunk + 2·overlap ≥ 3·taps`—that keeps streaming
EEG filtering zero-phase and artifact-free. A companion result shows quantitative features need
channel-aware montaging. All code and figures are released.

## Contributions (poster framing)
1. Overlap-add removes window-seam artifacts in streaming EEG filtering (~35–48 dB) at
   negligible latency, confirmed on synthetic and real EEG.
2. A previously-undocumented silent fallback: a zero-phase filter reverts to a causal path below a
   length floor, mis-timing events by up to ~500 ms.
3. A deterministic, generalized design rule (`chunk + 2·overlap ≥ 3·taps`) that
   prevents both failures, validated across FIR order and 12 seeds.

## Figures to feature (300 DPI, in `../experiments/results/`)
- **Primary:** `fig_seam.png` (synthetic seam) + `fig_realdata_seam.png` (real EEG seam) side by
  side — the headline before/after.
- **Mechanism:** `fig_transition.png` (boundary error collapses at the floor) and
  `fig_order_floor.png` (floor/shift scale with order).
- **Secondary panel:** `fig_transient_shift.png` (the +500 ms event mis-timing).
- Captions: reuse the corresponding `\caption{...}` text from `../paper.tex`.

## Summary-video outline (~3 min)
1. (0:00–0:30) Problem: streaming viewers filter window-by-window; what can go wrong.
2. (0:30–1:30) Overlap-add removes seam artifacts — show `fig_seam` / `fig_realdata_seam`,
   state 34.7/48.5 dB at negligible latency.
3. (1:30–2:20) The zero-phase length floor and the +500 ms fallback shift — show
   `fig_transient_shift` and `fig_transition`; give the `3×taps` rule.
4. (2:20–3:00) Takeaway: one inequality keeps streaming EEG filtering correct; code released.

## Path to the full paper
Adding the magnitude-fidelity, bandpower-equivalence (double-taper), and Siena
channel/feature results promotes this poster to the six-page paper in `../paper.tex`.
