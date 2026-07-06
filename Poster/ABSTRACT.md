# Poster - Abstract, Contributions, Figures, Video

**SPMB Poster track** (short abstract + poster + summary video). This is the de-risked slice of
the paper (`../paper.tex`), centered on the time-domain filtering correctness result: overlap-add
and the zero-phase length floor. The spectral-estimation arm (cascaded tapering) stays in the
paper.

---

## Title
*Keeping Windowed EEG Filtering Zero-Phase and Artifact-Free: Overlap-Add and the Zero-Phase
Length Floor*

## Short abstract (~220 words)
Networked clinical EEG viewers filter and re-reference signals on demand, one short window at a
time. Using a standard-library reference implementation (FIR bandpass with
`scipy.signal.filtfilt`, IIR notch) evaluated against synthetic ground truth and multiple open
clinical datasets at two sampling rates (PhysioNet CHB-MIT at 256 Hz, Sleep-EDF at 100 Hz), we
show that windowed filtering is correct only under
two conditions that are easy to violate. First, filtering display windows independently produces
visible chunk-seam artifacts; an overlap-add scheme that filters an extended window and keeps the
clean center removes them by 34.7 dB on synthetic data and more on real EEG, at negligible
latency. Second, zero-phase forward--backward filtering (`filtfilt`, FIR order 200) only runs when
the window is longer than its default pad length (606 samples; shortest valid input 607 samples);
a shorter window cannot be zero-phase, so a streaming filter must fall back to a causal path whose
uncompensated group delay shifts events by up to ~500 ms (measured +391 ms on 256 Hz real EEG,
matching theory). We show the pad length and the shift scale as `3×taps` and `(taps−1)/2/fs`
across FIR orders, and that the overlap-add benefit is robust (34.3 ± 2.4 dB over 24 seeds). We
distill a deterministic design rule-`(chunk + 2·overlap)·fs > 3·taps`-and implement a
safe-window scheduler that computes the needed overlap automatically. All code and figures are
released; clinical EDFs are fetched at run time.

## Contributions (poster framing)
1. Overlap-add removes window-seam artifacts in streaming EEG filtering (~35 dB, more on real
   EEG) at negligible latency, confirmed on synthetic and two real datasets (CHB-MIT, Sleep-EDF).
2. A zero-phase length floor: at or below `3·taps` samples, forward--backward filtering cannot run
   and a causal fallback mis-times events by up to ~500 ms; pad length and shift scale with FIR
   order.
3. A deterministic design rule (`(chunk + 2·overlap)·fs > 3·taps`) and scheduler that prevent both
   failures, validated across 8 FIR orders and 24 seeds.

## Figures to feature (300 DPI, in `../experiments/results/`)
- **Primary:** `fig_seam.png` (synthetic seam) + `fig_realdata_seam.png` (real CHB-MIT seam) side
  by side - the headline before/after.
- **Mechanism:** `fig_transition.png` (boundary error collapses at the floor) and
  `fig_order_floor.png` (floor/shift scale with order).
- **Secondary panel:** `fig_transient_shift.png` (the +500 ms event mis-timing).
- Captions: reuse the corresponding `\caption{...}` text from `../paper.tex`.

## Summary-video outline (~3 min)
1. (0:00–0:30) Problem: streaming viewers filter window-by-window; what can go wrong.
2. (0:30–1:30) Overlap-add removes seam artifacts - show `fig_seam` / `fig_realdata_seam`,
   state 34.7 dB (synthetic) and the larger real-EEG reduction at negligible latency.
3. (1:30–2:20) The zero-phase length floor and the ~500 ms fallback shift - show
   `fig_transient_shift` and `fig_transition`; give the `>3×taps` rule.
4. (2:20–3:00) Takeaway: one inequality keeps streaming EEG filtering correct; code released.

## Path to the full paper
Adding the EDF-duration runtime benchmark and the cascaded-tapering spectral-estimation result
(`R5`: tapering twice biases relative band power) promotes this poster to the six-page paper in
`../paper.tex`.
