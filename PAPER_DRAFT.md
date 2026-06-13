# Correct-by-Construction Real-Time EEG Filtering and Quantitative Features: A Characterization of Windowing, Phase, and Spectral Estimation in a Clinical Streaming Pipeline

*Draft for IEEE SPMB (Paper track). Citations are `[CITE: …]` placeholders mapped to the
literature-review themes in `PAPER_OUTLINE.md`; replace with verified references before
submission. All quantitative claims trace to `experiments/FINDINGS.md`.*

---

## Abstract

Browser- and cloud-based clinical EEG review now filters and re-references signals on demand,
window by window, rather than on whole recordings. We show, by auditing a deployed clinical EEG
platform against synthetic ground truth and two open clinical datasets, that three commonplace
design choices silently govern correctness. First, windowed filtering produces visible
chunk-seam artifacts unless overlap-add is used; overlap-add removes them by ~35 dB at
negligible latency. Second, the zero-phase `filtfilt` path has a length floor of three filter
lengths; below it the implementation silently reverts to a causal filter whose uncompensated
group delay shifts events by up to ~500 ms — a clinically meaningful temporal error that appears
only for short streamed chunks. Third, quantitative features require channel-aware processing:
failing to canonicalize and exclude non-EEG channels before montaging biases relative band power
by tens of percent, and a redundant analysis-window taper biases relative band power by up to
~12 percentage points relative to a reference estimator. We give deterministic design criteria
(`chunk + 2·overlap ≥ 3·taps`; single taper; canonicalize-then-montage), confirm every finding
on real EEG (CHB-MIT, Siena), and show the filtering results generalize across FIR order. The
contribution is a small, reproducible set of correctness invariants for streaming clinical EEG
signal processing.

## I. Introduction

Clinical electroencephalography is increasingly reviewed through networked, on-demand viewers
that fetch and process short windows of a recording rather than loading entire studies
[CITE: streaming/web EEG viewers; EDF/EDF+]. This shift moves standard digital-signal-processing
operations — bandpass and notch filtering, montage re-referencing, and quantitative-EEG (QEEG)
feature extraction — from a whole-signal, offline setting into a windowed, real-time, multi-user
setting. Operations that are textbook-correct offline can become subtly and silently wrong under
windowing.

This paper characterizes that gap on a *deployed* clinical EEG platform. Rather than propose a
new algorithm, we treat the production filtering, montage, and band-power code as the system
under test and quantify where streaming-time choices break correctness, using synthetic signals
with known spectra for ground truth and two open clinical datasets for external validity. We make
five contributions:

1. A reproducible methodology that turns a deployed pipeline into a correctness benchmark, with
   pinned source versions and released code.
2. The `filtfilt` length-floor result: a zero-phase filter implementation silently reverts to a
   causal path below a length threshold, producing a measured event mis-timing of up to ~500 ms
   for short streamed chunks, with a deterministic guard to prevent it.
3. A quantification of overlap-add windowed filtering (~35 dB seam-artifact reduction at
   negligible latency), generalized across FIR order.
4. Evidence that canonicalize-then-montage is necessary: mixing a real ECG channel into an
   average reference biases EEG relative alpha power by >100 % on real data.
5. A spectral-estimation equivalence against a reference library that uncovers a redundant-taper
   bias (~12 percentage points) affecting both Welch and multitaper estimates.

## II. Background and Related Work

**Streaming and real-time EEG visualization.** Networked EEG viewers and QEEG platforms serve
windowed signal segments to the browser [CITE: web EEG viewers; tele-EEG]. The European Data
Format (EDF/EDF+) is the de-facto storage format [CITE: EDF; EDF+]. Prior work emphasizes
interactivity and scale; the DSP correctness of windowed filtering and feature extraction in this
setting is largely unexamined.

**EEG filtering and phase.** Bandpass and notch filtering of EEG is standard, and the risks of
phase distortion and filter transients on event timing are well documented for offline analysis
[CITE: Widmann et al. 2015; de Cheveigné & Nelken 2019; Tanner et al. 2015]. Zero-phase filtering
(forward–backward `filtfilt`) removes phase distortion but is non-causal and requires sufficient
data length [CITE: Oppenheim & Schafer]. These works warn about phase offline; none address a
*silent* zero-phase-to-causal fallback inside a length-gated streaming filter.

**Block convolution and boundary artifacts.** Filtering long signals in blocks is classically
handled by overlap-add and overlap-save [CITE: Oppenheim & Schafer; Proakis & Manolakis], and the
relevant routines are provided by standard libraries [CITE: SciPy]. The textbook methods are not,
however, characterized for clinical EEG streaming, nor parameterized against a zero-phase length
floor.

**Montages and channel nomenclature.** Re-referencing (bipolar, average, Laplacian) is foundational
to EEG interpretation [CITE: 10–20 system; average/Laplacian referencing], and channel-naming
standardization is an active concern [CITE: EEG-BIDS]. Cross-vendor naming inconsistency as a
*quantitative feature-correctness* hazard is rarely measured.

**Spectral band power and QEEG.** Relative band power via Welch or multitaper spectral estimation,
and amplitude-integrated EEG, are common QEEG features [CITE: Welch 1967; Thomson 1982 multitaper;
aEEG]. Reference implementations are widely used [CITE: MNE-Python; SciPy]. Implementation-level
estimator errors and their bias on clinical relative band power are under-reported.

## III. Methods

### A. System under test

We evaluate the production signal-processing primitives of a deployed clinical EEG platform: an
FIR bandpass (order 200, designed with `firwin`) applied with forward–backward zero-phase
filtering (`scipy.signal.filtfilt`), an IIR notch (`iirnotch`) applied with `filtfilt`, a
channel-name canonicalization and family-classification module, and a Welch/multitaper power
spectral density (PSD) routine. A salient implementation detail is that the bandpass falls back
to causal filtering (`scipy.signal.lfilter` with reflective padding) when the input is shorter
than three filter lengths, because `filtfilt` requires `≥ 3·len(taps)` samples.

### B. Synthetic ground-truth signals

We generate deterministic (seeded) multichannel signals at 200 Hz with known spectral content:
band-limited oscillations (δ/θ/α/β) with randomized phases, slow amplitude-modulated bursts, 1/f
("pink") background, and a 50 Hz mains component. The signal is continuous and
stationary-in-expectation, so any discontinuity in a windowed reconstruction is attributable to
the windowing strategy rather than the signal. For feature experiments we add a synthetic ECG-like
channel (periodic QRS-shaped transients) to probe non-EEG contamination.

### C. Windowing strategies

For a 1-D signal we compare three strategies, all driving the *same* production filters:
(i) *whole-signal* filtering, used as the zero-phase reference (ground truth); (ii) *naive*
per-window filtering, in which each non-overlapping display window is filtered independently and
concatenated; and (iii) *overlap-add*, in which each display window is filtered within an extended
window that includes `overlap_s` seconds of real neighboring samples on each side, retaining only
the clean center. Strategy (iii) mirrors the platform's deployed fix.

### D. Metrics

*Boundary error* is the RMS reconstruction error within ±50 ms of each window seam, versus the
whole-signal reference; *interior error* excludes ±0.6 s around seams. We report seam-error
reduction in dB. *Magnitude fidelity* is measured with an equal-amplitude multitone probe across
0.2–60 Hz. *Event mis-timing* is the temporal shift of a 20 ms Gaussian transient, recovered by
peak location and by cross-correlation against the zero-phase output. *Canonicalization accuracy*
is the channel-family classification rate on curated and real channel-name variants. *Feature
equivalence* compares per-band relative power from the production PSD routine against a reference
(`mne.time_frequency.psd_array_welch` / `psd_array_multitaper`) by Pearson correlation and maximum
per-band discrepancy in percentage points (pp).

### E. Real data

We confirm findings on two open datasets: PhysioNet **CHB-MIT** `chb01_01` (256 Hz, bipolar
montage) for the filtering results, and PhysioNet **Siena Scalp EEG** `PN00-1` (512 Hz, 10–20
monopolar EEG plus a genuine `EKG` channel and SpO2/HR) for the channel/feature results. Raw EDFs
are not redistributed; download commands are provided with the code.

### F. Reproducibility

Experiments were run against pinned source revisions of the platform libraries and a fixed
environment (Python 3.12; NumPy 2.4.1; SciPy 1.17.0; MNE 1.11.0). All generators, metrics, and
figures are released; raw clinical EDFs are fetched from PhysioNet at run time.

## IV. Results

### R1. Overlap-add eliminates window-seam artifacts

With 8 s display windows, naive per-window filtering concentrates large error at seams (boundary
RMSE 14.93 µV) while leaving the interior near-exact. Overlap-add with 0.5 s of context reduces
the boundary RMSE to 0.275 µV — a 34.7 dB reduction — and 1.0 s of context renders it negligible
(≈ 3·10⁻⁴ µV). The per-window filtering-time overhead of overlap-add is within measurement noise
(sub-millisecond), so the correction is effectively free (Fig. 1–3).

### R2. A zero-phase length floor and a silent fallback regime

The bandpass uses zero-phase `filtfilt` only when the input has at least `3·len(taps) = 606`
samples (3.03 s at 200 Hz); shorter inputs silently use a causal path. The streaming filter
operates on a window of `chunk + 2·overlap`. The platform's *default* chunk (10 s) with 0.5 s
overlap yields 11 s and is safely zero-phase. However, the chunk size is configurable down to
0.5 s, so a sufficiently small chunk drops below the floor (e.g., a 1 s chunk with 0.5 s overlap
spans 400 < 606 samples) and the filter is no longer zero-phase, with large error throughout the
window rather than only at seams (interior RMSE 30.7 µV).

### R3. Magnitude fidelity, and up to ~500 ms event mis-timing in the fallback

On the zero-phase path the magnitude response realizes the intended design: passband (1–25 Hz)
gain −0.64 ± 1.54 dB, 50 Hz notch −120 dB, 60 Hz stopband −120 dB (Fig. 4). The temporal behavior,
however, is regime-dependent: a 20 ms transient is unshifted (0 ms) on the zero-phase path but is
displaced by **+500 ms** on the causal fallback path, equal to the uncompensated linear-phase FIR
group delay `(taps−1)/2/fs ≈ 502 ms` (Fig. 5). Combined with R2, whenever the streaming filter
enters the fallback regime, displayed EEG events are mis-timed by about half a second.

### R4. Channel canonicalization and feature integrity (synthetic)

On 19 curated real-world channel-name variants, family classification is 89.5% accurate. Of the
two disagreements, classifying `EEG A1-REF` as a reference site is correct (ear/mastoid reference,
not scalp EEG), while `POL Fp1` classified as auxiliary is a genuine gap: the Nihon-Kohden `POL`
prefix is not stripped, so such EEG channels would be wrongly excluded from EEG montages.
Computing an average-reference montage without excluding a mislabeled ECG channel inflates the
target channel's alpha band power by 48.7%, confirming that channel-aware classification must
precede montaging.

### R5. Bandpower equivalence versus a reference, and a double-taper bias

The production PSD routine (Welch, 256/128) and the reference `psd_array_welch` produce strongly
correlated per-band relative powers (Pearson r = 0.991), but the production estimate differs by up
to 11.9 pp on the dominant band. The cause, verified by ablation, is a *double taper*: the routine
applies a Hann window to the whole analysis window and then calls Welch, which applies its own
per-segment Hann. Disabling the redundant manual taper reduces the maximum discrepancy from
11.9 pp to 0.64 pp, at which point the production estimate matches the reference. The bias is
estimator-agnostic: on the multitaper path the discrepancy is 9.1 pp with the manual taper and
0.0 pp without it.

### R6. Real-data confirmation of R1–R3 (CHB-MIT)

On CHB-MIT `chb01_01` (256 Hz), overlap-add reduces seam boundary RMSE from 12.96 µV to 0.049 µV
(48.5 dB) — a larger effect than on synthetic data — with a near-zero interior error confirming
zero-phase behavior (Fig. 8). The `filtfilt` floor is 606 samples, i.e. 2.37 s at 256 Hz; because
the floor is in samples its duration scales with sampling rate. The fallback group-delay shift,
recovered by cross-correlation, is +391 ms, matching the predicted `(taps−1)/2/fs = 393 ms`; the
mis-timing is therefore sampling-rate-dependent (~393 ms at 256 Hz versus ~500 ms at 200 Hz).

### R7. Generalization across FIR order, the transition, and robustness

The zero-phase floor equals `3·(order+1)` taps and the fallback shift equals `(taps−1)/2/fs`; both
scale with FIR order. Sweeping the order over {100, 200, 300, 400} gives floors of
{306, 606, 906, 1206} samples and shifts of {255, 505, 755, 1005} ms (Fig. 6), generalizing the
design criterion `chunk + 2·overlap ≥ 3·taps` to any FIR order. Sweeping the total filtered length
through the floor shows the boundary error collapse sharply at the threshold (Fig. 7). Across 12
seeds, the overlap-add seam-error reduction is 35.1 ± 2.3 dB (naive 15.9 ± 4.1 µV; overlap-add
0.273 ± 0.035 µV), confirming the effect is robust.

### R8. Real-data confirmation of R4–R5 (Siena)

On Siena `PN00-1` (512 Hz), which contains 29 monopolar 10–20 EEG channels alongside a genuine
`EKG` channel and SpO2/HR, canonicalization separates all 29 EEG electrodes from the six genuine
non-EEG channels. Including the real ECG channel in the average reference biases the relative alpha
power of EEG C3 by +127.9% (3.10% → 7.08%) — larger than the synthetic case — confirming on real
data that channel-aware exclusion is required for correct quantitative features. (The bias
magnitude is recording-dependent; its direction and mechanism are robust.)

## V. Discussion

The results reduce to a small set of correctness invariants for streaming clinical EEG DSP.
(1) *Windowing:* always supply real context and retain only the center (overlap-add); never filter
display windows independently. (2) *Zero-phase length:* guarantee `(chunk + 2·overlap)·fs ≥
3·len(taps)` so the filter never silently leaves the zero-phase regime, and make the fallback
observable; otherwise event timing — central to spike review and annotation alignment — can be off
by hundreds of milliseconds. (3) *Spectral estimation:* taper exactly once. (4) *Montaging:*
canonicalize channels and exclude non-EEG families before averaging. These are implementation
invariants, not new algorithms, but each is violated by a plausible, deployed configuration, and
each violation is silent.

*Threats to validity.* R1–R3 and R7 rest on deterministic properties of the filter implementation
(group delay, the `filtfilt` length floor) and hold independent of signal content; we nonetheless
confirm them on real EEG (R6). The magnitudes in R4, R5, and R8 are configuration- and
recording-dependent (e.g., the +127.9% alpha bias reflects a low-alpha subject), but their
direction and mechanism are robust and reproduced on real data. We evaluate one platform's
implementation; the invariants, however, follow from properties of `filtfilt`, linear-phase FIR
group delay, and Welch/multitaper tapering common to widely used libraries.

## VI. Conclusion

Moving routine EEG DSP into a windowed, real-time, multi-user setting introduces silent
correctness failures that standard offline practice does not anticipate. By auditing a deployed
clinical platform against synthetic ground truth and two open datasets, we identify and quantify
four — seam artifacts, a zero-phase length-floor fallback with ~500 ms event mis-timing, a
double-taper spectral bias, and non-EEG montage contamination — and give deterministic,
low-cost remedies. Future work will extend the real-data confirmation to larger cohorts and to the
AI-detection path of the platform.

## References

*To be completed from the verified literature-review themes in `PAPER_OUTLINE.md` (§3).
Anchor candidates to confirm: EDF/EDF+ [Kemp & Värri 1992; Kemp & Olivan 2003]; Widmann, Schröger
& Maess 2015; de Cheveigné & Nelken 2019; Tanner et al. 2015; Oppenheim & Schafer (DTSP); Proakis
& Manolakis; Virtanen et al. 2020 (SciPy); Jasper 1958 / Klem et al. 1999 (10–20); Pernet et al.
2019 (EEG-BIDS); Welch 1967; Thomson 1982; Hellström-Westas et al. (aEEG); Gramfort et al. 2013
(MNE-Python); Obeid & Picone 2016 (TUH); Goldberger et al. 2000 (PhysioNet); CHB-MIT [Shoeb 2009];
Siena Scalp EEG [Detti et al. 2020]. Verify every entry before inclusion.*
