# Option 2 (Non-AI / Classical-DSP) — Paper Plan

**Working title:** *Artifact-Free Real-Time EEG Re-Montaging and Quantitative Feature Extraction
over a Windowed Datastore*

**SPMB track:** Paper (original, ~4–6 pages, oral presentation).

This plan collapses four classical signal-processing and systems capabilities into one
narrative, with no machine learning: channel canonicalization (`domain/helpers/channels.py`),
real-time overlap-add montage filtering (`domain/helpers/filters.py` plus
`file-service/src/services/streaming/helpers.py`), quantitative derived features
(bandpower and aEEG in `workers/`), and a memory-bounded streaming server
(`file-service/src/core/garbage_collector.py`).

---

## 1. Problem and motivation

Browser-based EEG review streams short signal windows on demand. Naïve per-window filtering
introduces boundary/edge artifacts; inconsistent channel naming corrupts montage computation
and every downstream quantitative feature; and serving many concurrent viewers can exhaust
memory. We present a deterministic pipeline that guarantees inter-window continuity (overlap-add),
correct re-referencing, and reproducible bandpower/aEEG, while bounding memory under concurrent
load — a signal-processing and systems contribution that needs no training data.

## 2. Research questions and hypotheses

- **RQ1 (continuity).** Does overlap-add windowed filtering eliminate boundary artifacts versus
  naïve per-window filtering, at equal end-to-end latency? (Motivated by the corrections recorded
  in `file-service/changelogs/OVERLAP_ADD_FIX_APPLIED.md` and the polarity/validation notes.)
- **RQ2 (correctness).** Does channel canonicalization (`normalize_channel_name`,
  `build_channel_map`) produce montage and feature outputs that match a reference implementation
  (e.g., MNE) on recordings whose raw channel names diverge?
- **RQ3 (scale).** What is the latency/throughput/peak-memory envelope of the connection-aware
  garbage collector under N concurrent WebSocket sessions?

## 3. Background and related work

Position against standard EEG filtering (FIR vs. IIR, zero-phase vs. causal), block-convolution
methods (overlap-add / overlap-save), montage/re-referencing conventions, and quantitative EEG
features (relative band power, amplitude-integrated EEG). The contribution is doing all of this
**correctly in a streaming, multi-tenant setting** and quantifying it.

## 4. Methods

**Channels.** `domain/helpers/channels.py` — `ChannelFamily`, `filter_channels_for_montage`,
and deterministic display ordering (`get_ordered_channel_sequence`, `order_plot_channels`).

**Filtering.** FIR bandpass (order ~200) and IIR notch at 50/60 Hz
(`_get_bandpass_coefficients` at `domain/helpers/filters.py:37`, `_get_notch_coefficients` at
`:60`), applied globally and per channel; overlap-add continuity in the streaming hot path
(`file-service/src/services/streaming/helpers.py`, `_query_window_direct_fast` at `:320`).

**Montage.** Bipolar, average, ear, Cz, Laplacian, and custom references
(`domain/helpers/montage.py`).

**Derived features.** Per-second δ/θ/α/β/γ bandpower and amplitude-integrated EEG
(`workers/src/utils/extraction/band_power_optimized.py`,
`workers/src/utils/extraction/aeeg_processor.py`).

**Systems.** Metadata cache, active-connection tracking, and RAM-pressure-triggered clearing
(`file-service/src/core/garbage_collector.py`: `get_memory_usage_mb` at `:58`,
`connection_opened`/`connection_closed` at `:128`/`:144`).

## 5. Data

Public EDF corpora used purely as signals (e.g., TUH normal/abnormal, CHB-MIT), plus
**synthetic signals with known spectra** to provide ground truth for filter-fidelity and
overlap-add boundary measurements.

## 6. Experimental design

1. **Continuity (RQ1):** measure boundary-artifact energy at window edges for overlap-add vs.
   naïve windowing, across multiple window sizes and filter settings, holding latency constant.
2. **Correctness (RQ2):** compare montage and bandpower/aEEG outputs against a reference (MNE)
   on recordings with harmonized vs. raw channel names.
3. **Scale (RQ3):** load-test the streaming server, sweeping the concurrent-session count with
   GC on and off; record latency, throughput, and peak memory.

## 7. Evaluation metrics

- Edge-artifact energy reduction (dB) and visible discontinuity at window boundaries.
- Spectral error versus synthetic ground truth (magnitude/phase fidelity).
- Feature agreement versus reference (intraclass correlation, Bland–Altman).
- End-to-end window latency (p50/p95), sustained throughput, and peak memory vs. session count.

## 8. Expected contributions

1. A reproducible real-time montage/filter method with **quantified** boundary-artifact
   elimination at no latency cost.
2. Evidence that channel canonicalization is necessary for correct quantitative features.
3. A characterized, memory-bounded multi-tenant streaming design.
4. An open reference implementation and measurement harness.

## 8b. Preliminary results (already obtained — see `experiments/`)

Driving the real `domain` production filters on synthetic ground truth
(`experiments/FINDINGS.md`):

1. **Overlap-add eliminates seam artifacts:** in the zero-phase regime (8 s windows), boundary
   error drops 14.93 µV → 0.275 µV (**≈35 dB**) at **negligible latency cost** (sub-millisecond
   per window; overhead within measurement noise).
2. **A `filtfilt` length floor governs the whole thing:** the FIR order-200 bandpass uses
   zero-phase `filtfilt` only at ≥ 606 samples (3.03 s); below that it silently falls back to
   causal `lfilter`. The streaming default chunk (10 s + 0.5 s overlap = 11 s) is safely above
   the floor, but `chunk_size` is configurable down to 0.5 s, and small chunks silently cross
   into the fallback regime.
3. **Event mis-timing in the fallback regime:** the fallback path imposes a **+500 ms**
   uncompensated group-delay shift (zero-phase path: 0 ms); magnitude response is otherwise
   faithful (passband −0.64 dB, notch −120 dB).
4. **Channel canonicalization matters for features:** 89.5% family-classification accuracy on
   real-world name variants (with a `POL`-prefix gap), and omitting channel-aware exclusion of a
   mislabeled ECG channel inflates average-reference alpha bandpower by **+48.7%**.

These elevate the paper from "overlap-add fixes seams" to a characterization of the
order/floor/window/overlap interaction with a concrete clinical-impact metric (event timing).

## 9. Timeline (≈12 weeks)

| Weeks | Milestone |
|---|---|
| 1–2 | Measurement harness + synthetic ground-truth signals |
| 3–5 | Overlap-add boundary study (RQ1) |
| 6–7 | Filter magnitude/phase fidelity |
| 8–9 | Montage/feature equivalence vs. reference (RQ2) |
| 10–11 | Load + memory characterization (RQ3) |
| 12 | Writing, figures, internal review |

## 10. Risks and mitigations

- **Reference-convention mismatch** (filter design, band edges) → fix shared coefficients and
  document band definitions before comparison.
- **Load-test variability** → fixed hardware, repeated trials, reported confidence intervals.
- **Feature-definition differences vs. MNE** → align and document band edges and windowing.

## 11. Reproducibility

Release the DSP reference implementation, synthetic-signal generators, the load-test harness,
and all configurations. No datasets with PHI are required.

## 12. Why this fits SPMB as a Paper

A complete signal-processing-plus-systems study with measurable results (artifact energy,
spectral fidelity, feature agreement, and a latency/memory envelope) and a strong
reproducibility story — appropriate for a 4–6 page paper and an oral. The companion **Poster**
(see `Poster/README.md`) isolates the single sharpest result.
