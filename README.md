# Windowed EEG Benchmark

This repository accompanies the SPMB submission titled *A Correctness and Runtime Benchmark for
Windowed Real-Time EEG Signal Processing: Phase, Windowing, and Spectral Estimation*.

It is a classical signal-processing study with no machine learning. The core question is what
breaks when standard EEG DSP is applied window-by-window instead of to a whole recording. The key
failure modes are silent: a zero-phase filter may stop being zero-phase on short windows, and
spectral estimates can be biased if a window is tapered twice.

The repository contains the paper, the experiments, and the poster materials:

- [Paper](paper.tex) and [references](references.bib)
- [Experiment scripts and figures](experiments/)
- [Benchmark findings](experiments/FINDINGS.md)
- [Poster plan](Poster/README.md)

## What this work shows

- Overlap-add removes window seam artifacts with only a small latency cost.
- `filtfilt` has a real length floor, so short windows need a deterministic safety check.
- Causal fallback introduces measurable event-timing error when the zero-phase floor is not met.
- Tapering twice biases relative band-power estimates; taper once.
- The safe-window scheduler keeps the streaming path far below real time on synthetic and clinical
  EDF data.

## Repository contents

- `paper.tex` - paper source.
- `references.bib` - bibliography.
- `experiments/` - NumPy/SciPy/MNE scripts for the DSP reference implementation, safe-window
  scheduler, EDF runtime benchmark, and figures.
- `Poster/` - poster-focused summary that isolates the strongest filtering result.

## Experiments at a glance

- `experiments/dsp.py` - reference FIR bandpass and notch filtering.
- `experiments/safe_window.py` - zero-phase length-floor guardrail.
- `experiments/windowing.py` - whole-signal, naive per-window, and overlap-add filtering.
- `experiments/synth.py` - synthetic ground truth signals.
- `experiments/benchmark_edf_lengths.py` - EDF-duration throughput benchmark.
- `experiments/benchmark_latency.py` - direct per-window latency benchmark.
- `experiments/exp_edge_policy.py` - true-boundary and interior edge policies.

## Data

The benchmark uses synthetic EEG plus two open clinical datasets for external validity:

- PhysioNet CHB-MIT EEG, 256 Hz
- Sleep-EDF Expanded, 100 Hz

The EDF files are downloaded at run time and are not committed to the repository.

## Reproducing the results

The findings are documented in [experiments/FINDINGS.md](experiments/FINDINGS.md). The exact
package versions are pinned in `experiments/requirements.txt`.

Typical entry points:

```bash
cd experiments
python3 run_boundary_experiment.py
python3 exp_filter_fidelity.py
python3 exp_regime.py
python3 exp_bandpower_equiv.py
python3 exp_multisubject.py
python3 exp_sleepedf.py
python3 benchmark_edf_lengths.py
python3 benchmark_latency.py
python3 exp_edge_policy.py
```

If `mne` fails to import because of `numba`, set `NUMBA_DISABLE_JIT=1` before running the
experiments.

## Status

The paper draft, benchmark scripts, and poster materials are all in place. The main results are
already summarized in `experiments/FINDINGS.md` and can be reused directly in the submission.
