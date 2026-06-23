"""
Adaptive safe-window scheduler for windowed EDF/EEG filtering.

The paper's core guardrail is that a display chunk plus its real-sample context must be long
enough for zero-phase filtering. SciPy's ``filtfilt`` uses a default padding length of
``3 * len(taps)`` samples and requires the input length to be strictly greater than that
padding length. This module turns that implementation fact into a small runtime contract:

  - identify whether a configured chunk/overlap is zero-phase safe;
  - compute the minimum overlap that keeps the filter out of the causal fallback regime;
  - expose the fallback group-delay that would affect event timing if the guard is ignored.

It is intentionally independent of any product code. The benchmark scripts import it, and it
can also be run directly:

    python3 safe_window.py --fs 256 --chunk-s 1.0 --overlap-s 0.5 --order 200
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


def fir_tap_count(order: int = 200) -> int:
    """Return the tap count used by ``dsp.bandpass_coefficients`` for a requested order."""
    effective_order = int(order)
    if effective_order % 2 == 0:
        effective_order += 1
    return effective_order + 1


def filtfilt_padlen_samples(taps: int) -> int:
    """SciPy default pad length for the FIR numerator and denominator ``[1.0]``."""
    return 3 * int(taps)


def filtfilt_min_valid_samples(taps: int) -> int:
    """Shortest input length accepted by ``filtfilt`` with the default pad length."""
    return filtfilt_padlen_samples(taps) + 1


def causal_group_delay_ms(taps: int, fs: float) -> float:
    """Uncompensated group delay of a causal linear-phase FIR fallback."""
    return 1e3 * ((int(taps) - 1) / 2.0) / float(fs)


def _samples(seconds: float, fs: float) -> int:
    return int(round(float(seconds) * float(fs)))


@dataclass(frozen=True)
class SafeWindowPlan:
    fs: float
    order: int
    taps: int
    chunk_s: float
    requested_overlap_s: float
    chunk_samples: int
    requested_overlap_samples: int
    padlen_samples: int
    min_valid_samples: int
    required_overlap_samples: int
    safe_overlap_samples: int
    safe_overlap_s: float
    configured_span_samples: int
    configured_span_s: float
    scheduled_span_samples: int
    scheduled_span_s: float
    configured_filtfilt_safe: bool
    scheduled_filtfilt_safe: bool
    one_sided_edge_span_samples: int
    one_sided_edge_filtfilt_safe: bool
    fallback_group_delay_ms: float
    action: str
    edge_note: str

    def to_dict(self) -> dict:
        return asdict(self)


def plan_window(fs: float, chunk_s: float, requested_overlap_s: float,
                order: int = 200) -> SafeWindowPlan:
    """Return the smallest overlap that keeps interior windows zero-phase safe.

    The interior-window span is ``chunk + 2 * overlap``. The first and last windows of a record
    may have only one-sided real context unless the caller prefetches outside the displayed
    interval or uses an explicit boundary policy. That edge status is reported separately so a
    pipeline can make boundary fallback observable instead of silently mixing regimes.
    """
    if fs <= 0:
        raise ValueError("fs must be positive")
    if chunk_s <= 0:
        raise ValueError("chunk_s must be positive")
    if requested_overlap_s < 0:
        raise ValueError("requested_overlap_s must be non-negative")

    taps = fir_tap_count(order)
    min_valid = filtfilt_min_valid_samples(taps)
    chunk_samples = _samples(chunk_s, fs)
    requested_overlap_samples = _samples(requested_overlap_s, fs)
    required_overlap_samples = max(0, math.ceil((min_valid - chunk_samples) / 2.0))
    safe_overlap_samples = max(requested_overlap_samples, required_overlap_samples)

    configured_span_samples = chunk_samples + 2 * requested_overlap_samples
    scheduled_span_samples = chunk_samples + 2 * safe_overlap_samples
    one_sided_edge_span_samples = chunk_samples + safe_overlap_samples

    configured_safe = configured_span_samples >= min_valid
    scheduled_safe = scheduled_span_samples >= min_valid
    edge_safe = one_sided_edge_span_samples >= min_valid
    action = "keep_configured_overlap" if configured_safe else "increase_overlap"
    edge_note = (
        "first/last windows are safe with one-sided real context"
        if edge_safe else
        "first/last windows need an explicit boundary policy or extra one-sided context"
    )

    return SafeWindowPlan(
        fs=float(fs),
        order=int(order),
        taps=taps,
        chunk_s=float(chunk_s),
        requested_overlap_s=float(requested_overlap_s),
        chunk_samples=chunk_samples,
        requested_overlap_samples=requested_overlap_samples,
        padlen_samples=filtfilt_padlen_samples(taps),
        min_valid_samples=min_valid,
        required_overlap_samples=required_overlap_samples,
        safe_overlap_samples=safe_overlap_samples,
        safe_overlap_s=safe_overlap_samples / float(fs),
        configured_span_samples=configured_span_samples,
        configured_span_s=configured_span_samples / float(fs),
        scheduled_span_samples=scheduled_span_samples,
        scheduled_span_s=scheduled_span_samples / float(fs),
        configured_filtfilt_safe=configured_safe,
        scheduled_filtfilt_safe=scheduled_safe,
        one_sided_edge_span_samples=one_sided_edge_span_samples,
        one_sided_edge_filtfilt_safe=edge_safe,
        fallback_group_delay_ms=causal_group_delay_ms(taps, fs),
        action=action,
        edge_note=edge_note,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Plan safe overlap for windowed EEG filtering.")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument("--chunk-s", type=float, default=1.0, help="Displayed chunk length")
    parser.add_argument("--overlap-s", type=float, default=0.5, help="Configured context per side")
    parser.add_argument("--order", type=int, default=200, help="Requested FIR order")
    args = parser.parse_args()

    plan = plan_window(args.fs, args.chunk_s, args.overlap_s, args.order)
    print(json.dumps(plan.to_dict(), indent=2))


if __name__ == "__main__":
    _main()
