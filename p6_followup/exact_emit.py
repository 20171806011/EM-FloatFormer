"""Optional exact-arithmetic R2 emission reduction; frozen P6 stays unchanged.

The R2 transmitter has no power amplifier.  Let E insert the occupied Fourier
bins into an oversampled spectrum, S extract those same bins, and F be the
unitary oversampled DFT.  The P6 emission is

    (1 / sqrt(2)) S F F* (sqrt(2) E x) = S E x = x.

Returning x removes that FFT round trip and its temporary arrays.  The value
and Jacobian agree in exact arithmetic, but FFT roundoff means this is not a
claim of bitwise agreement with the frozen implementation.  This does not
change the later sample-time Doppler/ICI waveform construction.
"""

from contextlib import contextmanager

from p6 import axial_em_transformer as solver


_FROZEN_EMIT = solver._emit


def emit(x, regime, radial):
    """Use the identity in R2 and the frozen P6 implementation in R1/R3.

    The R2 return aliases x.  Receiver callers must treat it as read-only,
    as the frozen P6 forward does.  The unused R2 radial input retains its
    original absent gradient; no zero-gradient dependency is introduced.
    """
    if regime == 2:
        return x
    return _FROZEN_EMIT(x, regime, radial)


@contextmanager
def use_exact_r2_emit(enabled=True):
    """Temporarily select the reduction for an existing P6 receiver.

    Restores the previous binding even on exceptions and supports nesting.
    This changes a process-global module binding: use it in a single receiver
    execution thread, or use separate processes for concurrent experiments.
    It never edits historical P6 source files or model/checkpoint parameters.
    """
    if not enabled:
        yield
        return
    previous = solver._emit
    solver._emit = emit
    try:
        yield
    finally:
        solver._emit = previous
