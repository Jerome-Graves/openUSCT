"""Transducer transmit model: from a pulser excitation to an acoustic wavelet.

A piezoelectric transducer does not emit the electrical drive directly; it emits
its impulse response to it. Modelling that response (a band-limited resonance)
turns the FPGA pulser's bipolar square-wave excitation into the acoustic wavelet
that actually propagates. This closes the transmit path: the pulse the digital
logic generates is the pulse the simulation launches.
"""

from __future__ import annotations

import numpy as np


def pulser_excitation(half_period, n_halfcycles, dead_time):
    """Digital bipolar excitation at clock resolution (+1 / -1 / 0 per clock).

    Matches ``tx_pulser.sv``: each half-cycle is ``dead_time`` idle clocks then
    the active polarity, alternating positive and negative.
    """
    x = np.zeros(int(n_halfcycles) * int(half_period))
    for hc in range(int(n_halfcycles)):
        for cnt in range(int(half_period)):
            if cnt >= dead_time:
                x[hc * half_period + cnt] = 1.0 if hc % 2 == 0 else -1.0
    return x


def transducer_impulse_response(fc, frac_bw, fs, n=None):
    """Gaussian-modulated cosine impulse response (a standard transducer model).

    fc : centre frequency (Hz); frac_bw : fractional -6 dB bandwidth; fs : Hz.
    """
    sigma_f = frac_bw * fc / 2.3548        # FWHM -> Gaussian sigma in frequency
    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)
    if n is None:
        n = int(6.0 * sigma_t * fs) | 1    # odd length window
    t = (np.arange(n) - n // 2) / fs
    h = np.exp(-t ** 2 / (2.0 * sigma_t ** 2)) * np.cos(2.0 * np.pi * fc * t)
    s = np.sum(np.abs(h))
    return h / s if s > 0 else h


def acoustic_wavelet(exc_clk, clk_hz, fs, fc, frac_bw, n_out=None):
    """Resample a clock-rate excitation to ``fs`` and convolve with the transducer.

    Returns the normalised acoustic transmit wavelet at sample rate ``fs``.
    """
    return transmit_chain(exc_clk, clk_hz, fs, fc, frac_bw, n_out=n_out)


# --- FPGA-producible excitations ---------------------------------------------
# A digital pulser cannot synthesise an arbitrary analytic wavelet: it drives
# discrete levels {-1, 0, +1} at clock resolution. These generators cover the
# standard experimental drive signals; everything downstream (filters and the
# transducer response) is what turns them into the acoustic output.

def unipolar_pulse(width_clk):
    """Single unipolar spike of ``width_clk`` clocks (spike/impulse excitation)."""
    return np.ones(int(max(1, width_clk)))


def square_chirp(clk_hz, f_lo, f_hi, dur_s):
    """Bipolar square linear-FM chirp at clock resolution (coded excitation)."""
    nclk = int(round(dur_s * clk_hz))
    t = np.arange(nclk) / clk_hz
    k = (f_hi - f_lo) / dur_s
    phase = 2.0 * np.pi * (f_lo * t + 0.5 * k * t * t)
    return np.sign(np.sin(phase) + 1e-12)


# --- Analogue filters ---------------------------------------------------------

def lowpass(sig, fs, f_cut, order=4):
    """Causal Butterworth low-pass (models the TX output/matching filter)."""
    from scipy.signal import butter, sosfilt

    wn = min(f_cut / (fs / 2.0), 0.99)
    sos = butter(order, wn, btype="low", output="sos")
    return sosfilt(sos, sig)


def bandpass(sig, fs, f_lo, f_hi, order=2, axis=-1):
    """Causal Butterworth band-pass (models the RX analogue front end)."""
    from scipy.signal import butter, sosfilt

    lo = max(f_lo / (fs / 2.0), 1e-6)
    hi = min(f_hi / (fs / 2.0), 0.99)
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfilt(sos, sig, axis=axis)


def transmit_chain(exc_clk, clk_hz, fs, fc, frac_bw, n_out=None,
                   tx_cut_hz=None, tx_order=4):
    """Full transmit path: discrete FPGA drive -> TX filter -> transducer.

    The clock-rate excitation is held zero-order onto the simulation sample
    rate ``fs`` (still a stair-step: what the pulser really outputs), optionally
    smoothed by the TX output filter, then convolved with the transducer
    impulse response. Returns the normalised acoustic transmit wavelet.
    """
    exc_clk = np.asarray(exc_clk, float)
    nclk = len(exc_clk)
    dur = nclk / clk_hz
    n_exc = int(np.ceil(dur * fs))
    t = np.arange(n_exc) / fs
    idx = np.floor(t * clk_hz).astype(int)
    exc = np.where(idx < nclk, exc_clk[np.clip(idx, 0, nclk - 1)], 0.0)

    if tx_cut_hz is not None:
        exc = lowpass(exc, fs, tx_cut_hz, tx_order)

    h = transducer_impulse_response(fc, frac_bw, fs)
    w = np.convolve(exc, h)                # transducer emits its response
    m = np.max(np.abs(w))
    if m > 0:
        w = w / m
    if n_out is not None:
        if len(w) >= n_out:
            w = w[:n_out]
        else:
            w = np.pad(w, (0, n_out - len(w)))
    return w
