"""Verify the FPGA-realistic transmit chain and RX filtering.

The pulser can only drive discrete levels {-1, 0, +1} at clock resolution;
the acoustic wavelet must come out of the chain excitation -> TX filter ->
transducer response, band-limited around the transducer centre frequency.
"""

from __future__ import annotations

import numpy as np

from ringfwi import transducer as td


CLK = 100e6
FS = 1.0 / 7.0e-8          # ~14.3 MHz simulation sample rate
F0 = 0.4e6


def _peak_freq(w, fs):
    spec = np.abs(np.fft.rfft(w))
    return np.fft.rfftfreq(len(w), 1.0 / fs)[int(np.argmax(spec))]


def test_excitations_are_discrete():
    hp = int(round(CLK / (2 * F0)))
    burst = td.pulser_excitation(hp, 4, 2)
    assert set(np.unique(burst)) <= {-1.0, 0.0, 1.0}
    spike = td.unipolar_pulse(hp)
    assert set(np.unique(spike)) <= {0.0, 1.0}
    chirp = td.square_chirp(CLK, 0.5 * F0, 1.5 * F0, 6 / F0)
    assert set(np.unique(chirp)) <= {-1.0, 1.0}


def test_chain_bandlimits_square_drive():
    hp = int(round(CLK / (2 * F0)))
    burst = td.pulser_excitation(hp, 4, 2)
    w = td.transmit_chain(burst, CLK, FS, F0, 0.6, n_out=600, tx_cut_hz=2.5 * F0)
    assert len(w) == 600 and np.isfinite(w).all()
    assert abs(np.abs(w).max() - 1.0) < 1e-12          # normalised
    fpk = _peak_freq(w, FS)
    assert 0.6 * F0 < fpk < 1.5 * F0                   # energy at the transducer fc
    # The transducer+filter must suppress the square wave's 3rd harmonic.
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), 1.0 / FS)
    h3 = spec[np.argmin(np.abs(freqs - 3 * F0))]
    assert h3 < 0.05 * spec.max()


def test_rx_bandpass_cleans_out_of_band():
    rng = np.random.default_rng(0)
    nt = 800
    sig = np.sin(2 * np.pi * F0 * np.arange(nt) / FS) + 0.8 * rng.standard_normal(nt)
    data = np.tile(sig, (2, 1)).T[None, :, :]           # (1, nt, 2) like FMC data
    out = td.bandpass(data, FS, 0.3 * F0, 2.2 * F0, order=2, axis=1)
    assert out.shape == data.shape
    S_in = np.abs(np.fft.rfft(data[0, :, 0]))
    S_out = np.abs(np.fft.rfft(out[0, :, 0]))
    freqs = np.fft.rfftfreq(nt, 1.0 / FS)
    hiband = freqs > 4 * F0
    assert S_out[hiband].sum() < 0.2 * S_in[hiband].sum()   # HF noise removed
    k0 = np.argmin(np.abs(freqs - F0))
    assert S_out[k0] > 0.5 * S_in[k0]                        # signal band kept


if __name__ == "__main__":
    test_excitations_are_discrete()
    test_chain_bandlimits_square_drive()
    test_rx_bandpass_cleans_out_of_band()
    print("transmit chain checks passed")
