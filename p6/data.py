"""Frame-generation boundary for Phase 6 development experiments."""
from p5.study import rng
from p5.wavefield_sim import simulate

PAIRS = (901, 902, 903)
SNR = {1: 12., 2: 9., 3: 7.}


def dev_seed(regime, stream, pair, batch=0):
    if type(regime) is not int or regime not in SNR:
        raise ValueError("unknown regime")
    if type(pair) is not int or pair not in PAIRS:
        raise ValueError("Phase 6 development requires pair 901, 902 or 903")
    if type(stream) is not int or stream not in ((5, 6, 8, 9) if regime == 3 else (5, 6)):
        raise ValueError("stream outside Phase 6 development split")
    if type(batch) is not int or not 0 <= batch < 1000:
        raise ValueError("batch index crosses pair seed range")
    return rng(regime, stream, pair, batch)


def dev_frame(regime, stream, pair, batch_size, batch=0, snr=None):
    seed = dev_seed(regime, stream, pair, batch)
    if stream == 9:
        raise ValueError("stream 9 supplies R3 permutation seeds only")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive integer batch size required")
    return simulate(regime, batch_size, SNR[regime] if snr is None else snr, seed)
