"""Q2 at the pinned C4 condition; inputs are the two integer cache codes and pilot matched sums."""
import argparse
import cmath
import json
import math
from pathlib import Path


KERNEL_PATH = Path(__file__).with_name("q2_s17_kernel.json")


def read(kernel, z1, z2, references):
    """Each reference is (acquisition ID, shares_channel, complex matched sum).

    This kernel is calibrated for same-time 2 dB pilots at t=4. Keep acquisition
    IDs when making copies; assign new IDs to independent measurement noise.
    """
    if any(not isinstance(z, int) or not 0 <= z < 16 for z in (z1, z2)):
        raise ValueError("Both cache indices must be integers in 0..15")
    seen = {}
    for identifier, same_channel, matched_sum in references:
        t = complex(matched_sum)
        if not math.isfinite(t.real) or not math.isfinite(t.imag):
            raise ValueError("Non-finite matched sum")
        record = (same_channel, t)
        if identifier in seen and seen[identifier] != record:
            raise ValueError("Conflicting metadata or value for a repeated acquisition ID")
        seen[identifier] = record
    valid = [t for same_channel, t in seen.values() if same_channel]
    if not valid:
        return [0.25]*4
    weighted_t = 2*kernel["rho"]*sum(valid)
    # Stabilize likelihood scale with one common maximum across all g, phase nodes.
    ll = [[(weighted_t*(1j**(-g))*cmath.exp(-1j*ph)).real for g in range(4)]
          for ph in kernel["pilot_phase_nodes"]]
    shift = max(max(row) for row in ll)
    carrier = [sum(w*math.exp(row[g]-shift)
                   for w,row in zip(kernel["weights_given_z2"][z2], ll)) for g in range(4)]
    norm = sum(carrier)
    return [sum(carrier[g]*kernel["u_given_z1"][z1][(k+g)%4] for g in range(4))/norm
            for k in range(4)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("z1", type=int)
    p.add_argument("z2", type=int)
    p.add_argument("--pilot", nargs=3, action="append", metavar=("ID","REAL","IMAG"), default=[])
    args = p.parse_args()
    kernel = json.loads(KERNEL_PATH.read_text(encoding="utf-8"))
    refs = [(idx, True, complex(float(a),float(b))) for idx,a,b in args.pilot]
    print(json.dumps(read(kernel, args.z1, args.z2, refs)))
