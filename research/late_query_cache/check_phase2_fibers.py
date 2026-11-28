"""Check the continuous sensor's explicit encoder on a deterministic grid."""

import itertools
import math


def midpoint(value, lower, upper, width):
    count = math.ceil((upper - lower) / width)
    index = min(count - 1, math.floor((value - lower) / width))
    left = lower + index * width
    return (left + min(left + width, upper)) / 2


def check(precision):
    radius = 2.0 ** -precision
    length = math.log(2)
    step = math.sqrt(radius / 2)
    anchor_count = 1 + math.floor(length / step)
    residual_count = math.ceil((1 + 2 * length) / radius)
    grid = [i / 16 for i in range(17)]
    questions = [0, 0.125, 0.5, 0.875, 1]
    worst = 0.0
    for x, y, z in itertools.product(grid, repeat=3):
        s, v = math.log1p(x), math.log1p(y)
        anchor = math.floor(s / step) * step
        weight = math.exp(anchor)
        quantized = midpoint(z - weight * v, -2 * length, 1, radius)
        for t in questions:
            observed = v - t * s
            estimate = quantized + weight * observed + t * weight * (anchor - 1)
            target = z - t * (1 + x)
            worst = max(worst, abs(estimate - target))
    assert worst <= radius + 1e-13
    return (precision, (anchor_count * residual_count - 1).bit_length(),
            len(grid) ** 3 * len(questions), worst / radius)


if __name__ == "__main__":
    print("log2(1/r) fixed_bits source_question_pairs max_abs_error_over_r")
    for precision in (4, 8, 12, 16):
        print(*check(precision))
