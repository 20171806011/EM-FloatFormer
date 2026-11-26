"""One-call quarter-turn equivariance for the published s17 two-window code.

External wrapper for four-candidate distributions. Pilot data and metadata
remain in their original coordinates. Free text is outside this interface.
"""


def canonical_input(z1, z2, task):
    """Return the two canonical indices and the output right-shift."""
    if not all(isinstance(z, int) and 0 <= z < 16 for z in (z1, z2)):
        raise ValueError("Expected two integer s17 indices in [0, 15]")
    turn1, inner1 = divmod(z1, 4)
    turn2, inner2 = divmod(z2, 4)
    if task == "q1":
        shift = turn2 - turn1
    elif task in ("q2", "q3_repeat"):
        shift = turn1
    else:
        raise ValueError("Supported tasks are q1, q2 and q3_repeat")
    return inner1, inner2, shift % 4


def rotate(values, shift):
    if len(values) != 4:
        raise ValueError("Expected four scores")
    return [values[(k-shift) % 4] for k in range(4)]


def one_call(reader, z1, z2, task, context):
    """reader takes canonical codes, task and public context, then returns four scores."""
    i, j, shift = canonical_input(z1, z2, task)
    return rotate(reader(i, j, task, context), shift)
