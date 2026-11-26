"""Finite checks for candidate decisions, hard branches and language rendering."""
import json
import math


def kl(p, q):
    return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)


def cross_entropy(p, q):
    return -sum(a * math.log(b) for a, b in zip(p, q) if a > 0)


def brier(p, q):
    return sum(weight * sum((r - (i == y)) ** 2 for i, r in enumerate(q))
               for y, weight in enumerate(p))


def close(x, y):
    if not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError((x, y))


def winner(p):
    return max(range(len(p)), key=p.__getitem__)


def check():
    branches = ((0.40, 0.35, 0.15, 0.10), (0.05, 0.80, 0.10, 0.05))
    p = tuple(0.51 * a + 0.49 * b for a, b in zip(*branches))
    hard, bayes = winner(branches[0]), winner(p)
    close(p[hard], 0.2285)
    close(p[bayes], 0.5705)
    approx = branches[0]
    excess = cross_entropy(p, approx) - cross_entropy(p, p)
    close(excess, kl(p, approx))
    close(brier(p, approx) - brier(p, p), sum((x-y)**2 for x, y in zip(p, approx)))
    assert p[bayes] - p[hard] <= math.sqrt(2 * excess)

    # Two equally weighted old observations; S=old winner is constant.
    posteriors = ((0.6, 0.4), (0.9, 0.1))
    p_s = tuple((x+y)/2 for x, y in zip(*posteriors))
    r_s = (0.7, 0.3)
    information_loss = sum(kl(row, p_s) for row in posteriors)/2
    total_excess = sum(cross_entropy(row, r_s) - cross_entropy(row, row)
                       for row in posteriors)/2
    close(total_excess, information_loss + kl(p_s, r_s))
    assert information_loss > 0
    close(sum(max(row) for row in posteriors)/2, max(p_s))

    # Finite enumeration over the pilot event and its complement.
    event_likelihood = (0.2, 0.6)
    joint_given_old = [tuple(x * t for x, t in zip(row, event_likelihood))
                       for row in posteriors]
    assert [winner(row) for row in joint_given_old] == [1, 0]
    paired_accuracy = 0.0
    compressed_accuracy = 0.0
    for observed in (False, True):
        likelihood = [t if observed else 1-t for t in event_likelihood]
        paired_accuracy += sum(max(x * t for x, t in zip(row, likelihood))
                               for row in posteriors)/2
        compressed_accuracy += max(x * t for x, t in zip(p_s, likelihood))
    close(paired_accuracy, 0.81)
    close(compressed_accuracy, 0.75)

    # Parse failure (None) counts as an error. Check the paired identity.
    triples = ((1, 1, None), (1, 0, 1), (2, 2, 2), (3, 3, 1))
    delta = sum(int(g == y) - int(b == y) for y, b, g in triples)/len(triples)
    gains = sum(g == y and b != y for y, b, g in triples)
    losses = sum(g != y and b == y for y, b, g in triples)
    close(delta, (gains-losses)/len(triples))
    return {"mixture_optimal_accuracy": p[bayes], "hard_branch_accuracy": p[hard],
            "hard_branch_accuracy_regret": p[bayes] - p[hard],
            "proper_score_excess_nats": excess, "compression_loss_nats": information_loss,
            "pilot_accuracy_with_full_prior": paired_accuracy,
            "pilot_accuracy_with_old_winner": compressed_accuracy,
            "toy_verbalizer_gains": gains, "toy_verbalizer_losses": losses}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2, sort_keys=True))
