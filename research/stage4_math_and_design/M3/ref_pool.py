"""Prior-aware RefSufficientPool for proper complex Gaussian observation blocks."""

from dataclasses import dataclass
import hashlib
import math
from operator import index
import struct
from types import MappingProxyType

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, roots_hermitenorm, xlogy


ROOTS = (1+0j, 1j, -1+0j, -1j)
MAX_REFERENCES = 64
WIRE_VERSION = 2


def readonly(array, dtype):
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.flags.writeable = False
    return out


def canonical_iq(value, frame):
    # Exact quarter-turns before any general floating-point arithmetic.
    a, b = value.real, value.imag
    return (complex(a, b), complex(b, -a), complex(-a, -b), complex(-b, a))[frame % 4]


def sample_digest(canonical_sample):
    a, b = canonical_sample.real, canonical_sample.imag
    return hashlib.sha256(struct.pack("<dd", a or 0.0, b or 0.0)).hexdigest()


@dataclass(frozen=True)
class PhaseContext:
    anchor_tick: int
    ticks_per_unit: float
    omega_hat: float
    v_omega: float
    diffusion_var_per_unit: float
    p_phi0: float = 0.0
    cov_phi0_omega: float = 0.0
    phi0_hat: float = 0.0

    def path_moments(self, ticks):
        vals = (self.ticks_per_unit, self.omega_hat, self.v_omega, self.diffusion_var_per_unit,
                self.p_phi0, self.cov_phi0_omega, self.phi0_hat)
        if (not np.isfinite(vals).all() or self.ticks_per_unit <= 0
                or min(self.v_omega, self.diffusion_var_per_unit, self.p_phi0) < 0
                or abs(self.cov_phi0_omega) > math.sqrt(self.v_omega*self.p_phi0)):
            raise ValueError("Invalid anchored Brownian/CFO prior")
        # Subtract integer ticks before conversion, preserving short lags at large epochs.
        t = np.array([index(tick) - self.anchor_tick for tick in ticks], dtype=np.float64) / self.ticks_per_unit
        if np.any(t < 0) or not np.isfinite(t).all():
            raise ValueError("Clock predates the Brownian prediction anchor or has invalid units")
        cov = (self.diffusion_var_per_unit * np.minimum.outer(t, t)
               + self.v_omega * np.outer(t, t)
               + self.p_phi0 + self.cov_phi0_omega*(t[:, None]+t))
        return self.phi0_hat+self.omega_hat*t, cov


@dataclass(frozen=True)
class ClockMeta:
    acquisition_id: str
    provenance: str
    physical_tick: int
    available_tick: int
    known_root: complex | None
    calibration_id: str
    raw_support: tuple = ()  # optional in fixtures; half-open raw-tick intervals in M4
    role: str = "reference"

    def __post_init__(self):
        if (not self.acquisition_id or not self.provenance or not self.calibration_id
                or not isinstance(self.physical_tick, int) or not isinstance(self.available_tick, int)
                or self.available_tick < self.physical_tick
                or self.known_root not in (*ROOTS, None)
                or self.role not in ("history", "reference", "probe")
                or any(a >= b or b - 1 > self.available_tick for a, b in self.raw_support)):
            raise ValueError("Invalid clock metadata or known QPSK root")


@dataclass(frozen=True)
class Observation:
    clock: ClockMeta
    matched: complex | None  # None only for history retained in the consumption ledger
    shares_channel: bool = True
    digest: str | None = None  # canonical sample digest, mandatory for retired history

    @property
    def acquisition_id(self):
        return self.clock.acquisition_id


@dataclass(frozen=True)
class NodeBank:
    # Columns of phase_grid are physical_ticks, including target clocks if required.
    hypothesis_ids: tuple
    physical_ticks: tuple
    phase_grid: np.ndarray   # (H, N, clocks), real radians before the unknown quarter-turn
    log_weights_by_turn: np.ndarray  # (H, 4, N), globally normalized when entering the pool


@dataclass(frozen=True)
class PoolState:
    owner: object
    base: np.ndarray
    history: tuple  # Every identifier consumed by the F0 snapshot
    added: tuple
    log_weights_by_turn: np.ndarray
    log_increment: float


@dataclass(frozen=True)
class GaussianLaw:
    mean: np.ndarray
    covariance: np.ndarray  # (H, d, d)

    def logpdf(self, values):
        """Return (..., H, 4, N) log densities for observations (..., d)."""
        x = np.asarray(values, dtype=np.complex128)
        if x.ndim == 0 or x.shape[-1] != self.mean.shape[-1]:
            raise ValueError("Observation has the wrong dimension")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite observation")
        d = x.shape[-1]
        out = []
        for mean, cov in zip(self.mean, self.covariance):
            l = np.linalg.cholesky(cov)
            residual = x[..., None, None, :] - mean
            whitened = solve_triangular(l, residual.reshape(-1, d).T, lower=True,
                                        check_finite=False).T.reshape(residual.shape)
            log_norm = d * math.log(math.pi) + 2*np.log(l.diagonal().real).sum()
            out.append(-np.sum(np.abs(whitened)**2, axis=-1)-log_norm)
        return np.stack(out, axis=-3)

    def project(self, weights):
        # The scored scalar is weights @ y, with no implicit conjugation.
        w = np.asarray(weights, dtype=np.complex128)
        cov = np.einsum("i,hij,j->h", w, self.covariance, w.conj()).real
        return GaussianLaw((self.mean @ w)[..., None], cov[:, None, None])


def conditional_gaussian(covariance, means, new, past, seen):
    """Condition on measured past coordinates, at each latent node."""
    k = covariance
    knn = k[:, new][:, :, new]
    mean = means[..., new].copy()
    if not len(past):
        return GaussianLaw(mean, knn)
    knp = k[:, new][:, :, past]
    kpp = k[:, past][:, :, past]
    kpn = k[:, past][:, :, new]
    # Missing boundary samples are harmless only for exact zero conditional coupling.
    a = np.linalg.solve(kpp, kpn).conj().swapaxes(-1, -2)
    missing = np.array([value is None for value in seen], dtype=bool)
    if np.any(a[..., missing] != 0):
        raise ValueError("F0 lacks a correlated Gaussian boundary observation")
    y = np.array([0 if value is None else value for value in seen], dtype=np.complex128)
    mean += np.einsum("hjp,hgnp->hgnj", a, y - means[..., past])
    s = knn - a @ kpn
    # Keep the Hermitian structure against solve roundoff.
    return GaussianLaw(mean, (s + s.conj().swapaxes(-1, -2))/2)


@dataclass(frozen=True)
class Forecast:
    state: PoolState
    acquisition_ids: tuple
    law: GaussianLaw
    reveal_deadline: int

    def log_densities(self, samples):
        ell = self.law.logpdf(samples)
        weights = self.state.log_weights_by_turn
        joint = logsumexp(ell + weights, axis=(-2, -1))
        log_prior = logsumexp(weights, axis=(-2, -1))
        marginal = logsumexp(joint, axis=-1)
        with np.errstate(invalid="ignore"):
            conditional = np.where(np.isfinite(log_prior), joint-log_prior, -np.inf)
        return conditional, marginal, joint-marginal[..., None]

    def eig(self, order=24):
        if len(self.acquisition_ids) != 1:
            raise ValueError("The Stage 6 query reads one complex acquisition")
        z, w = roots_hermitenorm(order)
        grid = (z[:, None]+1j*z).ravel()
        gh = (w[:, None]*w).ravel()/(2*np.pi)
        prior = np.exp(logsumexp(self.state.log_weights_by_turn, axis=(-2, -1)))
        log_prior = logsumexp(self.state.log_weights_by_turn, axis=(-2, -1))
        var = self.law.covariance[:, 0, 0].real
        mean = self.law.mean[..., 0]
        nodes = (mean[..., None]+np.sqrt(var[:, None, None, None]/2)*grid).reshape(-1, 1)
        weight = np.exp(self.state.log_weights_by_turn[..., None]) * gh
        _, _, log_post = self.log_densities(nodes)
        p = np.exp(log_post)
        # 0 log 0 is zero, including hypotheses with zero current prior.
        kl = np.sum(xlogy(p, p) - p*np.where(prior > 0, log_prior, 0), axis=-1)
        return float(weight.ravel() @ kl)


class RefSufficientPool:
    """Apply unique observations to an immutable conditional F0 snapshot."""

    def __init__(self, nodes, clocks, means, covariance, receiver_frame=0):
        self.hypothesis_ids = tuple(nodes.hypothesis_ids)
        self.physical_ticks = tuple(nodes.physical_ticks)
        self.phase_grid = readonly(nodes.phase_grid, np.float64)
        self.initial = readonly(nodes.log_weights_by_turn, np.float64)
        self.clocks = tuple(clocks)
        self.indices = MappingProxyType({item.acquisition_id: i for i, item in enumerate(clocks)})
        self.means = readonly(means, np.complex128)
        raw_cov = np.asarray(covariance, dtype=np.complex128)
        h, d = len(self.hypothesis_ids), len(self.clocks)
        if raw_cov.ndim == 2:
            raw_cov = np.broadcast_to(raw_cov, (h, d, d))
        self.covariance = readonly(raw_cov, np.complex128)
        self.frame = receiver_frame % 4
        self._owner = object()
        if (self.initial.ndim != 3 or self.initial.shape[:2] != (h, 4)
                or self.phase_grid.shape != (h, self.initial.shape[-1], len(self.physical_ticks))
                or len(self.indices) != d or len(set(self.hypothesis_ids)) != h
                or len(set(self.physical_ticks)) != len(self.physical_ticks)
                or self.means.shape != (*self.initial.shape, d)
                or not np.isfinite(logsumexp(self.initial))
                or not np.isfinite(self.phase_grid).all() or not np.isfinite(self.means).all()
                or self.covariance.shape != (h, d, d)
                or not np.isfinite(self.covariance).all()
                or not np.array_equal(self.covariance, self.covariance.conj().swapaxes(-1, -2))
                or any(item.physical_tick not in self.physical_ticks for item in self.clocks)):
            raise ValueError("Inconsistent phase bank or proper complex Gaussian design")
        np.linalg.cholesky(self.covariance)

    def start(self, history, as_of):
        # Initial weights are already conditional on all supplied F0 history.
        old = self._canonical_records(history, as_of, allow_missing=True, allow_outside=True)
        if any(obs.clock.available_tick > as_of for obs in old):
            raise ValueError("F0 contains an unavailable observation")
        base = readonly(self.initial - logsumexp(self.initial), np.float64)
        return PoolState(self._owner, base, old, (), base, 0.0)

    def _validate_state(self, state):
        if state.owner is not self._owner:
            raise ValueError("State belongs to a different Gaussian design")

    def _canonical_records(self, records, as_of, allow_missing=False, allow_outside=False):
        by_id = {}
        for raw in records:
            if raw is None:
                continue  # typed transport padding
            index = self.indices.get(raw.acquisition_id)
            if index is not None and self.clocks[index] != raw.clock:
                raise ValueError(f"Conflicting metadata for {raw.acquisition_id}")
            if index is None and raw.shares_channel and not allow_outside:
                raise ValueError(f"No validated provenance for {raw.acquisition_id}")
            if raw.shares_channel and raw.clock.available_tick > as_of:
                raise ValueError(f"Observation unavailable at deadline: {raw.acquisition_id}")
            if raw.matched is None:
                if not allow_missing or raw.digest is None:
                    raise ValueError(f"Missing observation payload: {raw.acquisition_id}")
                sample = None
                digest = raw.digest
            else:
                sample = canonical_iq(raw.matched, self.frame)
                if raw.shares_channel and not np.isfinite(sample):
                    raise ValueError(f"Nonfinite valid acquisition: {raw.acquisition_id}")
                digest = sample_digest(sample)
                if raw.digest is not None and raw.digest != digest:
                    raise ValueError(f"Conflicting payload digest for {raw.acquisition_id}")
            obs = Observation(raw.clock, sample, raw.shares_channel, digest)
            previous = by_id.setdefault(obs.acquisition_id, obs)
            if previous != obs:
                raise ValueError(f"Conflicting copies of {obs.acquisition_id}")
        return tuple(by_id[key] for key in sorted(by_id))

    def _conditional(self, ids, boundary):
        past = [(self.indices[o.acquisition_id], o.matched) for o in boundary
                if o.acquisition_id in self.indices]
        return conditional_gaussian(self.covariance, self.means,
                                    [self.indices[k] for k in ids],
                                    [index for index, _ in past], [val for _, val in past])

    def update(self, state, records, as_of):
        self._validate_state(state)
        consumed = {r.acquisition_id: r for r in (*state.history, *state.added)}
        additions = {}
        for obs in self._canonical_records(records, as_of, allow_outside=True):
            prior = consumed.get(obs.acquisition_id)
            if prior is not None:
                if (prior.clock != obs.clock or prior.shares_channel != obs.shares_channel
                        or prior.digest != obs.digest):
                    raise ValueError(f"Conflicts with consumed acquisition {obs.acquisition_id}")
                # A retired payload has no likelihood, even if someone returns its sample.
            elif obs.shares_channel:
                if obs.acquisition_id not in self.indices:
                    raise ValueError(f"No validated provenance for {obs.acquisition_id}")
                if obs.clock.role == "history":
                    raise ValueError(f"History ID missing from the F0 ledger: {obs.acquisition_id}")
                additions[obs.acquisition_id] = obs
        if not additions:
            return state
        merged = {r.acquisition_id: r for r in state.added}
        merged.update(additions)
        if len(merged) > MAX_REFERENCES:
            raise ValueError("More than 64 distinct revealed valid acquisitions")
        ids = tuple(sorted(merged))
        accepted = tuple(merged[key] for key in ids)
        law = self._conditional(ids, state.history)
        ell = law.logpdf([obs.matched for obs in accepted])
        joint = state.base + ell
        logz = float(logsumexp(joint))
        return PoolState(self._owner, state.base, state.history, accepted,
                         readonly(joint-logz, np.float64), logz)

    def innovation(self, state, acquisition_ids, reveal_deadline):
        """Return the sorted, normalized likelihood law before seeing new payloads."""
        self._validate_state(state)
        ids = tuple(sorted(set(acquisition_ids)))
        if not ids or len(ids) > MAX_REFERENCES:
            raise ValueError("Expected between 1 and 64 candidate IDs")
        for key in ids:
            index = self.indices.get(key)
            if (index is None or self.clocks[index].role == "history"
                    or self.clocks[index].available_tick > reveal_deadline):
                raise ValueError("Unknown candidate or unavailable by the reveal deadline")
        boundary = tuple(sorted((*state.history, *state.added), key=lambda o: o.acquisition_id))
        if set(ids) & {obs.acquisition_id for obs in boundary}:
            raise ValueError("Candidate is already consumed")
        law = self._conditional(ids, boundary)
        return ids, GaussianLaw(readonly(law.mean, np.complex128), readonly(law.covariance, np.complex128))

    def forecast(self, state, acquisition_id, reveal_deadline):
        ids, law = self.innovation(state, (acquisition_id,), reveal_deadline)
        return Forecast(state, ids, law, reveal_deadline)

    def reveal(self, state, forecast, obs, as_of):
        if (forecast.state is not state or forecast.acquisition_ids != (obs.acquisition_id,)
                or as_of > forecast.reveal_deadline):
            raise ValueError("Reveal differs from the forecast commitment")
        if not obs.shares_channel:
            raise ValueError("Forecast was for a validated same-channel candidate")
        return self.update(state, (obs,), as_of)


def task_readout(state, joint_given_node):
    target = np.asarray(joint_given_node, dtype=np.float64)
    weights = np.exp(state.log_weights_by_turn)
    if (target.shape != (*weights.shape, 4, 4) or not np.isfinite(target).all()
            or np.any(target < 0) or not np.allclose(target.sum(axis=(-2, -1)), 1, atol=2e-12, rtol=0)):
        raise ValueError("Expected normalized Q2/Q3 joint emission at each state")
    joint = np.einsum("hgn,hgnkl->kl", weights, target)
    return joint.sum(1), joint.sum(0), joint


def fixed_slots(canonical_codes, q2, q3):
    """Versioned canonical header + two always-present float32 evidence vectors."""
    if len(canonical_codes) != 3 or any(v not in range(4) for v in canonical_codes):
        raise ValueError("Expected S17 inner1, inner2 and relative turn")
    slots = readonly((q2, q3), "<f4")
    if (slots.shape != (2, 4) or not np.isfinite(slots).all() or np.any(slots < 0)
            or not np.allclose(slots.sum(axis=1), 1, atol=2e-7, rtol=0)):
        raise ValueError("Expected two normalized four-class evidence vectors")
    return readonly((WIRE_VERSION, *canonical_codes), np.uint8), slots
