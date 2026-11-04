"""Exact reference values: the TFIM chain from free fermions, and any small PauliSum by exact diagonalization.

The Jordan-Wigner transformation turns H = -sum X_i X_{i+1} - h sum Z_i into a
quadratic fermion Hamiltonian (Lieb, Schultz, Mattis 1961; Pfeuty 1970). For
the open chain its single-particle energies are 2 s_n, where s_n are the
singular values of the L x L bidiagonal matrix M with M_ii = h and
M_{i,i+1} = J, and the ground energy is E_0 = -sum_n s_n.
"""

import numpy as np
from scipy.integrate import quad
from scipy.sparse.linalg import LinearOperator, eigsh

from .models import PauliSum


def tfim_open_energy(L, h, J=1.0):
    """Exact ground-state energy of the open chain of L sites."""
    M = h * np.eye(L) + J * np.eye(L, k=1)
    return -float(np.sum(np.linalg.svd(M, compute_uv=False)))


def tfim_infinite_energy_density(h, J=1.0):
    """Exact ground-state energy per site of the infinite chain.

    e_0 = -(1/pi) int_0^pi sqrt(J^2 + h^2 - 2 J h cos k) dk, equal to -4/pi at h = J = 1.
    """
    val, _ = quad(lambda k: np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k)), 0, np.pi, limit=200)
    return -val / np.pi


def central_charge_fit(L, S):
    """Fit S(l) = (c/6) log2[(2L/pi) sin(pi l/L)] + const for an open critical chain.

    ``S[l-1]`` is the entropy of the first l sites. Calabrese and Cardy,
    https://arxiv.org/abs/hep-th/0405152. Returns ``(c, const)``.
    Odd and even l oscillate a little in the open chain, so we fit both together.
    """
    l = np.arange(1, L)
    x = np.log2(2 * L / np.pi * np.sin(np.pi * l / L))
    slope, const = np.polyfit(x, S, 1)
    return 6 * slope, const


# Any PauliSum (the cylinders of models.py): exact ground states by Lanczos on the full
# Hilbert space, up to about 24 spins.
#
# Bit order: site 0 is the most significant bit, as in PennyLane and as in ``mps.to_dense``.
# A Pauli string acts on a basis state as ``P|b> = i^{n_Y} (-1)^{popcount(b & m_YZ)} |b ^ m_XY>``,
# so each term is a bit flip plus a sign, applied to the whole vector at once.


def _parity(k: np.ndarray, mask: int, n: int) -> np.ndarray:
    out = np.zeros(k.shape, dtype=np.int64)
    for b in range(n):
        if mask >> b & 1:
            out ^= (k >> b) & 1
    return out


def _compile(H: PauliSum):
    n = H.n
    k = np.arange(2**n, dtype=np.int64)
    compiled = []
    for c, ops in H.terms:
        flip = sign_mask = 0
        n_y = 0
        for site, p in ops:
            bit = 1 << (n - 1 - site)
            if p in "XY":
                flip |= bit
            if p in "YZ":
                sign_mask |= bit
            n_y += p == "Y"
        if n_y % 2:
            raise ValueError("odd number of Y: the term is not real")
        phase = c * (-1) ** (n_y // 2)
        signs = phase * (1 - 2 * _parity(k ^ flip, sign_mask, n))
        compiled.append((flip, signs))
    return k, compiled


def matvec_operator(H: PauliSum) -> LinearOperator:
    """``H`` as a scipy LinearOperator on the ``2^n`` dimensional space."""
    k, compiled = _compile(H)

    def matvec(v):
        v = np.ravel(v)
        out = np.zeros_like(v)
        for flip, signs in compiled:
            out += signs * v[k ^ flip]
        return out

    dim = 2**H.n
    return LinearOperator((dim, dim), matvec=matvec, dtype=float)


def ground_state(H: PauliSum, seed: int = 0) -> tuple[float, np.ndarray]:
    """Ground energy and state of ``H`` (normalized, real)."""
    if H.n > 24:
        raise ValueError("exact diagonalization is limited to about 24 spins")
    v0 = np.random.default_rng(seed).standard_normal(2**H.n)
    vals, vecs = eigsh(matvec_operator(H), k=1, which="SA", v0=v0, tol=1e-12)
    return float(vals[0]), vecs[:, 0]
