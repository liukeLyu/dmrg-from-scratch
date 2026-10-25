"""Exact reference values for the TFIM, from the mapping to free fermions.

The Jordan-Wigner transformation turns H = -sum X_i X_{i+1} - h sum Z_i into a
quadratic fermion Hamiltonian (Lieb, Schultz, Mattis 1961; Pfeuty 1970). For
the open chain its single-particle energies are 2 s_n, where s_n are the
singular values of the L x L bidiagonal matrix M with M_ii = h and
M_{i,i+1} = J, and the ground energy is E_0 = -sum_n s_n.
"""

import numpy as np
from scipy.integrate import quad


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
