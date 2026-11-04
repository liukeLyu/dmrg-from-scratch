"""Fermions on a chain: the Jordan-Wigner map and the Hubbard chain as an MPO.

One spin orbital is one qubit: ``|0>`` empty, ``|1>`` occupied, so ``n = |1><1| = (1 - Z)/2``.
On a single orbital the annihilator is ``a = |0><1|``. Fermion operators on different orbitals
anticommute, spin operators commute; the Jordan-Wigner map [P. Jordan and E. Wigner,
Z. Phys. 47, 631 (1928), https://doi.org/10.1007/BF01331938] repairs that with a string,

    c_j = F_0 F_1 ... F_{j-1} a_j,      F = (-1)^n = Z,

which counts the parity of all orbitals to the left of ``j``. The TFIM exact solution
(``exact.py``) is the same map read backwards: there it turns spins into free fermions.

A hop then reads ``c+_p c_q = a+_p F_{p+1} ... F_{q-1} a_q`` (p < q): the strings left of ``p``
cancel, and only the orbitals in between keep an ``F``.

The Hubbard chain with L sites is stored as 2L orbitals, interleaved: orbital ``2i`` is site
``i`` spin up, ``2i + 1`` spin down (as in ``models.hubbard_cylinder``). Every tensor is then
2 x 2, as for the spin chains, and the sign is visible: a hop from ``2i`` to ``2i + 2`` passes
orbital ``2i + 1``, and the MPO channel that carries it applies ``F`` there instead of ``I``.
"""

import numpy as np

I2 = np.eye(2)
A = np.array([[0.0, 1.0], [0.0, 0.0]])  # a = |0><1|
AD = A.T  # a+ = |1><0|
N = np.diag([0.0, 1.0])
F = np.diag([1.0, -1.0])  # (-1)^n


def hubbard_mpo(L, U, t=1.0, mu=0.0):
    """MPO of H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i n_iu n_id - mu sum_i,s n_is, open chain.

    2L tensors ``W[w_left, w_right, out, in]``, all with bond dimension 7. The channels:

        0  start                     6  done
        1  a+ placed, pass one more orbital with F     3  ... then close with -t a
        2  a  placed, pass one more orbital with F     4  ... then close with -t a+
        5  n_up placed on this site, close with U n_down on the next orbital
    """
    W = np.zeros((2, 7, 7, 2, 2))  # [orbital parity (0 = up, 1 = down), w_left, w_right, out, in]
    for s in (0, 1):
        Ws = W[s]
        Ws[0, 0] = Ws[6, 6] = I2
        Ws[0, 6] = -mu * N
        Ws[0, 1], Ws[1, 3], Ws[3, 6] = AD, F, -t * A  # -t a+_p F_{p+1} a_{p+2}
        Ws[0, 2], Ws[2, 4], Ws[4, 6] = A, F, -t * AD  # -t a_p F_{p+1} a+_{p+2}
    W[0, 0, 5] = N  # spin up opens the interaction ...
    W[1, 5, 6] = U * N  # ... and the spin-down orbital of the same site closes it
    mpo = [W[p % 2].copy() for p in range(2 * L)]
    mpo[0] = mpo[0][0:1]
    mpo[-1] = mpo[-1][:, 6:7]
    return mpo


def free_chain_energy(L, t=1.0):
    """Exact ground energy of the open Hubbard chain at U = 0, mu = 0 (half filling for even L).

    The single-particle states are standing waves phi_k(j) = sin(pi k j / (L + 1)), j = 1..L: they
    satisfy -t (phi(j-1) + phi(j+1)) = -2t cos(pi k / (L + 1)) phi(j) with phi(0) = phi(L+1) = 0.
    Every negative level is filled once per spin.
    """
    eps = -2 * t * np.cos(np.pi * np.arange(1, L + 1) / (L + 1))
    return 2 * float(eps[eps < 0].sum())


def free_chain_correlation(L, i, j, t=1.0):
    """<c+_i c_j> for one spin species in the U = 0 ground state of the open chain (sites 0..L-1)."""
    k = np.arange(1, L + 1)
    eps = -2 * t * np.cos(np.pi * k / (L + 1))
    phi = np.sqrt(2 / (L + 1)) * np.sin(np.pi * np.outer(k, np.arange(1, L + 1)) / (L + 1))
    occ = eps < 0
    return float(phi[occ, i] @ phi[occ, j])
