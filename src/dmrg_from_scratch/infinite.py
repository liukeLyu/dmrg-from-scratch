"""Infinite DMRG (iDMRG) for a translation-invariant chain.

The idea (White 1992; McCulloch https://arxiv.org/abs/0804.2509): grow the
chain from the middle. At each step insert two new sites between a left
block and a right block, find the ground state of the two-site problem, split
it with an SVD, and absorb the left tensor into the left block and the right
tensor into the right block. The chain grows by two sites per step; once the
centre stops changing, the two tensors describe the infinite chain.

Starting Lanczos from a random vector at every step is slow. McCulloch's
trick gives a much better starting vector: the new centre looks like the old
one shifted by one site, which in terms of the last step's tensors is

    theta_guess = S . B . S_prev^{-1} . A . S

(S_prev are the Schmidt values of the step before). Set ``predict=False`` to
see the difference.

Same index conventions as finite.py. The energy per site comes from the
growth of the total energy, e = (E_{N+2} - E_N) / 2, which cancels the
boundary contribution.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from .finite import grow_left, grow_right, solve_two_site
from .mps import entropy, truncated_svd


def idmrg(W, chi_max, n_steps=200, cutoff=1e-12, predict=True, seed=0, verbose=False):
    """Run iDMRG with bulk MPO tensor ``W`` (left boundary state 0, right boundary state last).

    Returns a dict with the energy per site ``energy`` (last step), the
    history ``energies`` and ``entropies`` (centre-bond entropy), the last
    Schmidt values ``S`` and ``S_prev``, and the unit-cell tensors ``A`` and ``B``.
    """
    rng = np.random.default_rng(seed)
    D = W.shape[0]
    d = W.shape[2]
    L = np.zeros((1, D, 1))
    L[0, 0, 0] = 1.0  # start in MPO state 0
    R = np.zeros((1, D, 1))
    R[0, D - 1, 0] = 1.0  # end in the last MPO state

    E_old = 0.0
    S_prev = S_last = np.ones(1)  # Schmidt values of the previous and current centre
    A = B = None  # left and right tensors of the last step
    energies, entropies = [], []
    for step in range(n_steps):
        chi = L.shape[0]
        if predict and step >= 2:
            # theta0[a, s1, s2, e] = S[a] B[a, s1, c] / S_prev[c] A[c, s2, e] S[e]
            theta0 = np.einsum("a,asc,c,cte,e->aste", S_last, B, 1 / S_prev, A, S_last)
        else:
            theta0 = rng.normal(size=(chi, d, d, chi))
        E, theta = solve_two_site(L, W, W, R, theta0)
        U, S, Vh, _ = truncated_svd(theta.reshape(chi * d, d * chi), chi_max, cutoff)
        A = U.reshape(chi, d, -1)
        B = Vh.reshape(-1, d, chi)
        L = grow_left(L, A, W)
        R = grow_right(R, B, W)
        energies.append((E - E_old) / 2)  # two sites were added
        entropies.append(entropy(S))
        E_old = E
        if verbose and (step + 1) % 20 == 0:
            print(f"step {step + 1}: N = {2 * step + 2}, e = {energies[-1]:.12f}, S = {entropies[-1]:.6f}")
        S_prev, S_last = S_last, S
    return {
        "energy": float(energies[-1]),
        "energies": np.array(energies),
        "entropies": np.array(entropies),
        "S": S_last,
        "S_prev": S_prev,
        "A": A,
        "B": B,
    }


def correlation_length(res):
    """Correlation length in sites from the transfer matrix of the two-site unit cell.

    The infinite state reads ... A [S] B ..., and the bond to the right of B
    is the same as the bond to the left of A, whose Schmidt values are S_prev
    (the centre of the previous step). So the right-canonical unit cell is

        B1 = S_prev^{-1} A S,   B2 = B,

    and the transfer matrix T = sum_s B1 conj(B1) . sum_s B2 conj(B2) has a
    leading eigenvalue 1. The next one, lambda_2, sets xi = -2 / log|lambda_2|
    (two sites per unit cell).
    """
    A, B, S, S_prev = res["A"], res["B"], res["S"], res["S_prev"]
    if len(S_prev) != A.shape[0]:
        raise ValueError("bond dimension still growing; run more steps")
    B1 = np.einsum("a,asb,b->asb", 1 / S_prev, A, S)
    chi = B.shape[2]

    def matvec(v):
        # v[x, y] on the right bond; apply B2 then B1 from the right
        v = v.reshape(chi, chi)
        v = np.einsum("xsa,ysb,ab->xy", B, B.conj(), v)
        v = np.einsum("xsa,ysb,ab->xy", B1, B1.conj(), v)
        return v.ravel()

    T = LinearOperator((chi * chi, chi * chi), matvec=matvec, dtype=float)
    vals = eigs(T, k=3, which="LM", return_eigenvectors=False)
    vals = np.sort(np.abs(vals))[::-1]
    return -2.0 / np.log(vals[1] / vals[0])
