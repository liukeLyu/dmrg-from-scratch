"""Finite-size two-site DMRG.

The idea (White 1992; Schollwöck https://arxiv.org/abs/1008.3477, Sec. 6.3):
minimize <psi|H|psi> over MPS by optimizing two neighbouring tensors at a
time while all others are held fixed. With the rest of the chain in canonical
form, that local problem is an ordinary eigenvalue problem

    H_eff theta = E theta,

where theta is the two-site tensor and H_eff is H with everything except the
two sites contracted into a left environment L and a right environment R.
We solve it with Lanczos (scipy's eigsh), split theta with a truncated SVD,
move one site over, and sweep back and forth.

Index conventions (see mps.py and mpo.py):
    MPS tensor       A[left, phys, right]
    MPO tensor       W[w_left, w_right, out, in]
    environment      L[x, w, y] and R[x, w, y]: x = ket bond, w = MPO bond, y = bra bond
    two-site tensor  theta[left, s1, s2, right]
"""

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .mps import entropy, random_mps, right_canonicalize, truncated_svd


def grow_left(L, A, W):
    """Add one site to a left environment.

    L_new[a, v, b] = sum L[x, w, y] A[x, s, a] W[w, v, t, s] conj(A[y, t, b])
    One tensor at a time keeps each step cheap (cost chi^3).
    """
    T = np.einsum("xwy,xsa->wysa", L, A, optimize=True)
    T = np.einsum("wysa,wvts->yvta", T, W, optimize=True)
    return np.einsum("yvta,ytb->avb", T, A.conj(), optimize=True)


def grow_right(R, B, W):
    """Add one site to a right environment.

    R_new[x, w, y] = sum B[x, s, a] W[w, v, t, s] conj(B[y, t, b]) R[a, v, b]
    """
    T = np.einsum("xsa,avb->xsvb", B, R, optimize=True)
    T = np.einsum("xsvb,wvts->xwtb", T, W, optimize=True)
    return np.einsum("xwtb,ytb->xwy", T, B.conj(), optimize=True)


def apply_heff(L, W1, W2, R, theta):
    """H_eff acting on the two-site tensor theta[x, s1, s2, a].

    out[y, t1, t2, b] = sum L[x, w, y] W1[w, u, t1, s1] W2[u, v, t2, s2] R[a, v, b] theta[x, s1, s2, a]
    Done in four steps so each einsum is easy to read (and the cost stays chi^3).
    optimize=True lets NumPy hand each step to BLAS as a matrix product, which is
    10-20 times faster at chi ~ 100 than einsum's own loops.
    """
    T = np.einsum("xwy,xpqa->wypqa", L, theta, optimize=True)  # attach the left environment
    T = np.einsum("wypqa,wutp->uytqa", T, W1, optimize=True)  # first site's MPO tensor
    T = np.einsum("uytqa,uvrq->vytra", T, W2, optimize=True)  # second site's MPO tensor
    return np.einsum("vytra,avb->ytrb", T, R, optimize=True)  # attach the right environment


def solve_two_site(L, W1, W2, R, theta0, tol=1e-10, stats=None):
    """Lowest eigenpair of H_eff, starting Lanczos from theta0.

    If ``stats`` is a dict, ``stats["flops"]`` grows by the work of each H_eff application:
    its two big contractions cost 2 w d^2 chi_l chi_r (chi_l + chi_r) flops (w = MPO bond dimension).
    """
    shape = theta0.shape
    n = theta0.size
    chi_l, d1, d2, chi_r = shape
    w = max(L.shape[1], R.shape[1])
    per_call = 2.0 * w * d1 * d2 * chi_l * chi_r * (chi_l + chi_r)

    def matvec(v):
        if stats is not None:
            stats["flops"] += per_call
        return apply_heff(L, W1, W2, R, v.reshape(shape)).ravel()

    H = LinearOperator((n, n), matvec=matvec, dtype=float)
    if n <= 4:  # eigsh needs k < n; tiny problems are solved densely
        Hd = np.column_stack([matvec(e) for e in np.eye(n)])
        vals, vecs = np.linalg.eigh(Hd)
    else:
        vals, vecs = eigsh(H, k=1, which="SA", v0=theta0.ravel(), tol=tol)
    return vals[0], vecs[:, 0].reshape(shape)


def dmrg(mpo, chi_max, n_sweeps=5, cutoff=1e-12, seed=0, verbose=False, mps=None):
    """Ground state of an open-chain MPO by two-site DMRG.

    ``mps`` is an optional starting state, e.g. the result at a smaller ``chi_max``
    (see :func:`dmrg_ladder`); by default DMRG starts from a random MPS.

    Returns a dict with the final ``energy``, the right-canonical ``mps``,
    the ``energies`` after each half sweep, the ``entropies`` S(l) of the
    first l sites (l = 1..L-1) from the last sweep, and the largest
    ``discarded`` weight of that sweep. For pricing a run it also reports
    the largest bond dimension ``chi``, the ``flops`` spent in H_eff
    (and ``flops_per_sweep``) and the wall time ``seconds``.
    """
    t0 = time.perf_counter()
    stats = {"flops": 0.0}
    L = len(mpo)
    d = mpo[0].shape[2]
    if mps is None:
        mps = random_mps(L, d, chi=min(chi_max, 4), seed=seed)
    mps = right_canonicalize(mps)  # a copy: the caller's MPS is left alone

    # Lenv[i] contains sites 0..i-1, Renv[i] contains sites i..L-1.
    Lenv = [None] * (L + 1)
    Renv = [None] * (L + 1)
    Lenv[0] = np.ones((1, 1, 1))
    Renv[L] = np.ones((1, 1, 1))
    for i in range(L - 1, 0, -1):
        Renv[i] = grow_right(Renv[i + 1], mps[i], mpo[i])

    energies = []
    S_bonds = [None] * (L - 1)
    for sweep in range(n_sweeps):
        max_disc = 0.0
        # Left to right: after each step site i is a left isometry A.
        for i in range(L - 1):
            theta = np.einsum("asb,btc->astc", mps[i], mps[i + 1], optimize=True)
            E, theta = solve_two_site(Lenv[i], mpo[i], mpo[i + 1], Renv[i + 2], theta, stats=stats)
            chi_l, d1, d2, chi_r = theta.shape
            U, S, Vh, disc = truncated_svd(theta.reshape(chi_l * d1, d2 * chi_r), chi_max, cutoff)
            max_disc = max(max_disc, disc)
            mps[i] = U.reshape(chi_l, d1, -1)
            mps[i + 1] = (S[:, None] * Vh).reshape(-1, d2, chi_r)
            Lenv[i + 1] = grow_left(Lenv[i], mps[i], mpo[i])
        energies.append(E)
        # Right to left: after each step site i+1 is a right isometry B.
        for i in range(L - 2, -1, -1):
            theta = np.einsum("asb,btc->astc", mps[i], mps[i + 1], optimize=True)
            E, theta = solve_two_site(Lenv[i], mpo[i], mpo[i + 1], Renv[i + 2], theta, stats=stats)
            chi_l, d1, d2, chi_r = theta.shape
            U, S, Vh, disc = truncated_svd(theta.reshape(chi_l * d1, d2 * chi_r), chi_max, cutoff)
            max_disc = max(max_disc, disc)
            mps[i] = (U * S[None, :]).reshape(chi_l, d1, -1)
            mps[i + 1] = Vh.reshape(-1, d2, chi_r)
            Renv[i + 1] = grow_right(Renv[i + 2], mps[i + 1], mpo[i + 1])
            S_bonds[i] = S  # Schmidt values of the cut between sites i and i+1
        energies.append(E)
        if verbose:
            print(f"sweep {sweep + 1}: E = {E:.12f}, max discarded weight = {max_disc:.1e}")

    return {
        "energy": float(E),
        "mps": mps,
        "energies": np.array(energies),
        "entropies": np.array([entropy(S) for S in S_bonds]),
        "schmidt_values": S_bonds,
        "discarded": max_disc,
        "chi": max(A.shape[2] for A in mps),
        "flops": stats["flops"],
        "flops_per_sweep": stats["flops"] / n_sweeps,
        "seconds": time.perf_counter() - t0,
    }


def dmrg_ladder(mpo, chis, n_sweeps=4, seed=0):
    """DMRG at each bond dimension in ``chis`` (increasing), each started from the previous result.

    A ladder of growing ``chi`` costs little more than its last rung, and the energies along it
    are what one extrapolates to zero discarded weight.
    """
    out, mps = [], None
    for chi in chis:
        r = dmrg(mpo, chi, n_sweeps=n_sweeps, seed=seed, mps=mps)
        mps = r["mps"]
        out.append(r)
    return out
