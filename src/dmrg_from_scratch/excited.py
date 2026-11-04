"""Excited states by DMRG: the ground state again, for a Hamiltonian that penalizes the states found so far.

    H' = H + w sum_k |psi_k><psi_k|

With w larger than the gaps of interest, the ground state of H' is the lowest state of
H orthogonal to every psi_k (Schollwöck https://arxiv.org/abs/1008.3477, Sec. 6.6, and
Stoudenmire and White https://arxiv.org/abs/1105.1374, Sec. 7). In the two-site update
the projector becomes w |p_k><p_k|, where p_k is psi_k seen through the current left
and right isometries: an overlap environment, built exactly like L and R but with
psi_k as the bra instead of the MPO.

The sweep is the one of finite.py; only the effective Hamiltonian changes.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .finite import apply_heff, grow_left, grow_right
from .mps import entropy, overlap, random_mps, right_canonicalize, truncated_svd


def _ov_left(P, A, B):
    """P_new[a, b] = sum P[x, y] A[x, s, a] conj(B[y, s, b]) with A from the current MPS, B from psi_k."""
    return np.einsum("xy,xsa,ysb->ab", P, A, B.conj())


def _ov_right(P, A, B):
    return np.einsum("xsa,ab,ysb->xy", A, P, B.conj())


def _projected(PL, B1, B2, PR):
    """psi_k on the current two-site basis, p[x, s1, s2, a], so that <psi_k|psi> = p^dagger theta."""
    return np.einsum("xy,ysb,btc,ac->xsta", PL.conj(), B1, B2, PR.conj())


def dmrg_excited(mpo, chi_max, lower, weight=10.0, n_sweeps=6, cutoff=1e-12, seed=1):
    """Lowest state of ``mpo`` orthogonal to the normalized MPS in ``lower``.

    Returns a dict like :func:`finite.dmrg`: ``energy`` (of H, not of H'), ``mps``,
    ``entropies`` and ``discarded``, plus the ``overlaps`` |<psi_k|psi>| at the end.
    """
    L = len(mpo)
    d = mpo[0].shape[2]
    mps = right_canonicalize(random_mps(L, d, chi=min(chi_max, 4), seed=seed))
    K = len(lower)

    Lenv, Renv = [None] * (L + 1), [None] * (L + 1)
    Lenv[0], Renv[L] = np.ones((1, 1, 1)), np.ones((1, 1, 1))
    PL = [[None] * (L + 1) for _ in range(K)]
    PR = [[None] * (L + 1) for _ in range(K)]
    for k in range(K):
        PL[k][0], PR[k][L] = np.ones((1, 1)), np.ones((1, 1))
    for i in range(L - 1, 0, -1):
        Renv[i] = grow_right(Renv[i + 1], mps[i], mpo[i])
        for k in range(K):
            PR[k][i] = _ov_right(PR[k][i + 1], mps[i], lower[k][i])

    def solve(i, theta0):
        shape = theta0.shape
        ps = [_projected(PL[k][i], lower[k][i], lower[k][i + 1], PR[k][i + 2]).ravel() for k in range(K)]

        def matvec(v):
            out = apply_heff(Lenv[i], mpo[i], mpo[i + 1], Renv[i + 2], v.reshape(shape)).ravel()
            for p in ps:
                out = out + weight * p * (p.conj() @ v)
            return out

        n = theta0.size
        if n <= 4:
            Hd = np.column_stack([matvec(e) for e in np.eye(n)])
            vals, vecs = np.linalg.eigh(Hd)
        else:
            vals, vecs = eigsh(LinearOperator((n, n), matvec=matvec, dtype=float), k=1, which="SA", v0=theta0.ravel())
        v = vecs[:, 0]
        E = float(vals[0]) - weight * sum(abs(p.conj() @ v) ** 2 for p in ps)  # energy of H alone
        return E, v.reshape(shape)

    S_bonds = [None] * (L - 1)
    for _ in range(n_sweeps):
        max_disc = 0.0
        for i in range(L - 1):
            theta = np.einsum("asb,btc->astc", mps[i], mps[i + 1])
            E, theta = solve(i, theta)
            cl, d1, d2, cr = theta.shape
            U, S, Vh, disc = truncated_svd(theta.reshape(cl * d1, d2 * cr), chi_max, cutoff)
            max_disc = max(max_disc, disc)
            mps[i], mps[i + 1] = U.reshape(cl, d1, -1), (S[:, None] * Vh).reshape(-1, d2, cr)
            Lenv[i + 1] = grow_left(Lenv[i], mps[i], mpo[i])
            for k in range(K):
                PL[k][i + 1] = _ov_left(PL[k][i], mps[i], lower[k][i])
        for i in range(L - 2, -1, -1):
            theta = np.einsum("asb,btc->astc", mps[i], mps[i + 1])
            E, theta = solve(i, theta)
            cl, d1, d2, cr = theta.shape
            U, S, Vh, disc = truncated_svd(theta.reshape(cl * d1, d2 * cr), chi_max, cutoff)
            max_disc = max(max_disc, disc)
            mps[i], mps[i + 1] = (U * S[None, :]).reshape(cl, d1, -1), Vh.reshape(-1, d2, cr)
            Renv[i + 1] = grow_right(Renv[i + 2], mps[i + 1], mpo[i + 1])
            for k in range(K):
                PR[k][i + 1] = _ov_right(PR[k][i + 2], mps[i + 1], lower[k][i + 1])
            S_bonds[i] = S

    overlaps = [abs(overlap(psi, mps)) for psi in lower]
    return {
        "energy": E,
        "mps": mps,
        "entropies": np.array([entropy(S) for S in S_bonds]),
        "discarded": max_disc,
        "overlaps": overlaps,
    }
