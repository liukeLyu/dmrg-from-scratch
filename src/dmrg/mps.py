"""Matrix product states (MPS) with plain NumPy.

An MPS on L sites is a list of 3-index tensors. Tensor ``A`` on one site has
shape ``(chi_left, d, chi_right)``:

    index 0  left bond   (connects to the site on the left)
    index 1  physical    (the spin, d = 2)
    index 2  right bond  (connects to the site on the right)

The outer bonds of the first and last tensor have dimension 1.
Reference: U. Schollwöck, "The density-matrix renormalization group in the
age of matrix product states", https://arxiv.org/abs/1008.3477 (Sec. 4).
"""

import numpy as np


def random_mps(L, d=2, chi=8, seed=None):
    """Random MPS with bond dimension at most ``chi`` (not normalized)."""
    rng = np.random.default_rng(seed)
    # Bond k sits between site k-1 and site k; it cannot exceed the Hilbert
    # space dimension on either side, d^k or d^(L-k).
    dims = [min(chi, d**k, d ** (L - k)) for k in range(L + 1)]
    return [rng.normal(size=(dims[i], d, dims[i + 1])) for i in range(L)]


def product_mps(states):
    """MPS of a product state; ``states`` is a list of length-d vectors."""
    return [np.asarray(v, dtype=float).reshape(1, -1, 1) for v in states]


def from_dense(psi, L, d=2, chi_max=None):
    """Exact (or truncated) MPS of a state vector by successive SVDs, left to right.

    Each SVD splits off one site: psi[(left bond, s), (rest)] = U S Vh. U becomes
    the site tensor (a left isometry) and S Vh is carried on to the next step.
    With ``chi_max=None`` nothing is truncated and the MPS is exact.
    """
    chi_max = chi_max or d**L
    mps = []
    rest = np.asarray(psi).reshape(1, -1)  # rest[left bond, remaining physical indices]
    for _ in range(L - 1):
        chi_l = rest.shape[0]
        U, S, Vh, _ = truncated_svd(rest.reshape(chi_l * d, -1), chi_max, cutoff=0)
        mps.append(U.reshape(chi_l, d, -1))
        rest = S[:, None] * Vh
    mps.append(rest.reshape(rest.shape[0], d, 1))
    return mps


def to_dense(mps):
    """Contract the MPS into a full state vector of length d^L (site 0 = most significant)."""
    psi = np.ones((1, 1))  # (physical indices so far, right bond)
    for A in mps:
        # psi[p, a] A[a, s, b] -> psi[p, s, b], then merge (p, s)
        psi = np.einsum("pa,asb->psb", psi, A).reshape(-1, A.shape[2])
    return psi[:, 0]


def overlap(bra, ket):
    """<bra|ket> by zipping the two chains from left to right."""
    E = np.ones((1, 1))  # E[x, y]: x = ket bond, y = bra bond
    for A, B in zip(ket, bra, strict=True):
        E = np.einsum("xy,xsa,ysb->ab", E, A, B.conj())
    return E[0, 0]


def left_canonicalize(mps):
    """Make every tensor a left isometry with QR, sweeping left to right.

    After the sweep sum_{a,s} A[a,s,b] conj(A[a,s,b']) = delta(b,b') on every
    site except the last, which keeps the norm of the state.
    """
    mps = [A.copy() for A in mps]
    for i in range(len(mps) - 1):
        chi_l, d, chi_r = mps[i].shape
        Q, R = np.linalg.qr(mps[i].reshape(chi_l * d, chi_r))
        mps[i] = Q.reshape(chi_l, d, -1)
        # Push R into the next site: R[a, b] A[b, s, c]
        mps[i + 1] = np.einsum("ab,bsc->asc", R, mps[i + 1])
    return mps


def right_canonicalize(mps):
    """Mirror image of :func:`left_canonicalize`: right isometries, sweeping right to left."""
    flipped = [A.transpose(2, 1, 0) for A in reversed(mps)]
    return [A.transpose(2, 1, 0) for A in reversed(left_canonicalize(flipped))]


def normalize(mps):
    """Right-canonical, unit-norm copy of the MPS."""
    mps = right_canonicalize(mps)
    mps[0] = mps[0] / np.linalg.norm(mps[0])
    return mps


def truncated_svd(M, chi_max, cutoff=1e-12):
    """SVD of a matrix keeping at most ``chi_max`` singular values above ``cutoff``.

    Returns ``U, S, Vh, discarded`` with S normalized to sum(S^2) = 1 and
    ``discarded`` the weight sum(S^2) that was thrown away. Keeping the largest
    Schmidt values is the best low-rank approximation (Eckart-Young theorem).
    """
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    S = S / np.linalg.norm(S)
    keep = min(chi_max, max(1, int(np.sum(S > cutoff))))
    discarded = float(np.sum(S[keep:] ** 2))
    S = S[:keep] / np.linalg.norm(S[:keep])
    return U[:, :keep], S, Vh[:keep], discarded


def entropy(S):
    """Von Neumann entanglement entropy -sum p log2 p of Schmidt values S (p = S^2), in bits."""
    p = np.asarray(S) ** 2
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log2(p)))


def schmidt_values(mps):
    """Schmidt values on every bond 1..L-1, from a left-to-right SVD sweep.

    Start right-canonical, so the state is ``C A_1 ... ``with everything to the
    right of the current site an isometry; an SVD of the current site then
    gives the Schmidt values of the cut to its right.
    """
    mps = normalize(mps)
    out = []
    for i in range(len(mps) - 1):
        chi_l, d, chi_r = mps[i].shape
        U, S, Vh, _ = truncated_svd(mps[i].reshape(chi_l * d, chi_r), chi_max=chi_l * d, cutoff=0)
        out.append(S)
        mps[i] = U.reshape(chi_l, d, -1)
        mps[i + 1] = np.einsum("a,ab,bsc->asc", S, Vh, mps[i + 1])
    return out


def compress(mps, chi_max):
    """Truncate every bond to ``chi_max`` by sweeping SVDs (one pass, not variational)."""
    mps = normalize(mps)
    for i in range(len(mps) - 1):
        chi_l, d, chi_r = mps[i].shape
        U, S, Vh, _ = truncated_svd(mps[i].reshape(chi_l * d, chi_r), chi_max)
        mps[i] = U.reshape(chi_l, d, -1)
        mps[i + 1] = np.einsum("a,ab,bsc->asc", S, Vh, mps[i + 1])
    return mps
