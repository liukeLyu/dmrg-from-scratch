"""The transverse-field Ising chain as a matrix product operator (MPO).

    H = -J sum_i X_i X_{i+1} - h sum_i Z_i     (open chain, L sites)

This is the convention of ``medt.tfim`` and of TeNPy's ``TFIChain`` (J, g=h).
An MPO tensor ``W`` has shape ``(w_left, w_right, d_out, d_in)``:

    index 0  left MPO bond
    index 1  right MPO bond
    index 2  physical output (bra side)
    index 3  physical input  (ket side)

The bulk tensor is a 3x3 matrix of 2x2 operators, read as a finite-state
machine (Schollwöck https://arxiv.org/abs/1008.3477, Sec. 6.1). State 0 means
"nothing placed yet", state 2 means "a term is finished", and state 1 means
"an X was placed, the next site must close it":

    W = [[ I,  X,  -h Z ],
         [ 0,  0,  -J X ],
         [ 0,  0,   I   ]]

Multiplying W's along the chain and taking the (0, 2) entry sums every path
0 -> ... -> 2, and each path is one term of H.
"""

import numpy as np

I2 = np.eye(2)
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])


def tfim_bulk_tensor(h, J=1.0):
    """The 3x3 bulk tensor W[w_left, w_right, out, in]."""
    W = np.zeros((3, 3, 2, 2))
    W[0, 0] = I2
    W[0, 1] = X
    W[0, 2] = -h * Z
    W[1, 2] = -J * X
    W[2, 2] = I2
    return W


def tfim_mpo(L, h, J=1.0):
    """List of L MPO tensors; the first keeps row 0 and the last keeps column 2."""
    W = tfim_bulk_tensor(h, J)
    mpo = [W.copy() for _ in range(L)]
    mpo[0] = W[0:1, :, :, :]  # left boundary: start in state 0
    mpo[-1] = W[:, 2:3, :, :]  # right boundary: end in state 2
    return mpo


def mpo_to_dense(mpo):
    """Contract the MPO into a d^L x d^L matrix (only for small L, as a check)."""
    M = np.ones((1, 1, 1))  # M[out, in, right MPO bond]
    for W in mpo:
        # M[p, q, w] W[w, v, s, t] -> [p s, q t, v]
        M = np.einsum("pqw,wvst->psqtv", M, W)
        n = M.shape[0] * M.shape[1]
        M = M.reshape(n, n, -1)
    return M[:, :, 0]


def expectation(mps, mpo):
    """<psi|H|psi> / <psi|psi> by a left-to-right sweep of environments."""
    E = np.ones((1, 1, 1))  # E[x, w, y]: ket bond, MPO bond, bra bond
    N = np.ones((1, 1))
    for A, W in zip(mps, mpo, strict=True):
        E = np.einsum("xwy,xsa,wvts,ytb->avb", E, A, W, A.conj(), optimize=True)
        N = np.einsum("xy,xsa,ysb->ab", N, A, A.conj())
    return float(np.real(E[0, 0, 0] / N[0, 0]))
