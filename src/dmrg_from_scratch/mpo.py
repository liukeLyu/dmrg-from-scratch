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

from .models import PauliSum

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


# Any sum of one- and two-site Pauli terms, for the cylinders of notebooks 09 and 11-14.
#
# Any sum of one- and two-site Pauli terms as a matrix product operator.
#
# An MPO tensor ``W[w_left, w_right, out, in]`` is read as a finite-state machine
# (Schollwöck https://arxiv.org/abs/1008.3477, Sec. 6.1). Each MPO bond has the states
#
#     "start"     nothing placed yet
#     "done"      a term is complete; only identities follow
#     (P, i)      the operator P was placed on site i; a later site j closes it with c_ij Q_j
#
# Walking from left to right, every path start -> ... -> done spells one term of H. A channel
# (P, i) is open across a bond only if some term still has to close it on the right, so the MPO
# bond dimension is 2 plus the number of interactions that cross that bond. On a cylinder of
# width W about 3W Heisenberg bonds cross every cut, which is why the MPO, and much more the
# MPS, grows with W.
#
# All tensors are real: Y is stored as iY = [[0, 1], [-1, 0]], and each pair of Y's puts a
# factor (-i)^2 = -1 into the coefficient.

PAULI = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, 1.0], [-1.0, 0.0]]),  # i * Y, real
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


def build_mpo(H: PauliSum) -> list[np.ndarray]:
    """List of ``H.n`` MPO tensors; the first has one row (start), the last one column (done)."""
    n = H.n
    onsite = [np.zeros((2, 2)) for _ in range(n)]
    pairs = []  # (i, P, j, Q, c) with i < j
    for c, ops in H.terms:
        n_y = sum(p == "Y" for _, p in ops)
        c = c * (-1) ** (n_y // 2)
        if len(ops) == 1:
            ((i, p),) = ops
            onsite[i] += c * PAULI[p]
        elif len(ops) == 2:
            (i, p), (j, q) = sorted(ops)
            pairs.append((i, p, j, q, c))
        else:
            raise ValueError("only one- and two-site terms")

    # channels[b] = open channels on the bond between sites b and b + 1
    channels = []
    for b in range(n - 1):
        open_ = sorted({(p, i) for i, p, j, _, _ in pairs if i <= b < j}, key=lambda t: (t[1], t[0]))
        channels.append(["start", "done", *open_])
    left = [["start"]] + channels
    right = channels + [["done"]]

    mpo = []
    for s in range(n):
        li = {k: a for a, k in enumerate(left[s])}
        ri = {k: a for a, k in enumerate(right[s])}
        Wt = np.zeros((len(li), len(ri), 2, 2))
        if "start" in li and "start" in ri:
            Wt[li["start"], ri["start"]] = PAULI["I"]
        if "done" in li and "done" in ri:
            Wt[li["done"], ri["done"]] = PAULI["I"]
        if "start" in li and "done" in ri:
            Wt[li["start"], ri["done"]] = onsite[s]
        for key in ri:
            if isinstance(key, tuple):
                p, i = key
                if i == s and "start" in li:
                    Wt[li["start"], ri[key]] = PAULI[p]  # open a channel
                elif key in li:
                    Wt[li[key], ri[key]] = PAULI["I"]  # carry it along
        for i, p, j, q, c in pairs:
            if j == s:
                Wt[li[(p, i)], ri["done"]] += c * PAULI[q]  # close it
        mpo.append(Wt)
    return mpo


def pauli_string_mpo(H: PauliSum) -> list[np.ndarray]:
    """MPO of any sum of real Pauli strings, of any length: one channel per term.

    The same finite-state machine as :func:`build_mpo`, without sharing channels: a term that
    acts on sites i1 < ... < im opens its own channel at i1 (with its coefficient), applies its
    Pauli on each of its sites and identities in between, and closes at im. The MPO bond
    dimension is 2 plus the number of terms whose span covers the bond, larger than
    :func:`build_mpo` needs for two-site terms, but it takes the six-spin plaquette terms of
    notebook 18.
    """
    n = H.n
    onsite = [np.zeros((2, 2)) for _ in range(n)]
    strings = []  # (sites, paulis, coefficient), sites increasing
    for c, ops in H.terms:
        n_y = sum(p == "Y" for _, p in ops)
        if n_y % 2:
            raise ValueError("odd number of Y: the term is not real")
        c = c * (-1) ** (n_y // 2)
        ops = sorted(ops)
        if len(ops) == 1:
            ((i, p),) = ops
            onsite[i] += c * PAULI[p]
        else:
            strings.append(([i for i, _ in ops], dict(ops), c))

    # channels[b] = open channels on the bond between sites b and b + 1: every term whose span covers it
    channels = [
        ["start", "done", *[k for k, (st, _, _) in enumerate(strings) if st[0] <= b < st[-1]]] for b in range(n - 1)
    ]
    left = [["start"]] + channels
    right = channels + [["done"]]
    mpo = []
    for s in range(n):
        li = {k: a for a, k in enumerate(left[s])}
        ri = {k: a for a, k in enumerate(right[s])}
        Wt = np.zeros((len(li), len(ri), 2, 2))
        if "start" in li and "start" in ri:
            Wt[li["start"], ri["start"]] = PAULI["I"]
        if "done" in li and "done" in ri:
            Wt[li["done"], ri["done"]] = PAULI["I"]
        if "start" in li and "done" in ri:
            Wt[li["start"], ri["done"]] = onsite[s]
        for k, (sites, paulis, c) in enumerate(strings):
            op = PAULI[paulis.get(s, "I")]
            if s == sites[0]:
                Wt[li["start"], ri[k]] = c * op  # open the channel, with the coefficient
            elif sites[0] < s < sites[-1]:
                Wt[li[k], ri[k]] = op  # carry it, acting on the term's own sites
            elif s == sites[-1]:
                Wt[li[k], ri["done"]] += op  # close it
        mpo.append(Wt)
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
