"""Reduced density matrices of a few sites from an MPS: entanglement microscopy with DMRG.

Entanglement microscopy (Wang et al., https://arxiv.org/abs/2402.14916) looks at the reduced
density matrix (RDM) of a small region A, a few sites, and asks whether those sites are
entangled with each other: negativity for two parts, genuine multipartite negativity (GMN) for
three or four. Quantum Monte Carlo samples the RDM directly: a matrix element

    <a| rho_A |b> = Z^-1 sum_xi <a, xi| exp(-beta H) |b, xi>

is a partition function whose imaginary time is periodic everywhere except on the sites of A,
where the world lines are cut open, and the frequency with which each pair (a, b) of boundary
configurations appears is proportional to that element (Mao, Ding and Yan,
https://arxiv.org/abs/2310.16709). Pauli tomography, rho_A = 2^-n sum_P <P> P, is another way
to rebuild the same matrix from expectation values.

An MPS gives the matrix directly: contract the chain from the left, keep the physical
legs of the sites in A open (a ket and a bra leg each) and trace out every other site in
between. For n sites spread over a segment of m sites this costs about m d^(2n) chi^3, so a
2 x 2 plaquette on a cylinder of width W (a segment of W + 2 sites) is cheap.

Conventions are those of the package: tensors ``A[left, phys, right]``, basis index 0 is
Z = +1, and the first site of the region is the most significant bit of ``rho``.
"""

import itertools

import numpy as np

PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1j], [1j, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
}


def segment_rdm(tensors, sites, left=None):
    """RDM of ``sites`` (indices into ``tensors``) from a right-canonical segment of an MPS.

    ``tensors`` are consecutive site tensors ``B[left, phys, right]`` whose right part is in
    right-canonical form (each B is a right isometry), so the right end closes with the
    identity. ``left`` is the reduced density matrix of the bond to the left of ``tensors[0]``,
    ``diag(S^2)`` with S the Schmidt values there; ``None`` means the open left end of a finite
    chain. Start the segment at the first site of the region whenever ``left`` is known: the
    sites before it are traced out anyway. Returns ``rho`` with the qubits in the order given by
    ``sites``.
    """
    order = sorted(sites)
    d = tensors[0].shape[1]
    chi = tensors[0].shape[0]
    E = (np.ones((1, 1)) if left is None else np.asarray(left))[None, None]  # E[ket legs, bra legs, a, b]
    assert E.shape[2:] == (chi, chi)
    for i in range(order[-1] + 1):  # sites left of the region are traced out too
        B = tensors[i]
        if i in order:  # keep the leg: E[K, K', a, b] B[a, s, c] B*[b, t, e] -> E[(K s), (K' t), c, e]
            E = np.einsum("KLab,asc,bte->KsLtce", E, B, B.conj(), optimize=True)
            k = E.shape[0] * d
            E = E.reshape(k, k, B.shape[2], B.shape[2])
        else:  # trace the site out
            E = np.einsum("KLab,asc,bse->KLce", E, B, B.conj(), optimize=True)
    rho = np.einsum("KLaa->KL", E)  # right-canonical: the rest of the chain is the identity
    rho = rho / np.trace(rho)
    return permute_qubits(rho, [order.index(s) for s in sites])


def permute_qubits(rho, perm):
    """Reorder the qubits of ``rho``: qubit k of the result is qubit ``perm[k]`` of the input."""
    n = len(perm)
    T = rho.reshape((2,) * (2 * n))
    T = T.transpose(list(perm) + [n + p for p in perm])
    return T.reshape(2**n, 2**n)


def dense_rdm(psi, sites, n):
    """RDM of ``sites`` from a state vector on ``n`` qubits (site 0 = most significant bit)."""
    T = np.asarray(psi).reshape((2,) * n)
    rest = [s for s in range(n) if s not in sites]
    T = T.transpose(list(sites) + rest).reshape(2 ** len(sites), -1)
    return T @ T.conj().T


def partial_trace(rho, keep):
    """RDM of the qubits ``keep`` (in that order) of the multi-qubit density matrix ``rho``."""
    n = int(np.log2(rho.shape[0]))
    rest = [k for k in range(n) if k not in keep]
    T = rho.reshape((2,) * (2 * n)).transpose(list(keep) + rest + [n + k for k in keep] + [n + k for k in rest])
    a, b = 2 ** len(keep), 2 ** len(rest)
    return np.einsum("ikjk->ij", T.reshape(a, b, a, b))


def plaquette_measures(rho):
    """Negativities and GMN of a 2 x 2 plaquette RDM with qubits (x,y), (x,y+1), (x+1,y), (x+1,y+1).

    Returns the negativity N = (||rho^T||_1 - 1) / 2 of a nearest-neighbour pair along y and along
    x and of the diagonal pair, the GMN of the three-site corner (x,y), (x,y+1), (x+1,y), and the
    GMN of the whole plaquette. Needs ``medt`` (the GMN is a semidefinite program).

    A matrix with statistical errors, such as a QMC estimate, can have slightly negative
    eigenvalues, for which the GMN program is unbounded; ``rho`` is therefore first replaced by
    the nearest density matrix (negative eigenvalues set to zero, trace renormalized).
    """
    from medt.gmn import gmn, negativity  # optional dependency, like PennyLane in resources.py

    w, V = np.linalg.eigh((rho + rho.conj().T) / 2)
    w = np.clip(w, 0, None)
    rho = np.real_if_close((V * (w / w.sum())) @ V.conj().T)

    def robust_gmn(r):
        try:
            return gmn(r)
        except RuntimeError:  # the default interior-point solver gave up; try the first-order one
            return gmn(r, solver="SCS", eps=1e-9)

    return {
        "N_nn_y": negativity(partial_trace(rho, [0, 1])),
        "N_nn_x": negativity(partial_trace(rho, [0, 2])),
        "N_diag": negativity(partial_trace(rho, [0, 3])),
        "gmn_corner": robust_gmn(partial_trace(rho, [0, 1, 2])),
        "gmn_plaquette": robust_gmn(rho),
    }


def pauli_expectations(rho):
    """``{"XZ": <X Z>, ...}`` for every Pauli string on the qubits of ``rho``."""
    n = int(np.log2(rho.shape[0]))
    out = {}
    for word in itertools.product("IXYZ", repeat=n):
        P = PAULIS[word[0]]
        for p in word[1:]:
            P = np.kron(P, PAULIS[p])
        out["".join(word)] = float(np.real(np.trace(rho @ P)))
    return out


def rdm_from_paulis(expectations):
    """Pauli tomography: ``rho = 2^-n sum_P <P> P`` from the expectation values of every Pauli string."""
    n = len(next(iter(expectations)))
    rho = np.zeros((2**n, 2**n), dtype=complex)
    for word, value in expectations.items():
        P = PAULIS[word[0]]
        for p in word[1:]:
            P = np.kron(P, PAULIS[p])
        rho += value * P
    return rho / 2**n


def symmetrize(rho):
    """Average ``rho`` with its image under the global spin flip ``prod Z`` (the Z2 of -XX - hZ).

    In the ordered phase DMRG may pick one of the two symmetry-broken ground states. The finite
    system used by QMC is in their symmetric superposition (the cat state), whose local RDM is
    the equal mixture of the two: this function turns one into the other.
    """
    n = int(np.log2(rho.shape[0]))
    P = np.diag([(-1) ** bin(k).count("1") for k in range(2**n)]).astype(float)
    return (rho + P @ rho @ P) / 2


def to_zz_basis(rho):
    """Rotate from H = -sum XX - h sum Z (this package) to H = -sum ZZ - h sum X (the QMC papers).

    A Hadamard on every qubit swaps X and Z, so it maps one convention onto the other.
    """
    n = int(np.log2(rho.shape[0]))
    Hd = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
    U = Hd
    for _ in range(n - 1):
        U = np.kron(U, Hd)
    return U @ rho @ U


def tenpy_segment(psi, first, last):
    """``(left, tensors)`` of sites ``first..last`` of a TeNPy MPS, ready for :func:`segment_rdm`.

    The TeNPy object is only read (duck typing), so this package does not import TeNPy. TeNPy's
    spin-1/2 site with a conserved parity has basis index 0 = Z = -1, the reverse of ours; the
    physical index is flipped wherever that is the case. For an infinite MPS the indices may run
    past the unit cell.
    """
    S = np.asarray(psi.get_SL(first))
    tensors = []
    for i in range(first, last + 1):
        B = psi.get_B(i, form="B").itranspose(["vL", "p", "vR"]).to_ndarray()
        z = psi.sites[i % psi.L].get_op("Sigmaz").to_ndarray().diagonal()
        tensors.append(B[:, ::-1, :] if z[0] < 0 else B)
    return np.diag(S**2), tensors
