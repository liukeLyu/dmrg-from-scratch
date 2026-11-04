"""A matrix product state as a quantum circuit, exactly or with layers of two-qubit gates.

Exact: sequential generation (Schön, Solano, Verstraete, Cirac, Wolf, https://arxiv.org/abs/quant-ph/0501096).
An ancilla register of k = log2(chi) qubits carries the bond index. Site by site, one unitary acts on
the ancilla and on that site's qubit (still in |0>):

    U_i |a>|0> = sum_{s, b} B_i[a, s, b] |b>|s>

These columns are orthonormal exactly when B_i is a right isometry, sum_{s,b} B[a,s,b] conj(B[a',s,b]) = delta,
so a right-canonical MPS (what our DMRG returns) gives the unitaries directly; the remaining columns are any
orthonormal completion. The ancilla starts in |0> (the left bond has dimension 1) and ends in |0> (so does
the right bond), so it comes out unentangled. Each U_i acts on k + 1 qubits.

Approximate: for chi = 2 the ancilla can be the next site's qubit, so the MPS becomes a staircase of
n - 1 two-qubit gates. Ran (https://arxiv.org/abs/1908.07958) stacks such staircases: truncate the
state to chi = 2, build the staircase U_1, undo it, psi -> U_1^dagger psi, which is closer to a product
state; repeat. After K layers, U_1 ... U_K |0...0> approximates psi with fidelity
|<0...0| U_K^dagger ... U_1^dagger psi>|.

Conventions as in mps.py: A[left, phys, right], site 0 is the most significant bit. Gates are plain
NumPy matrices; ``mps_circuit`` and ``layers_circuit`` place them in a PennyLane circuit.
"""

import numpy as np
from scipy.linalg import null_space

from .mps import from_dense, normalize


def pad_bonds(mps):
    """Copy of the MPS with every bond padded with zeros to a power of two (the state is unchanged)."""
    dims = [1] + [A.shape[2] for A in mps[:-1]] + [1]
    padded = [1] + [int(2 ** np.ceil(np.log2(d))) for d in dims[1:-1]] + [1]
    out = []
    for i, A in enumerate(mps):
        P = np.zeros((padded[i], A.shape[1], padded[i + 1]), dtype=A.dtype)
        P[: A.shape[0], :, : A.shape[2]] = A
        out.append(P)
    return out


def right_canonical(mps):
    """Right-canonical, unit-norm MPS: unchanged if it already is (as DMRG returns it), else via QR."""
    isometries = all(np.allclose(np.einsum("asb,csb->ac", B, B.conj()), np.eye(B.shape[0])) for B in mps[1:])
    if not isometries:
        return normalize(mps)
    return [mps[0] / np.linalg.norm(mps[0]), *mps[1:]]


def _complete(U, cols):
    """Fill the columns of U outside ``cols`` with an orthonormal basis of the complement."""
    U[:, [c for c in range(U.shape[1]) if c not in cols]] = null_space(U[:, cols].conj().T)
    return U


def site_unitaries(mps):
    """Exact sequential circuit of an MPS: one unitary per site on ``(ancilla qubits..., site qubit)``.

    Returns ``(unitaries, k)``, with ``k`` ancilla qubits and each unitary of size ``2^(k+1)``.
    """
    mps = right_canonical(mps)
    k = int(np.ceil(np.log2(max(A.shape[2] for A in mps))))
    chi = 2**k  # the ancilla register; smaller bonds use only part of it
    unitaries = []
    for B in mps:
        chi_l, d, chi_r = B.shape
        U = np.zeros((chi * d, chi * d), dtype=B.dtype)
        cols = []
        for a in range(chi_l):
            for s in range(d):
                for b in range(chi_r):
                    U[b * d + s, a * d] = B[a, s, b]  # input |a>|0>, output |b>|s>
            cols.append(a * d)
        unitaries.append(_complete(U, cols))
    return unitaries, k


def staircase(mps):
    """Two-qubit gates on sites ``(i, i + 1)``, i = 0..n-2, that prepare a chi <= 2 MPS from |0...0>.

    Gate i takes the bond index a from qubit i and the fresh qubit i + 1 in |0>, and returns site i's
    spin s on qubit i and the next bond index b on qubit i + 1. The last gate writes both last spins.
    """
    mps = right_canonical(mps)
    if max(A.shape[2] for A in mps) > 2:
        raise ValueError("a staircase of two-qubit gates holds bond dimension 2 at most")
    n = len(mps)
    gates = []
    for i in range(n - 1):
        B = mps[i] if i < n - 2 else np.einsum("asb,bt->ast", mps[i], mps[i + 1][:, :, 0])
        U = np.zeros((4, 4), dtype=B.dtype)
        for a in range(B.shape[0]):
            U[:, a * 2] = B[a].reshape(-1)  # output index s * 2 + b (or s * 2 + t)
        gates.append(_complete(U, list(range(0, 2 * B.shape[0], 2))))
    return gates


def apply_staircase(psi, gates, dagger=False):
    """Apply a staircase (or its inverse) to a state vector of n = len(gates) + 1 qubits."""
    n = len(gates) + 1
    psi = np.asarray(psi).reshape([2] * n)
    order = range(n - 2, -1, -1) if dagger else range(n - 1)
    for i in order:
        G = gates[i].conj().T if dagger else gates[i]
        psi = np.tensordot(G.reshape(2, 2, 2, 2), psi, axes=([2, 3], [i, i + 1]))
        psi = np.moveaxis(psi, [0, 1], [i, i + 1])
    return psi.reshape(-1)


def disentangling_layers(psi, n_layers):
    """Ran's layers: returns the list of staircases and the fidelity after each number of layers."""
    n = int(np.log2(len(psi)))
    phi = np.asarray(psi) / np.linalg.norm(psi)
    layers, fidelities = [], []
    for _ in range(n_layers):
        gates = staircase(from_dense(phi, n, chi_max=2))
        phi = apply_staircase(phi, gates, dagger=True)
        layers.append(gates)
        fidelities.append(float(abs(phi[0])))  # overlap of U_K^dag ... U_1^dag psi with |0...0>
    return layers, fidelities


def prepare(layers, n):
    """State U_1 U_2 ... U_K |0...0> made by a list of staircases."""
    psi = np.zeros(2**n)
    psi[0] = 1.0
    for gates in reversed(layers):
        psi = apply_staircase(psi, gates)
    return psi


def to_pennylane_mps(mps):
    """The MPS in the format of ``qml.MPSPrep``: right-canonical, bonds padded, end tensors of rank 2."""
    mps = pad_bonds(right_canonical(mps))
    return [mps[0][0], *mps[1:-1], mps[-1][:, :, 0]]


def mps_circuit(mps, wires, work_wires):
    """Queue the exact sequential circuit in a PennyLane circuit (``len(work_wires) >= k``)."""
    import pennylane as qml

    unitaries, k = site_unitaries(mps)
    anc = list(work_wires)[:k]
    for U, w in zip(unitaries, wires, strict=True):
        qml.QubitUnitary(U, wires=[*anc, w])


def layers_circuit(layers, wires):
    """Queue Ran's layers in a PennyLane circuit: U_1 U_2 ... U_K |0...0>, so U_K acts first."""
    import pennylane as qml

    wires = list(wires)
    for gates in reversed(layers):
        for i, G in enumerate(gates):
            qml.QubitUnitary(G, wires=[wires[i], wires[i + 1]])
