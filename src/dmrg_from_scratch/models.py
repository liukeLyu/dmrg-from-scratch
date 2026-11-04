"""Spin models on a cylinder, written as sums of Pauli strings.

A cylinder is ``Lx`` columns of ``W`` sites, open along x and periodic around y. Sites are
numbered column by column, ``s = x * W + y``, which is the order an MPS walks through them.
Neighbours along y are then 1 apart (or ``W - 1`` across the seam) and neighbours along x are
``W`` apart, so the MPS has to carry correlations over a distance of about ``W`` sites. That is
the reason DMRG gets expensive as the cylinder gets wider.

The main model is the frustrated J1-J2 Heisenberg antiferromagnet,

    H = J1 sum_<ij> S_i . S_j + J2 sum_<<ij>> S_i . S_j,   S = sigma / 2,

with nearest neighbours <ij> and diagonal next-nearest neighbours <<ij>>. At J2/J1 = 0.5 it has
a sign problem for quantum Monte Carlo, so DMRG is the standard classical method. The
transverse-field Ising model is included as a simpler second example.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PauliSum:
    """``H = sum_k coeff_k P_k`` with each ``P_k`` a Pauli string given as ``((site, "X"), ...)``."""

    n: int
    terms: list[tuple[float, tuple[tuple[int, str], ...]]]
    name: str = ""
    shape: tuple[int, int] = field(default=(0, 0))  # (Lx, W) for a cylinder

    @property
    def one_norm(self) -> float:
        """LCU one-norm ``lambda = sum_k |coeff_k|``; it sets the cost of qubitization."""
        return float(sum(abs(c) for c, _ in self.terms))

    def pauli_counts(self) -> dict[str, int]:
        """How many terms of each Pauli word, e.g. ``{"XX": 72, "YY": 72, "ZZ": 72}``."""
        counts: dict[str, int] = {}
        for _, ops in self.terms:
            word = "".join(p for _, p in ops)
            counts[word] = counts.get(word, 0) + 1
        return counts


def cylinder_bonds(Lx: int, W: int, diagonal: bool = False) -> list[tuple[int, int]]:
    """Nearest-neighbour bonds (or diagonal bonds) of an ``Lx x W`` cylinder, each once, ``i < j``.

    For ``W = 2`` going up or down around the cylinder reaches the same site, so the
    cylinder is a ladder and each bond is kept once.
    """

    def site(x, y):
        return x * W + (y % W)

    bonds = set()
    for x in range(Lx):
        for y in range(W):
            if diagonal:
                if x + 1 < Lx and W > 1:
                    candidates = [(site(x, y), site(x + 1, y + 1)), (site(x, y), site(x + 1, y - 1))]
                else:
                    candidates = []
            else:
                candidates = [(site(x, y), site(x, y + 1))] if W > 1 else []
                if x + 1 < Lx:
                    candidates.append((site(x, y), site(x + 1, y)))
            for i, j in candidates:
                if i != j:
                    bonds.add((min(i, j), max(i, j)))
    return sorted(bonds)


def default_length(W: int) -> int:
    """``Lx = W``, or ``W + 1`` for odd ``W``: nearly square, with an even number of sites so the
    antiferromagnet has a unique singlet ground state."""
    return W + W % 2


def j1j2_cylinder(W: int, Lx: int | None = None, J2: float = 0.5, J1: float = 1.0) -> PauliSum:
    """J1-J2 Heisenberg model on an ``Lx x W`` cylinder (``Lx = default_length(W)`` by default)."""
    Lx = default_length(W) if Lx is None else Lx
    terms = []
    for J, diagonal in ((J1, False), (J2, True)):
        if J == 0:
            continue
        for i, j in cylinder_bonds(Lx, W, diagonal):
            for p in "XYZ":
                terms.append((J / 4, ((i, p), (j, p))))
    return PauliSum(Lx * W, terms, name=f"J1-J2 Heisenberg, J2/J1 = {J2 / J1:g}", shape=(Lx, W))


def tfim_cylinder(W: int, Lx: int | None = None, h: float = 3.0, J: float = 1.0) -> PauliSum:
    """``H = -J sum_<ij> X_i X_j - h sum_i Z_i`` on an ``Lx x W`` cylinder (2D critical point h ~ 3.04)."""
    Lx = default_length(W) if Lx is None else Lx
    terms = [(-J, ((i, "X"), (j, "X"))) for i, j in cylinder_bonds(Lx, W)]
    terms += [(-h, ((i, "Z"),)) for i in range(Lx * W)]
    return PauliSum(Lx * W, terms, name=f"TFIM, h = {h:g}", shape=(Lx, W))


def n_bonds(H: PauliSum) -> int:
    return int(np.sum([len(ops) == 2 for _, ops in H.terms]))


def neel_pinning(H: PauliSum, h: float = 0.5) -> PauliSum:
    """``H`` plus a staggered field ``h (-1)^(x+y) S^z`` on the first and last column of its cylinder.

    The field stands in for a frozen Néel neighbour beyond each open end: an edge spin coupled
    with ``J = 1`` to a spin fixed at ``S^z = -/+ 1/2`` feels ``h = 1/2`` [White and Chernyshev,
    arXiv:0705.2746]. It picks one of the Néel states, so the order parameter can be read off as
    a plain ``<S^z>`` in the middle instead of from a correlation function.
    """
    Lx, W = H.shape
    terms = list(H.terms)
    for x in (0, Lx - 1):
        for y in range(W):
            terms.append((0.5 * h * (-1) ** (x + y), ((x * W + y, "Z"),)))  # h S^z = (h/2) Z
    return PauliSum(H.n, terms, name=f"{H.name}, edge pinning {h:g}", shape=H.shape)


def hubbard_cylinder(W: int, Lx: int, U: float, t: float = 1.0) -> PauliSum:
    """The Hubbard model on an ``Lx x W`` cylinder, as qubits by Jordan-Wigner, for exact diagonalization.

        H = -t sum_<ij>,s (c+_is c_js + h.c.) + U sum_i (n_i,up - 1/2)(n_i,dn - 1/2)

    Site ``s`` holds two qubits, ``2s`` (spin up) and ``2s + 1`` (spin down), and a qubit in
    ``|1>`` (``Z = -1``) is occupied, so ``n = (1 - Z)/2``. Jordan-Wigner turns a hop between qubits
    ``p < q`` into ``(X_p Z...Z X_q + Y_p Z...Z Y_q) / 2``, with a ``Z`` on every qubit in between:
    the fermion sign. The interaction, written particle-hole symmetrically, is ``(U/4) Z_up Z_dn``.
    That form puts the ground state at half filling; with ``U n_up n_dn`` instead (TeNPy's
    convention) the half-filled energy is higher by ``U/4`` per site.
    """
    terms = []
    for i, j in cylinder_bonds(Lx, W):
        for spin in (0, 1):
            p, q = 2 * i + spin, 2 * j + spin
            string = tuple((r, "Z") for r in range(p + 1, q))
            for P in "XY":
                terms.append((-t / 2, ((p, P), *string, (q, P))))
    for s in range(Lx * W):
        terms.append((U / 4, ((2 * s, "Z"), (2 * s + 1, "Z"))))
    return PauliSum(2 * Lx * W, terms, name=f"Hubbard, U = {U:g}, t = {t:g}", shape=(Lx, W))
