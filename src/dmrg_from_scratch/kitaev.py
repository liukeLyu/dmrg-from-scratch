"""Kitaev's honeycomb model on a brick wall: the lattice, the Hamiltonian and the regions of notebook 18.

Kitaev's model (Kitaev, https://arxiv.org/abs/cond-mat/0506438) puts a spin 1/2 on each site of
the honeycomb lattice and couples only one spin component per bond, which component depending on
the bond's direction:

    H = sum_<ij>_gamma K_gamma S_i^gamma S_j^gamma,    S = sigma / 2,    gamma = x, y, z,

the convention of Lyu et al. (https://arxiv.org/abs/2505.18124, Eq. 1, with zero field). With
K > 0 (antiferromagnetic) and K_x = K_y = K_z the ground state is a gapless spin liquid; with
K_z much larger than K_x = K_y it is a gapped Z2 spin liquid of the toric-code type.

A honeycomb is a brick wall. Sites sit on an Lx x Ly grid, numbered column by column,
s = x * Ly + y, which is the order an MPS walks through them (as for the cylinders of
``models.py``). Every site has two horizontal bonds and one vertical bond:

    horizontal  (x, y) - (x + 1, y)   type x if x + y is even, type y if it is odd
    vertical    (x, y) - (x, y + 1)   type z, only if x + y is even

so each site has exactly one bond of each type. The grid is periodic in y (Ly even). It is
either open in x (a cylinder) or periodic with a twist: column Lx is column 0 shifted down by
``twist`` rows, (x + Lx, y) = (x, y - twist), with Lx + twist even so that the bond types close.
The twist lets the brick wall make the symmetric clusters of exact diagonalization: the 24-site
Nauru graph is Lx = 6, Ly = 4, twist = 2, and the 32-site Dyck graph is Lx = 4, Ly = 8,
twist = 4 (tests/test_kitaev.py checks both against the graphs' LCF notation,
[5, -9, 7, -7, 9, -5]^4 and [5, -5, 13, -13]^8). A hexagon is a 3 x 2 rectangle of the brick
wall whose lower-left site has x + y even.

On an open edge a site loses a bond, and the Kitaev model then has exact degeneracies (the
spin component of the missing bond is left unpaired), so exact comparisons use tori.
"""

import numpy as np

from .models import PauliSum

# (Lx, Ly, twist) of the clusters used by Lyu et al. (Fig. 2a and SM Sec. VII D)
NAURU = (6, 4, 2)
DYCK = (4, 8, 4)


def _site_function(Lx, Ly, twist):
    def site(x, y):
        return (x % Lx) * Ly + (y - twist * (x // Lx)) % Ly

    return site


def brickwall_bonds(Lx, Ly, periodic_x=True, twist=0):
    """Bonds ``(i, j, gamma)`` of the brick-wall honeycomb, each once, ``gamma`` in ``"xyz"``."""
    if Ly % 2 or (periodic_x and (Lx + twist) % 2):
        raise ValueError("Ly (and Lx + twist when periodic in x) must be even for the bond types to close")
    site = _site_function(Lx, Ly, twist)
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            if x + 1 < Lx or periodic_x:
                bonds.append((site(x, y), site(x + 1, y), "x" if (x + y) % 2 == 0 else "y"))
            if (x + y) % 2 == 0:
                bonds.append((site(x, y), site(x, y + 1), "z"))
    return bonds


def hexagon(x0, y0, Lx, Ly, twist=0):
    """The six sites of the hexagon whose lower-left corner is ``(x0, y0)`` (``x0 + y0`` even), in loop order.

    The order is that of Lyu et al. (SM Tables II-IV): bonds 1-2 z, 2-3 x, 3-4 y, 4-5 z, 5-6 x, 6-1 y,
    so the plaquette operator is ``W_p = X1 Y2 Z3 X4 Y5 Z6``.
    """
    if (x0 + y0) % 2:
        raise ValueError("the lower-left corner of a hexagon has x + y even")
    site = _site_function(Lx, Ly, twist)
    A, B, C = site(x0, y0), site(x0 + 1, y0), site(x0 + 2, y0)
    D, E, F = site(x0 + 2, y0 + 1), site(x0 + 1, y0 + 1), site(x0, y0 + 1)
    return (C, D, E, F, A, B)


HEXAGON_W = "XYZXYZ"  # the Pauli on each hexagon site in W_p: the type of the bond that leaves the hexagon


def plaquettes(Lx, Ly, periodic_x=True, twist=0):
    """Every hexagon of the brick wall (all of them on a torus, those inside the cylinder otherwise)."""
    xs = range(Lx) if periodic_x else range(Lx - 2)
    return [hexagon(x0, y0, Lx, Ly, twist) for x0 in xs for y0 in range(Ly) if (x0 + y0) % 2 == 0]


def torus_loops(Lx, Ly, twist=0):
    """The two loop operators of the torus, as Pauli words ``((site, "X"), ...)``.

    A loop operator is built like W_p, but on a closed path that winds around the torus: on each
    site of the path, the Pauli of the one bond the path does not use (Gohlke, Moessner and Pollmann,
    https://arxiv.org/abs/1804.06811, Sec. II, for the cylinder). It commutes with H and with every
    W_p, and the two loops (one per cycle of the torus) commute with each other, so their
    eigenvalues +-1 label four topological sectors. Each path is a shortest path in the unwrapped
    brick wall from (0, 0) to one of its images a (0, Ly) + b (Lx, twist); two images whose (a, b)
    have determinant +-1 give the two cycles. On some tori the shortest path to an image crosses
    itself on the torus, so a few images are tried and the shortest two simple loops are kept.
    """

    def neighbours(x, y):
        yield (x + 1, y), "x" if (x + y) % 2 == 0 else "y"
        yield (x - 1, y), "x" if (x - 1 + y) % 2 == 0 else "y"
        yield (x, y + 1) if (x + y) % 2 == 0 else (x, y - 1), "z"

    site = _site_function(Lx, Ly, twist)

    def loop(tx, ty):
        parent, frontier = {(0, 0): None}, [(0, 0)]
        while (tx, ty) not in parent:  # breadth-first search
            new = []
            for p in frontier:
                for q, _ in neighbours(*p):
                    if q not in parent:
                        parent[q] = p
                        new.append(q)
            frontier = new
        path = [parent[(tx, ty)]]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        path = path[::-1]  # (0, 0), ..., the site before the image of (0, 0)
        x, y = path[-1]
        around = [(x - tx, y - ty), *path, (tx, ty)]  # each site's two neighbours on the loop
        word = []
        for k, p in enumerate(path):
            used = {g for q, g in neighbours(*p) if q in (around[k], around[k + 2])}
            word.append((site(*p), ({"x", "y", "z"} - used).pop().upper()))
        return tuple(sorted(word)) if len({s for s, _ in word}) == len(word) else None  # None: crosses itself

    found = {}
    for a, b in [(1, 0), (0, 1), (1, 1), (-1, 1), (1, 2), (2, 1), (-1, 2), (-2, 1)]:
        word = loop(b * Lx, a * Ly + b * twist)
        if word is not None:
            found[a, b] = word
    pairs = [(p, q) for p in found for q in found if p < q and abs(p[0] * q[1] - p[1] * q[0]) == 1]
    if not pairs:
        raise ValueError("no two simple loops found on this torus")
    p, q = min(pairs, key=lambda pq: len(found[pq[0]]) + len(found[pq[1]]))
    return [found[p], found[q]]


def kitaev_honeycomb(
    Lx,
    Ly,
    K=(1.0, 1.0, 1.0),
    periodic_x=True,
    twist=0,
    flux_penalty=0.0,
    sector=None,
    parity=None,
    sector_penalty=0.5,
):
    """Kitaev's model ``sum K_gamma S^gamma S^gamma`` on the brick wall, as a PauliSum (each term K / 4).

    ``flux_penalty`` adds ``-lambda sum_p W_p`` over every hexagon. The plaquette operators commute
    with H and with each other, so this keeps every eigenstate and only lowers the flux-free
    sector (W_p = +1, where Lieb's theorem puts the ground state of the infinite lattice) below
    the others; on small tori a flux sector can otherwise be lower or degenerate with it.

    ``sector = (s1, s2)`` (torus only) adds ``-mu (s1 W_l1 + s2 W_l2)`` with ``mu = sector_penalty``
    and the loops of ``torus_loops``: the same trick for the topological sectors. It lowers the
    states with W_l1 = s1, W_l2 = s2 by 2 mu and raises or keeps the others, so DMRG on it finds
    the ground state of that sector (add 2 mu to its energy).

    ``parity = (px, pz)`` does the same for the global spin parities P_x = prod_i X_i and
    P_z = prod_i Z_i, which H also conserves (every term flips spins in pairs) and which DMRG can
    lose in the same way; with both ``sector`` and ``parity``, add 4 mu.
    """
    Kd = dict(zip("xyz", K, strict=True))
    bonds = brickwall_bonds(Lx, Ly, periodic_x, twist)
    terms = [(Kd[g] / 4, ((i, g.upper()), (j, g.upper()))) for i, j, g in bonds]
    if flux_penalty:
        for sites in plaquettes(Lx, Ly, periodic_x, twist):
            terms.append((-flux_penalty, tuple(zip(sites, HEXAGON_W, strict=True))))
    if sector is not None:
        terms += [(-sector_penalty * s, word) for s, word in zip(sector, torus_loops(Lx, Ly, twist), strict=True)]
    if parity is not None:
        n = Lx * Ly
        terms += [(-sector_penalty * p, tuple((s, a) for s in range(n))) for p, a in zip(parity, "XZ", strict=True)]
    kind = f"torus (twist {twist})" if periodic_x else "cylinder"
    return PauliSum(Lx * Ly, terms, name=f"Kitaev honeycomb, K = {tuple(K)}, {Lx} x {Ly} {kind}", shape=(Lx, Ly))


def anisotropic_couplings(g):
    """``(K_x, K_y, K_z) = (1 - g/2, 1 - g/2, 1 + g)``: the anisotropy cut of Lyu et al., Fig. 4.

    g = 0 is the isotropic gapless spin liquid; the gapped Z2 (toric-code) phase starts at g = 1/2,
    where K_z = K_x + K_y (Kitaev, Sec. 6).
    """
    return (1 - g / 2, 1 - g / 2, 1 + g)


def fork(x0, y0, Lx, Ly, twist=0):
    """Six sites with no loop, the "fork" of Lyu et al., Fig. 3b, as three consecutive pairs.

    A centre c = (x0, y0) (x0 + y0 even), its z neighbour n1 above, its two horizontal neighbours
    n2 and n3, and the z neighbours n2', n3' of those (below). Returned as (c, n1, n2, n2', n3, n3'):
    the three parties of the figure are (c, n1), (n2, n2') and (n3, n3').
    """
    if (x0 + y0) % 2:
        raise ValueError("the centre of a fork has x + y even")
    site = _site_function(Lx, Ly, twist)
    return (
        site(x0, y0),
        site(x0, y0 + 1),
        site(x0 - 1, y0),
        site(x0 - 1, y0 - 1),
        site(x0 + 1, y0),
        site(x0 + 1, y0 - 1),
    )


def pauli_string_operator(word):
    """Dense matrix of a Pauli word such as ``"XYZ"`` (first letter = most significant qubit)."""
    P = {
        "I": np.eye(2),
        "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
        "Y": np.array([[0.0, -1j], [1j, 0.0]]),
        "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
    }
    out = np.ones((1, 1))
    for c in word:
        out = np.kron(out, P[c])
    return out


# The exact hexagon correlations of the isotropic model in the thermodynamic limit, from the
# Majorana solution: Lyu et al., arXiv:2505.18124, SM Tables II, III and IV (J_x = J_y = J_z = -1 in
# their H = -sum J sigma sigma, i.e. the antiferromagnet K > 0 here). Each entry is a Pauli word on
# the hexagon sites 1..6 (in the order of ``hexagon``) and its expectation value, to four digits.
# Every other Pauli string on the hexagon has zero expectation value (flux conservation and time
# reversal, SM Sec. VIII D).
EXACT_HEXAGON_CORRELATIONS = {
    # two-point functions, Table II
    "ZZIIII": -0.5249,
    "IXXIII": -0.5249,
    "IIYYII": -0.5249,
    "IIIZZI": -0.5249,
    "IIIIXX": -0.5249,
    "YIIIIY": -0.5249,
    # four-point functions, Table III
    "ZYZYII": -0.1858,
    "ZZYYII": +0.3730,
    "IXZXZI": -0.1858,
    "IXXZZI": +0.3730,
    "IIYXYX": -0.1858,
    "IIYYXX": +0.3730,
    "YIIZYZ": -0.1858,
    "YIIZZY": +0.3730,
    "XZIIXZ": -0.1858,
    "ZZIIXX": +0.3730,
    "XYXIIY": -0.1858,
    "YXXIIY": +0.3730,
    "IXXIXX": +0.2410,
    "YIYYIY": +0.2410,
    "ZZIZZI": +0.2410,
    # six-point functions, Table IV
    "YXZXYZ": -0.5249,
    "XZYXYZ": -0.5249,
    "XYXZYZ": -0.5249,
    "XYZYXZ": -0.5249,
    "XYZXZY": -0.5249,
    "ZYZXYX": -0.5249,
    "YXXZYZ": +0.3730,
    "XZYYXZ": +0.3730,
    "XYXZZY": +0.3730,
    "ZYZYXX": +0.3730,
    "YXZXZY": +0.3730,
    "ZZYXYX": +0.3730,
    "XZYXZY": +0.2410,
    "ZYXZYX": +0.2410,
    "YXZYXZ": +0.2410,
    "ZZYYXX": -0.4363,
    "YXXZZY": -0.4363,
    "XYZXYZ": +1.0,
}


def exact_hexagon_rdm():
    """The hexagon RDM of the isotropic model in the thermodynamic limit, ``2^-6 sum_P <P> P`` from the tables."""
    rho = np.eye(64, dtype=complex)
    for word, value in EXACT_HEXAGON_CORRELATIONS.items():
        rho += value * pauli_string_operator(word)
    return rho / 64


def region_measures(rho, parties):
    """GMN of ``rho`` with the given parties (tuples of qubit positions in ``rho``), with medt.

    Two parties give the negativity. The qubits are first reordered so that each party is a
    block, then the parties are passed to medt as subsystems of dimension 2^(party size).
    """
    from medt.gmn import gmn, negativity  # optional dependency, as in microscopy.py

    from .microscopy import permute_qubits

    order = [q for p in parties for q in p]
    if sorted(order) != list(range(round(np.log2(rho.shape[0])))):
        raise ValueError("every qubit of rho must belong to exactly one party")
    r = permute_qubits(rho, order)
    dims = [2 ** len(p) for p in parties]
    if len(parties) == 2:
        return negativity(r, (0,), dims)  # for two parties GMN is the negativity, an eigenvalue problem
    # Up to 6 qubits, the interior-point solver MOSEK, as in Lyu et al. (Methods), when it is
    # installed (it needs a licence, free for academics): 2 s for three parties, about 100 s for six
    # single spins (31 bipartitions). Otherwise Clarabel, also interior-point. The first-order
    # solver SCS can stall for nearly pure states, as the RDMs of single ground states here are,
    # but above 6 qubits it is the one that fits in memory.
    import cvxpy

    if len(order) > 6:
        return gmn(r, dims, solver="SCS", eps=1e-8)
    return gmn(r, dims, solver="MOSEK" if "MOSEK" in cvxpy.installed_solvers() else "CLARABEL")
