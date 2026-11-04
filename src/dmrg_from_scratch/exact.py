"""Exact reference values: the TFIM chain from free fermions, and any small PauliSum by exact diagonalization.

The Jordan-Wigner transformation turns H = -sum X_i X_{i+1} - h sum Z_i into a
quadratic fermion Hamiltonian (Lieb, Schultz, Mattis 1961; Pfeuty 1970). For
the open chain its single-particle energies are 2 s_n, where s_n are the
singular values of the L x L bidiagonal matrix M with M_ii = h and
M_{i,i+1} = J, and the ground energy is E_0 = -sum_n s_n.
"""

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigsh

from .models import PauliSum


def tfim_open_energy(L, h, J=1.0):
    """Exact ground-state energy of the open chain of L sites."""
    M = h * np.eye(L) + J * np.eye(L, k=1)
    return -float(np.sum(np.linalg.svd(M, compute_uv=False)))


def tfim_infinite_energy_density(h, J=1.0):
    """Exact ground-state energy per site of the infinite chain.

    e_0 = -(1/pi) int_0^pi sqrt(J^2 + h^2 - 2 J h cos k) dk, equal to -4/pi at h = J = 1.
    """
    val, _ = quad(lambda k: np.sqrt(J**2 + h**2 - 2 * J * h * np.cos(k)), 0, np.pi, limit=200)
    return -val / np.pi


def central_charge_fit(L, S):
    """Fit S(l) = (c/6) log2[(2L/pi) sin(pi l/L)] + const for an open critical chain.

    ``S[l-1]`` is the entropy of the first l sites. Calabrese and Cardy,
    https://arxiv.org/abs/hep-th/0405152. Returns ``(c, const)``.
    Odd and even l oscillate a little in the open chain, so we fit both together.
    """
    l = np.arange(1, L)
    x = np.log2(2 * L / np.pi * np.sin(np.pi * l / L))
    slope, const = np.polyfit(x, S, 1)
    return 6 * slope, const


# Time evolution and temperature, from Majorana fermions.
#
# With the Jordan-Wigner strings S_i = Z_0 ... Z_{i-1}, the 2L Majorana operators
#     a_{2i} = S_i X_i,   a_{2i+1} = S_i Y_i      ({a_m, a_n} = 2 delta_mn)
# give Z_i = -i a_{2i} a_{2i+1} and X_i X_{i+1} = -i a_{2i+1} a_{2i+2}, so
#     H = (i/4) sum_mn M_mn a_m a_n,   M real antisymmetric, M_{2i,2i+1} = 2h, M_{2i+1,2i+2} = 2J.
# Three facts follow by direct calculation and are all we need:
#   1. Heisenberg picture: d a / dt = i [H, a] = M a, so a(t) = R(t) a with R = exp(M t), a rotation.
#   2. The eigenvalues of the Hermitian matrix iM come in pairs +-Lambda_n; the Lambda_n are the
#      single-particle energies, H = sum_n Lambda_n (n_n - 1/2) (Lambda_n = 2 s_n above).
#   3. The states below (ground, thermal, product) are Gaussian: they are fixed by the correlation
#      matrix G_mn = <a_m a_n>, and higher correlators follow from Wick's theorem. With
#      iM = U diag(eps) U^dagger, a thermal state has G = 2 conj(U diag(f(eps)) U^dagger), with f
#      the Fermi function (a step at T = 0).
# References: Lieb, Schultz and Mattis, Ann. Phys. 16, 407 (1961), https://doi.org/10.1016/0003-4916(61)90115-4;
# Pfeuty, Ann. Phys. 57, 79 (1970); block entropy from G: Vidal, Latorre, Rico and Kitaev,
# https://arxiv.org/abs/quant-ph/0211074. tests/test_dynamics.py checks all of it against dense ED.


def tfim_majorana_matrix(L, h, J=1.0):
    """The 2L x 2L real antisymmetric M with H = (i/4) a^T M a, for the open chain."""
    M = np.zeros((2 * L, 2 * L))
    for i in range(L):
        M[2 * i, 2 * i + 1] = 2 * h
        if i < L - 1:
            M[2 * i + 1, 2 * i + 2] = 2 * J
    return M - M.T


def tfim_single_particle_energies(L, h, J=1.0):
    """The L single-particle energies Lambda_n >= 0, smallest first."""
    eps = np.linalg.eigvalsh(1j * tfim_majorana_matrix(L, h, J))
    return np.sort(eps[eps.size // 2 :])


def tfim_open_gap(L, h, J=1.0):
    """E_1 - E_0 of the open chain: one fermion in the lowest mode (exponentially small for h < J)."""
    return float(tfim_single_particle_energies(L, h, J)[0])


def tfim_thermal(L, h, T, J=1.0):
    """Energy, specific heat and free energy of the open chain at temperature(s) ``T`` (k_B = 1).

    With H = sum_n Lambda_n (n_n - 1/2) and x_n = Lambda_n / 2T:
        E = -sum_n (Lambda_n/2) tanh(x_n),   C = dE/dT = sum_n x_n^2 / cosh^2(x_n),
        F = -T sum_n log(2 cosh(x_n)).
    Returns a dict of arrays shaped like ``T``.
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    lam = tfim_single_particle_energies(L, h, J)
    x = lam[None, :] / (2 * T[:, None])
    E = -np.sum(lam / 2 * np.tanh(x), axis=1)
    C = np.sum(x**2 / np.cosh(x) ** 2, axis=1)
    F = -T * np.sum(np.logaddexp(x, -x), axis=1)
    return {"energy": E, "specific_heat": C, "free_energy": F}


def tfim_ground_correlations(L, h, J=1.0):
    """Majorana correlation matrix G_mn = <a_m a_n> of the ground state of the open chain."""
    eps, U = np.linalg.eigh(1j * tfim_majorana_matrix(L, h, J))
    filled = (eps < 0).astype(float)
    return 2 * np.conj((U * filled) @ U.conj().T)


def majorana_entropy(G, l):
    """Entanglement entropy (bits) of the first l sites of a Gaussian state with correlations G.

    Gamma = -i (G - 1) on the first 2l Majoranas is real antisymmetric with eigenvalues +-i nu_k;
    each pair contributes the binary entropy of (1 + nu_k) / 2.
    """
    Gamma = (-1j * (G - np.eye(len(G))))[: 2 * l, : 2 * l]
    nu = np.linalg.eigvalsh(1j * Gamma)
    p = np.clip((1 + nu[nu > 0]) / 2, 1e-15, 1 - 1e-15)
    return float(-np.sum(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def tfim_quench(L, h, times, J=1.0):
    """Quench from the all-up state |0...0> (every Z_i = +1) to the TFIM at field h.

    Returns ``z[t, i] = <Z_i(t)>`` and the half-chain entropy ``entropy[t]`` in bits.
    The initial state has <a_{2i} a_{2i+1}> = <X_i Y_i> = i and nothing else off the diagonal.
    """
    G0 = np.eye(2 * L, dtype=complex)
    for i in range(L):
        G0[2 * i, 2 * i + 1], G0[2 * i + 1, 2 * i] = 1j, -1j
    M = tfim_majorana_matrix(L, h, J)
    z, S = [], []
    for t in times:
        R = expm(M * t)
        G = R @ G0 @ R.T
        z.append(np.real(-1j * np.diagonal(G, 1)[::2]))
        S.append(majorana_entropy(G, L // 2))
    return {"z": np.array(z), "entropy": np.array(S)}


def tfim_zz_dynamics(L, h, c, times, J=1.0):
    """Connected ground-state correlation C[t, j] = <Z_j(t) Z_c(0)> - <Z_j><Z_c>, for every site j.

    Wick's theorem with A, B = a_{2j}(t), a_{2j+1}(t) and C, D = a_{2c}, a_{2c+1}: <Z_j(t) Z_c> = -<ABCD>,
    and its connected part is <AC><BD> - <AD><BC>, with <a_m(t) a_n> = (R(t) G)_mn.
    """
    G = tfim_ground_correlations(L, h, J)
    M = tfim_majorana_matrix(L, h, J)
    out = np.zeros((len(times), L), dtype=complex)
    for k, t in enumerate(times):
        RG = expm(M * t) @ G
        A, B = 2 * np.arange(L), 2 * np.arange(L) + 1
        out[k] = RG[A, 2 * c] * RG[B, 2 * c + 1] - RG[A, 2 * c + 1] * RG[B, 2 * c]
    return out


# Any PauliSum (the cylinders of models.py): exact ground states by Lanczos on the full
# Hilbert space, up to about 24 spins.
#
# Bit order: site 0 is the most significant bit, as in PennyLane and as in ``mps.to_dense``.
# A Pauli string acts on a basis state as ``P|b> = i^{n_Y} (-1)^{popcount(b & m_YZ)} |b ^ m_XY>``,
# so each term is a bit flip plus a sign, applied to the whole vector at once.


def _parity(k: np.ndarray, mask: int, n: int) -> np.ndarray:
    out = np.zeros(k.shape, dtype=np.int64)
    for b in range(n):
        if mask >> b & 1:
            out ^= (k >> b) & 1
    return out


def _compile(H: PauliSum):
    n = H.n
    k = np.arange(2**n, dtype=np.int64)
    compiled = []
    for c, ops in H.terms:
        flip = sign_mask = 0
        n_y = 0
        for site, p in ops:
            bit = 1 << (n - 1 - site)
            if p in "XY":
                flip |= bit
            if p in "YZ":
                sign_mask |= bit
            n_y += p == "Y"
        if n_y % 2:
            raise ValueError("odd number of Y: the term is not real")
        phase = c * (-1) ** (n_y // 2)
        signs = phase * (1 - 2 * _parity(k ^ flip, sign_mask, n))
        compiled.append((flip, signs))
    return k, compiled


def matvec_operator(H: PauliSum) -> LinearOperator:
    """``H`` as a scipy LinearOperator on the ``2^n`` dimensional space."""
    k, compiled = _compile(H)

    def matvec(v):
        v = np.ravel(v)
        out = np.zeros_like(v)
        for flip, signs in compiled:
            out += signs * v[k ^ flip]
        return out

    dim = 2**H.n
    return LinearOperator((dim, dim), matvec=matvec, dtype=float)


def ground_state(H: PauliSum, seed: int = 0) -> tuple[float, np.ndarray]:
    """Ground energy and state of ``H`` (normalized, real)."""
    if H.n > 24:
        raise ValueError("exact diagonalization is limited to about 24 spins")
    v0 = np.random.default_rng(seed).standard_normal(2**H.n)
    vals, vecs = eigsh(matvec_operator(H), k=1, which="SA", v0=v0, tol=1e-12)
    return float(vals[0]), vecs[:, 0]
