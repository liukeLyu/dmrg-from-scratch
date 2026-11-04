"""Time evolution of an MPS by TEBD, in real time (dynamics) and imaginary time (temperature).

The idea (Vidal https://arxiv.org/abs/quant-ph/0310089; White and Feiguin
https://arxiv.org/abs/cond-mat/0403310): write H = sum_b h_b as a sum of two-site
terms on the bonds b = (b, b+1). Each exp(-i dt h_b) is a 4x4 gate. Applying the
gates bond by bond is the same move as a DMRG sweep, with "solve for the lowest
eigenvector" replaced by "multiply by the gate":

    contract sites b, b+1 into theta -> apply the gate -> truncated SVD -> move on.

One step sweeps left to right with dt/2 and back with dt/2. The product of gates is
then symmetric, U_0 U_1 ... U_{L-2} U_{L-2} ... U_1 U_0, so the Trotter error per
unit time is O(dt^2) (Paeckel et al. https://arxiv.org/abs/1901.05824, Sec. 3).
The truncated SVD is the only approximation besides Trotter: entanglement grows
with time, and once it needs more than log2(chi_max) bits the discarded weight grows.

Finite temperature uses the same code. A purification is a pure state on the chain
plus one ancilla per site (local dimension 4) whose partial trace over the ancillas
is rho = exp(-beta H) / Z (Verstraete, Garcia-Ripoll and Cirac
https://arxiv.org/abs/cond-mat/0406426; Feiguin and White https://arxiv.org/abs/cond-mat/0510124).
At beta = 0 every spin is maximally entangled with its own ancilla, a product state
of pairs; imaginary-time gates exp(-tau h_b) on the physical spins then cool it.

Conventions as in finite.py: A[left, phys, right]; a gate g[out1, out2, in1, in2].
During a step the canonical centre starts and ends on site 0.
"""

import numpy as np
from scipy.linalg import expm

from .finite import grow_left
from .mpo import I2, X, Z
from .mps import entropy, truncated_svd


def tfim_bond_hamiltonians(L, h, J=1.0):
    """The TFIM as L - 1 two-site terms h_b[out1, out2, in1, in2] that sum to H.

    Each field term -h Z_i is shared by the two bonds touching site i; the end sites
    have only one bond, which takes their whole field.
    """
    out = []
    for b in range(L - 1):
        hl = h if b == 0 else h / 2
        hr = h if b == L - 2 else h / 2
        hb = -J * np.kron(X, X) - hl * np.kron(Z, I2) - hr * np.kron(I2, Z)
        out.append(hb.reshape(2, 2, 2, 2))
    return out


def gates(bond_hams, dt, imaginary=False):
    """exp(-i dt h_b) for every bond, or exp(-dt h_b) with ``imaginary=True``."""
    out = []
    for hb in bond_hams:
        d = hb.shape[0]
        H = hb.reshape(d * d, d * d)
        g = expm(-dt * H) if imaginary else expm(-1j * dt * H)
        out.append(g.reshape(d, d, d, d))
    return out


def _apply(mps, i, g, chi_max, cutoff, centre_right):
    theta = np.einsum("asb,btc->astc", mps[i], mps[i + 1], optimize=True)
    theta = np.einsum("stuv,auvc->astc", g, theta, optimize=True)  # gate on the two physical legs
    chi_l, d1, d2, chi_r = theta.shape
    U, S, Vh, disc = truncated_svd(theta.reshape(chi_l * d1, d2 * chi_r), chi_max, cutoff)
    if centre_right:  # sweeping right: site i becomes a left isometry
        mps[i] = U.reshape(chi_l, d1, -1)
        mps[i + 1] = (S[:, None] * Vh).reshape(-1, d2, chi_r)
    else:  # sweeping left: site i + 1 becomes a right isometry
        mps[i] = (U * S[None, :]).reshape(chi_l, d1, -1)
        mps[i + 1] = Vh.reshape(-1, d2, chi_r)
    return disc


def tebd_step(mps, half_gates, chi_max, cutoff=1e-12):
    """One second-order step, in place: ``half_gates`` (for dt/2) left to right, then right to left.

    ``mps`` must have its canonical centre on site 0 (e.g. right-canonical); it does again
    afterwards, normalized. Returns the largest discarded weight of the step.
    """
    disc = 0.0
    for i in range(len(mps) - 1):
        disc = max(disc, _apply(mps, i, half_gates[i], chi_max, cutoff, centre_right=True))
    for i in range(len(mps) - 2, -1, -1):
        disc = max(disc, _apply(mps, i, half_gates[i], chi_max, cutoff, centre_right=False))
    return disc


def half_chain_entropy(mps):
    """Entropy (bits) of the left half, for an MPS with its centre on site 0.

    Move the centre to the middle by SVDs; the Schmidt values appear on the way.
    """
    mps = [A.copy() for A in mps]
    for i in range(len(mps) // 2):
        chi_l, d, chi_r = mps[i].shape
        U, S, Vh, _ = truncated_svd(mps[i].reshape(chi_l * d, chi_r), chi_l * d, cutoff=0)
        mps[i + 1] = np.einsum("a,ab,bsc->asc", S, Vh, mps[i + 1])
    return entropy(S)


def transition_elements(bra, ket, op):
    """<bra| op_j |ket> for every site j, in one sweep (op[out, in] acts on one site)."""
    L = len(ket)
    R = [None] * (L + 1)
    R[L] = np.ones((1, 1))
    for j in range(L - 1, 0, -1):  # R[j][x, y]: sites j..L-1, x = ket bond, y = bra bond
        R[j] = np.einsum("xsb,ysb->xy", np.einsum("xsa,ab->xsb", ket[j], R[j + 1]), bra[j].conj())
    E = np.ones((1, 1))
    out = np.zeros(L, dtype=complex)
    for j in range(L):
        T = np.einsum("xy,xsa->ysa", E, ket[j])  # the left environment on the ket tensor
        out[j] = np.einsum("ytb,ytb->", np.einsum("ts,ysa,ab->ytb", op, T, R[j + 1], optimize=True), bra[j].conj())
        E = np.einsum("ysa,ysb->ab", T, bra[j].conj())
    return out


def mpo_expectation(mps, mpo):
    """<psi|H|psi> / <psi|psi>, one tensor at a time (cost chi^3 per site, like the DMRG environments)."""
    E, N = np.ones((1, 1, 1)), np.ones((1, 1))
    for A, W in zip(mps, mpo, strict=True):
        E = grow_left(E, A, W)
        N = np.einsum("ysa,ysb->ab", np.einsum("xy,xsa->ysa", N, A), A.conj())
    return float(np.real(E[0, 0, 0] / N[0, 0]))


def local_expectations(mps, op):
    """<op_j> for every site j of a (not necessarily normalized) MPS."""
    norm = np.real(transition_elements(mps, mps, np.eye(op.shape[0]))[0])
    return np.real(transition_elements(mps, mps, op)) / norm


def apply_local(mps, op, j):
    """A copy of the MPS with the one-site operator op[out, in] applied on site j."""
    out = [A.copy() for A in mps]
    out[j] = np.einsum("ts,asb->atb", op, mps[j])
    return out


def dynamical_correlations(psi0, op, c, half_gates, n_steps, chi_max, every=1):
    """Connected C[t, j] = <op_j(t) op_c(0)> - <op_j><op_c> in the state psi0 (an eigenstate of H).

    <op_j(t) op_c> = <psi0| e^{iHt} op_j e^{-iHt} op_c |psi0>: evolve phi = op_c |psi0> and psi0
    itself with the same gates, and take <psi0(t)| op_j |phi(t)> for every j at once. Evolving
    psi0 too (rather than multiplying by e^{i E0 t}) cancels the Trotter error of its phase.
    Returns the times (in units of steps; multiply by dt) and C, measured every ``every`` steps.
    """
    psi = [A.astype(complex) for A in psi0]
    phi = apply_local(psi, op, c)
    mean = local_expectations(psi, op)
    steps, C = [], []
    for k in range(n_steps + 1):
        if k % every == 0:
            steps.append(k)
            C.append(transition_elements(psi, phi, op) - mean * mean[c])
        if k < n_steps:
            tebd_step(psi, half_gates, chi_max)
            tebd_step(phi, half_gates, chi_max)
    return np.array(steps), np.array(C)


def structure_factor(C, dt, c, ks, omegas, width=None):
    """S(k, w) = sum_r e^{-ikr} int dt e^{iwt} C(r, t), from C[n, j] at times t = n dt >= 0, r = j - c.

    For an eigenstate, C(r, -t) = C(-r, t)^* when the chain looks the same on both sides of c,
    so the negative times add the complex conjugate: S = 2 Re sum_r e^{-ikr} int_0^tmax. A
    Gaussian window exp(-t^2 / 2 width^2) (width = tmax / 3 by default) hides the cut at tmax;
    it broadens every line to a Gaussian of width about 1 / width in frequency.
    """
    t = dt * np.arange(C.shape[0])
    r = np.arange(C.shape[1]) - c
    width = width or t[-1] / 3
    window = np.exp(-0.5 * (t / width) ** 2)
    window[0] *= 0.5  # trapezoid rule at t = 0
    Ck = (C * window[:, None]) @ np.exp(-1j * np.outer(r, ks))  # [t, k]
    return 2 * dt * np.real(np.exp(1j * np.outer(omegas, t)) @ Ck).T  # [k, w]


# Purification: site index s = 2 p + a combines the physical spin p and its ancilla a.


def infinite_temperature_state(L):
    """beta = 0: each spin maximally entangled with its ancilla, (|00> + |11>) / sqrt 2 on every site."""
    pair = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    return [pair.reshape(1, 4, 1) for _ in range(L)]


def purify_bond_hamiltonians(bond_hams):
    """h_b acting on the physical spins only: h_b (x) 1_ancilla (x) 1_ancilla, as a 4-dim-site gate."""
    out = []
    for hb in bond_hams:
        H = np.einsum("pqrs,ab,cd->paqcrbsd", hb, I2, I2)  # [p1 a1 p2 a2 (out), r1 b1 s2 d2 (in)]
        out.append(H.reshape(4, 4, 4, 4))
    return out


def purify_mpo(mpo):
    """An MPO acting on the physical spins of a purification (identity on the ancillas)."""
    return [np.einsum("wvts,ab->wvtasb", W, I2).reshape(W.shape[0], W.shape[1], 4, 4) for W in mpo]


def thermal_states(L, h, betas, chi_max, dbeta=0.05, J=1.0, cutoff=1e-12):
    """Purified thermal states of the open TFIM, cooled from beta = 0 by imaginary-time TEBD.

    rho(beta) needs exp(-beta H / 2) on the purification, so each step of imaginary time
    tau = dbeta / 2 lowers the temperature by dbeta. ``betas`` must be multiples of dbeta.
    Yields ``(beta, mps, discarded)`` at each requested beta.
    """
    half = gates(purify_bond_hamiltonians(tfim_bond_hamiltonians(L, h, J)), dbeta / 4, imaginary=True)
    mps = infinite_temperature_state(L)
    beta, disc = 0.0, 0.0
    for target in betas:
        while beta < target - 1e-9:
            disc = max(disc, tebd_step(mps, half, chi_max, cutoff))
            beta += dbeta
        yield target, [A.copy() for A in mps], disc
