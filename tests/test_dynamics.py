"""Excited states, time evolution and temperature, checked against dense ED and free fermions."""

import numpy as np
from scipy.linalg import expm

import dmrg_from_scratch as dmrg

L, H_FIELD = 8, 1.3


def dense_H(L=L, h=H_FIELD):
    return dmrg.mpo_to_dense(dmrg.tfim_mpo(L, h))


def z_on(j, L=L):
    ops = [np.eye(2)] * L
    ops[j] = np.diag([1.0, -1.0])
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def test_single_particle_energies_give_ground_energy_and_gap():
    lam = dmrg.tfim_single_particle_energies(L, H_FIELD)
    E = np.linalg.eigvalsh(dense_H())
    assert np.isclose(-lam.sum() / 2, E[0])
    assert np.isclose(dmrg.tfim_open_gap(L, H_FIELD), E[1] - E[0])


def test_thermal_matches_dense():
    E = np.linalg.eigvalsh(dense_H())
    for T in [0.3, 1.0, 4.0]:
        w = np.exp(-(E - E[0]) / T)
        Eth = (E * w).sum() / w.sum()
        C = ((E**2 * w).sum() / w.sum() - Eth**2) / T**2
        r = dmrg.tfim_thermal(L, H_FIELD, T)
        assert np.isclose(r["energy"][0], Eth) and np.isclose(r["specific_heat"][0], C)
        assert np.isclose(r["free_energy"][0], E[0] - T * np.log(w.sum()))


def test_quench_matches_dense():
    psi0 = np.zeros(2**L)
    psi0[0] = 1.0  # all up
    H = dense_H()
    ex = dmrg.tfim_quench(L, H_FIELD, [0.7])
    psi = expm(-1j * 0.7 * H) @ psi0
    z = [np.real(psi.conj() @ z_on(j) @ psi) for j in range(L)]
    assert np.allclose(ex["z"][0], z)
    s = np.linalg.svd(psi.reshape(2 ** (L // 2), -1), compute_uv=False)
    assert np.isclose(ex["entropy"][0], dmrg.entropy(s))


def test_zz_dynamics_matches_dense():
    H = dense_H()
    E, V = np.linalg.eigh(H)
    g = V[:, 0]
    c, t = 3, 0.9
    U = expm(-1j * t * H)
    Zc = z_on(c)
    exact = [np.exp(1j * E[0] * t) * (g @ z_on(j) @ U @ Zc @ g) - (g @ z_on(j) @ g) * (g @ Zc @ g) for j in range(L)]
    assert np.allclose(dmrg.tfim_zz_dynamics(L, H_FIELD, c, [t])[0], exact)


def test_tebd_quench_matches_free_fermions():
    up = np.array([1.0, 0.0])
    mps = [A.astype(complex) for A in dmrg.product_mps([up] * 16)]
    dt = 0.025
    half = dmrg.gates(dmrg.tfim_bond_hamiltonians(16, H_FIELD), dt / 2)
    for _ in range(80):  # t = 2
        dmrg.tebd_step(mps, half, chi_max=64)
    ex = dmrg.tfim_quench(16, H_FIELD, [2.0])
    z = dmrg.local_expectations(mps, np.diag([1.0, -1.0]))
    assert np.max(np.abs(z - ex["z"][0])) < 2e-4  # Trotter error, 1.3 dt^2
    assert abs(dmrg.half_chain_entropy(mps) - ex["entropy"][0]) < 1e-3


def test_excited_state_gap():
    mpo = dmrg.tfim_mpo(16, 1.5)
    g = dmrg.dmrg(mpo, chi_max=32, n_sweeps=4)
    e = dmrg.dmrg_excited(mpo, chi_max=32, lower=[g["mps"]], n_sweeps=6)
    assert abs(e["energy"] - g["energy"] - dmrg.tfim_open_gap(16, 1.5)) < 1e-7
    assert e["overlaps"][0] < 1e-6


def test_purification_energy():
    Lp = 10
    mpo = dmrg.purify_mpo(dmrg.tfim_mpo(Lp, H_FIELD))
    betas = [0.5, 2.0]
    ex = dmrg.tfim_thermal(Lp, H_FIELD, 1 / np.array(betas))["energy"]
    for (beta, mps, _), e in zip(dmrg.thermal_states(Lp, H_FIELD, betas, chi_max=64, dbeta=0.025), ex, strict=True):
        assert abs(dmrg.mpo_expectation(mps, mpo) - e) < 2e-3 * Lp, beta
