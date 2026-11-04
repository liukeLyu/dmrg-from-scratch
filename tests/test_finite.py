import numpy as np
import pytest

from dmrg_from_scratch import central_charge_fit, dmrg, expectation, schmidt_values, tfim_mpo, tfim_open_energy
from dmrg_from_scratch.finite import apply_heff


def test_heff_is_symmetric():
    rng = np.random.default_rng(0)
    L = rng.normal(size=(3, 3, 3))
    L = L + L.transpose(2, 1, 0)  # environments are Hermitian in (ket, bra)
    R = rng.normal(size=(4, 3, 4))
    R = R + R.transpose(2, 1, 0)
    W = tfim_mpo(4, 0.7)[1]
    n = 3 * 2 * 2 * 4
    H = np.column_stack([apply_heff(L, W, W, R, e.reshape(3, 2, 2, 4)).ravel() for e in np.eye(n)])
    assert np.allclose(H, H.T)


@pytest.mark.parametrize("h", [0.5, 1.0, 1.5])
def test_dmrg_energy_is_exact(h):
    L = 20
    res = dmrg(tfim_mpo(L, h), chi_max=16, n_sweeps=3)
    assert abs(res["energy"] - tfim_open_energy(L, h)) < 1e-9
    assert np.isclose(expectation(res["mps"], tfim_mpo(L, h)), res["energy"])


def test_entropies_match_schmidt_values_of_final_mps():
    res = dmrg(tfim_mpo(12, 1.0), chi_max=16, n_sweeps=3)
    S = [-(s**2 * np.log2(s**2)).sum() for s in schmidt_values(res["mps"])]
    assert np.allclose(S, res["entropies"], atol=1e-8)


def test_central_charge_of_critical_chain():
    L = 40
    res = dmrg(tfim_mpo(L, 1.0), chi_max=24, n_sweeps=3)
    c, _ = central_charge_fit(L, res["entropies"])
    assert abs(c - 0.5) < 0.1
