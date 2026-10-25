import numpy as np
import pytest

from dmrg import correlation_length, idmrg, tfim_bulk_tensor, tfim_infinite_energy_density


def test_exact_energy_density_at_critical_point():
    assert np.isclose(tfim_infinite_energy_density(1.0), -4 / np.pi)


@pytest.mark.parametrize("h", [0.5, 1.5, 2.0])
def test_idmrg_energy_in_gapped_phases(h):
    res = idmrg(tfim_bulk_tensor(h), chi_max=16, n_steps=40)
    assert abs(res["energy"] - tfim_infinite_energy_density(h)) < 1e-10


@pytest.mark.parametrize("h", [1.5, 2.0])
def test_correlation_length_in_paramagnet(h):
    # For h > 1 the X-X correlations decay as h^(-r), so xi = 1/log h.
    res = idmrg(tfim_bulk_tensor(h), chi_max=16, n_steps=40)
    assert abs(correlation_length(res) - 1 / np.log(h)) < 0.05 / np.log(h)


def test_ordered_phase_gives_cat_state():
    # For h < 1 the Z2-symmetric ground state is a cat (|up..> + |down..>) / sqrt(2) dressed by
    # fluctuations: its entropy is about 1 bit and the transfer matrix has a second eigenvalue 1.
    res = idmrg(tfim_bulk_tensor(0.5), chi_max=16, n_steps=40)
    assert abs(res["entropies"][-1] - 1) < 0.01
    assert correlation_length(res) > 1e4


def test_idmrg_critical_point():
    res = idmrg(tfim_bulk_tensor(1.0), chi_max=8, n_steps=300)
    assert abs(res["energy"] - tfim_infinite_energy_density(1.0)) < 1e-4


def test_wavefunction_prediction_gives_same_answer():
    a = idmrg(tfim_bulk_tensor(0.8), chi_max=10, n_steps=30, predict=True)
    b = idmrg(tfim_bulk_tensor(0.8), chi_max=10, n_steps=30, predict=False)
    assert np.isclose(a["energy"], b["energy"], atol=1e-10)


@pytest.mark.slow
def test_central_charge_from_finite_entanglement_scaling():
    S, xi = [], []
    for chi in [6, 8, 10, 12]:
        res = idmrg(tfim_bulk_tensor(1.0), chi_max=chi, n_steps=1000)
        S.append(res["entropies"][-1])
        xi.append(correlation_length(res))
    c = 6 * np.polyfit(np.log2(xi), S, 1)[0]
    assert abs(c - 0.5) < 0.05
