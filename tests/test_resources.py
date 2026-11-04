import numpy as np

from dmrg_from_scratch.models import j1j2_cylinder
from dmrg_from_scratch.resources import chi_needed, dmrg_seconds, error_fit, qpe_cost, reference_energy


def fake_runs(E0=-10.0, A=2.0, b=1.5):
    chis = [8, 16, 32, 64]
    return [
        {"chi": c, "energy": E0 + A * c**-b, "discarded": 0.1 * A * c**-b, "flops_per_sweep": 1e3 * c**3} for c in chis
    ]


def test_reference_energy_is_linear_extrapolation():
    assert np.isclose(reference_energy(fake_runs()), -10.0)


def test_power_law_fit_and_inversion():
    runs = fake_runs()
    A, b = error_fit(runs, -10.0)
    assert np.isclose(A, 2.0) and np.isclose(b, 1.5)
    assert np.isclose(chi_needed(runs, 2.0 * 128**-1.5, E_ref=-10.0), 128)


def test_dmrg_seconds_scale_as_chi_cubed():
    runs = fake_runs()
    assert np.isclose(dmrg_seconds(runs, 128, n_sweeps=1, flops_per_second=1.0) / (1e3 * 64**3), 8)


def test_qpe_cost_scales():
    H = j1j2_cylinder(4)
    c = qpe_cost(H, eps=0.01)
    assert c["walk_calls"] == int(np.ceil(np.pi * H.one_norm / 0.02))
    assert c["toffoli"] > c["walk_calls"]
    half = qpe_cost(H, eps=0.01, overlap=0.5)
    assert np.isclose(half["seconds"] / c["seconds"], 4)
    assert qpe_cost(H, eps=0.001)["walk_calls"] > 9 * c["walk_calls"]


def test_crossover_from_saved_runs():
    from dmrg_from_scratch.resources import classical_cost, crossover, load_ladders

    ladders = load_ladders()
    chis = [classical_cost(lad)["chi"] for lad in ladders if lad["W"] in (4, 6, 8)]
    assert chis[0] < chis[1] < chis[2]
    r = crossover(ladders)
    assert r["kappa"] > 0
    assert r["W_star"] is not None and 6 <= r["W_star"] <= 12
    assert np.all(np.diff(np.log(r["classical"])) > 0)
