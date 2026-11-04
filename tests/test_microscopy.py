import numpy as np
import pytest

import dmrg_from_scratch as dmrg
from dmrg_from_scratch import microscopy as mic


@pytest.mark.parametrize("sites", [(2, 3), (1, 4), (5, 2, 7), (0, 1, 3, 4), (6, 2, 8, 3)])
def test_segment_rdm_is_the_partial_trace(sites):
    rng = np.random.default_rng(0)
    n = 9
    psi = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    mps = dmrg.right_canonicalize(dmrg.from_dense(psi, n))
    assert np.allclose(mic.segment_rdm(mps, sites), mic.dense_rdm(psi, sites, n), atol=1e-13)


def test_segment_with_schmidt_values_on_the_left():
    # a right-canonical chain cut at bond k: start at site k with diag(S^2) on the left
    rng = np.random.default_rng(1)
    n = 8
    psi = rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    mps = dmrg.right_canonicalize(dmrg.from_dense(psi, n))
    S = dmrg.schmidt_values(mps)[2]  # cut between sites 2 and 3
    rho = mic.segment_rdm(mps[3:], [0, 2], left=np.diag(S**2))
    assert np.allclose(rho, mic.dense_rdm(psi, (3, 5), n), atol=1e-13)


def test_tomography_round_trip_and_basis_change():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = A @ A.conj().T
    rho /= np.trace(rho)
    assert np.allclose(mic.rdm_from_paulis(mic.pauli_expectations(rho)), rho)
    # the Hadamard rotation swaps <XXI> and <ZZI>
    e, f = mic.pauli_expectations(rho), mic.pauli_expectations(mic.to_zz_basis(rho))
    assert np.isclose(e["XXI"], f["ZZI"]) and np.isclose(e["ZIZ"], f["XIX"]) and np.isclose(e["YII"], -f["YII"])


def test_symmetrize_averages_the_two_broken_states():
    # for -XX - hZ at h = 0 the two ordered states are |++> and |-->; prod Z maps one to the other
    plus = np.full(4, 0.5)
    minus = np.array([0.5, -0.5, -0.5, 0.5])
    rho = mic.symmetrize(np.outer(plus, plus))
    assert np.allclose(rho, (np.outer(plus, plus) + np.outer(minus, minus)) / 2)


def test_permute_qubits():
    a, b, c = (np.diag([1.0, 0]), np.diag([0.3, 0.7]), np.array([[0.5, 0.5], [0.5, 0.5]]))
    rho = np.kron(np.kron(a, b), c)
    assert np.allclose(mic.permute_qubits(rho, [2, 0, 1]), np.kron(np.kron(c, a), b))


def test_partial_trace_matches_the_state_vector():
    rng = np.random.default_rng(3)
    n = 5
    psi = rng.normal(size=2**n)
    psi /= np.linalg.norm(psi)
    rho = mic.dense_rdm(psi, (0, 1, 2, 3, 4), n)
    for keep in [(1, 3), (4, 0), (2, 0, 3)]:
        assert np.allclose(mic.partial_trace(rho, keep), mic.dense_rdm(psi, keep, n))


def test_plaquette_measures_of_a_product_and_a_ghz_state():
    pytest.importorskip("medt")
    prod = np.zeros((16, 16))
    prod[0, 0] = 1
    m = mic.plaquette_measures(prod)
    assert all(abs(v) < 1e-6 for v in m.values())
    ghz = np.zeros(16)
    ghz[0] = ghz[15] = 1 / np.sqrt(2)
    m = mic.plaquette_measures(np.outer(ghz, ghz))
    assert np.isclose(m["gmn_plaquette"], 0.5, atol=1e-5)  # equal to its negativity across every cut, 1/2
    assert abs(m["N_nn_y"]) < 1e-9  # but no pair is entangled: tracing out two qubits leaves a classical mixture
