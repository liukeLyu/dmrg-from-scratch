import numpy as np
import pytest

from dmrg import expectation, mpo_to_dense, random_mps, tfim_mpo, tfim_open_energy, to_dense

X = np.array([[0, 1], [1, 0]])
Z = np.diag([1, -1])


def dense_tfim(L, h):
    """Build H = -sum X X - h sum Z with Kronecker products, the slow obvious way."""

    def op(o, i):
        return np.kron(np.kron(np.eye(2**i), o), np.eye(2 ** (L - i - 1)))

    H = -sum(op(X, i) @ op(X, i + 1) for i in range(L - 1))
    return H - h * sum(op(Z, i) for i in range(L))


@pytest.mark.parametrize("L,h", [(2, 0.3), (5, 1.0), (7, 1.7)])
def test_mpo_equals_kronecker_hamiltonian(L, h):
    assert np.allclose(mpo_to_dense(tfim_mpo(L, h)), dense_tfim(L, h))


@pytest.mark.parametrize("L,h", [(3, 0.5), (8, 1.0), (10, 2.0)])
def test_free_fermion_energy_matches_exact_diagonalization(L, h):
    assert np.isclose(np.linalg.eigvalsh(dense_tfim(L, h))[0], tfim_open_energy(L, h))


def test_free_fermion_energy_matches_medt():
    medt_tfim = pytest.importorskip("medt.tfim")
    E, _ = medt_tfim.tfim_ground_state(12, 0.8, periodic=False)
    assert np.isclose(E, tfim_open_energy(12, 0.8))


def test_expectation_value():
    mps, H = random_mps(6, seed=0), mpo_to_dense(tfim_mpo(6, 0.9))
    psi = to_dense(mps)
    assert np.isclose(expectation(mps, tfim_mpo(6, 0.9)), psi @ H @ psi / (psi @ psi))
