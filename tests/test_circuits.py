import numpy as np
import pennylane as qml
import pytest

from dmrg_from_scratch import ground_state, normalize, random_mps, tfim_cylinder, to_dense
from dmrg_from_scratch.circuits import (
    apply_staircase,
    disentangling_layers,
    layers_circuit,
    mps_circuit,
    prepare,
    site_unitaries,
    staircase,
    to_pennylane_mps,
)


def unit(psi):
    return psi / np.linalg.norm(psi)


@pytest.mark.parametrize("chi", [1, 2, 3, 4])
def test_sequential_circuit_prepares_the_mps(chi):
    mps = random_mps(6, chi=chi, seed=chi)
    unitaries, k = site_unitaries(mps)
    assert all(np.allclose(U.conj().T @ U, np.eye(len(U))) for U in unitaries)
    dev = qml.device("default.qubit", wires=k + 6)

    @qml.qnode(dev)
    def circuit():
        mps_circuit(mps, wires=range(k, k + 6), work_wires=range(k))
        return qml.state()

    state = circuit()
    assert np.isclose(abs(np.vdot(unit(to_dense(mps)), state[: 2**6])), 1)  # ancilla back in |0>


def test_matches_pennylane_mpsprep():
    mps = random_mps(5, chi=3, seed=7)
    dev = qml.device("default.qubit", wires=2 + 5)

    @qml.qnode(dev)
    def circuit():
        qml.MPSPrep(to_pennylane_mps(mps), wires=range(2, 7), work_wires=range(2))
        return qml.state()

    assert np.isclose(abs(np.vdot(unit(to_dense(mps)), circuit()[: 2**5])), 1)


def test_staircase_prepares_chi2_mps_and_inverts():
    mps = normalize(random_mps(7, chi=2, seed=3))
    gates = staircase(mps)
    zero = np.eye(2**7)[0]
    assert np.isclose(abs(np.vdot(to_dense(mps), apply_staircase(zero, gates))), 1)
    psi = unit(np.random.default_rng(0).normal(size=2**7))
    assert np.allclose(apply_staircase(apply_staircase(psi, gates), gates, dagger=True), psi)


def test_layers_improve_and_match_circuit():
    H = tfim_cylinder(1, Lx=8, h=1.0)
    _, psi0 = ground_state(H)
    layers, F = disentangling_layers(psi0, 3)
    assert F[0] > 0.99 and F[-1] >= F[0] - 1e-12
    assert np.isclose(abs(np.vdot(psi0, prepare(layers, 8))), F[-1])

    @qml.qnode(qml.device("default.qubit", wires=8))
    def circuit():
        layers_circuit(layers, range(8))
        return qml.state()

    assert np.isclose(abs(np.vdot(psi0, circuit())), F[-1])
