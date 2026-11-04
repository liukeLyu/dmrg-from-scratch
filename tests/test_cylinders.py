import numpy as np
import pytest

from dmrg_from_scratch.exact import ground_state, matvec_operator
from dmrg_from_scratch.finite import dmrg, dmrg_ladder
from dmrg_from_scratch.models import PauliSum, j1j2_cylinder, tfim_cylinder
from dmrg_from_scratch.mpo import build_mpo, mpo_to_dense
from dmrg_from_scratch.mps import overlap, to_dense


@pytest.mark.parametrize(
    "H",
    [j1j2_cylinder(3, Lx=3), j1j2_cylinder(2, Lx=4), tfim_cylinder(3, Lx=3, h=1.3)],
    ids=["j1j2_3x3", "j1j2_ladder", "tfim_3x3"],
)
def test_mpo_matches_exact_operator(H):
    dense = matvec_operator(H) @ np.eye(2**H.n)
    assert np.allclose(mpo_to_dense(build_mpo(H)), dense)


def test_mpo_long_range_and_y_terms():
    rng = np.random.default_rng(1)
    terms = [(rng.normal(), ((0, "Y"), (4, "Y"))), (rng.normal(), ((1, "X"), (3, "Z"))), (rng.normal(), ((2, "Z"),))]
    H = PauliSum(5, terms)
    assert np.allclose(mpo_to_dense(build_mpo(H)), matvec_operator(H) @ np.eye(32))


def test_dmrg_matches_exact():
    H = j1j2_cylinder(3, Lx=4)
    E0, psi0 = ground_state(H)
    r = dmrg(build_mpo(H), chi_max=64, n_sweeps=4)
    assert abs(r["energy"] - E0) < 1e-8
    assert abs(abs(to_dense(r["mps"]) @ psi0) - 1) < 1e-6
    assert r["flops"] > 0


def test_ladder_improves_and_overlaps():
    H = j1j2_cylinder(3)
    runs = dmrg_ladder(build_mpo(H), [2, 4, 8, 32], n_sweeps=4)
    E = [r["energy"] for r in runs]
    assert all(a >= b - 1e-10 for a, b in zip(E, E[1:], strict=False))
    assert abs(overlap(runs[-1]["mps"], runs[-1]["mps"]) - 1) < 1e-10
