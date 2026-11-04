import numpy as np
import pytest

import dmrg_from_scratch as dmrg
from dmrg_from_scratch.exact import matvec_operator
from dmrg_from_scratch.fermions import AD, A, F, N, free_chain_correlation, free_chain_energy, hubbard_mpo
from dmrg_from_scratch.mps import product_expectation


@pytest.mark.parametrize("L, U", [(2, 3.0), (3, 2.0)])
def test_mpo_matches_jordan_wigner_pauli_sum(L, U):
    # hubbard_cylinder uses U (n_up - 1/2)(n_dn - 1/2) = U n_up n_dn - (U/2)(n_up + n_dn) + U/4 per site
    dense = matvec_operator(dmrg.hubbard_cylinder(1, L, U)) @ np.eye(4**L)
    M = dmrg.mpo_to_dense(hubbard_mpo(L, U, mu=U / 2))
    assert np.allclose(M + U * L / 4 * np.eye(4**L), dense)


def test_free_chain_dmrg_and_correlations():
    L = 4
    r = dmrg.dmrg(hubbard_mpo(L, 0.0), chi_max=32, n_sweeps=4)
    assert abs(r["energy"] - free_chain_energy(L)) < 1e-9
    for j in (1, 2, 3):
        string = {k: F for k in range(1, 2 * j)}
        assert np.isclose(product_expectation(r["mps"], {0: AD, 2 * j: A, **string}), free_chain_correlation(L, 0, j))


def test_half_filling_with_chemical_potential():
    L, U = 4, 4.0
    E_ph, _ = dmrg.ground_state(dmrg.hubbard_cylinder(1, L, U))
    r = dmrg.dmrg(hubbard_mpo(L, U, mu=U / 2), chi_max=32, n_sweeps=4)
    assert abs(r["energy"] + U / 2 * L - (E_ph + U * L / 4)) < 1e-9
    assert np.isclose(sum(product_expectation(r["mps"], {p: N}) for p in range(2 * L)), L)
