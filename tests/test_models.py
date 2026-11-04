import numpy as np

from dmrg_from_scratch.exact import ground_state
from dmrg_from_scratch.models import cylinder_bonds, j1j2_cylinder, tfim_cylinder


def test_bond_counts():
    # W x Lx cylinder: W*Lx bonds around, W*(Lx-1) along, 2W*(Lx-1) diagonals
    assert len(cylinder_bonds(4, 4)) == 16 + 12
    assert len(cylinder_bonds(4, 4, diagonal=True)) == 24
    # W = 2 is a ladder: each rung once
    assert len(cylinder_bonds(3, 2)) == 3 + 4


def test_one_norm_and_counts():
    H = j1j2_cylinder(4, J2=0.5)
    assert H.pauli_counts() == {"XX": 52, "YY": 52, "ZZ": 52}
    assert np.isclose(H.one_norm, 0.75 * (28 + 0.5 * 24))


def test_four_site_ring():
    # the 2 x 2 cylinder with J2 = 0 is a 4-site Heisenberg ring, E0 = -2 J (spins 1/2)
    E0, _ = ground_state(j1j2_cylinder(2, J2=0.0))
    assert np.isclose(E0, -2.0)


def test_tfim_ladder_limits():
    # h = 0: classical Ising ground state, -J per bond
    H = tfim_cylinder(2, Lx=3, h=0.0)
    E0, _ = ground_state(H)
    assert np.isclose(E0, -len(cylinder_bonds(3, 2)))
