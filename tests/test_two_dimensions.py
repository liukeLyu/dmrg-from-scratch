import numpy as np

import dmrg_from_scratch as dmrg
from dmrg_from_scratch.exact import ground_state
from dmrg_from_scratch.models import cylinder_bonds, hubbard_cylinder, neel_pinning
from dmrg_from_scratch.mps import product_mps, site_expectations

Z = np.diag([1.0, -1.0])


def test_site_expectations_of_a_product_state():
    up, down = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    assert np.allclose(site_expectations(product_mps([up, down, plus]), Z), [1, -1, 0])


def test_pinned_cylinder_matches_exact():
    H = neel_pinning(dmrg.j1j2_cylinder(4, Lx=3, J2=0.0))
    E0, psi0 = ground_state(H)
    r = dmrg.dmrg(dmrg.build_mpo(H), chi_max=64, n_sweeps=4)
    assert abs(r["energy"] - E0) < 1e-8
    # <Z_i> from the exact vector: site 0 is the most significant bit
    n = H.n
    bits = (np.arange(2**n)[:, None] >> (n - 1 - np.arange(n))) & 1
    z_exact = (psi0**2) @ (1 - 2 * bits)
    assert np.allclose(site_expectations(r["mps"], Z), z_exact, atol=1e-6)
    # the pinning picks one Néel state: alternating signs
    z = site_expectations(r["mps"], Z).reshape(3, 4)
    stag = z * (-1.0) ** np.add.outer(np.arange(3), np.arange(4))
    assert np.all(stag < 0)  # the field + h (-1)^(x+y) S^z points even sites down


def test_hubbard_free_fermions():
    # U = 0: fill every negative level of the hopping matrix, once per spin
    for W, Lx in [(2, 3), (3, 2)]:
        T = np.zeros((W * Lx, W * Lx))
        for i, j in cylinder_bonds(Lx, W):
            T[i, j] = T[j, i] = -1.0
        ev = np.linalg.eigvalsh(T)
        E0, _ = ground_state(hubbard_cylinder(W, Lx, U=0.0))
        assert np.isclose(E0, 2 * ev[ev < 0].sum())


def test_hubbard_atomic_limit():
    # t = 0: every site singly occupied, (U/4) Z_up Z_dn = -U/4 per site
    E0, _ = ground_state(hubbard_cylinder(2, 2, U=4.0, t=0.0))
    assert np.isclose(E0, -4.0)
