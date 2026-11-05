import networkx as nx
import numpy as np
import pytest

import dmrg_from_scratch as dmrg
from dmrg_from_scratch import kitaev as kt
from dmrg_from_scratch.exact import ground_state, matvec_operator
from dmrg_from_scratch.microscopy import dense_rdm, pauli_expectations
from dmrg_from_scratch.models import PauliSum
from dmrg_from_scratch.mpo import mpo_to_dense, pauli_string_mpo


def graph(Lx, Ly, twist):
    G = nx.Graph()
    G.add_edges_from((i, j) for i, j, _ in kt.brickwall_bonds(Lx, Ly, True, twist))
    return G


def test_every_site_has_one_bond_of_each_type():
    for Lx, Ly, tw in [(4, 4, 0), kt.NAURU, kt.DYCK]:
        seen = {}
        for i, j, g in kt.brickwall_bonds(Lx, Ly, True, tw):
            for s in (i, j):
                seen.setdefault(s, []).append(g)
        assert all(sorted(v) == ["x", "y", "z"] for v in seen.values()) and len(seen) == Lx * Ly


def test_nauru_and_dyck_clusters():
    # LCF notation of the Nauru and Dyck graphs (e.g. the table in the Wikipedia article on LCF notation)
    assert nx.is_isomorphic(graph(*kt.NAURU), nx.LCF_graph(24, [5, -9, 7, -7, 9, -5], 4))
    assert nx.is_isomorphic(graph(*kt.DYCK), nx.LCF_graph(32, [5, -5, 13, -13], 8))
    assert not nx.is_isomorphic(graph(6, 4, 0), nx.LCF_graph(24, [5, -9, 7, -7, 9, -5], 4))


def test_hexagon_bond_types_and_plaquette_count():
    Lx, Ly, tw = kt.NAURU
    types = {frozenset((i, j)): g for i, j, g in kt.brickwall_bonds(Lx, Ly, True, tw)}
    for h in kt.plaquettes(Lx, Ly, True, tw):
        loop = [types[frozenset((h[k], h[(k + 1) % 6]))] for k in range(6)]
        assert loop == list("zxyzxy")
    assert len(kt.plaquettes(Lx, Ly, True, tw)) == Lx * Ly // 2


def test_pauli_string_mpo_matches_exact_operator():
    H = kt.kitaev_honeycomb(4, 2, K=(0.7, 1.1, 1.3), flux_penalty=0.3)
    assert np.allclose(mpo_to_dense(pauli_string_mpo(H)), matvec_operator(H) @ np.eye(2**H.n))
    rng = np.random.default_rng(0)
    G = PauliSum(5, [(rng.normal(), ((0, "Y"), (2, "Z"), (4, "Y"))), (rng.normal(), ((1, "X"),))])
    assert np.allclose(mpo_to_dense(pauli_string_mpo(G)), matvec_operator(G) @ np.eye(32))


def test_plaquettes_commute_with_kitaev():
    H = matvec_operator(kt.kitaev_honeycomb(4, 2)) @ np.eye(256)
    for h in kt.plaquettes(4, 2):
        W = matvec_operator(PauliSum(8, [(1.0, tuple(zip(h, kt.HEXAGON_W, strict=True)))])) @ np.eye(256)
        assert np.allclose(H @ W, W @ H)


def commute(a, b):
    a, b = dict(a), dict(b)
    return sum(a[s] != b[s] for s in a if s in b) % 2 == 0


def gf2_rank(rows):
    rows, rank = [int("".join(map(str, r)), 2) for r in rows], 0
    while rows:
        pivot = max(rows)
        if pivot == 0:
            break
        rank += 1
        top = pivot.bit_length() - 1
        rows = [r ^ pivot if r >> top & 1 else r for r in rows if r != pivot]
    return rank


@pytest.mark.parametrize("cluster", [(4, 4, 0), kt.NAURU, kt.DYCK])
def test_torus_loops_are_conserved_and_wind_around(cluster):
    Lx, Ly, tw = cluster
    H = kt.kitaev_honeycomb(Lx, Ly, twist=tw, flux_penalty=1.0)
    loops = kt.torus_loops(Lx, Ly, tw)
    assert all(commute(word, l) for _, word in H.terms for l in loops)  # with every bond and every W_p
    assert commute(*loops)

    # as sets of sites (mod 2), neither loop nor their product is a product of hexagons
    def sites(ss):
        return [int(s in ss) for s in range(Lx * Ly)]

    hexes = [sites(set(h)) for h in kt.plaquettes(Lx, Ly, True, tw)]
    l1, l2 = (sites({s for s, _ in l}) for l in loops)
    assert gf2_rank(hexes + [l1, l2]) == gf2_rank(hexes) + 2


def test_sector_penalty_selects_each_topological_sector():
    loops = [matvec_operator(PauliSum(16, [(1.0, l)])) for l in kt.torus_loops(4, 4)]
    E0 = ground_state(kt.kitaev_honeycomb(4, 4, flux_penalty=0.2))[0]
    energies = []
    for sector in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        E, psi = ground_state(kt.kitaev_honeycomb(4, 4, flux_penalty=0.2, sector=sector))
        assert np.allclose([psi.conj() @ (W @ psi) for W in loops], sector)
        energies.append(E + 2 * 0.5)
    assert np.isclose(min(energies), E0)
    parities = [matvec_operator(PauliSum(16, [(1.0, tuple((s, a) for s in range(16)))])) for a in "XZ"]
    for parity in [(1, 1), (-1, -1)]:
        _, psi = ground_state(kt.kitaev_honeycomb(4, 4, flux_penalty=0.2, sector=(1, 1), parity=parity))
        assert np.allclose([psi.conj() @ (W @ psi) for W in loops + parities], (1, 1, *parity))


def test_exact_table_rdm_is_a_state():
    rho = kt.exact_hexagon_rdm()
    assert np.isclose(np.trace(rho).real, 1)
    assert np.linalg.eigvalsh(rho).min() > -1e-3  # four-digit table entries
    assert np.isclose(np.real(np.trace(rho @ kt.pauli_string_operator("XYZXYZ"))), 1)


def test_dmrg_matches_ed_on_flux_free_16_site_torus():
    H = kt.kitaev_honeycomb(4, 4, flux_penalty=0.2)
    E0, psi0 = ground_state(H)
    r = dmrg.dmrg(pauli_string_mpo(H), 128, n_sweeps=6)  # chi = 64 is 0.08 off; 128 truncates nothing
    assert abs(r["energy"] - E0) < 1e-8
    hexagon = kt.hexagon(0, 0, 4, 4)
    rho_ed = dense_rdm(psi0, hexagon, H.n)
    rho_mps = dense_rdm(dmrg.to_dense(r["mps"]), hexagon, H.n)
    assert np.abs(rho_ed - rho_mps).max() < 1e-6
    assert np.isclose(np.real(pauli_expectations(rho_ed)["XYZXYZ"]), 1)


def test_left_end_rdm_matches_dense():
    from dmrg_from_scratch.microscopy import left_end_rdm

    mps = dmrg.right_canonicalize(dmrg.random_mps(10, chi=8, seed=3))
    psi = dmrg.to_dense(mps)
    psi /= np.linalg.norm(psi)
    for sites in [(0, 1, 5, 6, 8, 9), (7, 2, 4)]:
        assert np.allclose(left_end_rdm(mps, list(sites)), dense_rdm(psi, sites, 10))


@pytest.mark.parametrize("parties", [[(0,), (1,)], [(0, 1), (2, 3), (4, 5)]])
def test_region_measures_on_simple_states(parties):
    n = sum(len(p) for p in parties)
    product = np.zeros(2**n)
    product[0] = 1
    assert abs(kt.region_measures(np.outer(product, product), parties)) < 1e-6
    if n == 2:
        bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
        assert np.isclose(kt.region_measures(np.outer(bell, bell), parties), 0.5, atol=1e-5)
