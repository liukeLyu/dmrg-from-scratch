# ruff: noqa: F821  (md, code, cells, nbf, WT come from build_nb18.py)

md(r"""
## 2. The machinery: a honeycomb as a brick wall, fluxes, and degenerate ground states

**The lattice.** A honeycomb is a brick wall: put the sites on a grid, give every site its two horizontal neighbours and one vertical neighbour (up or down, alternating). In Kitaev's model the bond direction decides the coupling: horizontal bonds alternate between $S^xS^x$ and $S^yS^y$, vertical bonds are $S^zS^z$, so every spin has one bond of each type. Sites are numbered column by column, $s = xL_y + y$, the order the MPS walks through them. Wrapping the grid in both directions, with column $L_x$ glued to column 0 shifted by `twist` rows, gives a torus. The paper's two clusters are twisted brick walls: the 24-site Nauru graph is $6\times4$ with twist 2 and the 32-site Dyck graph is $4\times8$ with twist 4 (`tests/test_kitaev.py` checks both against the standard LCF notation of the two graphs).
""")

code(r"""
def draw(Lx, Ly, twist, regions, ax, title):
    colours = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    pos = {x * Ly + y: (x, y) for x in range(Lx) for y in range(Ly)}
    for i, j, g in kt.brickwall_bonds(Lx, Ly, True, twist):
        (x1, y1), (x2, y2) = pos[i], pos[j]
        if abs(x1 - x2) + abs(y1 - y2) == 1:  # bonds inside the drawing; the others wrap around the torus
            ax.plot([x1, x2], [y1, y2], color=colours[g], lw=2, zorder=1)
    for s, (x, y) in pos.items():
        ax.text(x, y, str(s), ha="center", va="center", fontsize=7, zorder=3,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="0.5"))
    for sites, colour in regions:
        ax.scatter([pos[s][0] for s in sites], [pos[s][1] for s in sites], s=380, color=colour, alpha=0.35, zorder=2)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
Lx, Ly, tw = kt.NAURU
draw(Lx, Ly, tw, [(kt.hexagon(0, 0, Lx, Ly, tw), "tab:orange"), (kt.fork(1, 1, Lx, Ly, tw), "tab:purple")], ax[0],
     "Nauru, 24 sites: hexagon (orange), fork (purple)")
Lx, Ly, tw = kt.DYCK
draw(Lx, Ly, tw, [(kt.hexagon(0, 0, Lx, Ly, tw), "tab:orange")], ax[1], "Dyck, 32 sites")
fig.text(0.5, 0.02, "bonds: x red, y green, z blue; bonds that wrap around the torus are not drawn", ha="center", fontsize=8)
fig.tight_layout()
""")

md(r"""
**Fluxes.** Around every hexagon $p$ the product $W_p = X_1Y_2Z_3X_4Y_5Z_6$ (on each site, the Pauli of the bond that leaves the hexagon) commutes with $H$ and with every other $W_q$ (Kitaev, [arXiv:cond-mat/0506438](https://arxiv.org/abs/cond-mat/0506438)). So the Hilbert space splits into flux sectors with $W_p = \pm1$, and in each sector the model is free Majorana fermions in a static $\mathbb Z_2$ gauge field: this is why it is exactly solvable. On the infinite lattice the ground state has $W_p = +1$ everywhere (Lieb's theorem, cited by Kitaev). On a small torus a sector with some $W_p = -1$ can be as low: on the 16-site torus it is exactly degenerate with the flux-free state. Adding $-\lambda\sum_p W_p$ leaves every eigenstate alone and lowers the flux-free sector. That is the `flux_penalty=0.2` above; in an MPO it needs six-spin terms, which `pauli_string_mpo` builds with one channel per term.
""")

code(r"""
from scipy.sparse.linalg import eigsh

for lam in (0.0, 0.2):
    Hl = kt.kitaev_honeycomb(4, 4, flux_penalty=lam)
    vals, vecs = eigsh(matvec_operator(Hl), k=4, which="SA", tol=1e-12)
    order = np.argsort(vals)
    W = [np.real(pauli_expectations(dense_rdm(vecs[:, k], hexagon, Hl.n))["XYZXYZ"]) for k in order]
    print(f"lambda = {lam}: lowest energies {np.round(vals[order], 5)}, <W_p> on one hexagon {np.round(W, 3)}")
""")

md(r"""
Without the penalty the lowest level is four-fold degenerate, and the eigenvectors the solver returns are arbitrary mixtures of flux sectors. With it the flux-free ground state is unique and every hexagon has $W_p = +1$.

**Topological degeneracy.** Even inside the flux-free sector, a torus has several ground states, one per topological sector (the Majorana fermions can be periodic or antiperiodic around each of the two cycles). On the Nauru cluster, ED finds the lowest level three-fold degenerate in the isotropic model and two-fold in the gapped phase. The degenerate states are equally good ground states, but their RDMs differ, so "the RDM" has to be defined. The paper uses one symmetric ground state, or equivalently the $T\to0$ limit of the thermal state, the equal mixture over the degenerate manifold (its Methods). Here both methods use the mixture: ED over every eigenvector within $10^{-6}$ of the lowest, DMRG over the lowest states it finds one after the other (each orthogonal to the earlier ones) that lie within $10^{-3}$ of each other.

**Which DMRG.** For the 24- and 32-site clusters the same MPO goes to TeNPy's DMRG ([arXiv:1805.00055](https://arxiv.org/abs/1805.00055)): the algorithm of notebook 3, about four times faster (README, *How fast is it?*), and its mixer lets a sweep move between topological sectors. This package's `dmrg` cannot: in the gapped phase of the 16-site torus it settles $0.014$ above the ground energy, in another sector (exercise 2).
""")

md(r"""
## 3. The paper's cluster: Nauru, 24 sites, DMRG against ED

Precomputed with `python scripts/make_results.py kitaev_loops` (RUNTIME; the ED of $2^{24}$ amplitudes takes about 10 minutes and 10 GB per point). Two points: the isotropic gapless spin liquid ($g = 0$) and the gapped $\mathbb Z_2$ phase ($g = 1$: $K_x = K_y = 1/2$, $K_z = 2$, on the anisotropy cut of the paper's Fig. 4, $K_{x,y} = 1 - g/2$, $K_z = 1 + g$).
""")

code(r"""
nauru = {c["g"]: c for c in RES["clusters"] if c["n"] == 24}
for g, c in nauru.items():
    ed = c["ed"]
    print(f"g = {g}: ED ground manifold {ed['degeneracy']}-fold at E = {ed['energies'][0]:.8f}; DMRG keeps {c['dmrg_degeneracy']} states")
    for row in c["dmrg"]:
        dE = min(row["energies"]) - ed["energies"][0]
        d = row["max_diff_ed"]
        print(f"   chi = {row['chi']:3d}: E - E_ED = {dE:8.1e}   max|rho_DMRG - rho_ED|: hexagon {d['hexagon']:.1e},"
              f" fork {d['fork']:.1e}   ({row['seconds']:.0f} s)")
""")

md(r"""
TEXT_AFTER_NAURU
""")

code(r"""
names = {"hexagon_3": "hexagon, 3 pairs", "hexagon_6": "hexagon, 6 spins", "fork_3": "fork, 3 pairs",
         "fork_6": "fork, 6 spins", "pair": "two neighbours", "three": "three consecutive"}
print(f"{'':20s}" + "".join(f"{'g = ' + str(g) + ': ED':>12s}{'DMRG':>9s}" for g in nauru))
for key, label in names.items():
    print(f"{label:20s}" + "".join(f"{nauru[g]['ed']['measures'][key]:12.4f}{nauru[g]['dmrg'][-1]['measures'][key]:9.4f}"
                                   for g in nauru))
""")

md(r"""
## 4. Beyond ED: the 32-site Dyck cluster and the infinite lattice

The paper's main cluster has 32 spins, $2^{32}$ amplitudes: beyond the ED of this package (the paper used a parallel Lanczos code). DMRG treats it like the Nauru cluster. The infinite lattice is exact too: the paper lists every nonzero correlation function of a hexagon in the Majorana solution (SM Tables II–IV), and $\rho = 2^{-6}\sum_P \langle P\rangle P$ rebuilds the hexagon RDM from them (`kt.exact_hexagon_rdm()`, to four digits).
""")

code(r"""
rho_inf = kt.exact_hexagon_rdm().real
print(f"exact infinite-lattice hexagon: trace {np.trace(rho_inf):.4f}, smallest eigenvalue {np.linalg.eigvalsh(rho_inf).min():.1e}")
print(f"  GMN, 3 pairs = {kt.region_measures(rho_inf, pairs):.4f}   (live)")
print(f"  GMN, 6 spins = {RES['exact_infinite']['measures']['hexagon_6']:.4f}   (from results/: a minute of SCS)")
print(f"  two neighbours {kt.region_measures(partial_trace(rho_inf, [0, 1]), [(0,), (1,)]):.4f},"
      f" three consecutive {kt.region_measures(partial_trace(rho_inf, [0, 1, 2]), [(0,), (1,), (2,)]):.4f}")
""")

code(r"""
dyck = {c["g"]: c for c in RES["clusters"] if c["n"] == 32}
inf = RES["exact_infinite"]["measures"]
print("hexagon GMN, 3 pairs / 6 spins, isotropic model (g = 0):")
print(f"   16-site torus, ED (live)   {kt.region_measures(rho_ed.real, pairs):.4f}")
print(f"   Nauru, 24, ED              {nauru[0.0]['ed']['measures']['hexagon_3']:.4f} / {nauru[0.0]['ed']['measures']['hexagon_6']:.4f}")
print(f"   Dyck, 32, DMRG             {dyck[0.0]['dmrg'][-1]['measures']['hexagon_3']:.4f} / {dyck[0.0]['dmrg'][-1]['measures']['hexagon_6']:.4f}")
print(f"   infinite lattice, exact    {inf['hexagon_3']:.4f} / {inf['hexagon_6']:.4f}")
print("   paper, Fig. 3a at h = 0 (Dyck, ED; read off the plot): about 0.20 / 0.065")
for g, c in dyck.items():
    rows = c["dmrg"]
    print(f"Dyck g = {g}: DMRG keeps {c['dmrg_degeneracy']} states; energies at chi = {rows[-1]['chi']}: "
          f"{np.round(rows[-1]['energies'], 6)}; hexagon RDM moved {rows[-1]['max_diff_previous_chi']['hexagon']:.1e}"
          f" in the last doubling of chi")
""")

md(r"""
TEXT_AFTER_DYCK
""")

md(r"""
## 5. Why this is a spin liquid, and what an entanglement loop is

TEXT_MEANING
""")

md(r"""
## 6. What DMRG buys here

TEXT_BUYS

## Exercises

1. **Fluxes.** Compute $\langle W_p\rangle$ for every hexagon of the 16-site ground state with `dense_rdm`. Then set `flux_penalty=-0.2`, which favours $W_p = -1$: what happens to the energy and to the hexagon GMN?
2. **Topological sectors.** Run this package's `dmrg` on the 16-site torus at $g = 1$ (`K=kt.anisotropic_couplings(1.0)`) from several seeds and compare the energies with the ED spectrum. Which ED eigenvalues does it find? Why can a sweep of two-site updates not change the sector, and what does TeNPy's mixer add?
3. **A shorter loop.** On the $4\times4$ torus, a vertical loop around the torus has only four sites. Compute the GMN of those four spins as four parties and compare with the hexagon. Is the rule about hexagons, or about closed loops of bonds?
4. **The fork in the gapped phase.** On the 16-site torus at $g = 1$ the fork GMN is about 0.05. Is that a finite-size effect? Check on the Nauru and Dyck clusters with the results file, then argue from the toric-code limit $K_z\to\infty$ what the fork RDM should become.
5. **Adding the field.** Add $-h\sum_i (S^x_i + S^y_i + S^z_i)/\sqrt3$ to the 16-site Hamiltonian with TeNPy (the $S^y$ term makes it complex; `pauli_string_mpo` is real only). Follow the hexagon and fork GMN up to $h = 1$ and compare with the paper's Fig. 3.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.15"},
}
nbf.write(nb, WT + "/examples/18_kitaev_entanglement_loops.ipynb")
print("wrote notebook 18 with", len(cells), "cells")
