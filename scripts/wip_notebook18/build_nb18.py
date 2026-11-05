"""Write examples/18_kitaev_entanglement_loops.ipynb (unexecuted); run it afterwards with nbconvert."""

import nbformat as nbf

WT = str(__import__("pathlib").Path(__file__).resolve().parents[2])  # the repo root
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# 18. Entanglement loops in Kitaev's spin liquid

**Goal.** Take one of the quantum spin liquids of Lyu, Chandorkar, Kapoor, Takei, Sørensen and Witczak-Krempa (*Multiparty entanglement loops in quantum spin liquids*, [arXiv:2505.18124](https://arxiv.org/abs/2505.18124); [Nat. Commun. 17, 5650 (2026)](https://doi.org/10.1038/s41467-026-71113-8)), find its ground state with DMRG on the very clusters the paper diagonalized exactly, and check the reduced density matrices (RDMs) and their multiparty entanglement against exact diagonalization (ED).

The paper's finding, in one line: in a spin liquid, small groups of neighbouring spins are *not* genuinely entangled with each other ("entanglement frustration"); genuine multiparty entanglement only appears in regions that contain a closed loop, like a hexagon of the honeycomb lattice.

**Which of the paper's models suit DMRG?** The paper studies several spin liquids, mostly with ED on periodic clusters of 24 to 36 spins. DMRG handles a model well when its couplings are short-ranged and real, it maps onto a cylinder without long bonds, its ground state is gapped (so the entanglement across a cut stays small), and it has symmetries a package can use.

| Model in the paper | What was computed there | For DMRG |
|---|---|---|
| **Kitaev honeycomb, no field** (Eq. 1 with $h = 0$): $\sum K_\gamma S^\gamma_i S^\gamma_j$, isotropic gapless spin liquid | ED on the 32-site Dyck and 24-site Nauru clusters; the exact Majorana solution of the infinite lattice (SM Tables II–IV); GMN of the hexagon, a "fork", pairs and triples (Fig. 3 at $h = 0$) | **Easy to medium.** Real, nearest-neighbour, spin 1/2, conserved fluxes that can be pinned; gapless, so the 24-site torus needs $\chi \approx 500$ and the 32-site one is out of reach (section 4); exact checks available. **This notebook.** |
| **Kitaev, anisotropic** ($K_z > K_x = K_y$), gapped $\mathbb Z_2$ (toric-code) spin liquid | exact Majorana solution, hexagon GMN against anisotropy (Fig. 4) | **Easy** on 24 sites. Gapped, little entanglement ($\chi = 128$ already gives the RDMs to $10^{-6}$); DMRG can sit in the wrong topological sector. **This notebook.** |
| Kitaev in a [111] field $h$ | ED on 32 and 24 sites, GMN of many regions against $h$ (Figs. 3, 7, 8) | **Medium to hard.** The field term $S^y$ makes $H$ complex and breaks flux conservation; the intermediate phase ($0.38 < h < 0.64$, Fig. 3) has competing states. ED here would need a complex solver. |
| Kitaev plus the three-spin term $\kappa S^xS^yS^z$ (the small-field effective model, SM Eq. 10) | exact Majorana solution (light curves in Fig. 3a) | Medium: complex three-site terms, but gapped, with an exact check. |
| Kagome Heisenberg $J_1$–$J_2$ (Eq. 2) | ED on 36 sites (12–36 in SM Fig. 14); GMN of the bowtie, hexagon, triangles (Fig. 5) | **Hard.** Frustrated, nearly degenerate competing orders, probably gapless (a Dirac spin liquid); the classic DMRG challenge needs cylinders of circumference 8–12. |
| Kagome with scalar chirality $\lambda\,\mathbf S_i\cdot(\mathbf S_j\times\mathbf S_k)$ (Eq. 3) | ED on 24 and 36 sites (Fig. 6) | Medium to hard: complex three-spin terms; the chiral phase is gapped but has chiral edge modes on cylinders. |
| RVB states (kagome, square lattice) | 36-site wavefunctions | Not a DMRG target: these are wavefunctions, not ground states of a simple Hamiltonian. |
| String nets (toric code, double semion, Fibonacci, ...) | exact RDMs, separability proofs (SM Sec. X) | The toric code is an exact small-$\chi$ MPS on a cylinder; the non-abelian nets have large local dimension and many-spin plaquette terms. Their RDMs are known exactly anyway. |

Kitaev's honeycomb model at zero field is the natural first step: a genuine spin liquid, a real Hamiltonian, and two exact references, ED on the paper's clusters and the exact solution of the infinite lattice.
""")

code(r"""
import json
import logging
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dmrg_from_scratch as dmrg
from dmrg_from_scratch import kitaev as kt
from dmrg_from_scratch.exact import ground_state, matvec_operator
from dmrg_from_scratch.microscopy import dense_rdm, partial_trace, pauli_expectations

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")
RES = json.loads(Path("../results/kitaev_loops.json").read_text())
""")

md(r"""
## 1. One cluster, one call each

Kitaev's model on a small honeycomb torus of 16 spins: build the Hamiltonian, find the ground state with exact diagonalization and with this package's DMRG, and cut out the six spins of one hexagon. The model is $H = \sum_{\langle ij\rangle_\gamma} K_\gamma S_i^\gamma S_j^\gamma$ with $\mathbf S = \boldsymbol\sigma/2$ and $K_x = K_y = K_z = 1$ (antiferromagnetic), the convention of the paper's Eq. 1. The term $-0.2\sum_p W_p$ is explained in section 2; it does not change the ground state here, it only makes it unique.
""")

code(r"""
H = kt.kitaev_honeycomb(4, 4, flux_penalty=0.2)          # 4 x 4 brick wall = 16-site honeycomb torus
E_ed, psi_ed = ground_state(H)
r = dmrg.dmrg(dmrg.pauli_string_mpo(H), chi_max=128, n_sweeps=6)
psi_dmrg = dmrg.to_dense(r["mps"])

hexagon = kt.hexagon(0, 0, 4, 4)                        # six sites around one hexagon
rho_ed, rho_dmrg = dense_rdm(psi_ed, hexagon, H.n), dense_rdm(psi_dmrg, hexagon, H.n)
print(f"energy: ED {E_ed:.12f}, DMRG {r['energy']:.12f}")
print(f"hexagon RDM, largest difference of an entry: {np.abs(rho_ed - rho_dmrg).max():.1e}")
""")

md(r"""
The two RDMs agree to machine precision: at $\chi = 128$ this small torus is represented exactly. Now the entanglement. The genuine multiparty negativity (GMN; Jungnitsch, Moroder and Gühne, [arXiv:1010.6049](https://arxiv.org/abs/1010.6049)) is positive only if the parties are genuinely multipartite entangled; for two parties it is the negativity. Four regions, as in the paper's Fig. 3: the hexagon split into three pairs, a "fork" of six spins without a loop split the same way, two neighbours, and three consecutive spins.
""")

code(r"""
fork = kt.fork(1, 1, 4, 4)
rho_fork = dense_rdm(psi_ed, fork, H.n)
pairs = [(0, 1), (2, 3), (4, 5)]
t0 = time.perf_counter()
print(f"hexagon, 3 pairs        GMN = {kt.region_measures(rho_ed.real, pairs):.4f}")
print(f"fork, 3 pairs           GMN = {kt.region_measures(rho_fork.real, pairs):.4f}")
print(f"two neighbours          negativity = {kt.region_measures(partial_trace(rho_ed, [0, 1]).real, [(0,), (1,)]):.4f}")
print(f"three consecutive spins GMN = {kt.region_measures(partial_trace(rho_ed, [0, 1, 2]).real, [(0,), (1,), (2,)]):.4f}")
print(f"({time.perf_counter() - t0:.0f} s)")
print(f"yet the two neighbours are strongly correlated: <Z1 Z2> = {np.real(pauli_expectations(rho_ed)['ZZIIII']):.3f}")
""")

md(r"""
That is the paper's result in miniature. The hexagon is genuinely entangled; the fork, also six spins and also built from the same bonds, is not; neither is a pair nor a triple, even though neighbouring spins are strongly correlated. Entanglement sits on the loop.

The counterexample is any ordinary magnet: in the transverse-field Ising model, two neighbours are entangled at every field (notebook 17, and the paper's refs. 14 and 30).
""")

exec(open(__file__.replace("build_nb18.py", "build_nb18_part2.py")).read())
