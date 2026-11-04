# dmrg-from-scratch

*Why I built this.* I made these tutorials for my own learning. I have been around DMRG for years: I know what a matrix product state is and roughly what infinite DMRG does, and I have run the standard packages. But running a package is not the same as understanding it. I wanted two things: to understand the algorithm at a high level, well enough to know why it works and when it fails, and to use it to understand materials and molecules. So I first wrote it myself, one small piece at a time, on a model I know well, checking each step against the exact answer. Then I took it, step by step, to the problems DMRG is used for: molecules, two-dimensional magnets and electrons, dynamics and temperature, and the question of when a quantum computer would do better.

The density-matrix renormalization group (DMRG), written from scratch in short, readable NumPy and SciPy on the transverse-field Ising chain $H = -\sum_i X_iX_{i+1} - h\sum_i Z_i$, then used through [TeNPy](https://github.com/tenpy/tenpy) and [block2](https://github.com/block-hczhai/block2-preview) on molecules, cylinders and the Hubbard model. Every result is checked against an exact answer: free fermions for the Ising chain, exact diagonalization for small clusters and molecules, or published benchmarks.

```python
import dmrg_from_scratch as dmrg

r = dmrg.dmrg(dmrg.tfim_mpo(L=100, h=1.0), chi_max=32)   # 100 spins at the critical point
r["energy"] - dmrg.tfim_open_energy(100, 1.0)              # 9e-11: error against the exact free-fermion energy
```

New to DMRG? Start with [`00_start_here`](examples/00_start_here.ipynb): why a laptop can find the ground state of 60 interacting spins (the exponential wall, the area law). Already know how DMRG works, for example from the [pocket course](docs/pocket-course.html)? Start at part 2.

**Part 1. How DMRG works**, built from scratch on the Ising chain.

| Step | What you learn | Notebook |
|---|---|---|
| 1 | Matrix product states: one SVD per cut, canonical forms, Schmidt values, truncation | [`01_mps_basics`](examples/01_mps_basics.ipynb) |
| 2 | The Hamiltonian as a matrix product operator (a 3-state machine); exact free-fermion energies | [`02_tfim_mpo`](examples/02_tfim_mpo.ipynb) |
| 3 | Finite two-site DMRG with a Lanczos eigensolver | [`03_finite_dmrg`](examples/03_finite_dmrg.ipynb) |
| 4 | Infinite DMRG, correlation length and finite-entanglement scaling | [`04_idmrg`](examples/04_idmrg.ipynb) |

**Part 2. DMRG in practice**, with well-optimized packages.

| Step | What you learn | Notebook |
|---|---|---|
| 5 | TeNPy: the same objects by their package names, what to check before trusting a result, symmetries | [`05_dmrg_in_practice`](examples/05_dmrg_in_practice.ipynb) |
| 6 | Molecules with block2: molecules as DMRG chains, breaking the N₂ bond where CCSD(T) fails, orbital order and orbital entanglement (phone companion: [quantum chemistry pocket course](docs/qchem-pocket-course.html)) | [`06_quantum_chemistry_block2`](examples/06_quantum_chemistry_block2.ipynb) |

**Part 3. Towards materials**: what experiments measure, and systems that are not chains.

| Step | What you learn | Notebook |
|---|---|---|
| 7 | Beyond the ground state: excited states by DMRG with a penalty, real-time TEBD, a quench and the entanglement barrier, the dynamical structure factor $S^{zz}(k,\omega)$ against free fermions | [`07_excitations_and_dynamics`](examples/07_excitations_and_dynamics.ipynb) |
| 8 | Finite temperature: purification with ancillas, cooling by imaginary-time TEBD, energy and specific heat at all temperatures, how much entanglement a thermal state needs | [`08_finite_temperature`](examples/08_finite_temperature.ipynb) |
| 9 | Electrons and two dimensions: Jordan–Wigner by hand (the Hubbard chain as an MPO, what a missing string does to correlators, fixing the filling), cylinders, Néel order from a pinning field, extrapolating in truncation error and width, the Hubbard model with $N$ and $S^z$ conserved | [`09_two_dimensions`](examples/09_two_dimensions.ipynb) |
| 10 | Calibration: DMRG next to AFQMC, DMET, fixed-node, coupled cluster and other methods on the 2D Hubbard benchmark; reproducing the published width-4 energy; why the width, not $\chi$, limits a 2D number | [`10_calibration`](examples/10_calibration.ipynb) |

**Part 4. DMRG and the quantum computer**: when does a fault-tolerant quantum computer beat DMRG, and how DMRG helps it.

| Step | What you learn | Notebook |
|---|---|---|
| 11 | Who wins? Both prices for one ground-state energy, in one call each | [`11_dmrg_vs_qpe`](examples/11_dmrg_vs_qpe.ipynb) |
| 12 | The classical price: the bond dimension needed grows exponentially with cylinder width; flops and seconds | [`12_classical_price`](examples/12_classical_price.ipynb) |
| 13 | The quantum price: block encodings, the one-norm $\lambda$, walk calls, Toffolis, logical qubits, hours (`pennylane.estimator`) | [`13_quantum_price`](examples/13_quantum_price.ipynb) |
| 14 | The hidden cost: phase estimation needs a good starting state, a small DMRG state provides one; where the curves cross | [`14_overlap_and_crossover`](examples/14_overlap_and_crossover.ipynb) |
| 15 | The DMRG starting state as a circuit: sequential generation from the right-canonical tensors, PennyLane's `MPSPrep`, Ran's layers of two-qubit gates; preparation cost against $1/\gamma^2$ | [`15_mps_to_circuit`](examples/15_mps_to_circuit.ipynb) |
| 16 | How much does DMRG cost on your machine? Count the flops, measure the matrix-multiply rate and the overhead of small calls, memory, the SVD's share, a GPU if you have one: predict a run time, then measure it | [`16_cost_intuition`](examples/16_cost_intuition.ipynb) |

Part 4 uses the frustrated J1-J2 Heisenberg model on a cylinder of width $W$, $H = J_1 \sum_{\langle ij\rangle} \mathbf S_i\cdot\mathbf S_j + J_2 \sum_{\langle\langle ij\rangle\rangle} \mathbf S_i\cdot\mathbf S_j$ at $J_2/J_1 = 0.5$, where quantum Monte Carlo has a sign problem and DMRG is the method to beat:

```python
from dmrg_from_scratch import j1j2_cylinder
from dmrg_from_scratch.resources import qpe_cost

qpe_cost(j1j2_cylinder(6), eps=0.01)   # 6 x 6 cylinder, 36 spins
# {'lambda': 72.0, 'walk_calls': 11310, 'toffoli': 14624742, 'T': 2002748,
#  'logical_qubits': 202, 'repetitions': 1.0, 'seconds': 2826.7}      <- 47 minutes
```

DMRG does the same 36 spins in seconds. The quantum computer only wins once the cylinder is wide enough for entanglement to make DMRG explode.

**What next.** [`docs/ROADMAP.md`](docs/ROADMAP.md) explains why these steps were chosen and what it takes to use DMRG on real materials and molecules; [`IDEAS.md`](IDEAS.md) lists the stages planned for later (real molecules with transition metals, and the step from a material to a model).

## What you will see

**Part 1.**

* Finite DMRG reaches the exact ground energy of the open chain to $10^{-12}$ ($L = 100$, $\chi = 64$, $h = 1$), and when $\chi$ is too small the energy error tracks the discarded weight.
* At the critical point $h = 1$ the entanglement profile $S(l)$ follows the Calabrese–Cardy formula with central charge $c \approx 0.53$ ($L = 100$; exact value $1/2$).
* iDMRG gets the infinite-chain energy density to machine precision in the gapped phases and to $6\times10^{-7}$ at $h = 1$ with $\chi = 16$. Finite-entanglement scaling $S = (c/6)\log_2\xi$ gives $c \approx 0.51$.
* In the ordered phase, finite DMRG settles on one symmetry-broken state ($S \approx 0$), while iDMRG with a symmetric Hamiltonian returns the cat state ($S \approx 1$ bit, degenerate transfer matrix).
**Part 3.**

* DMRG with a penalty finds the first excited state of 24 spins at $h = 1.5$; its gap matches free fermions to ten digits. After a global quench the entanglement grows linearly, so doubling $\chi$ buys only a fixed stretch of time: on 64 spins the error in $\langle Z\rangle$ passes $10^{-3}$ at $t = 1.1, 1.7, 2.5, 4.3$ for $\chi = 4, 8, 16, 64$.
* $S^{zz}(k,\omega)$ of 64 spins fills the exact two-particle continuum. Purification gives the energy of the critical chain at every temperature to $10^{-5}$ (the Trotter error); $\chi = 16$ is as good as $\chi = 64$.
* The Hubbard chain, built as an MPO by hand and solved with our own DMRG, matches free fermions and exact diagonalization to $10^{-12}$. Without the Jordan–Wigner string, $\langle c^\dagger_0 c_j\rangle$ comes out zero where the exact value is up to 0.44.
* Pinned cylinders up to width 8 bracket the square-lattice Néel order, $0.3026 < M_0 < 0.3173$, around the quantum Monte Carlo value 0.30743; different width fits spread from 0.299 to 0.328, so the width, not $\chi$, is the error that matters.
* On the width-4 Hubbard cylinder our energies are within $10^{-3}$ of the published DMRG values ($-0.8651$ against $-0.8645$ at $U = 4$), and conserving $N$ and $S^z$ cuts the cost $\chi^3 = 1.7\times10^7$ to $\sum \text{block}^3 = 1.2\times10^5$.

**Part 4.**

* On J1-J2 cylinders DMRG needs bond dimension $\chi \approx 18$, $110$ and $800$ for $W = 4, 6, 8$: a factor of about 33 in run time per unit of width.
* Qubitized phase estimation grows only like $N^{1.8}$ in Toffolis: $1.5\times10^7$ Toffolis on about 200 logical qubits for 36 spins, $10^9$ on about 700 for 400 spins.
* A product state has overlap $\gamma^2 \approx 3\times10^{-9}$ with the 64-spin ground state; an MPS with $\chi_0 = 32$ keeps $\gamma^2 \approx 0.34$.
* The curves cross between $W = 6$ and $W = 10$ (36 to 100 spins), where phase estimation takes minutes to a day, depending on how fast the quantum computer makes Toffolis and how fast the classical computer runs.

## Install

```bash
pip install git+https://github.com/liukeLyu/multipartite-entanglement-detection-toolkit   # optional, install first: medt, an independent check of the exact energies
pip install -e ".[dev,examples]"
pytest
```

Each notebook cell runs in seconds. Results that need longer runs appear twice: a small live run you can change, then the full run loaded from a small file in `results/`. `python scripts/make_results.py` remakes those files (about 40 minutes on a 12-core laptop, set by the 8 x 8 cylinder of steps 12 and 14 and the width-4 Hubbard cylinder of step 10).

`benchmarks/benchmark_tenpy.py` solves the same problems with [TeNPy](https://github.com/tenpy/tenpy) as a cross-check (same accuracy).

The package imports as `import dmrg_from_scratch as dmrg`. Entropies are in bits (log base 2) throughout. Index conventions: MPS tensors `A[left, phys, right]`, MPO tensors `W[w_left, w_right, out, in]`, environments `L[ket, mpo, bra]`. Site 0 is the most significant bit. On a cylinder, spins are numbered column by column, $s = xW + y$, open along the cylinder and periodic around it; $\mathbf S = \boldsymbol\sigma/2$, so each Heisenberg bond is $\tfrac14(XX + YY + ZZ)$.

## References

* S. R. White, *Density matrix formulation for quantum renormalization groups*, [Phys. Rev. Lett. 69, 2863 (1992)](https://doi.org/10.1103/PhysRevLett.69.2863)
* U. Schollwöck, *The density-matrix renormalization group in the age of matrix product states*, [arXiv:1008.3477](https://arxiv.org/abs/1008.3477)
* I. P. McCulloch, *Infinite size density matrix renormalization group, revisited*, [arXiv:0804.2509](https://arxiv.org/abs/0804.2509)
* P. Calabrese, J. Cardy, *Entanglement entropy and quantum field theory*, [arXiv:hep-th/0405152](https://arxiv.org/abs/hep-th/0405152)
* F. Pollmann, S. Mukerjee, A. M. Turner, J. E. Moore, *Theory of finite-entanglement scaling at one-dimensional quantum critical points*, [arXiv:0812.2903](https://arxiv.org/abs/0812.2903)
* S. R. White, A. L. Chernyshev, *Néel order in square and triangular lattice Heisenberg models*, [arXiv:0705.2746](https://arxiv.org/abs/0705.2746)
* G. H. Low, I. L. Chuang, *Hamiltonian simulation by qubitization*, [arXiv:1610.06546](https://arxiv.org/abs/1610.06546)
* R. Babbush, C. Gidney, D. W. Berry, N. Wiebe, J. McClean, A. Paler, A. Fowler, H. Neven, *Encoding electronic spectra in quantum circuits with linear T complexity*, [arXiv:1805.03662](https://arxiv.org/abs/1805.03662)
* R. Babbush, J. McClean, M. Newman, C. Gidney, S. Boixo, H. Neven, *Focus beyond quadratic speedups for error-corrected quantum advantage*, [arXiv:2011.04149](https://arxiv.org/abs/2011.04149)
* S. Lee et al., *Evaluating the evidence for exponential quantum advantage in ground-state quantum chemistry*, [arXiv:2208.02199](https://arxiv.org/abs/2208.02199); [Nat. Commun. 14, 1952 (2023)](https://doi.org/10.1038/s41467-023-37587-6)
* N. Yoshioka, T. Okubo, Y. Suzuki, Y. Koizumi, W. Mizukami, *Hunting for quantum-classical crossover in condensed matter problems*, [arXiv:2210.14109](https://arxiv.org/abs/2210.14109)
* P. Jordan, E. Wigner, *Über das Paulische Äquivalenzverbot*, [Z. Phys. 47, 631 (1928)](https://doi.org/10.1007/BF01331938)
* E. Lieb, T. Schultz, D. Mattis, *Two soluble models of an antiferromagnetic chain*, [Ann. Phys. 16, 407 (1961)](https://doi.org/10.1016/0003-4916(61)90115-4)
* G. Vidal, *Efficient simulation of one-dimensional quantum many-body systems*, [arXiv:quant-ph/0310089](https://arxiv.org/abs/quant-ph/0310089); [Phys. Rev. Lett. 93, 040502 (2004)](https://doi.org/10.1103/PhysRevLett.93.040502)
* S. R. White, A. E. Feiguin, *Real time evolution using the density matrix renormalization group*, [arXiv:cond-mat/0403310](https://arxiv.org/abs/cond-mat/0403310); [Phys. Rev. Lett. 93, 076401 (2004)](https://doi.org/10.1103/PhysRevLett.93.076401)
* P. Calabrese, J. Cardy, *Evolution of entanglement entropy in one-dimensional systems*, [arXiv:cond-mat/0503393](https://arxiv.org/abs/cond-mat/0503393)
* F. Verstraete, J. J. García-Ripoll, J. I. Cirac, *Matrix product density operators: simulation of finite-temperature and dissipative systems*, [arXiv:cond-mat/0406426](https://arxiv.org/abs/cond-mat/0406426); [Phys. Rev. Lett. 93, 207204 (2004)](https://doi.org/10.1103/PhysRevLett.93.207204)
* A. E. Feiguin, S. R. White, *Finite-temperature density matrix renormalization using an enlarged Hilbert space*, [arXiv:cond-mat/0510124](https://arxiv.org/abs/cond-mat/0510124); [Phys. Rev. B 72, 220401 (2005)](https://doi.org/10.1103/PhysRevB.72.220401)
* R. Coldea et al., *Quantum criticality in an Ising chain: experimental evidence for emergent E8 symmetry*, [arXiv:1103.3694](https://arxiv.org/abs/1103.3694); [Science 327, 177 (2010)](https://doi.org/10.1126/science.1180085)
* M. M. Wolf, *Violation of the entropic area law for fermions*, [arXiv:quant-ph/0503219](https://arxiv.org/abs/quant-ph/0503219); [Phys. Rev. Lett. 96, 010404 (2006)](https://doi.org/10.1103/PhysRevLett.96.010404)
* A. W. Sandvik, H. G. Evertz, *Loop updates for variational and projector quantum Monte Carlo simulations in the valence-bond basis*, [arXiv:0807.0682](https://arxiv.org/abs/0807.0682); [Phys. Rev. B 82, 024407 (2010)](https://doi.org/10.1103/PhysRevB.82.024407)
* J. P. F. LeBlanc et al., *Solutions of the two-dimensional Hubbard model: benchmarks and results from a wide range of numerical algorithms*, [arXiv:1505.02290](https://arxiv.org/abs/1505.02290); [Phys. Rev. X 5, 041041 (2015)](https://doi.org/10.1103/PhysRevX.5.041041)
* C. Schön, E. Solano, F. Verstraete, J. I. Cirac, M. M. Wolf, *Sequential generation of entangled multiqubit states*, [arXiv:quant-ph/0501096](https://arxiv.org/abs/quant-ph/0501096); [Phys. Rev. Lett. 95, 110503 (2005)](https://doi.org/10.1103/PhysRevLett.95.110503)
* S.-J. Ran, *Encoding of matrix product states into quantum circuits of one- and two-qubit gates*, [arXiv:1908.07958](https://arxiv.org/abs/1908.07958); [Phys. Rev. A 101, 032310 (2020)](https://doi.org/10.1103/PhysRevA.101.032310)
* S. Fomichev et al., *Initial state preparation for quantum chemistry on quantum computers*, [arXiv:2310.18410](https://arxiv.org/abs/2310.18410); [PRX Quantum 5, 040339 (2024)](https://doi.org/10.1103/PRXQuantum.5.040339)
