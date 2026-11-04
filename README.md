# dmrg-from-scratch

*Why I built this.* I have been around DMRG for years. I know what a matrix product state is and roughly what infinite DMRG does, and I have run the standard packages. But running a package is not the same as understanding it, and I never felt I understood DMRG at a deep level. So I wrote it myself, one small piece at a time, on a model I know well, checking each step against the exact answer.

The density-matrix renormalization group (DMRG), written from scratch in short, readable NumPy and SciPy, on the transverse-field Ising chain $H = -\sum_i X_iX_{i+1} - h\sum_i Z_i$. Every result is checked against the exact free-fermion solution or, on small cylinders, exact diagonalization.

New to DMRG? Start with [`00_start_here`](examples/00_start_here.ipynb): why a laptop can find the ground state of 60 interacting spins (the exponential wall, the area law).

Already know how DMRG works, for example from the [pocket course](docs/pocket-course.html), and want to run it with a well-optimized package? Go straight to steps 5 (TeNPy) and 6 (block2, for molecules).

| Step | What you learn | Notebook |
|---|---|---|
| 1 | Matrix product states: one SVD per cut, canonical forms, Schmidt values, truncation | [`01_mps_basics`](examples/01_mps_basics.ipynb) |
| 2 | The Hamiltonian as a matrix product operator (a 3-state machine); exact free-fermion energies | [`02_tfim_mpo`](examples/02_tfim_mpo.ipynb) |
| 3 | Finite two-site DMRG with a Lanczos eigensolver | [`03_finite_dmrg`](examples/03_finite_dmrg.ipynb) |
| 4 | Infinite DMRG, correlation length and finite-entanglement scaling | [`04_idmrg`](examples/04_idmrg.ipynb) |
| 5 | DMRG in practice with TeNPy: the same objects by their package names, what to check before trusting a result, symmetries | [`05_dmrg_in_practice`](examples/05_dmrg_in_practice.ipynb) |
| 6 | Quantum chemistry with block2: molecules as DMRG chains, breaking the N₂ bond where CCSD(T) fails, orbital order and orbital entanglement (phone companion: [quantum chemistry pocket course](docs/qchem-pocket-course.html)) | [`06_quantum_chemistry_block2`](examples/06_quantum_chemistry_block2.ipynb) |
| 11 | Who wins, DMRG or a quantum computer? Both prices for one ground-state energy, in one call each | [`11_dmrg_vs_qpe`](examples/11_dmrg_vs_qpe.ipynb) |
| 12 | The classical price: the bond dimension needed grows exponentially with cylinder width; flops and seconds | [`12_classical_price`](examples/12_classical_price.ipynb) |
| 13 | The quantum price: block encodings, the one-norm $\lambda$, walk calls, Toffolis, logical qubits, hours (`pennylane.estimator`) | [`13_quantum_price`](examples/13_quantum_price.ipynb) |
| 14 | The hidden cost: phase estimation needs a good starting state, a small DMRG state provides one; where the curves cross | [`14_overlap_and_crossover`](examples/14_overlap_and_crossover.ipynb) |

Steps 11 to 14 ask what the promise "a fault-tolerant quantum computer will beat classical methods for quantum materials" means in numbers, for the frustrated J1-J2 Heisenberg model on a cylinder of width $W$, $H = J_1 \sum_{\langle ij\rangle} \mathbf S_i\cdot\mathbf S_j + J_2 \sum_{\langle\langle ij\rangle\rangle} \mathbf S_i\cdot\mathbf S_j$ at $J_2/J_1 = 0.5$, where quantum Monte Carlo has a sign problem and DMRG is the method to beat:

```python
from dmrg_from_scratch import j1j2_cylinder
from dmrg_from_scratch.resources import qpe_cost

qpe_cost(j1j2_cylinder(6), eps=0.01)   # 6 x 6 cylinder, 36 spins
# {'lambda': 72.0, 'walk_calls': 11310, 'toffoli': 14624742, 'T': 2002748,
#  'logical_qubits': 202, 'repetitions': 1.0, 'seconds': 2826.7}      <- 47 minutes
```

DMRG does the same 36 spins in seconds. The quantum computer only wins once the cylinder is wide enough for entanglement to make DMRG explode.

Where next: [`docs/ROADMAP.md`](docs/ROADMAP.md) maps what else it takes to use DMRG on real materials and molecules; [`IDEAS.md`](IDEAS.md) lists stages planned for later.

## What you will see

* Finite DMRG reaches the exact ground energy of the open chain to $10^{-12}$ ($L = 100$, $\chi = 64$, $h = 1$), and when $\chi$ is too small the energy error tracks the discarded weight.
* At the critical point $h = 1$ the entanglement profile $S(l)$ follows the Calabrese–Cardy formula with central charge $c \approx 0.53$ ($L = 100$; exact value $1/2$).
* iDMRG gets the infinite-chain energy density to machine precision in the gapped phases and to $6\times10^{-7}$ at $h = 1$ with $\chi = 16$. Finite-entanglement scaling $S = (c/6)\log_2\xi$ gives $c \approx 0.51$.
* In the ordered phase, finite DMRG settles on one symmetry-broken state ($S \approx 0$), while iDMRG with a symmetric Hamiltonian returns the cat state ($S \approx 1$ bit, degenerate transfer matrix).
* On J1-J2 cylinders DMRG needs bond dimension $\chi \approx 18$, $110$ and $800$ for $W = 4, 6, 8$: a factor of about 35 in run time per unit of width.
* Qubitized phase estimation grows only like $N^{1.8}$ in Toffolis: $1.5\times10^7$ Toffolis on about 200 logical qubits for 36 spins, $10^9$ on about 700 for 400 spins.
* A product state has overlap $\gamma^2 \approx 3\times10^{-9}$ with the 64-spin ground state; an MPS with $\chi_0 = 32$ keeps $\gamma^2 \approx 0.35$.
* The curves cross between $W = 6$ and $W = 10$ (36 to 100 spins), where phase estimation takes minutes to a day, depending on how fast the quantum computer makes Toffolis and how fast the classical computer runs.

## Install

```bash
pip install git+https://github.com/liukeLyu/multipartite-entanglement-detection-toolkit   # optional, install first: medt, an independent check of the exact energies
pip install -e ".[dev,examples]"
pytest
```

Each notebook cell runs in seconds. Results that need longer runs appear twice: a small live run you can change, then the full run loaded from a small file in `results/`. `python scripts/make_results.py` remakes those files (about two hours on 4 cores, set by the 8 x 8 cylinder of steps 12 and 14; the rest takes about 3 minutes).

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
* S. Lee et al., *Evaluating the evidence for exponential quantum advantage in ground-state quantum chemistry*, [arXiv:2208.02199](https://arxiv.org/abs/2208.02199); [Nat. Commun. (2023)](https://doi.org/10.1038/s41467-023-37587-6)
* N. Yoshioka, T. Okubo, Y. Suzuki, Y. Koizumi, W. Mizukami, *Hunting for quantum-classical crossover in condensed matter problems*, [arXiv:2210.14109](https://arxiv.org/abs/2210.14109)
