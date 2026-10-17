# dmrg-from-scratch

The density-matrix renormalization group (DMRG), written from scratch in short, readable NumPy and SciPy, on the transverse-field Ising chain $H = -\sum_i X_iX_{i+1} - h\sum_i Z_i$. Every result is checked against the exact free-fermion solution.

New to DMRG? Start with [`00_start_here`](examples/00_start_here.ipynb): why a laptop can find the ground state of 60 interacting spins (the exponential wall, the area law).

| Step | What you learn | Notebook |
|---|---|---|
| 1 | Matrix product states: one SVD per cut, canonical forms, Schmidt values, truncation | [`01_mps_basics`](examples/01_mps_basics.ipynb) |
| 2 | The Hamiltonian as a matrix product operator (a 3-state machine); exact free-fermion energies | [`02_tfim_mpo`](examples/02_tfim_mpo.ipynb) |
| 3 | Finite two-site DMRG with a Lanczos eigensolver | [`03_finite_dmrg`](examples/03_finite_dmrg.ipynb) |
| 4 | Infinite DMRG, correlation length and finite-entanglement scaling | [`04_idmrg`](examples/04_idmrg.ipynb) |

## What you will see

* Finite DMRG reaches the exact ground energy of the open chain to $10^{-12}$ ($L = 100$, $\chi = 64$, $h = 1$), and when $\chi$ is too small the energy error tracks the discarded weight.
* At the critical point $h = 1$ the entanglement profile $S(l)$ follows the Calabrese–Cardy formula with central charge $c \approx 0.53$ ($L = 100$; exact value $1/2$).
* iDMRG gets the infinite-chain energy density to machine precision in the gapped phases and to $6\times10^{-7}$ at $h = 1$ with $\chi = 16$. Finite-entanglement scaling $S = (c/6)\log\xi$ gives $c \approx 0.51$.
* In the ordered phase, finite DMRG settles on one symmetry-broken state ($S \approx 0$), while iDMRG with a symmetric Hamiltonian returns the cat state ($S \approx \log 2$, degenerate transfer matrix).

## Install

```bash
pip install git+https://github.com/liukeLyu/multipartite-entanglement-detection-toolkit   # optional, install first: medt, an independent check of the exact energies
pip install -e ".[dev,examples]"
pytest
```

`benchmarks/benchmark_tenpy.py` solves the same problems with [TeNPy](https://github.com/tenpy/tenpy) as a cross-check (same accuracy).

Index conventions: MPS tensors `A[left, phys, right]`, MPO tensors `W[w_left, w_right, out, in]`, environments `L[ket, mpo, bra]`.

## References

* S. R. White, *Density matrix formulation for quantum renormalization groups*, [Phys. Rev. Lett. 69, 2863 (1992)](https://doi.org/10.1103/PhysRevLett.69.2863)
* U. Schollwöck, *The density-matrix renormalization group in the age of matrix product states*, [arXiv:1008.3477](https://arxiv.org/abs/1008.3477)
* I. P. McCulloch, *Infinite size density matrix renormalization group, revisited*, [arXiv:0804.2509](https://arxiv.org/abs/0804.2509)
* P. Calabrese, J. Cardy, *Entanglement entropy and quantum field theory*, [arXiv:hep-th/0405152](https://arxiv.org/abs/hep-th/0405152)
* F. Pollmann, S. Mukerjee, A. M. Turner, J. E. Moore, *Theory of finite-entanglement scaling at one-dimensional quantum critical points*, [arXiv:0812.2903](https://arxiv.org/abs/0812.2903)
