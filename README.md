# dmrg-from-scratch

DMRG for the transverse-field Ising chain, $H = -\sum_i X_iX_{i+1} - h\sum_i Z_i$, written from scratch with NumPy and SciPy only, as a learning project. The goal is the simplest readable code: every function is short, and every tensor index is named in a comment.

New to DMRG? Start with [`00_start_here`](examples/00_start_here.ipynb): a short, plain-language introduction to why DMRG works (the exponential wall, the area law) that runs it on 60 spins in under a minute.

| Step | What you learn | Module | Notebook |
|---|---|---|---|
| 1 | Matrix product states: SVD per cut, canonical forms, Schmidt values, entanglement, truncation | `dmrg.mps` | [`01_mps_basics`](examples/01_mps_basics.ipynb) |
| 2 | The TFIM as a matrix product operator (a 3-state machine), and exact free-fermion energies | `dmrg.mpo`, `dmrg.exact` | [`02_tfim_mpo`](examples/02_tfim_mpo.ipynb) |
| 3 | Finite two-site DMRG with Lanczos (`scipy.sparse.linalg.eigsh` on a matvec) | `dmrg.finite` | [`03_finite_dmrg`](examples/03_finite_dmrg.ipynb) |
| 4 | Infinite DMRG with McCulloch's wavefunction prediction, transfer-matrix correlation length | `dmrg.infinite` | [`04_idmrg`](examples/04_idmrg.ipynb) |

Each notebook gives the intuition first, then shows the code (often its source, printed), then checks it against an exact answer. Citations are arXiv links; the main references are Schollwöck's review [arXiv:1008.3477](https://arxiv.org/abs/1008.3477) and McCulloch's iDMRG paper [arXiv:0804.2509](https://arxiv.org/abs/0804.2509).

## What the code reproduces

* **Finite DMRG** gets the exact free-fermion ground energy of the open chain to $10^{-12}$ ($L = 100$, $\chi = 64$, $h = 1$), and the energy error tracks the discarded weight when $\chi$ is too small.
* **Entanglement at criticality:** the profile $S(l)$ of the open chain at $h = 1$ fits the Calabrese–Cardy formula with $c \approx 0.53$ ($L = 100$).
* **iDMRG** gets the infinite-chain energy density to machine precision in the gapped phases and to $6\times10^{-7}$ at $h = 1$ with $\chi = 16$. Finite-entanglement scaling $S = (c/6)\log\xi$ over $\chi = 6$–$16$ gives $c \approx 0.51$.
* In the ordered phase iDMRG returns the cat state: $S \approx \log 2$ and a degenerate transfer matrix.

## Comparison with TeNPy

`benchmarks/benchmark_tenpy.py` solves the same problems with [TeNPy](https://github.com/tenpy/tenpy) ([arXiv:1805.00055](https://arxiv.org/abs/1805.00055)). TeNPy is only imported there. One CPU core, $h = 1$:

| Problem | Ours: error, time | TeNPy: error, time |
|---|---|---|
| finite, $L = 40$, $\chi = 32$, 4 sweeps | $1\times10^{-13}$, 4.5 s | $2\times10^{-13}$, 3.3 s |
| finite, $L = 100$, $\chi = 32$, 4 sweeps | $9\times10^{-11}$, 17 s | $8\times10^{-11}$, 11 s |
| finite, $L = 100$, $\chi = 64$, 4 sweeps | $8\times10^{-13}$, 53 s | $7\times10^{-13}$, 16 s |
| infinite, $\chi = 8$ | $1.7\times10^{-5}$, 4.8 s | $1.7\times10^{-5}$, 13 s |
| infinite, $\chi = 16$ | $5.8\times10^{-7}$, 10 s | $5.8\times10^{-7}$, 12 s |

Same accuracy everywhere. TeNPy is up to 3× faster for finite DMRG at large $\chi$; most of that gap is `np.einsum` versus BLAS-backed `np.tensordot` in the effective-Hamiltonian matvec, which we keep for readability.

## Install and test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,examples]"
pip install -e ../multipartite-entanglement-detection-toolkit   # optional: medt, one cross-check of the exact energies
pytest            # about 20 s
pytest -m slow    # also re-runs the notebooks and the iDMRG central-charge fit (a few minutes)
ruff check . && ruff format --check .
pip install -e ".[benchmark]" && python benchmarks/benchmark_tenpy.py   # about 3 minutes
```

Conventions: MPS tensors are `A[left, phys, right]`, MPO tensors `W[w_left, w_right, out, in]`, environments `L[ket, mpo, bra]`; site 0 is the most significant bit of a dense vector, as in `medt`.

Private for now: nothing goes to GitHub without Liuke's OK.
