# Notebook 18 (Kitaev entanglement loops): handoff, 2026-11-04

Work in progress on branch `worktree-agent-a937c88c3cb4af707` (based on main `f5f0abe`). Nothing is merged or pushed. **Delete this folder (`scripts/wip_notebook18/`) before merging into main.**

## The task (from Liuke)

Which quantum-spin-liquid models of Lyu et al., *Multiparty entanglement loops in quantum spin liquids* ([arXiv:2505.18124](https://arxiv.org/abs/2505.18124); Nat. Commun. 17, 5650 (2026)), are easy for DMRG? Build notebook 18 on the easiest, with DMRG checked against ED. Local PDF: `~/Documents/Entanglement/publications/Multiparty_entanglement_loops_QSL.pdf`.

Answer and choice: the difficulty table is the opening cell of `build_nb18.py`. The chosen model is Kitaev's honeycomb at zero field: isotropic (gapless) and anisotropic $g = 1$ (gapped Z2), $K = (1 - g/2, 1 - g/2, 1 + g)$ as in the paper's Fig. 4.

## Done and committed

* `src/dmrg_from_scratch/kitaev.py`:
  * brick-wall honeycomb with a twist; the paper's clusters `NAURU = (6, 4, 2)` (24 sites) and `DYCK = (4, 8, 4)` (32 sites), checked against the LCF notation in `tests/test_kitaev.py`;
  * `kitaev_honeycomb(..., flux_penalty)` and `plaquettes`;
  * the paper's region shapes: `hexagon` (site order of SM Tables II–IV) and `fork` (Fig. 3b);
  * `exact_hexagon_rdm()`, the infinite-lattice hexagon RDM rebuilt from SM Tables II–IV;
  * `region_measures`, GMN with medt (SCS).
* `mpo.pauli_string_mpo`: an MPO for Pauli strings of any length, needed for the six-spin $W_p$ terms.
* `microscopy.left_end_rdm`: the RDM of sites near the open left end of a finite MPS, as θθ†. Fast at χ = 512, whereas `segment_rdm` needs about 10 GB there.
* Tests: `tests/test_kitaev.py` (10 pass). Ruff is clean.
* `scripts/make_results.py`: a `kitaev_loops` entry (functions `kitaev_cluster_run`, `kitaev_tenpy_states`, `kitaev_measures`, `kitaev_exact_tl`). TeNPy DMRG runs on our MPO, with `orthogonal_to` passed as a keyword, not as an option.

## Verified numbers (reproduce them, don't trust blindly)

* **16-site torus `kitaev_honeycomb(4, 4, flux_penalty=0.2)`, g = 0:** our `dmrg` at χ = 128 equals ED to 1e-15.
  * Hexagon GMN (3 pairs) 0.2404; fork, pair and three spins 0 (the loop result).
  * Hexagon 6-party GMN 0.155.
  * Without the penalty the lowest level is 4-fold degenerate across flux sectors.
* **16-site torus, g = 1:** our `dmrg` gets stuck 0.014 above ED (another topological sector). Fork GMN there is 0.054 (exercise 4).
* **Exact infinite lattice (tables):** hexagon GMN 0.2045 (3 pairs), 0.0653 (6 spins); pair and three spins 0. This matches the paper's Fig. 3a and Fig. 4 at h = 0 (≈0.20 / 0.065, read off the plots).
* **Nauru ED (24 sites, about 10 min and 10 GB each):**
  * g = 0: the lowest level is 3-fold degenerate in the flux-free sector, E = −4.764298 without the penalty, so −7.164298 with λ = 0.2 (12 plaquettes).
  * g = 1: the lowest level is 2-fold, E = −8.5998246 with the penalty; the next sector lies 0.02 above.
* **Nauru DMRG energies, g = 0 (TeNPy on our MPO):** χ = 64: −7.098 (3 s); 128: −7.154 (8 s); 256: −7.1630 (34 s); 512: −7.1642984 (109 s), equal to ED. Our own `dmrg`: χ = 256 gives −7.1630 in 158 s.
* **GMN cost:** SCS with 3 pairs takes about 1–5 s; with 6 parties, about 40–150 s each.

## Next steps

1. **Results.** `kitaev_loops` was started on 2026-11-04 and had not finished at handoff. Check whether `results/kitaev_loops.json` exists in the branch's working tree. If not, rerun it:
   `python scripts/wip_notebook18/make_runner.py kitaev_loops`.
   This puts the repo's `src` on `PYTHONPATH` for the pool workers; from a normal checkout with `pip install -e .`, plain `python scripts/make_results.py kitaev_loops` works. Expect about 30–60 min and 2 × 10 GB of RAM.
   * An earlier attempt was slow only because of `segment_rdm` at χ = 512, now fixed with `left_end_rdm`.
   * If it is still too slow, drop the 6-party regions (`hexagon_6`, `fork_6`) from `KITAEV_REGIONS`, or reduce the Dyck runs to χ ≤ 256.
2. **Notebook.** Run `python scripts/wip_notebook18/build_nb18.py` to write `examples/18_kitaev_entanglement_loops.ipynb`. Fill in the placeholders in `build_nb18_part2.py` from the results: `RUNTIME`, `TEXT_AFTER_NAURU`, `TEXT_AFTER_DYCK`, `TEXT_MEANING` (why it is a spin liquid: conserved fluxes, Majoranas, ⟨σ⟩ = 0, only nearest-neighbour bond correlations, Table II; what a loop is: W_p = +1 is a six-body correlation of value 1, and the GME of a loop survives measuring one spin, paper Sec. III) and `TEXT_BUYS`. Execute it with nbconvert (the PYTHONPATH must point at the repo's `src` if this branch is not the installed package), and check that live cells take seconds.
3. **Cite in the notebook:** Lieb's flux theorem, [arXiv:cond-mat/9410025](https://arxiv.org/abs/cond-mat/9410025), [PRL 73, 2158 (1994)](https://doi.org/10.1103/PhysRevLett.73.2158) (verified). Kitaev, [arXiv:cond-mat/0506438](https://arxiv.org/abs/cond-mat/0506438), [Ann. Phys. 321, 2 (2006)](https://doi.org/10.1016/j.aop.2005.10.005) (verified). Also Jungnitsch et al. and Hofmann et al. for GMN, and TeNPy.
4. **Commits:** the notebook goes in a notebook-only commit authored by Liuke (liukelyu@gmail.com), no trailer; the results JSON in a "Scripts:" commit. Then report the README row and bullets to Liuke. No README edits and no push without Liuke's OK.
