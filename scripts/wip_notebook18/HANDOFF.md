# Notebook 18 (Kitaev entanglement loops): handoff, 2026-11-05

Branch `worktree-agent-a937c88c3cb4af707` (based on main `f5f0abe`). Not merged, not pushed. **Delete this folder (`scripts/wip_notebook18/`) before merging into main.**

## State

Done and committed (`d0a8ade`, `7d75807`, `102e370`):

* `examples/18_kitaev_entanglement_loops.ipynb`, built by `build_nb18.py` (+ `build_nb18_part2.py`), executed: no errors, 28 s. No placeholders left.
* `results/kitaev_loops.json`: Nauru (24 sites) with ED, 16 sectors per point; Dyck (32) with 4 loop sectors; exact infinite hexagon.
* Package: `kt.torus_loops`, `kitaev_honeycomb(..., sector=, parity=)`, `region_measures` uses MOSEK up to 6 qubits (Clarabel if MOSEK is missing), SCS above. Tests in `tests/test_kitaev.py`.

## What was learned (in the notebook, sections 2-4)

* DMRG treats every conserved quantity like a hidden charge: once a sweep loses a sector, nothing brings it back. On tori this hits the two loop operators and the global parities P_x, P_z. Fix: one run per sector, penalising the others (mu = 0.5).
* With that, Nauru matches ED: g = 0 three ground sectors, RDMs to 2e-13 at chi = 512; g = 1 two sectors, 9e-7 at chi = 256; all GMNs to 4 digits.
* A sector run can miss its sector when its lowest state is > 2 mu above a neighbour's, or when DMRG is not converged; `kitaev_assemble` only picks runs whose measured labels are within 1e-3.
* Measure after `psi.canonical_form()`: a non-canonical final MPS gave wrong loop values and a fake energy below the ground state.
* Dyck does not converge (worst cut 18 bonds against 10 for Nauru): it is the notebook's failure case.
* SCS stalled (> 20 min) on a nearly pure 6-qubit RDM; MOSEK takes about 100 s per six-party GMN.

## Runtime

A fresh `python scripts/make_results.py kitaev_loops`: 43 tasks, 535 CPU-minutes, about an hour on 12 cores; the 4 Dyck g = 0 runs take 44 min each, the 2 EDs 26-34 min and 10 GB each. `make_results.py` keeps nothing if one task fails; the run used a scratch driver with a per-task cache and the same assembly.

## Open for Liuke

1. README row, bullets and references for notebook 18: approved and committed (`9586d90`).
2. Paper Sec. III: after measuring one hexagon spin, the five others have GMN 0 as five single parties (exact infinite RDM, X, Y or Z), positive only for some three-party splits. Exercise 6.
3. Fork GMN is nonzero at g = 1 on small tori (0.054 on 16, 0.041 on 24 sites), against the paper's "loopy only" for the Z2 phase. Exercise 4.

## Before merging

Delete `scripts/wip_notebook18/`, run `pytest -m slow tests/test_notebooks.py` for notebook 18, update the repo's status.
