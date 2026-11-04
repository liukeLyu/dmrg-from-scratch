# From DMRG on a chain to DMRG on real materials and molecules

**The question:** after notebooks 00–06 a reader knows how DMRG works and can run TeNPy and block2. What else do they need before they can use DMRG to learn something new about a material or a molecule, and trust the answer?

2026-11-04. Every reference below was checked on arXiv or the journal page; entries marked *(to confirm)* were not.

## What the repo covers now, and what is missing

| Skill | Now | Missing |
|---|---|---|
| Ground state of a 1D model, checked against an exact answer | 01–04 (TFIM, free fermions) | — |
| Running a package, symmetries, convergence checks | 05 (TeNPy) | extrapolation in the discarded weight; when a run is stuck |
| Molecules as chains | 06 (block2: N₂, orbital order, orbital entanglement) | active-space choice, dynamic correlation, transition metals |
| Two dimensions | 12 (J1–J2 cylinders: χ against width) | pinning fields, order parameters, width extrapolation, fermions in 2D |
| Anything beyond the ground state | — | excitations, real-time dynamics, spectral functions, finite temperature |
| From a real material to a model | — | downfolding: DFT, Wannier orbitals, effective Hamiltonian |
| Judging DMRG against other methods | partly (06: CCSD(T) fails on N₂) | benchmarks across methods; where DMRG loses |
| The link to quantum computing | 11–14 (DMRG against phase estimation, starting-state overlap) | the MPS starting state compiled to a circuit |

Proficiency means more than one item in the right-hand column: it means being able to say how wrong a number might be. In practice that rests on four habits, which run through every stage below:

1. **Extrapolate, don't report.** Run a sequence of bond dimensions, plot the energy against the discarded weight, extrapolate to zero, and quote the extrapolation as the error bar. This is standard practice for 2D DMRG ([Stoudenmire & White](https://arxiv.org/abs/1105.1374)) and for ab initio DMRG ([Olivares-Amaya et al.](https://doi.org/10.1063/1.4905329)).
2. **Know the cost before you run.** Cost grows with χ³, and the χ needed grows with the entanglement across the worst cut: exponentially in cylinder width, and with how badly the orbitals are ordered in a molecule.
3. **Check against something independent.** An exact answer when the system is small, another method (QMC, coupled cluster, AFQMC, DMET) when it is not.
4. **Know the physics question.** The ground-state energy is rarely the goal: the goal is an order parameter, a gap, a spin-state ordering, a spectrum that a measurement can see.

## The journey, in stages

Each stage is a notebook, numbered as in `examples/`, and keeps the repo's rule: a small run with an exact answer first, then the real problem. Two further stages, real molecules and the step from a material to a model, are in [`IDEAS.md`](../IDEAS.md) for later.

### 07. Beyond the ground state: excitations and dynamics (stay on the TFIM)

*Why:* experiments measure spectra (neutrons, ARPES, optics), not ground-state energies. The TFIM is a good place to learn this because its dynamics are exact (free fermions), and it is realized in a real material: CoNb₂O₆ in a transverse field shows the E₈ bound-state spectrum near criticality ([Coldea et al.](https://arxiv.org/abs/1103.3694); [Science 327, 177 (2010)](https://doi.org/10.1126/science.1180085)).

*Learn:* excited states (DMRG with orthogonality to lower states); real-time evolution with TEBD and TDVP; the dynamical structure factor S(k, ω) from a time-evolved local excitation; why entanglement growth limits the reachable time.

*Check:* a quench and S(k, ω) of the TFIM against free fermions.

*Read:* [Paeckel et al., time-evolution methods for MPS](https://arxiv.org/abs/1901.05824) (the practical review); [Haegeman et al., TDVP](https://arxiv.org/abs/1408.5056) (DMRG as TDVP with infinite imaginary time step); [Ganahl et al., Chebyshev MPS](https://arxiv.org/abs/1403.1209) (spectral functions without time evolution).

### 08. Finite temperature

*Why:* most experiments run at finite temperature (specific heat, susceptibility), and finite-T tensor networks are how one compares with them.

*Learn:* purification (an MPS for the thermal density matrix on a doubled chain) versus METTS (sampling minimally entangled states).

*Check:* the TFIM free energy and energy density against free fermions at all temperatures.

*Read:* [White, METTS](https://arxiv.org/abs/0902.4475); [Stoudenmire & White, METTS algorithms](https://arxiv.org/abs/1002.1305).

### 09. Two dimensions and fermions

*Why:* almost every open question in quantum materials (spin liquids, stripes and superconductivity in the Hubbard model) is 2D. DMRG on cylinders is one of the main methods where QMC has a sign problem.

*Learn:* mapping a cylinder onto a chain and why χ must grow exponentially with width; boundary pinning fields to read off an order parameter; width extrapolation; fermionic signs in the MPO; U(1) × SU(2) symmetries, which become essential rather than optional.

*Check:* a 4 × 4 or 2-leg Hubbard ladder against exact diagonalization; the 2D Hubbard energy at half filling against the cross-method benchmark.

*Landmark results to reproduce in miniature:* the kagome spin liquid on cylinders up to circumference 12 ([Yan, Huse & White](https://arxiv.org/abs/1011.6114); [Science 332, 1173 (2011)](https://doi.org/10.1126/science.1201080)); stripes at 1/8 doping, where DMRG, iPEPS, AFQMC and DMET agree ([Zheng et al.](https://arxiv.org/abs/1701.00054); [Science 358, 1155 (2017)](https://doi.org/10.1126/science.aam7127)).

*Read:* [Stoudenmire & White, 2D DMRG](https://arxiv.org/abs/1105.1374); [Hauschild & Pollmann, TeNPy notes](https://arxiv.org/abs/1805.00055) (abelian symmetries in practice); [Hubig et al., single-site DMRG with subspace expansion](https://arxiv.org/abs/1501.05504) (the algorithm modern codes run); [ITensor](https://arxiv.org/abs/2007.14822).

### 10. Calibration: what other methods say

*Why:* a DMRG number in a paper is persuasive only next to other methods. Knowing where DMRG wins and where it loses (wide 2D systems, dynamic correlation in large basis sets) is part of being proficient.

*Learn:* read the 2D Hubbard benchmarks across ~10 methods and see where each one fails ([LeBlanc et al.](https://arxiv.org/abs/1505.02290); [PRX 5, 041041 (2015)](https://doi.org/10.1103/PhysRevX.5.041041)); the consensus and open questions for the Hubbard model ([Qin et al.](https://arxiv.org/abs/2104.00064)).

### 11–15. DMRG and the quantum computer

*Why:* in industry, DMRG has two jobs on the quantum side. It is the classical baseline a quantum algorithm must beat, and it supplies the starting state for phase estimation.

*Learn:* the evidence for and against exponential quantum advantage in ground-state chemistry, built largely on DMRG baselines ([Lee et al.](https://arxiv.org/abs/2208.02199); [Nat. Commun. 14, 1952 (2023)](https://doi.org/10.1038/s41467-023-37587-6) *(article number to confirm)*); preparing initial states for phase estimation, including from MPS ([Fomichev et al.](https://arxiv.org/abs/2310.18410); [PRX Quantum 5, 040339 (2024)](https://doi.org/10.1103/PRXQuantum.5.040339)); compiling an MPS into one- and two-qubit gates ([Ran](https://arxiv.org/abs/1908.07958)).

*Done in notebooks 11–14* (merged from `dmrg-vs-qpe`): both prices, the starting-state overlap and the crossover width. *New, notebook 15:* compile the DMRG starting state into a circuit in PennyLane and check its overlap exactly. Related: `qchem-from-scratch` (the Hartree–Fock overlap collapses on stretched hydrogen chains).

## Where each stage lives

| Notebook | Topic | Exact check |
|---|---|---|
| 07 | Excitations and dynamics | TFIM free fermions |
| 08 | Finite temperature | TFIM free fermions |
| 09 | Two dimensions and fermions | ED on small clusters |
| 10 | Calibration against other methods | published benchmarks |
| 11–14 | DMRG versus phase estimation (merged from `dmrg-vs-qpe`) | ED on small cylinders |
| 15 | MPS starting state as a circuit | overlap computed exactly on small systems |

Notebooks 07, 08 and 15 build the skills that matter most in a quantum-algorithms role: time evolution is Hamiltonian simulation, and MPS state preparation feeds phase estimation.
