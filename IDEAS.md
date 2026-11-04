# Ideas for later

Stages of the [roadmap](docs/ROADMAP.md) that are planned but not yet started. Each would become one or more notebooks, with the same rule as the rest of the repo: a small run checked against an exact answer first, then the real problem.

## Molecules for real

*Why:* notebook 06 shows the idea on N₂. Real chemistry problems (transition-metal complexes, enzyme active sites) add three questions: which orbitals go in the active space, in which order, and how to add the dynamic correlation of everything left out.

*Learn:*
* active-space selection from orbital entropies of a cheap DMRG run ([Stein & Reiher](https://arxiv.org/abs/1602.03835));
* orbital ordering and its effect on χ (begun in 06);
* spin adaptation (SU(2)) for spin-state energetics;
* DMRG plus perturbation theory for dynamic correlation, e.g. DMRG-NEVPT2 on the chromium dimer ([Guo et al.](https://arxiv.org/abs/1512.08137));
* excited states and properties (spin densities, gaps).

*Check:* a small transition-metal model against CASSCF/FCI from PySCF; then a published benchmark.

*Landmark results:* the Mn₄CaO₅ cluster of photosystem II ([Kurashige, Chan & Yanai, Nature Chem. 5, 660 (2013)](https://doi.org/10.1038/nchem.1677)); the low-energy spectrum of [2Fe-2S] and [4Fe-4S] clusters, with many more low-lying states than the model Hamiltonians used before ([Sharma et al.](https://arxiv.org/abs/1408.5080); [Nature Chem. 6, 927 (2014)](https://doi.org/10.1038/nchem.2041)).

*Read:* [Chan & Sharma, Annu. Rev. Phys. Chem. 62, 465 (2011)](https://doi.org/10.1146/annurev-physchem-032210-103338); [Olivares-Amaya et al., ab initio DMRG in practice](https://doi.org/10.1063/1.4905329) (the user's guide); [Baiardi & Reiher, recent developments](https://arxiv.org/abs/1910.00137); [block2](https://arxiv.org/abs/2310.03920).

## From a material to a model

*Why:* DMRG does not take a crystal structure as input. For a material, someone first derives a lattice model (which orbitals, which hoppings, how big U is). The answer DMRG gives is only as good as that model.

*Learn:* the pipeline DFT → Wannier orbitals → effective interactions (constrained RPA) → lattice model → DMRG; how sensitive conclusions are to the parameters; DMRG as an impurity solver inside DMFT ([Ganahl et al.](https://arxiv.org/abs/1403.1209)).

*Read:* Y. Chang et al., *Downfolding from ab initio to interacting model Hamiltonians: comprehensive analysis and benchmarking of the DFT+cRPA approach*, [arXiv:2311.05987](https://arxiv.org/abs/2311.05987). This stage needs a further literature search before it is taught.

