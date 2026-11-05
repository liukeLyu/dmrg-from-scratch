"""DMRG from scratch: MPS, the TFIM as an MPO, finite DMRG and iDMRG, with NumPy and SciPy only.

Spin models on cylinders (``models``) and an MPO for any of them (``build_mpo``) serve the
two-dimensional notebooks. ``dmrg_from_scratch.resources`` prices DMRG against quantum phase
estimation; it needs PennyLane, so it is not imported here.
"""

from .exact import central_charge_fit, ground_state, tfim_infinite_energy_density, tfim_open_energy
from .fermions import free_chain_energy, hubbard_mpo
from .finite import dmrg, dmrg_ladder
from .infinite import correlation_length, idmrg
from .models import PauliSum, cylinder_bonds, hubbard_cylinder, j1j2_cylinder, neel_pinning, tfim_cylinder
from .mpo import build_mpo, expectation, mpo_to_dense, tfim_bulk_tensor, tfim_mpo
from .mps import (
    compress,
    entropy,
    from_dense,
    left_canonicalize,
    normalize,
    overlap,
    product_expectation,
    product_mps,
    random_mps,
    right_canonicalize,
    schmidt_values,
    site_expectations,
    to_dense,
    truncated_svd,
)

__all__ = [
    "build_mpo",
    "central_charge_fit",
    "compress",
    "correlation_length",
    "cylinder_bonds",
    "dmrg",
    "dmrg_ladder",
    "entropy",
    "expectation",
    "free_chain_energy",
    "from_dense",
    "ground_state",
    "hubbard_cylinder",
    "hubbard_mpo",
    "idmrg",
    "j1j2_cylinder",
    "left_canonicalize",
    "mpo_to_dense",
    "neel_pinning",
    "normalize",
    "overlap",
    "PauliSum",
    "product_expectation",
    "product_mps",
    "random_mps",
    "right_canonicalize",
    "schmidt_values",
    "site_expectations",
    "tfim_bulk_tensor",
    "tfim_cylinder",
    "tfim_infinite_energy_density",
    "tfim_mpo",
    "tfim_open_energy",
    "to_dense",
    "truncated_svd",
]

# Notebooks 07 and 08: excited states, time evolution and temperature.
from .dynamics import (  # noqa: E402
    apply_local,
    dynamical_correlations,
    gates,
    half_chain_entropy,
    infinite_temperature_state,
    local_expectations,
    mpo_expectation,
    purify_mpo,
    structure_factor,
    tebd_step,
    tfim_bond_hamiltonians,
    thermal_states,
    transition_elements,
)
from .exact import (  # noqa: E402
    majorana_entropy,
    tfim_ground_correlations,
    tfim_open_gap,
    tfim_quench,
    tfim_single_particle_energies,
    tfim_thermal,
    tfim_zz_dynamics,
)
from .excited import dmrg_excited  # noqa: E402

__all__ += [
    "apply_local",
    "dmrg_excited",
    "dynamical_correlations",
    "gates",
    "half_chain_entropy",
    "infinite_temperature_state",
    "local_expectations",
    "majorana_entropy",
    "mpo_expectation",
    "purify_mpo",
    "structure_factor",
    "tebd_step",
    "tfim_bond_hamiltonians",
    "tfim_ground_correlations",
    "tfim_open_gap",
    "tfim_quench",
    "tfim_single_particle_energies",
    "tfim_thermal",
    "tfim_zz_dynamics",
    "thermal_states",
    "transition_elements",
]

# Notebook 18: Kitaev honeycomb clusters and an MPO for longer Pauli strings
from .kitaev import kitaev_honeycomb  # noqa: E402
from .mpo import pauli_string_mpo  # noqa: E402

__all__ += ["kitaev_honeycomb", "pauli_string_mpo"]
