"""Two prices for one ground-state energy: DMRG on a classical computer, phase estimation on a quantum one.

The classical price
-------------------
The classical price: how big a bond dimension DMRG needs for a given accuracy, and what it costs.

An MPS with bond dimension ``chi`` can hold at most ``log2 chi`` bits of entanglement across a
cut. On a cylinder of width ``W`` the ground state obeys an area law, ``S ~ a W``, so the bond
dimension needed for a fixed accuracy grows like ``2^(a W)`` and the cost, ``chi^3``, like
``2^(3 a W)``. Everything below turns a few DMRG runs into numbers for that statement:

* ``reference_energy``: the energy at zero truncation, from the standard linear extrapolation of
  ``E(chi)`` against the discarded weight [White and Chernyshev, arXiv:0705.2746].
* ``chi_needed``: fit the error ``E(chi) - E_ref`` to a power law in ``chi`` and solve for ``eps``.
* ``dmrg_seconds``: flops of a sweep scale as ``chi^3``; times a number of sweeps, at a given
  machine speed.

The quantum price
-----------------
The quantum price: ground-state energy by qubitized phase estimation, in Toffolis and in hours.

Recipe (as in ``pennylane-hamsim``):

1. Write ``H = sum_k c_k P_k`` and block-encode ``H / lambda`` with PREP (prepare the weights
   ``|c_k| / lambda`` by coherent alias sampling) and SELECT (apply ``P_k`` controlled on ``k``).
   The qubitized walk operator then has eigenphases ``arccos(E / lambda)``
   [Low and Chuang, arXiv:1610.06546].
2. Phase estimation reads ``E`` to precision ``eps`` with about ``pi lambda / (2 eps)`` walk calls
   [Babbush et al., arXiv:1805.03662].
3. Phase estimation returns the ground energy only with probability ``gamma^2``, the overlap
   squared between the starting state and the ground state, so it is repeated ``1 / gamma^2``
   times on average.
4. In a surface-code computer the Clifford gates are cheap and the time is set by the
   non-Clifford gates, one Toffoli (or T) at a time from a magic-state factory.

``pennylane.estimator`` does steps 1 and 2 from the shape of ``H`` alone (which Pauli words, how
many, and ``lambda``), so it costs milliseconds even for thousands of spins.

Side by side
------------
Put the two prices side by side and find the width where the quantum computer wins.

Inputs are the DMRG ladders of ``scripts/make_results.py`` (one per cylinder width). For each
width we get the classical time from the bond dimension needed for precision ``eps``, and the
quantum time from ``qpe_cost`` with the overlap of a cheap MPS starting state. Both are then
extrapolated to wider cylinders, where only the quantum estimate can still be computed directly:

* classical: ``log(seconds)`` linear in ``W`` (the area law),
* overlap: ``log(gamma)`` linear in the number of spins ``N`` (an orthogonality catastrophe).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pennylane.estimator as qre

from .finite import dmrg_ladder
from .models import PauliSum, j1j2_cylinder
from .mpo import build_mpo

RESULTS = Path(__file__).resolve().parents[2] / "results"


# The classical price


def run_ladder(H: PauliSum, chis=(4, 8, 16, 32), n_sweeps: int = 4) -> list[dict]:
    """DMRG on ``H`` at each bond dimension in ``chis``: summaries with the half-cylinder ``entropy``, no MPS."""
    keys = ("chi", "energy", "discarded", "flops", "flops_per_sweep", "seconds")
    runs = dmrg_ladder(build_mpo(H), list(chis), n_sweeps=n_sweeps)
    return [{**{k: r[k] for k in keys}, "entropy": r["entropies"][H.n // 2 - 1]} for r in runs]


def reference_energy(runs: list[dict], n_fit: int = 3) -> float:
    """Extrapolate ``E`` linearly in the discarded weight to zero, using the ``n_fit`` largest ``chi``."""
    pts = runs[-n_fit:]
    dw = np.array([r["discarded"] for r in pts])
    E = np.array([r["energy"] for r in pts])
    if np.ptp(dw) < 1e-14:
        return float(E[-1])
    return float(np.polyfit(dw, E, 1)[1])


def error_fit(runs: list[dict], E_ref: float, n_fit: int = 3, floor: float = 1e-10) -> tuple[float, float]:
    """Fit ``E(chi) - E_ref = A chi^(-b)`` on the ``n_fit`` largest ``chi``; returns ``(A, b)``."""
    pts = [r for r in runs if r["energy"] - E_ref > floor][-n_fit:]
    x = np.log([r["chi"] for r in pts])
    y = np.log([r["energy"] - E_ref for r in pts])
    slope, intercept = np.polyfit(x, y, 1)
    return float(np.exp(intercept)), float(-slope)


def chi_needed(runs: list[dict], eps: float, E_ref: float | None = None, n_fit: int = 3) -> float:
    """Bond dimension at which the DMRG energy error reaches ``eps`` (extrapolated if needed)."""
    E_ref = reference_energy(runs) if E_ref is None else E_ref
    A, b = error_fit(runs, E_ref, n_fit)
    return float((A / eps) ** (1 / b))


def dmrg_seconds(runs: list[dict], chi: float, n_sweeps: int = 10, flops_per_second: float = 1e12) -> float:
    """Time for ``n_sweeps`` sweeps at bond dimension ``chi``, scaling the largest run's flops by ``chi^3``."""
    last = runs[-1]
    flops = last["flops_per_sweep"] * (chi / last["chi"]) ** 3
    return float(n_sweeps * flops / flops_per_second)


# The quantum price

# Seconds per Toffoli in a superconducting surface code. Babbush et al. (arXiv:2011.04149) take
# a 1 us code cycle, distance 30 and 5.5 d cycles per Toffoli from one factory: about 170 us.
# Running many factories in parallel is faster by a factor of about 10 to 100.
SECONDS_PER_TOFFOLI = {"one factory": 170e-6, "many factories": 1.7e-6}


def walk_operator(H: PauliSum, prep_precision: float = 1e-6) -> qre.Qubitization:
    """Qubitized walk operator: alias-sampling PREP and ``SelectPauli`` for SELECT."""
    pauli = qre.PauliHamiltonian(num_qubits=H.n, pauli_terms=H.pauli_counts(), one_norm=H.one_norm)
    return qre.Qubitization(qre.AliasSampling(len(H.terms), precision=prep_precision), qre.SelectPauli(pauli))


def _non_clifford(res) -> tuple[int, int]:
    gates = {str(k): v for k, v in res.gate_counts.items()}
    return int(gates.get("Toffoli", 0)), int(gates.get("T", 0))


def qpe_cost(H: PauliSum, eps: float = 0.01, overlap: float = 1.0, seconds_per_toffoli: float = 170e-6) -> dict:
    """Cost of estimating the ground energy of ``H`` to ``eps`` by qubitized QPE.

    ``overlap`` is ``gamma = |<start|ground>|``. Returns the walk calls, the Toffoli and T counts of
    one phase estimation, the logical qubits, the expected number of repetitions ``1 / gamma^2``
    and the expected wall-clock ``seconds``, counting every T as one more Toffoli (a slight
    overestimate, since a T is cheaper to distill than a Toffoli).
    """
    W = walk_operator(H)
    calls = int(np.ceil(np.pi * H.one_norm / (2 * eps)))
    toffoli, t = _non_clifford(qre.estimate(qre.UnaryIterationQPE(W, num_iterations=calls)))
    # one controlled walk step holds every wire the computation needs at once
    wires = qre.estimate(qre.Controlled(W, num_ctrl_wires=1, num_zero_ctrl=0)).total_wires
    qubits = wires - 1 + int(np.ceil(np.log2(calls + 1)))
    reps = 1.0 / overlap**2
    return {
        "lambda": H.one_norm,
        "walk_calls": calls,
        "toffoli": toffoli,
        "T": t,
        "logical_qubits": int(qubits),
        "repetitions": reps,
        "seconds": reps * (toffoli + t) * seconds_per_toffoli,
    }


# Side by side


def load_ladders(path=None) -> list[dict]:
    """The DMRG ladders saved by ``python scripts/make_results.py dmrg_ladders``."""
    path = RESULTS / "dmrg_ladders.json" if path is None else Path(path)
    return json.loads(Path(path).read_text())["ladders"]


def best_energy(ladder: dict) -> float:
    """Exact energy when the cylinder is small enough, otherwise the extrapolated one."""
    if "exact_energy" in ladder:
        return ladder["exact_energy"]
    return reference_energy([r for r in ladder["runs"] if r["discarded"] > 0])


def classical_cost(ladder: dict, eps: float = 0.01, n_sweeps: int = 10, flops_per_second: float = 1e12) -> dict:
    """Bond dimension DMRG needs for energy error ``eps``, and the time for ``n_sweeps`` sweeps."""
    runs = [r for r in ladder["runs"] if r["chi"] >= 8]
    chi = chi_needed(runs, eps, E_ref=best_energy(ladder))
    return {
        "W": ladder["W"],
        "n": ladder["n"],
        "chi": chi,
        "seconds": dmrg_seconds(runs, chi, n_sweeps=n_sweeps, flops_per_second=flops_per_second),
    }


def overlap_of(ladder: dict, chi0: int) -> float:
    """Overlap of the bond-dimension-``chi0`` MPS with the ground state (exact one when available)."""
    (row,) = [r for r in ladder["runs"] if r["chi"] == chi0]
    return row.get("overlap_with_exact", row["overlap_with_best"])


def fit_line(x, y) -> tuple[float, float]:
    """Least-squares ``y = a + b x``; returns ``(a, b)``."""
    b, a = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(a), float(b)


def crossover(
    ladders,
    eps=0.01,
    chi0=32,
    seconds_per_toffoli=170e-6,
    flops_per_second=1e12,
    widths=range(4, 17, 2),
    fit_widths=(6, 8),
) -> dict:
    """Classical and quantum time against width, and the first width where the quantum one is smaller.

    Only even widths are used for the fits: on an odd-width cylinder the Neel pattern cannot close
    around the circumference, which makes a different and more entangled problem. The default fit
    leaves out W = 4, where DMRG is still in its cheap small-system regime.
    """
    fit = [lad for lad in ladders if lad["W"] in fit_widths]
    cl = [classical_cost(lad, eps, flops_per_second=flops_per_second) for lad in fit]
    a_c, b_c = fit_line([c["W"] for c in cl], np.log([c["seconds"] for c in cl]))
    # -log(gamma) per spin, fitted through the origin (gamma = 1 for zero spins)
    n = np.array([lad["n"] for lad in fit], float)
    loss = -np.log([overlap_of(lad, chi0) for lad in fit])
    kappa = float(n @ loss / (n @ n))

    widths = np.asarray(list(widths))
    classical = np.exp(a_c + b_c * widths)
    quantum = []
    for W in widths:
        H = j1j2_cylinder(int(W))
        q = qpe_cost(H, eps, overlap=np.exp(-kappa * H.n), seconds_per_toffoli=seconds_per_toffoli)
        quantum.append(q["seconds"])
    quantum = np.array(quantum)
    wins = widths[quantum < classical]
    return {
        "widths": widths,
        "classical": classical,
        "quantum": quantum,
        "W_star": int(wins[0]) if len(wins) else None,
        "measured": cl,
        "kappa": kappa,
        "classical_fit": (a_c, b_c),
    }
