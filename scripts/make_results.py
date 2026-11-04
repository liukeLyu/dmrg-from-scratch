"""Full-size runs behind the notebook figures, saved as small JSON summaries in results/.

    python scripts/make_results.py              # every result, spread over all CPU cores
    python scripts/make_results.py critical_entropy

Each notebook runs a seconds-long live version of a result, then loads the full one from results/.
DMRG is deterministic (fixed seed), so the files are reproducible. About two hours in total on 4 cores,
set by the 8 x 8 cylinder of ``dmrg_ladders``; everything else takes about 3 minutes.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import dmrg_from_scratch as dmrg

RESULTS = Path(__file__).resolve().parent.parent / "results"


def finite_run(L, h, chi, n_sweeps):
    r = dmrg.dmrg(dmrg.tfim_mpo(L, h), chi_max=chi, n_sweeps=n_sweeps)
    return {"h": h, "energy": r["energy"], "exact": dmrg.tfim_open_energy(L, h), "entropies": list(r["entropies"])}


def infinite_run(h, chi, n_steps):
    r = dmrg.idmrg(dmrg.tfim_bulk_tensor(h), chi_max=chi, n_steps=n_steps)
    return {
        "chi": chi,
        "S": r["entropies"][-1],
        "xi": dmrg.correlation_length(r),
        "de": r["energy"] - dmrg.tfim_infinite_energy_density(h),
    }


def ladder_run(W, chis, n_sweeps, J2=0.5):
    """DMRG at growing bond dimension on the W-wide J1-J2 cylinder, with overlaps against the best state."""
    H = dmrg.j1j2_cylinder(W, J2=J2)
    runs = dmrg.dmrg_ladder(dmrg.build_mpo(H), chis, n_sweeps=n_sweeps)
    best = runs[-1]["mps"]
    out = {"W": W, "n": H.n, "one_norm": H.one_norm, "runs": []}
    psi0 = None
    if H.n <= 20:
        out["exact_energy"], psi0 = dmrg.ground_state(H)
    for r in runs:
        row = {k: r[k] for k in ("chi", "energy", "discarded", "flops_per_sweep", "seconds")}
        row["entropy"] = r["entropies"][H.n // 2 - 1]  # the cut through the middle of the cylinder
        row["overlap_with_best"] = abs(float(dmrg.overlap(r["mps"], best)))
        if psi0 is not None:
            row["overlap_with_exact"] = float(abs(dmrg.to_dense(r["mps"]) @ psi0))
        out["runs"].append(row)
    return out


def quench_run(L, h, chi, dt, t_max):
    """Quench from all up: half-chain entropy and <Z> in the middle against time, TEBD at bond dimension chi."""
    mps = [A.astype(complex) for A in dmrg.product_mps([np.array([1.0, 0.0])] * L)]
    half = dmrg.gates(dmrg.tfim_bond_hamiltonians(L, h), dt / 2)
    n = round(t_max / dt)
    out = {"chi": chi, "t": [], "S": [], "z_mid": [], "discarded": []}
    disc = 0.0
    for k in range(n + 1):
        if k % round(0.1 / dt) == 0:
            out["t"].append(k * dt)
            out["S"].append(dmrg.half_chain_entropy(mps))
            out["z_mid"].append(dmrg.local_expectations(mps, np.diag([1.0, -1.0]))[L // 2])
            out["discarded"].append(disc)
        if k < n:
            disc = max(disc, dmrg.tebd_step(mps, half, chi))
    return out


def zz_spectrum_run(L, h, chi, dt, t_max, n_k, omegas):
    """S^zz(k, w) of the ground state from TEBD, and the largest deviation of C(r, t) from free fermions."""
    g = dmrg.dmrg(dmrg.tfim_mpo(L, h), chi_max=chi, n_sweeps=4)
    half = dmrg.gates(dmrg.tfim_bond_hamiltonians(L, h), dt / 2)
    c, every = L // 2, round(0.1 / dt)
    steps, C = dmrg.dynamical_correlations(g["mps"], np.diag([1.0, -1.0]), c, half, round(t_max / dt), chi, every)
    exact = dmrg.tfim_zz_dynamics(L, h, c, steps * dt)
    ks = np.linspace(0, np.pi, n_k)
    S = dmrg.structure_factor(C, every * dt, c, ks, omegas)
    return {
        "ks": ks.tolist(),
        "omegas": list(omegas),
        "S": np.round(S, 5).tolist(),
        "max_error_C": float(np.max(np.abs(C - exact))),
    }


def thermal_run(L, h, chi, dbeta, betas):
    """Energy and half-chain entropy of the purified thermal state against beta."""
    mpo = dmrg.purify_mpo(dmrg.tfim_mpo(L, h))
    out = {"chi": chi, "beta": [], "energy": [], "S": [], "discarded": []}
    for beta, mps, disc in dmrg.thermal_states(L, h, betas, chi, dbeta):
        out["beta"].append(beta)
        out["energy"].append(dmrg.mpo_expectation(mps, mpo))
        out["S"].append(dmrg.half_chain_entropy(mps))
        out["discarded"].append(disc)
    return out


def chis_up_to(top):
    return [c for c in (1, 2, 4, 8, 16, 32, 64, 128, 256) if c <= top]


# Each result: (list of tasks, function assembling the task outputs into a JSON-ready dict).
RESULTS_SPEC = {
    # notebook 00: 60 spins at three fields, chi = 24
    "start_L60": (
        [(finite_run, 60, h, 24, 3) for h in [0.5, 1.0, 2.0]],
        lambda out: {"L": 60, "chi_max": 24, "n_sweeps": 3, "runs": out},
    ),
    # notebook 03: entanglement profile of L = 100 at h = 1 (chi = 32) and in the gapped phases (chi = 16)
    "critical_entropy": (
        [(finite_run, 100, 1.0, 32, 3), (finite_run, 100, 0.5, 16, 3), (finite_run, 100, 1.5, 16, 3)],
        lambda out: {"L": 100, "chi_max": [32, 16, 16], "n_sweeps": 3, "runs": out},
    ),
    # notebook 04: finite-entanglement scaling at h = 1, 1000 iDMRG steps per chi
    "finite_entanglement_scaling": (
        [(infinite_run, 1.0, chi, 1000) for chi in [6, 8, 10, 12, 16]],
        lambda out: {"h": 1.0, "n_steps": 1000, "runs": out},
    ),
    # notebook 07: entanglement growth after a quench, L = 64, h = 1.5, at four bond dimensions
    "quench_entropy": (
        [(quench_run, 64, 1.5, chi, 0.05, 12.0) for chi in [4, 8, 16, 64]],
        lambda out: {"L": 64, "h": 1.5, "dt": 0.05, "runs": out},
    ),
    # notebook 07: S^zz(k, w) of 64 spins at h = 1.5, chi = 48, up to t = 14
    "zz_spectrum": (
        [(zz_spectrum_run, 64, 1.5, 48, 0.05, 14.0, 33, np.linspace(0, 12, 241))],
        lambda out: {"L": 64, "h": 1.5, "chi": 48, "dt": 0.05, "t_max": 14.0, **out[0]},
    ),
    # notebook 08: purified thermal states of 32 spins at h = 1, beta up to 10, three bond dimensions
    "thermal": (
        [(thermal_run, 32, 1.0, chi, 0.05, np.round(np.arange(0.1, 10.001, 0.1), 6)) for chi in [8, 16, 64]],
        lambda out: {"L": 32, "h": 1.0, "dbeta": 0.05, "runs": out},
    ),
    # notebooks 12 and 14: DMRG ladders on W x W cylinders, J2/J1 = 0.5
    "dmrg_ladders": (
        [
            (ladder_run, 3, chis_up_to(64), 6),
            (ladder_run, 4, chis_up_to(256), 5),
            (ladder_run, 5, chis_up_to(256), 4),
            (ladder_run, 6, chis_up_to(256), 3),
            (ladder_run, 8, chis_up_to(256), 3),
        ],
        lambda out: {"model": "J1-J2 Heisenberg, J2/J1 = 0.5, nearly square cylinders", "ladders": out},
    ),
}


def run(task):
    return task[0](*task[1:])


def main(names):
    names = names or list(RESULTS_SPEC)
    tasks = [(name, k, t) for name in names for k, t in enumerate(RESULTS_SPEC[name][0])]
    with ProcessPoolExecutor() as pool:
        outputs = dict(zip([(n, k) for n, k, _ in tasks], pool.map(run, [t for _, _, t in tasks]), strict=True))
    RESULTS.mkdir(exist_ok=True)
    for name in names:
        out = [outputs[name, k] for k in range(len(RESULTS_SPEC[name][0]))]
        data = {"made_by": f"python scripts/make_results.py {name}", **RESULTS_SPEC[name][1](out)}
        (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, default=float) + "\n")
        print("wrote", RESULTS / f"{name}.json")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(sys.argv[1:])
