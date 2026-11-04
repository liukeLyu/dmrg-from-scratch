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


def chis_up_to(top):
    return [c for c in (1, 2, 4, 8, 16, 32, 64, 128, 256) if c <= top]


def tenpy_ladder(psi, model, chis, sweeps, measure):
    """TeNPy two-site DMRG at each chi in turn (each from the last), with the largest truncation error of the
    last sweep and whatever ``measure(psi)`` returns."""
    import time

    from tenpy.algorithms import dmrg as tdmrg

    rows, t0 = [], time.perf_counter()
    for chi in chis:
        opts = {"trunc_params": {"chi_max": chi, "svd_min": 1e-12}, "mixer": chi == chis[0], "max_trunc_err": 1.0}
        opts.update(sweeps)
        eng = tdmrg.TwoSiteDMRGEngine(psi, model, opts)
        E, _ = eng.run()
        trunc = float(eng.sweep_stats["max_trunc_err"][-1])
        rows.append(
            {"chi": chi, "energy": float(E), "trunc_err": trunc, **measure(psi), "seconds": time.perf_counter() - t0}
        )
    return rows


def neel_run(W, Lx, chis):
    """Square-lattice Heisenberg model on an Lx x W cylinder, edges pinned by a staggered field 1/2 (TeNPy, S^z
    conserved): energy and the magnetization profile |<S^z>| column by column."""
    import warnings

    from tenpy.models.spins import SpinModel
    from tenpy.networks.mps import MPS

    warnings.filterwarnings("ignore")
    hz = np.zeros((Lx, W))
    for x in (0, Lx - 1):
        hz[x] = [-0.5 * (-1) ** (x + y) for y in range(W)]  # SpinModel adds -hz S^z: the field of neel_pinning
    model = SpinModel(
        {
            "lattice": "Square",
            "Lx": Lx,
            "Ly": W,
            "bc_x": "open",
            "bc_y": "periodic",
            "bc_MPS": "finite",
            "S": 0.5,
            "Jx": 1.0,
            "Jy": 1.0,
            "Jz": 1.0,
            "hz": hz,
            "conserve": "Sz",
        }
    )
    init = ["down" if (k // W + k % W) % 2 == 0 else "up" for k in range(W * Lx)]
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc="finite")

    def measure(psi):
        return {"M_x": list(np.abs(psi.expectation_value("Sz")).reshape(Lx, W).mean(axis=1))}

    return {"W": W, "Lx": Lx, "runs": tenpy_ladder(psi, model, chis, {"max_sweeps": 2, "min_sweeps": 2}, measure)}


def hubbard_infinite_run(U, phi_ext, chis, W=4):
    """iDMRG for the half-filled Hubbard model on an infinite cylinder of width W (TeNPy, N and S^z conserved).
    ``phi_ext = 0`` is periodic around the cylinder, ``1/2`` antiperiodic. Energies are per site."""
    import warnings

    from tenpy.models.hubbard import FermiHubbardModel
    from tenpy.networks.mps import MPS

    warnings.filterwarnings("ignore")
    model = FermiHubbardModel(
        {
            "lattice": "Square",
            "Lx": 2,
            "Ly": W,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "bc_MPS": "infinite",
            "U": U,
            "t": 1.0,
            "phi_ext": phi_ext,
        }
    )
    init = ["up" if (k // W + k % W) % 2 == 0 else "down" for k in range(2 * W)]
    psi = MPS.from_product_state(model.lat.mps_sites(), init, bc="infinite")
    sweeps = {"max_sweeps": 30, "min_sweeps": 6, "max_E_err": 1e-9, "max_S_err": 1e-5}
    rows = tenpy_ladder(
        psi, model, chis, sweeps, lambda psi: {"S_bits": float(max(psi.entanglement_entropy())) / np.log(2)}
    )
    return {"U": U, "phi_ext": phi_ext, "W": W, "runs": rows}


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
    # notebook 09: pinned Neel order, widths 4, 6, 8 at aspect ratios Lx/W = 1, 1.5, 2 (TeNPy)
    "neel_pinned": (
        [(neel_run, 4, Lx, [64, 128, 256]) for Lx in (4, 6, 8)]
        + [(neel_run, 6, Lx, [64, 128, 256, 512]) for Lx in (6, 9, 12)]
        + [(neel_run, 8, Lx, [128, 256, 512, 768]) for Lx in (8, 12, 16)],
        lambda out: {"model": "square-lattice Heisenberg, staggered edge field 1/2", "clusters": out},
    ),
    # notebook 10: half-filled Hubbard model on the infinite width-4 cylinder, periodic and antiperiodic (TeNPy)
    "hubbard_width4": (
        [(hubbard_infinite_run, U, phi, [128, 256, 512, 768]) for U in (4.0, 8.0) for phi in (0.0, 0.5)],
        lambda out: {"model": "Hubbard, t = 1, half filling, infinite cylinder of width 4", "runs": out},
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
