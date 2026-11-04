"""Full-size runs behind the notebook figures, saved as small JSON summaries in results/.

    python scripts/make_results.py              # every result, spread over all CPU cores
    python scripts/make_results.py critical_entropy

Each notebook runs a seconds-long live version of a result, then loads the full one from results/.
DMRG is deterministic (fixed seed), so the files are reproducible. About 40 minutes in total on a 12-core
laptop, set by the 8 x 8 cylinder of ``dmrg_ladders`` and by ``hubbard_width4`` (about 30 minutes each).
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


QMC_RDMS = (
    "https://raw.githubusercontent.com/songmengh/Ising_RDMs/main/2d/4sites"  # Song et al., ref. 41 of arXiv:2412.12533
)


def microscopy_idmrg(W, h, chis):
    """2 x 2 plaquette RDM of the 2D TFIM on an infinite cylinder of width W (TeNPy iDMRG, parity conserved),
    at each chi in turn, rotated to the -ZZ - hX basis of the QMC papers, with its negativities and GMN."""
    import time

    from tenpy.algorithms import dmrg as tdmrg
    from tenpy.models.tf_ising import TFIModel
    from tenpy.networks.mps import MPS

    from dmrg_from_scratch import microscopy as mic

    model = TFIModel(
        {
            "lattice": "Square",
            "Lx": 2,
            "Ly": W,
            "bc_x": "periodic",
            "bc_y": "periodic",
            "bc_MPS": "infinite",
            "J": 1.0,
            "g": h,
            "conserve": "parity",
        }
    )
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * (2 * W), bc="infinite")
    rows, t0 = [], time.perf_counter()
    for chi in chis:
        opts = {
            "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
            "mixer": True,
            "max_E_err": 1e-9,
            "max_S_err": 1e-5,
            "max_sweeps": 20,
            "max_trunc_err": 1.0,
        }
        E = tdmrg.run(psi, model, opts)["E"]
        left, tensors = mic.tenpy_segment(psi, 0, W + 1)
        rho = mic.to_zz_basis(mic.segment_rdm(tensors, [0, 1, W, W + 1], left)).real
        rows.append(
            {
                "chi": chi,
                "energy": float(E),
                "chi_used": int(max(psi.chi)),
                "rho": rho,
                **mic.plaquette_measures(rho),
                "seconds": time.perf_counter() - t0,
            }
        )
    rows[-1]["rho_change"] = float(np.abs(rows[-1]["rho"] - rows[-2]["rho"]).max())  # last step in chi
    for r in rows[:-1]:
        del r["rho"]
    return {"W": W, "h": h, "runs": rows}


def microscopy_qmc(L, h, n_samples=20, seed=0):
    """Negativities and GMN of the published QMC plaquette RDM (L x L torus, beta = L), with error bars from
    resampling every matrix element within its quoted error."""
    import urllib.request

    from dmrg_from_scratch import microscopy as mic

    def fetch(name):
        return np.loadtxt(urllib.request.urlopen(f"{QMC_RDMS}/L{L}/h{h:g}/{name}"))

    rho, err = fetch("rdm.txt"), fetch("rdm_error.txt")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        noise = np.triu(rng.normal(size=rho.shape) * err)
        try:
            samples.append(mic.plaquette_measures(rho + noise + np.triu(noise, 1).T))
        except RuntimeError:  # both SDP solvers failed on this sample: leave it out
            continue
    best = mic.plaquette_measures(rho)
    return {
        "L": L,
        "h": h,
        "rho": rho,
        "rho_error_max": float(err.max()),
        **best,
        "n_samples": len(samples),
        **{f"{k}_error": float(np.std([s[k] for s in samples])) for k in best},
    }


MICRO_H = [1.0, 2.0, 2.5, 2.8, 3.0, 3.044, 3.2, 3.5, 4.0, 5.0, 6.0, 7.6, 10.7]
MICRO_KEEP = (3.044, 5.0, 10.7)  # fields whose full matrices the notebook uses (width 6 and QMC L = 24)


def microscopy_assemble(out):
    """Compare every matrix with the QMC one at L = 24, then keep only the matrices the notebook needs."""
    qmc24 = {o["h"]: o["rho"] for o in out if o.get("L") == 24}
    for o in out:
        last = o["runs"][-1] if "W" in o else o
        if o.get("L") != 24:
            last["diff_qmc24"] = float(np.abs(last["rho"] - qmc24[o["h"]]).max())
        keep = o["h"] in MICRO_KEEP and (o.get("W") == 6 or o.get("L") == 24)
        last["rho"] = np.round(last["rho"], 10).tolist() if keep else None
        if last["rho"] is None:
            del last["rho"]
    return {
        "model": "H = -sum ZZ - h sum X, square lattice; plaquette (x,y) (x,y+1) (x+1,y) (x+1,y+1)",
        "dmrg": [o for o in out if "W" in o],
        "qmc": [o for o in out if "L" in o],
    }


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
    # notebook 17: plaquette RDMs of the 2D TFIM from iDMRG (widths 4, 6, 8), next to the published QMC RDMs
    "microscopy": (
        [(microscopy_idmrg, W, h, [32, 64, 128]) for W in (4, 6) for h in MICRO_H]
        + [(microscopy_idmrg, 8, h, [64, 128]) for h in (2.0, 2.8, 3.044, 3.5, 5.0)]
        + [(microscopy_qmc, L, h) for L in (16, 24) for h in MICRO_H],
        microscopy_assemble,
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
