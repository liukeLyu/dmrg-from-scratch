"""Compare our from-scratch DMRG with TeNPy (https://arxiv.org/abs/1805.00055) on the TFIM.

Run with:  python benchmarks/benchmark_tenpy.py   (needs pip install physics-tenpy)

Both codes solve H = -sum X_i X_{i+1} - h sum Z_i at h = 1 (TeNPy's TFIChain with J = 1, g = h).
We report the energy error against the exact free-fermion result and the wall time.
TeNPy is used only here, as a reference; the package itself never imports it.
"""

import logging
import time
import warnings

import numpy as np
from tenpy.algorithms import dmrg as tenpy_dmrg
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

import dmrg

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


def tenpy_finite(L, h, chi, n_sweeps):
    model = TFIChain({"L": L, "J": 1.0, "g": h, "bc_MPS": "finite", "conserve": None})
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * L, bc="finite")
    params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
        "mixer": None,
        "min_sweeps": n_sweeps,
        "max_sweeps": n_sweeps,
    }
    E, psi = tenpy_dmrg.run(psi, model, params)["E"], psi
    return E


def tenpy_infinite(h, chi, n_sweeps):
    model = TFIChain({"L": 2, "J": 1.0, "g": h, "bc_MPS": "infinite", "conserve": None})
    psi = MPS.from_product_state(model.lat.mps_sites(), ["up"] * 2, bc="infinite")
    params = {
        "trunc_params": {"chi_max": chi, "svd_min": 1e-12},
        "mixer": None,
        "min_sweeps": n_sweeps,
        "max_sweeps": n_sweeps,
        "max_E_err": 0,
    }
    E = tenpy_dmrg.run(psi, model, params)["E"]
    S_bits = float(psi.entanglement_entropy()[0]) / np.log(2)  # TeNPy uses the natural log; we use bits
    return E, S_bits, float(psi.correlation_length())


def timed(f, *args, **kwargs):
    t = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t


def main():
    h = 1.0
    print("Finite open chain, h = 1, 4 sweeps")
    print(f"{'L':>4} {'chi':>4} | {'ours: dE':>10} {'time':>7} | {'TeNPy: dE':>10} {'time':>7}")
    for L, chi in [(40, 32), (100, 32), (100, 64)]:
        E_exact = dmrg.tfim_open_energy(L, h)
        ours, t_ours = timed(dmrg.dmrg, dmrg.tfim_mpo(L, h), chi, n_sweeps=4)
        E_tp, t_tp = timed(tenpy_finite, L, h, chi, 4)
        dE, dE_tp = ours["energy"] - E_exact, E_tp - E_exact
        print(f"{L:>4} {chi:>4} | {dE:>10.1e} {t_ours:>6.1f}s | {dE_tp:>10.1e} {t_tp:>6.1f}s")

    print("\nInfinite chain, h = 1 (energy per site)")
    e_exact = dmrg.tfim_infinite_energy_density(h)
    cols = f"{'S':>7} {'xi':>7} {'time':>7}"
    print(f"{'chi':>4} | {'ours: de':>10} {cols} | {'TeNPy: de':>10} {cols}")
    for chi in [8, 16]:
        res, t_ours = timed(dmrg.idmrg, dmrg.tfim_bulk_tensor(h), chi, n_steps=1500)
        xi = dmrg.correlation_length(res)
        (e_tp, S_tp, xi_tp), t_tp = timed(tenpy_infinite, h, chi, 300)
        print(
            f"{chi:>4} | {res['energy'] - e_exact:>10.1e} {res['entropies'][-1]:>7.4f} {xi:>7.1f} {t_ours:>6.1f}s"
            f" | {e_tp - e_exact:>10.1e} {S_tp:>7.4f} {xi_tp:>7.1f} {t_tp:>6.1f}s"
        )


if __name__ == "__main__":
    np.random.seed(0)
    main()
