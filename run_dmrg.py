"""
tDMRG for 4×8 TFIM OBC — three driving protocols.
Loads hx/hz from 4by8_inference_results.pkl, runs tDMRG for each protocol,
measures <X>, <Z>, <ZZ> at every time step, saves all results to one pkl.

Usage:
    nohup python run_dmrg.py > dmrg.log 2>&1 &
"""

import os
N_THREADS = "64"
os.environ["OMP_NUM_THREADS"]      = N_THREADS
os.environ["MKL_NUM_THREADS"]      = N_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = N_THREADS
os.environ["NUMEXPR_NUM_THREADS"]  = N_THREADS

import numpy as np
import pickle
import logging
import time

import tenpy
logging.getLogger('tenpy').setLevel(logging.WARNING)

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Square
from tenpy.networks.mps import MPS
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

# ── Parameters ────────────────────────────────────────────────────────────────
LX      = 4
LY      = 8
J_ZZ    = 1.0
T_STEPS = 200      # time steps in input arrays
T_DMRG  = 200      # tDMRG steps (equal → no subsampling)
CHI_MAX = 128
SVD_MIN = 1e-12

INPUT_PKL  = "4by8_inference_results.pkl"
OUTPUT_PKL = f"4by8_dmrg_results_chi{CHI_MAX}.pkl"

# ── Load driving protocols ────────────────────────────────────────────────────
with open(INPUT_PKL, "rb") as f:
    inference_results = pickle.load(f)

protocol_names = list(inference_results.keys())
print(f"Loaded {INPUT_PKL}  —  protocols: {protocol_names}")

# ── TeNPy model ───────────────────────────────────────────────────────────────
class TFIM2D_OBC(CouplingMPOModel):
    def init_sites(self, model_params):
        return SpinHalfSite(conserve='None')

    def init_lattice(self, model_params):
        Lx   = model_params.get('Lx', 4)
        Ly   = model_params.get('Ly', 4)
        site = self.init_sites(model_params)
        return Square(Lx, Ly, site, bc='open', bc_MPS='finite')

    def init_terms(self, model_params):
        J  = model_params.get('J_zz', J_ZZ)
        hx = model_params.get('hx',   1.0)
        hz = model_params.get('hz',   0.0)
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-4.0 * J, u1, 'Sz', u2, 'Sz', dx)
        self.add_onsite(-2.0 * hx, 0, 'Sx')
        self.add_onsite(-2.0 * hz, 0, 'Sz')


def _build_bond_pairs(model, Lx, Ly):
    bond_pairs = []
    for y in range(Ly):
        for x in range(Lx):
            if x + 1 < Lx:
                i = model.lat.lat2mps_idx((x,     y, 0))
                j = model.lat.lat2mps_idx((x + 1, y, 0))
                bond_pairs.append((min(i, j), max(i, j)))
            if y + 1 < Ly:
                i = model.lat.lat2mps_idx((x, y,     0))
                j = model.lat.lat2mps_idx((x, y + 1, 0))
                bond_pairs.append((min(i, j), max(i, j)))
    return bond_pairs


def _init_plus_state(model, Lx, Ly):
    L      = Lx * Ly
    sites  = model.lat.mps_sites()
    B_plus = np.ones((2, 1, 1), dtype=complex) / np.sqrt(2.0)
    B_list = [B_plus.copy() for _ in range(L)]
    SVs    = [np.ones(1)] * (L + 1)
    psi = MPS.from_Bflat(sites, B_list, SVs=SVs, bc='finite', dtype=complex)
    psi.canonical_form()
    return psi


def _measure_mx(psi):
    """<X> per spin  (σ^x = 2 S^x)"""
    return 2.0 * float(np.mean(np.array(psi.expectation_value('Sx'), dtype=float)))


def _measure_mz(psi):
    """<Z> per spin  (σ^z = 2 S^z)"""
    return 2.0 * float(np.mean(np.array(psi.expectation_value('Sz'), dtype=float)))


def _measure_zz(psi, bond_pairs):
    """<ZZ> per bond  (σ^z_i σ^z_j = 4 S^z_i S^z_j)"""
    total = sum(
        psi.expectation_value_term([('Sz', i), ('Sz', j)])
        for i, j in bond_pairs
    )
    return 4.0 * float(np.real(total)) / len(bond_pairs)

def _measure_ent(psi):
    """Half-chain Rényi-2 entanglement entropy."""
    ee = psi.entanglement_entropy(n=2)   # n=2 → Rényi-2, matches NOQS SWAP estimator
    return float(ee[len(ee) // 2])


def run_tdmrg(h_fields):
    """
    Run tDMRG for a single protocol.
    h_fields : (T_DMRG, 2)  columns = [hx, hz]
    Returns t_axis, mx, mz, zz  each (T_DMRG,)
    """
    n_steps = h_fields.shape[0]
    T_MAX   = 0.5
    t_axis  = np.linspace(0.0, T_MAX, n_steps)
    dt_val  = T_MAX / (n_steps - 1)

    model_0    = TFIM2D_OBC(dict(Lx=LX, Ly=LY, J_zz=J_ZZ,
                                  hx=float(h_fields[0, 0]),
                                  hz=float(h_fields[0, 1])))
    psi        = _init_plus_state(model_0, LX, LY)
    bond_pairs = _build_bond_pairs(model_0, LX, LY)

    tdvp_params = {
        'trunc_params': {'chi_max': CHI_MAX, 'svd_min': SVD_MIN},
        'dt': dt_val,
        'N_steps': 1,
    }

    mx_list, mz_list, zz_list, ent_list = [], [], [], []
    t_start = time.time()

    for step in range(n_steps):
        mx_list.append(_measure_mx(psi))
        mz_list.append(_measure_mz(psi))
        zz_list.append(_measure_zz(psi, bond_pairs))
        ent_list.append(_measure_ent(psi))

        elapsed  = time.time() - t_start
        time_str = (f"  elapsed={elapsed/60:.1f}m  eta={elapsed/step*(n_steps-step)/60:.1f}m"
                    if step > 0 else "")
        print(f"  step {step+1:3d}/{n_steps}  "
              f"mx={mx_list[-1]:.4f}  mz={mz_list[-1]:.4f}  zz={zz_list[-1]:.4f}  "
              f"max_chi={max(psi.chi)}{time_str}", flush=True)

        if step < n_steps - 1:
            model_t = TFIM2D_OBC(dict(Lx=LX, Ly=LY, J_zz=J_ZZ,
                                       hx=float(h_fields[step, 0]),
                                       hz=float(h_fields[step, 1])))
            engine = TwoSiteTDVPEngine(psi, model_t, tdvp_params)
            engine.run()

    return t_axis, np.array(mx_list), np.array(mz_list), np.array(zz_list), np.array(ent_list)


# ── Run all three protocols ───────────────────────────────────────────────────
dmrg_results = {}

for name in protocol_names:
    print(f"\n{'='*60}")
    print(f"Protocol: {name}")
    print(f"{'='*60}")

    h_fields = inference_results[name]["h"]   # (T_STEPS, 2)

    # subsample if T_DMRG differs from T_STEPS
    if T_DMRG != T_STEPS:
        idx      = np.round(np.linspace(0, T_STEPS - 1, T_DMRG)).astype(int)
        h_fields = h_fields[idx]

    t_axis, mx, mz, zz, ent = run_tdmrg(h_fields)

    dmrg_results[name] = {
        "t":   t_axis,
        "h":   h_fields,        # (T_DMRG, 2)
        "mx":  mx,
        "mz":  mz,
        "zz":  zz,
        "ent": ent,
    }
    print(f"  Done. mx[-1]={mx[-1]:.4f}  mz[-1]={mz[-1]:.4f}  zz[-1]={zz[-1]:.4f}  ent[-1]={ent[-1]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
dmrg_results["_meta"] = {"chi_max": CHI_MAX, "svd_min": SVD_MIN,
                          "T_DMRG": T_DMRG, "LX": LX, "LY": LY}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(dmrg_results, f)

print(f"\nSaved → {OUTPUT_PKL}")
