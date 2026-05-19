"""
tDMRG for 6×6 TFIM OBC — random Z-basis initial states, three driving protocols.

Loads hx(t) and hz(t) for the GRF, tanh, and Gaussian protocols from INPUT_NPZ
(4by4_results.npz).  For each protocol a fresh random Z-basis product state is
sampled.  tDMRG is run for every (state, protocol) pair and <X>, <Z>, <ZZ>,
entanglement entropy are recorded at every time step.  Everything — initial
state, driving fields, observables — is saved to OUTPUT_PKL.

Usage:
    nohup python dmrg_propagator.py > dmrg_propagator.log 2>&1 &
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
from pathlib import Path

import tenpy
logging.getLogger('tenpy').setLevel(logging.WARNING)

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import SpinHalfSite
from tenpy.models.lattice import Square
from tenpy.networks.mps import MPS
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine

# ── Parameters ────────────────────────────────────────────────────────────────
LX      = 6
LY      = 6
J_ZZ    = 1.0
T_STEPS = 200
T_DMRG  = 200
CHI_MAX = 128
SVD_MIN = 1e-12

STATE_SEED = 69     # PRNG seed for sampling random Z-basis initial states

INPUT_NPZ  = "4by4_results.npz"
OUTPUT_PKL = f"dmrg_6x6_chi{CHI_MAX}_FineTuning.pkl"

PROTOCOLS  = ["GRF", "tanh", "Gaussian"]

# ── Load driving protocols from npz ──────────────────────────────────────────
data = np.load(INPUT_NPZ)
time_grid = data["time_grid"].astype(float)          # (T_STEPS,)

runs = []
for name in PROTOCOLS:
    hx = data[f"{name}_hx"].astype(float)            # (T_STEPS,)
    hz = data[f"{name}_hz"].astype(float)            # (T_STEPS,)
    h_fields = np.stack([hx, hz], axis=1)            # (T_STEPS, 2)
    runs.append((name, h_fields))

print(f"Loaded {INPUT_NPZ}  —  {len(runs)} protocol(s): {[n for n, _ in runs]}")
print(f"  time_grid: {time_grid[0]:.4f} → {time_grid[-1]:.4f}  ({len(time_grid)} steps)")

# ── Sample random Z-basis initial states ─────────────────────────────────────
rng   = np.random.default_rng(STATE_SEED)
L     = LX * LY
bits_list = [rng.integers(0, 2, size=L) for _ in runs]   # 0=up, 1=down

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


def _init_z_product_state(model, bits):
    """Product state in the Z basis. bits[i]=0 → |↑⟩, bits[i]=1 → |↓⟩ (MPS site order)."""
    sites   = model.lat.mps_sites()
    p_state = ['up' if b == 0 else 'down' for b in bits]
    psi     = MPS.from_product_state(sites, p_state, bc='finite', dtype=complex)
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
    ee = psi.entanglement_entropy(n=2)
    return float(ee[len(ee) // 2])


def run_tdmrg(h_fields, bits, t_axis):
    """
    Run tDMRG for a single (protocol, initial state) pair.
    h_fields : (T_DMRG, 2)  columns = [hx, hz]
    bits     : (L,)          0=up, 1=down, in MPS site order
    t_axis   : (T_DMRG,)    time values from the npz time_grid
    Returns t_axis, mx, mz, zz, ent  each (T_DMRG,)
    """
    n_steps = h_fields.shape[0]
    dt_val  = float(t_axis[1] - t_axis[0])

    model_0    = TFIM2D_OBC(dict(Lx=LX, Ly=LY, J_zz=J_ZZ,
                                  hx=float(h_fields[0, 0]),
                                  hz=float(h_fields[0, 1])))
    psi        = _init_z_product_state(model_0, bits)
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


# ── Run all (state, protocol) pairs ──────────────────────────────────────────
dmrg_results = {}

for (name, h_fields), bits in zip(runs, bits_list):
    bitstring = "".join(str(int(b)) for b in bits)
    print(f"\n{'='*60}")
    print(f"Protocol: {name}   initial state: {bitstring}")
    print(f"{'='*60}")

    t_sub = time_grid
    if T_DMRG != T_STEPS:
        idx      = np.round(np.linspace(0, T_STEPS - 1, T_DMRG)).astype(int)
        h_fields = h_fields[idx]
        t_sub    = time_grid[idx]

    t_axis, mx, mz, zz, ent = run_tdmrg(h_fields, bits, t_sub)

    dmrg_results[name] = {
        "bits":      bits.tolist(),
        "bitstring": bitstring,
        "t":         t_axis,
        "hx":        h_fields[:, 0],
        "hz":        h_fields[:, 1],
        "h":         h_fields,
        "mx":        mx,
        "mz":        mz,
        "zz":        zz,
        "ent":       ent,
    }
    print(f"  Done. mx[-1]={mx[-1]:.4f}  mz[-1]={mz[-1]:.4f}  "
          f"zz[-1]={zz[-1]:.4f}  ent[-1]={ent[-1]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
dmrg_results["_meta"] = {
    "chi_max":    CHI_MAX,
    "svd_min":    SVD_MIN,
    "T_DMRG":     T_DMRG,
    "LX":         LX,
    "LY":         LY,
    "state_seed": STATE_SEED,
    "input_npz":  INPUT_NPZ,
    "protocols":  PROTOCOLS,
    "time_grid":  time_grid,
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(dmrg_results, f)

print(f"\nSaved → {OUTPUT_PKL}")
