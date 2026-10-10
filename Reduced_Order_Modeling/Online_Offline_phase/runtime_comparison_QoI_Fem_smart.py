"""
Comparing the execution speed / Quantities of Interest
FOM with precomputed affine operators (Xu_1..Xu_5, B)  vs  ROM
"""

import os
import sys
import time
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(os.path.dirname(script_dir))

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from Solvers.Exchanger_Device import compute_U_P_solution_exchanger_device
from Utilities.Mesh_processing import refine
from Utilities.Plot_functions import Plot_Initial_Refined_meshes
from Utilities.ROM import solve_ROM
from Utilities.Stokes_felib import calculate_pressure_B

RUN_SCRATCH_FOM = False

#────────────────────────────────────────────────────────────────────────────────────────────────
# Paths
#────────────────────────────────────────────────────────────────────────────────────────────────

data_online = os.path.join(fem_dir, 'Reduced_Order_Modeling', 'Online_Offline_phase', 'Data')
data_rom    = os.path.join(fem_dir, 'Reduced_Order_Modeling', 'No_Online_Offline_phase', 'Data')

#────────────────────────────────────────────────────────────────────────────────────────────────
# Regions of interest (FOM-side QoI helpers)
#────────────────────────────────────────────────────────────────────────────────────────────────

omega_1 = np.array([[-2.00, 1/3], [-1.05, 1/3], [-1.05, 1.0], [-2.00, 1.0], [-2.00, 1/3]])
omega_2 = np.array([[-0.95, -1/3], [-0.05, -1/3], [-0.05, 1/3], [-0.95, 1/3], [-0.95, -1/3]])
omega_3 = np.array([[0.05, -1/3], [0.95, -1/3], [0.95, 1/3], [0.05, 1/3], [0.05, -1/3]])
omega_4 = np.array([[1.05, 1/3], [2.00, 1/3], [2.00, 1.0], [1.05, 1.0], [1.05, 1/3]])
regions = [omega_1, omega_2, omega_3, omega_4]

fom_region_data = {}
for reg_id in [1, 2, 3, 4]:
    glob_ids = np.load(os.path.join(data_rom, f'Glob_id{reg_id}.npy'))
    M_mat    = load_npz(os.path.join(data_rom, f'Mass{reg_id}_p.npz'))

    reg_poly = regions[reg_id - 1]
    width  = reg_poly[:, 0].max() - reg_poly[:, 0].min()
    height = reg_poly[:, 1].max() - reg_poly[:, 1].min()
    area   = width * height

    fom_region_data[reg_id] = {'glob_ids': glob_ids,
                               'weight_vec': (M_mat @ np.ones(len(glob_ids))) / area}


def compute_fom_region_avg_fast(field_full, region_id):
    glob_ids   = fom_region_data[region_id]['glob_ids']
    weight_vec = fom_region_data[region_id]['weight_vec']
    return np.dot(field_full[glob_ids], weight_vec)

#────────────────────────────────────────────────────────────────────────────────────────────────
# Mesh, ROM matrices & mu test set
#────────────────────────────────────────────────────────────────────────────────────────────────

mesh_path = os.path.join(fem_dir, 'Meshes', 'exchanger_device_altered_mesh_data.npz')
p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(data_path=mesh_path,
                                                           num_of_refinements=3,
                                                           plot=False,
                                                           figsize=(16, 4))
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

mu_test_set = np.load(os.path.join(data_online, 'mu_test_set.npy'))

basis_data = np.load(os.path.join(data_online, 'basis_fields.npz'))
psi_field = basis_data['psi_field']

Reduced_Affine_Operators = np.load(os.path.join(data_rom, 'Reduced_Affine_Operators.npy'), allow_pickle=True)
Vu = np.load(os.path.join(data_rom, 'Vu.npy'))
Vp = np.load(os.path.join(data_rom, 'Vp.npy'))
Div_N = np.load(os.path.join(data_rom, 'Div_N.npy'))
g_N = np.load(os.path.join(data_rom, 'g_N.npy'))
Reduced_lifting_vectors = np.load(os.path.join(data_rom, 'Reduced_lifting_vectors.npy'))
lf = np.load(os.path.join(data_rom, 'lifting.npy'))

#────────────────────────────────────────────────────────────────────────────────────────────────
# QoI operators (ROM side)
#────────────────────────────────────────────────────────────────────────────────────────────────

E_p = load_npz(os.path.join(data_rom, 'E_pressure.npz'))
E_u = load_npz(os.path.join(data_rom, 'E_velocity.npz'))
E_uxy = sp.block_diag([E_u, E_u])

E_pN = np.load(os.path.join(data_rom, 'E_pN.npy'))
E_uN = np.load(os.path.join(data_rom, 'E_uN.npy'))

with open(os.path.join(data_online, 'reduced_qoi_vectors.pkl'), 'rb') as f:
    reduced_vectors = pickle.load(f)

sensor_lift = E_uxy @ lf

#────────────────────────────────────────────────────────────────────────────────────────────────
# FOM WITH PRECOMPUTED OPERATORS (offline part)
#────────────────────────────────────────────────────────────────────────────────────────────────

t_offline_start = time.perf_counter()

Nv = p_fine.shape[0]
Np = p_coarse.shape[0]
n_params = psi_field.shape[1]
eps = 1e-10

# ── STEP 1: five Dirichlet-enforced Xu_k = blockdiag(A_k, A_k) and B_h^T ────────────────────────
Xu_k_list = []
BT_D = None

for k in range(n_params):
    _, _, _, Xu_k, B_hT_k = compute_U_P_solution_exchanger_device(
        p_fine, t_fine, e_fine, p_coarse, t_coarse,
        alpha=3,
        kinematic_viscosity=np.ascontiguousarray(psi_field[:, k]),
        return_matrices=True
    )
    Xu_k_list.append(Xu_k.tocsr())

    if BT_D is None:
        BT_D = B_hT_k.tocsr()

print(f"STEP 1 done: {n_params} x Xu_k {Xu_k_list[0].shape}, B_hT {BT_D.shape}")

# ── STEP 2: B = [Bx By] (no Dirichlet enforcement) + pressure pinning, precomputed ──────────────
xmin = p_fine[:, 0].min()
xmax = p_fine[:, 0].max()
inlet_idx  = np.where(np.abs(p_fine[:, 0] - xmin) < eps)[0]
outlet_idx = np.where(np.abs(p_fine[:, 0] - xmax) < eps)[0]
boundary_nodes  = np.unique(e_fine[e_fine[:, 2] > 0, 0:2])
v_wall_idx      = np.setdiff1d(boundary_nodes, np.concatenate([inlet_idx, outlet_idx]))
dirichlet_nodes = np.unique(np.concatenate([inlet_idx, v_wall_idx]))
dirichlet_rows  = np.concatenate([dirichlet_nodes, dirichlet_nodes + Nv])

# pressure pin: same node as in the solver
p_ref_candidates = np.where(np.abs(p_coarse[:, 0] - xmax) < eps)[0]
if len(p_ref_candidates) == 0:
    p_ref_candidates = [np.argmin(np.abs(p_coarse[:, 0] - xmax))]
p_pin = int(p_ref_candidates[0])

Bx, By = calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse)
B_full = sp.hstack([Bx, By], format='csr')

p_mask = np.ones(Np)
p_mask[p_pin] = 0.0
B_pinned = (sp.diags(p_mask) @ B_full).tocsr()                                 # pin row erased
Zero_pp = sp.csr_matrix(([1.0], ([p_pin], [p_pin])), shape=(Np, Np))           # zero block, only the pin entry = 1

print(f"STEP 2 done: B {B_pinned.shape}, pinned pressure node {p_pin}")

# ── STEP 3: lifting contributions to the RHS, using the five stored blocks ──────────────────────
R_lift = np.column_stack([Xu_k @ lf for Xu_k in Xu_k_list])
R_lift[dirichlet_rows, :] = 0.0

G_lift = -(B_full @ lf)
G_lift[p_pin] = 0.0

print(f"STEP 3 done: R_lift {R_lift.shape}, G_lift {G_lift.shape}")

t_offline = time.perf_counter() - t_offline_start
print(f"FOM offline precomputation total: {t_offline:.1f} s\n")


# ── STEP 4: online FOM function ─────────────────────────────────────────────────────────────────
def solve_FOM_affine(mu):

    Xu = mu[0] * Xu_k_list[0]
    for mu_k, Xu_k in zip(mu[1:], Xu_k_list[1:]):
        Xu = Xu + mu_k * Xu_k

    K = sp.bmat([[Xu,       BT_D   ],
                 [B_pinned, Zero_pp]], format='csc')

    rhs = np.concatenate([-(R_lift @ mu), G_lift])

    sol = spsolve(K, rhs)

    ux = sol[:Nv]          + lf[:Nv]
    uy = sol[Nv:2 * Nv]    + lf[Nv:]
    p  = sol[2 * Nv:]
    return ux, uy, p

#────────────────────────────────────────────────────────────────────────────────────────────────
# Check: Affine FOM <===> Standard Solver
#────────────────────────────────────────────────────────────────────────────────────────────────

mu_warmup = mu_test_set[0]
ux_ref, uy_ref, p_ref_sol = compute_U_P_solution_exchanger_device(
    p_fine, t_fine, e_fine, p_coarse, t_coarse, 3, psi_field @ mu_warmup)
ux_aff, uy_aff, p_aff = solve_FOM_affine(mu_warmup)

chk_u = (np.linalg.norm(np.hstack([ux_aff - ux_ref, uy_aff - uy_ref]))
         / np.linalg.norm(np.hstack([ux_ref, uy_ref])))
chk_p = np.linalg.norm(p_aff - p_ref_sol) / np.linalg.norm(p_ref_sol)
print(f"Affine FOM vs standard solver | rel. diff  u: {chk_u:.2e}   p: {chk_p:.2e}")
assert chk_u < 1e-8 and chk_p < 1e-8, "Affine FOM does not match the standard solver."

#────────────────────────────────────────────────────────────────────────────────────────────────
# Warm-up
#────────────────────────────────────────────────────────────────────────────────────────────────

num_params = len(mu_test_set)
rom_repeats_per_param = 50

fem_times, rom_times, scratch_times = [], [], []

print("Starting benchmark warm-up (untimed)...")
_ = solve_FOM_affine(mu_warmup)
_ = solve_ROM(mu_warmup, Reduced_Affine_Operators, Vu, Vp, Div_N, g_N, Reduced_lifting_vectors, lf,
              return_reduced_solution_only=True)
print(f"Warm-up complete. Starting timed runs for {num_params} parameters...\n")

#────────────────────────────────────────────────────────────────────────────────────────────────
# Runtime test loop & error calculation
#────────────────────────────────────────────────────────────────────────────────────────────────

error_sensors_p, error_sensors_u = [], []
error_p_o2, error_p_o3, error_u_o1, error_u_o4 = [], [], [], []

for i, mu_unseen in enumerate(tqdm(mu_test_set)):

    # ── FOM (precomputed operators) ─────────────────────────────────────────────────────────
    start_fem = time.perf_counter()
    ux_fom, uy_fom, p_fom = solve_FOM_affine(mu_unseen)
    u_fom = np.hstack([ux_fom, uy_fom])
    p_sensor_vals_fom = E_p @ p_fom                         # SENSORS
    u_sensor_vals_fom = E_uxy @ u_fom
    p_avg_fom_o2 = compute_fom_region_avg_fast(p_fom, 2)    # AVG P
    p_avg_fom_o3 = compute_fom_region_avg_fast(p_fom, 3)
    u_avg_fom_o1 = np.sqrt(compute_fom_region_avg_fast(ux_fom, 1)**2 + compute_fom_region_avg_fast(uy_fom, 1)**2)  # AVG U
    u_avg_fom_o4 = np.sqrt(compute_fom_region_avg_fast(ux_fom, 4)**2 + compute_fom_region_avg_fast(uy_fom, 4)**2)
    fem_time = time.perf_counter() - start_fem
    fem_times.append(fem_time)

    # ── FOM from scratch (optional, timing only) ────────────────────────────────────────────
    if RUN_SCRATCH_FOM:
        start_scr = time.perf_counter()
        _ = compute_U_P_solution_exchanger_device(p_fine, t_fine, e_fine, p_coarse, t_coarse, 3, psi_field @ mu_unseen)
        scratch_times.append(time.perf_counter() - start_scr)

    # ── ROM ─────────────────────────────────────────────────────────────────────────────────
    start_rom = time.perf_counter()
    for _ in range(rom_repeats_per_param):
        a_u, a_p = solve_ROM(mu_unseen, Reduced_Affine_Operators, Vu, Vp, Div_N, g_N,
                             Reduced_lifting_vectors, lf, return_reduced_solution_only=True)
        p_sensor_vals_rom = E_pN @ a_p                      # SENSORS
        u_sensor_vals_rom = sensor_lift + E_uN @ a_u
        p_avg_rom_o2 = a_p @ reduced_vectors[2]['p']        # AVG P
        p_avg_rom_o3 = a_p @ reduced_vectors[3]['p']
        u_avg_rom_o1 = np.sqrt((a_u @ reduced_vectors[1]['ux'])**2 + (a_u @ reduced_vectors[1]['uy'])**2)  # AVG U
        u_avg_rom_o4 = np.sqrt((a_u @ reduced_vectors[4]['ux'])**2 + (a_u @ reduced_vectors[4]['uy'])**2)
    rom_time = (time.perf_counter() - start_rom) / rom_repeats_per_param
    rom_times.append(rom_time)

    tqdm.write(f"Param {i+1}/{num_params} | FEM: {fem_time:.4f}s | ROM: {rom_time:.6f}s | "
               f"Speedup: {fem_time/rom_time:.2f}x")

    # ── Relative errors ─────────────────────────────────────────────────────────────────────
    error_sensors_p.append(np.linalg.norm(p_sensor_vals_fom - p_sensor_vals_rom) / (np.linalg.norm(p_sensor_vals_fom) + 1e-12))
    error_sensors_u.append(np.linalg.norm(u_sensor_vals_fom - u_sensor_vals_rom) / (np.linalg.norm(u_sensor_vals_fom) + 1e-12))
    error_p_o2.append(abs(p_avg_fom_o2 - p_avg_rom_o2) / (abs(p_avg_fom_o2) + 1e-12))
    error_p_o3.append(abs(p_avg_fom_o3 - p_avg_rom_o3) / (abs(p_avg_fom_o3) + 1e-12))
    error_u_o1.append(abs(u_avg_fom_o1 - u_avg_rom_o1) / (abs(u_avg_fom_o1) + 1e-12))
    error_u_o4.append(abs(u_avg_fom_o4 - u_avg_rom_o4) / (abs(u_avg_fom_o4) + 1e-12))

#────────────────────────────────────────────────────────────────────────────────────────────────
# Summary statistics
#────────────────────────────────────────────────────────────────────────────────────────────────

fem_times = np.array(fem_times)
rom_times = np.array(rom_times)

print("\n" + "─" * 90)
print("COMPUTATIONAL SPEEDUP (FOM with precomputed operators vs ROM)")
print("=" * 50)
print(f"FOM time  mean: {fem_times.mean():.4f} s (±{fem_times.std():.4f}) | median: {np.median(fem_times):.4f} s")
print(f"ROM time  mean: {rom_times.mean():.6f} s (±{rom_times.std():.6f}) | median: {np.median(rom_times):.6f} s")
print(f"Speedup (ratio of means):   {fem_times.mean() / rom_times.mean():.2f}x")
print(f"Speedup (ratio of medians): {np.median(fem_times) / np.median(rom_times):.2f}x")

if RUN_SCRATCH_FOM:
    scratch_times = np.array(scratch_times)
    print(f"\nFOM from scratch  mean: {scratch_times.mean():.4f} s (±{scratch_times.std():.4f}) | "
          f"median: {np.median(scratch_times):.4f} s")
    print(f"Speedup vs scratch (ratio of means): {scratch_times.mean() / rom_times.mean():.2f}x")

print("\n" + "─" * 90)
print("QUANTITIES OF INTEREST: MEAN RELATIVE ERROR")
print("=" * 50)
print(f"Sensors (P only):      {np.mean(error_sensors_p):.2e} ± {np.std(error_sensors_p):.2e}")
print(f"Sensors (U only):      {np.mean(error_sensors_u):.2e} ± {np.std(error_sensors_u):.2e}")
print(f"Region 2 (Pressure):   {np.mean(error_p_o2):.2e} ± {np.std(error_p_o2):.2e}")
print(f"Region 3 (Pressure):   {np.mean(error_p_o3):.2e} ± {np.std(error_p_o3):.2e}")
print(f"Region 1 (Velocity):   {np.mean(error_u_o1):.2e} ± {np.std(error_u_o1):.2e}")
print(f"Region 4 (Velocity):   {np.mean(error_u_o4):.2e} ± {np.std(error_u_o4):.2e}")
print("\n" + "─" * 90)


"""
STEP 1 done: 5 x Xu_k (182546, 182546), B_hT (182546, 23077)
STEP 2 done: B (23077, 182546), pinned pressure node 17
STEP 3 done: R_lift (182546, 5), G_lift (23077,)
FOM offline precomputation total: 59.6 s

Affine FOM vs standard solver | rel. diff  u: 2.66e-11   p: 7.41e-11
Starting benchmark warm-up (untimed)...
Warm-up complete. Starting timed runs for 30 parameters...

Param 1/30  | FEM: 11.4271s | ROM: 0.000054s | Speedup: 212393.99x                                                                                                                                              
Param 2/30  | FEM: 10.9322s | ROM: 0.000057s | Speedup: 190347.47x                                                                                                                                              
Param 3/30  | FEM: 10.4869s | ROM: 0.000055s | Speedup: 189674.19x                                                                                                                                              
Param 4/30  | FEM: 10.5011s | ROM: 0.000056s | Speedup: 186706.96x                                                                                                                                              
Param 5/30  | FEM: 10.1682s | ROM: 0.000057s | Speedup: 178403.11x                                                                                                                                              
Param 6/30  | FEM: 10.4463s | ROM: 0.000055s | Speedup: 191587.51x                                                                                                                                              
Param 7/30  | FEM: 10.8625s | ROM: 0.000057s | Speedup: 189464.01x                                                                                                                                              
Param 8/30  | FEM: 11.3230s | ROM: 0.000052s | Speedup: 217045.73x                                                                                                                                              
Param 9/30  | FEM: 10.9931s | ROM: 0.000053s | Speedup: 206960.20x                                                                                                                                              
Param 10/30 | FEM: 10.6472s | ROM: 0.000055s | Speedup: 191899.78x                                                                                                                                             
Param 11/30 | FEM: 11.5497s | ROM: 0.000054s | Speedup: 214549.02x                                                                                                                                             
Param 12/30 | FEM: 11.0288s | ROM: 0.000055s | Speedup: 199304.50x                                                                                                                                             
Param 13/30 | FEM: 11.3143s | ROM: 0.000057s | Speedup: 198020.12x                                                                                                                                             
Param 14/30 | FEM: 12.4458s | ROM: 0.000055s | Speedup: 227441.74x                                                                                                                                             
Param 15/30 | FEM: 11.0955s | ROM: 0.000055s | Speedup: 202727.23x                                                                                                                                             
Param 16/30 | FEM: 11.0614s | ROM: 0.000060s | Speedup: 184712.96x                                                                                                                                             
Param 17/30 | FEM: 10.9843s | ROM: 0.000052s | Speedup: 211124.45x                                                                                                                                             
Param 18/30 | FEM: 11.4844s | ROM: 0.000054s | Speedup: 212204.54x                                                                                                                                             
Param 19/30 | FEM: 11.7098s | ROM: 0.000056s | Speedup: 209832.11x                                                                                                                                             
Param 20/30 | FEM: 9.7876s  | ROM: 0.000053s | Speedup: 183625.22x                                                                                                                                              
Param 21/30 | FEM: 10.2147s | ROM: 0.000053s | Speedup: 191843.22x                                                                                                                                             
Param 22/30 | FEM: 10.2876s | ROM: 0.000061s | Speedup: 167427.50x                                                                                                                                             
Param 23/30 | FEM: 9.7953s  | ROM: 0.000053s | Speedup: 184952.47x                                                                                                                                              
Param 24/30 | FEM: 10.2753s | ROM: 0.000051s | Speedup: 202336.81x                                                                                                                                             
Param 25/30 | FEM: 9.8244s  | ROM: 0.000053s | Speedup: 185776.07x                                                                                                                                              
Param 26/30 | FEM: 9.7877s  | ROM: 0.000055s | Speedup: 179362.12x                                                                                                                                              
Param 27/30 | FEM: 9.7561s  | ROM: 0.000053s | Speedup: 183167.42x                                                                                                                                              
Param 28/30 | FEM: 9.7671s  | ROM: 0.000052s | Speedup: 188688.22x                                                                                                                                              
Param 29/30 | FEM: 9.8016s  | ROM: 0.000051s | Speedup: 190658.36x                                                                                                                                              
Param 30/30 | FEM: 9.8992s  | ROM: 0.000055s | Speedup: 179335.99x                                                                                                                                              
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [05:16<00:00, 10.55s/it]

──────────────────────────────────────────────────────────────────────────────────────────
COMPUTATIONAL SPEEDUP (FOM with precomputed operators vs ROM)
==================================================
FOM time  mean: 10.6553 s (±0.7015) | median: 10.5742 s
ROM time  mean: 0.000055 s (±0.000002) | median: 0.000055 s
Speedup (ratio of means):   194817.54x
Speedup (ratio of medians): 193853.58x

──────────────────────────────────────────────────────────────────────────────────────────
QUANTITIES OF INTEREST: MEAN RELATIVE ERROR
==================================================
Sensors (P only):      6.45e-04 ± 2.50e-04
Sensors (U only):      9.54e-03 ± 2.37e-03
Region 2 (Pressure):   1.77e-04 ± 2.10e-04
Region 3 (Pressure):   1.95e-04 ± 2.13e-04
Region 1 (Velocity):   2.07e-02 ± 1.08e-02
Region 4 (Velocity):   3.17e-02 ± 1.49e-02

──────────────────────────────────────────────────────────────────────────────────────────
"""