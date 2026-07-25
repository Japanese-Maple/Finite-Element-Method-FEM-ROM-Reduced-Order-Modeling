import os
import sys
import time

import numpy as np
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(script_dir)

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from Solvers.Exchanger_Device import compute_U_P_solution
from Utilities.Mesh_processing import refine
from Utilities.Plot_functions import Plot_Initial_Refined_meshes

# ──────────────────────────────────────────────────────────────────────────────
# Mesh Initialization & Data Loading
# ──────────────────────────────────────────────────────────────────────────────

mesh_path = os.path.join(fem_dir, 'Meshes', 'exchanger_device_altered_mesh_data.npz')
p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(
    data_path=mesh_path, 
    num_of_refinements=3, 
    plot=False,
    figsize=(16, 4)
)
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

npz_path = os.path.join(fem_dir, 'Reduced_Order_Modeling', 'Data', 'viscosity_snapshots.npz')
with np.load(npz_path) as data:
    v_t_snapshots = data['v_t_snapshots']
    parameters    = data['parameters']

# ──────────────────────────────────────────────────────────────────────────────
# System Dimensionality Setup
# ──────────────────────────────────────────────────────────────────────────────
num_snapshots = v_t_snapshots.shape[0]
Nv = p_fine.shape[0]
Np = p_coarse.shape[0]
total_dof = (2 * Nv) + Np

print("─" * 60)
print(" High-Fidelity Stokes FOM Computation Initiated")
print("─" * 60)
print(f" Snapshot configurations: {num_snapshots}")
print(f" Parameter space dim:     {parameters.shape[1]}")
print("─" * 60)
print(f" Velocity DOFs (N_u):     {2 * Nv}")
print(f" Pressure DOFs (N_p):     {Np}")
print(f" Total DOFs per state:    {total_dof}")
print("─" * 60)

# Pre-allocate snapshot matrices 
ux_snapshots = np.zeros((num_snapshots, Nv))
uy_snapshots = np.zeros((num_snapshots, Nv))
p_snapshots  = np.zeros((num_snapshots, Np))

# ──────────────────────────────────────────────────────────────────────────────
# Full Order Model (FOM) Batch Execution
# ──────────────────────────────────────────────────────────────────────────────
start_time = time.time()

# Wrapped in tqdm for robust progress tracking and ETA
for i in tqdm(range(num_snapshots), desc="Evaluating FOM Snapshots", unit="state", ncols=100):
    
    v_t = v_t_snapshots[i]
    
    ux, uy, p_sol = compute_U_P_solution(
        p_fine, t_fine, e_fine, 
        p_coarse, t_coarse,
        inlet_velocity=1.0,
        kinematic_viscosity=v_t
    )
    
    ux_snapshots[i, :] = ux
    uy_snapshots[i, :] = uy
    p_snapshots[i, :]  = p_sol

# ──────────────────────────────────────────────────────────────────────────────
# Output Serialization & Diagnostics
# ──────────────────────────────────────────────────────────────────────────────
elapsed_time = time.time() - start_time
avg_time_per_snapshot = elapsed_time / num_snapshots

print("\n" + "─" * 60)
print(" Computation Concluded")
print("─" * 60)
print(f" Total runtime:           {elapsed_time / 60:.2f} minutes")
print(f" Average iteration time:  {avg_time_per_snapshot:.3f} seconds/state")

output_dir = os.path.join(fem_dir, 'Reduced_Order_Modeling', 'Data')
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, 'stokes_solution_snapshots.npz')

np.savez_compressed(
    out_path,
    ux_snapshots=ux_snapshots,
    uy_snapshots=uy_snapshots,
    p_snapshots=p_snapshots,
    parameters=parameters,
    p_fine=p_fine, 
    t_fine=t_fine, 
    e_fine=e_fine,
    p_coarse=p_coarse, 
    t_coarse=t_coarse
)

print(f" Snapshot dataset successfully archived to:\n > {out_path}")
print("=" * 60)