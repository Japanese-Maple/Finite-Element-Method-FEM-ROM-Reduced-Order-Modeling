import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(script_dir)

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from Utilities.Mesh_processing import refine
from Utilities.Plot_functions import Plot_Initial_Refined_meshes
from Utilities.Stokes_felib import (
     calculate_velocity_A,
     calculate_pressure_B,
     calculate_stabilization_Cp,
     calculate_Saddle_point_K,
     save_simulation_data
     )

#_____________________________________________________________________________________________________________________________

# Solver

def parabolic_inlet_profile(x, x_1, x_2, alpha):
        return -alpha*(x - x_1)*(x - x_2)

def compute_U_P_solution_exchanger_device(p_fine, t_fine, e_fine, p_coarse, t_coarse,
                                          alpha: float = 3.0,
                                          kinematic_viscosity: float = 1,
                                          return_matrices: bool = False,
                                          print_timings: bool = False):
    t_start = time.perf_counter()

    Nv = p_fine.shape[0]
    Np = p_coarse.shape[0]
    eps = 1e-10   

    xmin = p_fine[:, 0].min()
    xmax = p_fine[:, 0].max()

    inlet_idx  = np.where(np.abs(p_fine[:, 0] - xmin) < eps)[0]
    outlet_idx = np.where(np.abs(p_fine[:, 0] - xmax) < eps)[0]

    corner_idx  = [inlet_idx[np.argmax(p_fine[inlet_idx, 1])], 
                inlet_idx[np.argmin(p_fine[inlet_idx, 1])]]

    boundary_nodes = np.unique(e_fine[e_fine[:, 2] > 0, 0:2])
    v_wall_idx = np.setdiff1d(boundary_nodes, np.concatenate([inlet_idx, outlet_idx]))
    dirichlet_nodes = np.unique(np.concatenate([inlet_idx, v_wall_idx]))

    lf_x = np.zeros(Nv)
    lf_y = np.zeros(Nv)

    y_inlet = p_fine[inlet_idx, 1]
    y_top, y_bottom = p_fine[corner_idx[0], 1], p_fine[corner_idx[1], 1]
    lf_x[inlet_idx] = parabolic_inlet_profile(y_inlet, y_top, y_bottom, alpha=alpha)

    t_setup = time.perf_counter() - t_start

    # ── A matrix ────────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    A  = calculate_velocity_A(p_fine, t_fine, kinematic_viscosity)
    t_A = time.perf_counter() - t0

    # ── B matrix ────────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    Bx, By = calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse)
    t_B = time.perf_counter() - t0

    # ── K matrix and right-hand side ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    # gamma = 0.05
    # Cp = calculate_stabilization_Cp(p_coarse, t_coarse, gamma=gamma)
    # K = calculate_Saddle_point_K(A, Bx, By, Cp=Cp)
    K  = calculate_Saddle_point_K(A, Bx, By)

    F = np.zeros(2 * Nv + Np)
    F[:Nv]    -= A.dot(lf_x)
    F[2*Nv:]  -= Bx.dot(lf_x)

    t_K = time.perf_counter() - t0

    # ── row erasure (Dirichlet rows + pressure pin) ─────────────────────────────────────────
    t0 = time.perf_counter()

    is_outlet_p = np.abs(p_coarse[:, 0] - xmax) < eps
    p_ref_idx   = np.where(is_outlet_p)[0]
    if len(p_ref_idx) == 0:
        p_ref_idx = [np.argmin(np.abs(p_coarse[:, 0] - xmax))]
    p_row = 2 * Nv + p_ref_idx[0]

    Ntot = 2 * Nv + Np
    dirichlet_rows = np.concatenate([dirichlet_nodes, dirichlet_nodes + Nv, [p_row]])

    K = K.tocsr()
    mask = np.ones(Ntot)
    mask[dirichlet_rows] = 0.0
    K = sp.diags(mask) @ K

    diag_fix = np.zeros(Ntot)
    diag_fix[dirichlet_rows] = 1.0
    K = K + sp.diags(diag_fix)

    F[dirichlet_rows] = 0.0

    t_erase = time.perf_counter() - t0

    Xu_bc, B_hT = None, None
    if return_matrices:
        Xu_bc = K[:2*Nv, :2*Nv].tocsc()
        B_hT  = K[:2*Nv, 2*Nv:].tocsc()

    # ── CSC conversion (kept separate so that the solve time is the pure spsolve) ───────────
    t0 = time.perf_counter()
    K_csc = K.tocsc()
    t_csc = time.perf_counter() - t0

    # ── solve ───────────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sol = spsolve(K_csc, F)
    t_solve = time.perf_counter() - t0

    if np.any(np.isnan(sol)):
        raise RuntimeError("NaNs in solution.")

    u0_x     = sol[:Nv]
    u0_y     = sol[Nv:2*Nv]
    pressure = sol[2*Nv:]

    ux = u0_x + lf_x
    uy = u0_y + lf_y

    t_total = time.perf_counter() - t_start

    if print_timings:
        print("\n" + "─" * 60)
        print("FOM TIME:")
        print("─" * 60)
        print(f"Setup (boundaries, lifting):   {t_setup:8.4f} s  ({100*t_setup/t_total:5.1f} %)")
        print(f"Assemble A:                    {t_A:8.4f} s  ({100*t_A/t_total:5.1f} %)")
        print(f"Assemble Bx, By:               {t_B:8.4f} s  ({100*t_B/t_total:5.1f} %)")
        print(f"Assemble K and RHS:            {t_K:8.4f} s  ({100*t_K/t_total:5.1f} %)")
        print(f"Row erasure (BCs + pin):       {t_erase:8.4f} s  ({100*t_erase/t_total:5.1f} %)")
        print(f"CSC conversion:                {t_csc:8.4f} s  ({100*t_csc/t_total:5.1f} %)")
        print(f"Solve K (spsolve):             {t_solve:8.4f} s  ({100*t_solve/t_total:5.1f} %)")
        print("─" * 60)
        print(f"TOTAL:                         {t_total:8.4f} s")
        print("─" * 60 + "\n")

    if return_matrices:
        return ux, uy, pressure, Xu_bc, B_hT
    
    return ux, uy, pressure

#_____________________________________________________________________________________________________________________________

if __name__ == "__main__":
    
    print('generating the mesh...')
    p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(
        data_path='Meshes/exchanger_device_altered_mesh_data.npz',
        num_of_refinements=3,
        figsize=(16, 4),
        plot=False
    )
    p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

    print('generating the solution...')
    ux, uy, p_sol = compute_U_P_solution_exchanger_device(p_fine, t_fine, e_fine, p_coarse, t_coarse)

    print('saving the solution...')
    save_simulation_data(p_fine, e_fine, t_fine,
                         p_coarse, e_coarse, t_coarse,
                         ux, uy, p_sol,
                         name='Exchanger_device')