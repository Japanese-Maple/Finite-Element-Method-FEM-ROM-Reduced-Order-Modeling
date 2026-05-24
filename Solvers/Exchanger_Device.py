import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # Points to .../FEM/Solvers
fem_dir = os.path.dirname(script_dir)                    # Points to .../FEM

# 2. Fix the Python path import safely
if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from scipy.sparse.linalg import spsolve
from Utilities.Stokes_felib import *
from Utilities.Mesh_processing import *

#==============================================================================================================================

# Mesh

p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(data_path='Meshes/exchanger_device_altered_mesh_data.npz', 
                                                           num_of_refinements=3, 
                                                           figsize=(16,4), plot=False)
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

#==============================================================================================================================

def compute_U_P_solution(p_fine, t_fine, e_fine, p_coarse, t_coarse):

    Nv = p_fine.shape[0]
    Np = p_coarse.shape[0]
    eps = 1e-17

    xmin = p_fine[:, 0].min()
    xmax = p_fine[:, 0].max()

    inlet_idx  = np.where(np.abs(p_fine[:, 0] - xmin) < eps)[0]
    outlet_idx = np.where(np.abs(p_fine[:, 0] - xmax) < eps)[0]

    # Extract all physical boundary nodes from edge flags
    boundary_nodes = np.unique(e_fine[e_fine[:, 2] > 0, 0:2])

    # Exclude inlet and outlet from solid walls
    v_wall_idx = np.setdiff1d(boundary_nodes, np.concatenate([inlet_idx, outlet_idx]))

    lf_x = np.zeros(Nv)
    lf_y = np.zeros(Nv)
    lf_x[inlet_idx] = -1.0  

    A = calculate_velocity_A(p_fine, t_fine, kinematic_viscosity=0.01)
    Bx, By = calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse)
    F = calculate_F(A, Bx, By, (lf_x, lf_y))
    K = calculate_Saddle_point_K(A, Bx, By)

    dirichlet_nodes = np.unique(np.concatenate([inlet_idx, v_wall_idx]))
    K = K.tolil()

    for i in dirichlet_nodes:
        # X-velocity tracks
        K[i, :] = 0.0
        K[i, i] = 1.0
        F[i] = 0.0  

        # Y-velocity tracks (shifted by Nv)
        iy = i + Nv
        K[iy, :] = 0.0
        K[iy, iy] = 1.0
        F[iy] = 0.0  

    is_outlet_p = (np.abs(p_coarse[:, 0] - xmax) < eps)
    p_ref_idx = np.where(is_outlet_p)[0]
    p_ref = p_ref_idx[0]
    p_row = 2 * Nv + p_ref

    K[p_row, :] = 0.0
    K[p_row, p_row] = 1.0
    F[p_row] = 0.0

    print("Solving lifted system...")
    sol = spsolve(K.tocsc(), F)

    if np.any(np.isnan(sol)):
        print("Warning: NaNs detected!")

    u0_x = sol[:Nv]
    u0_y = sol[Nv:2*Nv]
    pressure = sol[2*Nv:]

    ux = u0_x + lf_x
    uy = u0_y + lf_y

    div = Bx @ ux + By @ uy
    print("||div|| =", np.linalg.norm(div))
    print("max div =", np.max(np.abs(div)))

    return ux, uy, pressure

#==============================================================================================================================

# Solution computation and logging

if __name__ == "__main__":
    
    ux, uy, p_sol = compute_U_P_solution(p_fine, t_fine, e_fine, p_coarse, t_coarse)

    save_simulation_data(p_fine, e_fine, t_fine, 
                        p_coarse, e_coarse, t_coarse, 
                        ux, uy, p_sol,
                        name='Exchanger_device')