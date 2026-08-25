import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(script_dir)

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

from Utilities.Stokes_felib import *
from Utilities.Mesh_processing import *
from Utilities.Plot_functions import *

#_____________________________________________________________________________________________________________________________

# Mesh

p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(
    data_path='Meshes/Winding_pipe_fixed_mesh_data.npz',
    num_of_refinements=2,
    figsize=(16, 4),
    plot=False
)
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

#_____________________________________________________________________________________________________________________________

# Solver

def parabolic_inlet_profile(x, x_1, x_2, alpha):
        return -alpha*(x - x_1)*(x - x_2)

def compute_U_P_solution_winding_pipe(p_fine, t_fine, e_fine, p_coarse, t_coarse,
                                      alpha: float = 3.0,
                                      kinematic_viscosity: float = 1.0):

    Nv = p_fine.shape[0]
    Np = p_coarse.shape[0]
    eps = 1e-3

    y_bottom_inlet  = 0.6
    y_top_inlet     = 1.3
    y_bottom_outlet = 10.5
    y_top_outlet    = 11.5

    xmin = p_fine[:, 0].min()
    xmax = p_fine[:, 0].max()

    inlet_idx = np.where(
        (np.abs(p_fine[:, 0] - xmin) < eps) &
        (p_fine[:, 1] >= y_bottom_inlet) &
        (p_fine[:, 1] <= y_top_inlet)
    )[0]

    outlet_idx = np.where(
        (np.abs(p_fine[:, 0] - xmax) < eps) &
        (p_fine[:, 1] >= y_bottom_outlet) &
        (p_fine[:, 1] <= y_top_outlet)
    )[0]

    corner_idx  = [inlet_idx[np.argmax(p_fine[inlet_idx, 1])], 
                   inlet_idx[np.argmin(p_fine[inlet_idx, 1])]]
    
    boundary_nodes = np.unique(e_fine[e_fine[:, 2] > 0, 0:2])
    v_wall_idx     = np.setdiff1d(boundary_nodes, np.concatenate([inlet_idx, outlet_idx]))
    dirichlet_nodes = np.unique(np.concatenate([inlet_idx, v_wall_idx]))

    lf_x = np.zeros(Nv)
    lf_y = np.zeros(Nv)

    y_inlet = p_fine[inlet_idx, 1]
    y_top, y_bottom = p_fine[corner_idx[0], 1], p_fine[corner_idx[1], 1]
    lf_x[inlet_idx] = parabolic_inlet_profile(y_inlet, y_top, y_bottom, alpha)

    A  = calculate_velocity_A(p_fine, t_fine, kinematic_viscosity)
    Bx, By = calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse)
    K  = calculate_Saddle_point_K(A, Bx, By)

    F = np.zeros(2 * Nv + Np)
    F[:Nv]   -= A.dot(lf_x)
    F[2*Nv:] -= Bx.dot(lf_x)

    K = K.tolil()

    for i in dirichlet_nodes:
        K[i, :]  = 0.0;  K[i, i]  = 1.0;  F[i]  = 0.0    # ux
        iy = i + Nv
        K[iy, :] = 0.0;  K[iy, iy] = 1.0;  F[iy] = 0.0   # uy

    outlet_p_idx = np.where(
        (np.abs(p_coarse[:, 0] - xmax) < eps) &
        (p_coarse[:, 1] >= y_bottom_outlet) &
        (p_coarse[:, 1] <= y_top_outlet)
    )[0]

    p_row = 2 * Nv + outlet_p_idx[0]
    K[p_row, :] = 0.0
    K[p_row, p_row] = 1.0
    F[p_row] = 0.0

    sol = spsolve(K.tocsc(), F)

    if np.any(np.isnan(sol)):
        raise RuntimeError("NaNs in solution.")

    u0_x     = sol[:Nv]
    u0_y     = sol[Nv:2*Nv]
    pressure = sol[2*Nv:]

    ux = u0_x + lf_x
    uy = u0_y + lf_y

    return ux, uy, pressure

#_____________________________________________________________________________________________________________________________

if __name__ == "__main__":

    print('generating the solution...')
    ux, uy, p_sol = compute_U_P_solution_winding_pipe(p_fine, t_fine, e_fine, p_coarse, t_coarse)

    print('saving the solution...')
    save_simulation_data(p_fine, e_fine, t_fine,
                         p_coarse, e_coarse, t_coarse,
                         ux, uy, p_sol,
                         name='Winding_pipe')