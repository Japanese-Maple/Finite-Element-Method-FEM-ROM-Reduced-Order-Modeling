import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(script_dir)

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from scipy.sparse.linalg import spsolve
from Utilities.Stokes_felib import *
from Utilities.Mesh_processing import *
from Utilities.Plot_functions import *

#_____________________________________________________________________________________________________________________________

# Mesh

p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(
    data_path='Meshes/exchanger_device_altered_mesh_data.npz',
    num_of_refinements=3,
    figsize=(16, 4),
    plot=False
)
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

#_____________________________________________________________________________________________________________________________
def compute_U_P_solution(p_fine, t_fine, e_fine, p_coarse, t_coarse,
                         inlet_velocity: float = 1.0,
                         kinematic_viscosity: float = 0.01,
                         return_A: bool = False):

    Nv = p_fine.shape[0]
    Np = p_coarse.shape[0]
    eps = 1e-10   

    xmin = p_fine[:, 0].min()
    xmax = p_fine[:, 0].max()

    inlet_idx  = np.where(np.abs(p_fine[:, 0] - xmin) < eps)[0]
    outlet_idx = np.where(np.abs(p_fine[:, 0] - xmax) < eps)[0]

    boundary_nodes = np.unique(e_fine[e_fine[:, 2] > 0, 0:2])
    v_wall_idx = np.setdiff1d(boundary_nodes, np.concatenate([inlet_idx, outlet_idx]))
    dirichlet_nodes = np.unique(np.concatenate([inlet_idx, v_wall_idx]))

    lf_x = np.zeros(Nv)
    lf_y = np.zeros(Nv)
    lf_x[inlet_idx] = inlet_velocity          

    A  = calculate_velocity_A(p_fine, t_fine, kinematic_viscosity)
    Bx, By = calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse)
    K  = calculate_Saddle_point_K(A, Bx, By)

    F = np.zeros(2 * Nv + Np)
    F[:Nv]    -= A.dot(lf_x)
    F[2*Nv:]  -= Bx.dot(lf_x)  

    # print(f"K is of the shape {K.shape}")

    K = K.tolil()

    for i in dirichlet_nodes:
        K[i, :]  = 0.0
        K[i, i]  = 1.0
        F[i]     = 0.0

        iy = i + Nv
        K[iy, :] = 0.0
        K[iy, iy] = 1.0
        F[iy]    = 0.0

    Xu_bc = K[:2*Nv, :2*Nv].tocsc()

    is_outlet_p = np.abs(p_coarse[:, 0] - xmax) < eps
    p_ref_idx   = np.where(is_outlet_p)[0]

    if len(p_ref_idx) == 0:
        p_ref_idx = [np.argmin(np.abs(p_coarse[:, 0] - xmax))]
        # print("Warning: no coarse node exactly on outlet x; using nearest.")

    p_row = 2 * Nv + p_ref_idx[0]
    K[p_row, :] = 0.0
    K[p_row, p_row] = 1.0
    F[p_row] = 0.0        

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    # print("Solving lifted system...")
    sol = spsolve(K.tocsc(), F)

    if np.any(np.isnan(sol)):
        raise RuntimeError("NaNs in solution — check mesh connectivity and BCs.")

    u0_x     = sol[:Nv]
    u0_y     = sol[Nv:2*Nv]
    pressure = sol[2*Nv:]

    # Recover full velocity
    ux = u0_x + lf_x
    uy = u0_y + lf_y

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    # div = Bx @ ux + By @ uy
    # print(f"||div u|| = {np.linalg.norm(div):.3e}")
    # print(f"max |div| = {np.max(np.abs(div)):.3e}")
    # print(f"pressure  min/mean/max = "
    #       f"{pressure.min():.4f} / {pressure.mean():.4f} / {pressure.max():.4f}")
    if return_A:
        return ux, uy, pressure, Xu_bc

    return ux, uy, pressure

#_____________________________________________________________________________________________________________________________

if __name__ == "__main__":

    ux, uy, p_sol = compute_U_P_solution(p_fine, t_fine, e_fine, p_coarse, t_coarse)

    save_simulation_data(p_fine, e_fine, t_fine,
                         p_coarse, e_coarse, t_coarse,
                         ux, uy, p_sol,
                         name='Exchanger_device')