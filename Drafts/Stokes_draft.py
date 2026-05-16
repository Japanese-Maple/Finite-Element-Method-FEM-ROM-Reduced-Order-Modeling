import numpy as np
from scipy import sparse
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.sparse import bmat, linalg

#-------------------------------------------------------------------------------------------------------------------------

def calculate_velocity_A(p, t, kinematic_viscosity):
    """
    This is a matrix for the first integral in the Stokes Equations - The Stiffness Matrix A
    """

    Np = p.shape[0]
    Nt = t.shape[0]

    # we calculate all jacobians simultaneously           J = |x_2 - x_1   y_2 - y_1|
    #                                                         |x_3 - x_1   y_3 - y_1|

    jacobian = np.zeros(shape=(Nt, 2, 2))
    jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]] 
    jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]] 

    det_J = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    # Cofactor multiplication matrix Q:                   Q = |(y_3 - y_1)^2 + (x_3 - x_1)^2                   (y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x1 - x_3)|
    #                                                         |(y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x1 - x_3)  (y_2 - y_1)^2 + (x_2 - x_1)^2                 |
    #
    #                                                     Q = Cof(J).T @ Cof(J)

    q11 = jacobian[:, 1, 1]**2 + jacobian[:, 1, 0]**2
    q12 = -(jacobian[:, 0, 1] * jacobian[:, 1, 1] + jacobian[:, 0, 0] * jacobian[:, 1, 0])
    q22 = jacobian[:, 0, 1]**2 + jacobian[:, 0, 0]**2

    Q_mat = np.zeros_like(jacobian)
    Q_mat[:, 0, 0] = q11
    Q_mat[:, 1, 0], Q_mat[:, 0, 1] = q12, q12
    Q_mat[:, 1, 1] = q22
    
    test_function_derivatives = np.array([[-1, -1],   # ф1 = 1 - s_1 - s_2
                                          [1, 0],     # ф2 = s_1
                                          [0, 1]])    # ф3 = s_2


    # We can now construct a local matrix A for each triangle:
    
    A_local = np.zeros(shape=(Nt, 3, 3))
    for i in range(3):
        for j in range(i, 3):

            grad_i = test_function_derivatives[i]
            grad_j = test_function_derivatives[j]

            val = np.einsum(
                'i, nij, j->n', 
                grad_j,
                Q_mat,
                grad_i
            )

            val *= kinematic_viscosity / (2 * det_J)

            A_local[:, i, j] = val
            A_local[:, j, i] = val

    # INVESTIGATE:
    # A_local = np.einsum('mi,tkj,nj->tmn', test_function_derivatives, Q_mat, test_function_derivatives)
    # A_local *= (kinematic_viscosity / (2.0 * np.abs(det_J)))[:, None, None]

    rowidx = np.einsum("ni,j->nij", t[:,0:3], [1,1,1])
    colidx = np.einsum("nj,i->nij", t[:,0:3], [1,1,1])
    
    # Return corresponding csc_matrix
    return sparse.csc_matrix((np.ravel(A_local),(np.ravel(rowidx),np.ravel(colidx))),shape=(Np,Np))

#-------------------------------------------------------------------------------------------------------------------------

def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):
    """Calculates the pressure matrices Bx, By"""

    Np_fine = p_fine.shape[0]
    Nt_fine = t_fine.shape[0]

    Np_coarse = p_coarse.shape[0]
    Nt_coarse = t_coarse.shape[0]

    jacobian = np.zeros(shape=(Nt_fine, 2, 2))
    jacobian[:, 0, :] = p_fine[t_fine[:, 1]] - p_fine[t_fine[:, 0]] 
    jacobian[:, 1, :] = p_fine[t_fine[:, 2]] - p_fine[t_fine[:, 0]] 

    test_function_derivatives = np.array([[-1, -1],   # ф1 = 1 - s_1 - s_2
                                          [1, 0],     # ф2 = s_1
                                          [0, 1]])    # ф3 = s_2
    
    # We can now construct local matrices Bx, By for each triangle:

    Bx_local = np.zeros(shape=(Nt_fine, 3, 3))
    By_local = np.zeros(shape=(Nt_fine, 3, 3))

    # for i in [0, 1, 2]:
    #     for j in [0, 1, 2]:
    #         Bx_local[:, i, j] = 1/6 * (  jacobian[:, 0, 1] * test_function_derivatives[i, 1]       
    #                                    - jacobian[:, 1, 1] * test_function_derivatives[i, 0])

    #         By_local[:, i, j] = 1/6 * (  jacobian[:, 1, 0] * test_function_derivatives[i, 0] 
    #                                    - jacobian[:, 0, 0] * test_function_derivatives[i, 1])
            
    # print(Bx_local[0],'\n')
    # print(By_local[0],'\n')
            
    Bx_vals = 1/6 * (  jacobian[:, 0, 1, None] * test_function_derivatives[:, 1]       
                     - jacobian[:, 1, 1, None] * test_function_derivatives[:, 0])
    
    By_vals = 1/6 * (  jacobian[:, 1, 0, None] * test_function_derivatives[:, 0] 
                     - jacobian[:, 0, 0, None] * test_function_derivatives[:, 1])
    
    Bx_local = np.repeat(Bx_vals[:, :, None], 3, axis=2)
    By_local = np.repeat(By_vals[:, :, None], 3, axis=2)

    # print(Bx_local[0], '\n !!!\n ', 
    #       jacobian[:, 1, 0, None][:5], '\n !!!\n ', 
    #       jacobian[:, 1, 0][:5], '\n !!!\n ',
    #       test_function_derivatives[:, 1])

    # We now adress the global matrix problem for Bx and By:
    
    # INVESTIGATE:
    fine_to_coarse_idx = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]
    colidx = np.tile(t_fine[:, :3, None], (1, 1, 3)).ravel()
    parent_coarse_nodes = t_coarse[fine_to_coarse_idx, :3]
    rowidx = np.tile(parent_coarse_nodes[:, None, :], (1, 3, 1)).ravel()
    
    B_x = sparse.csc_matrix((np.ravel(Bx_local),
                            (np.ravel(rowidx), np.ravel(colidx))),
                            shape=(Np_coarse, Np_fine))
    
    B_y = sparse.csc_matrix((np.ravel(By_local),
                            (np.ravel(rowidx), np.ravel(colidx))),
                            shape=(Np_coarse, Np_fine))

    return B_x, B_y

#===============================================================================================================================================================

def calculate_Saddle_point_K(A, B_x, B_y):
    """Calculates the Saddle-Point matrix **K**"""

    Np_coarse = B_x.shape[0]
    Zero_pp = sparse.lil_matrix((Np_coarse, Np_coarse))

    K_mat = bmat([
        [A,       None,    B_x.T],
        [None,    A,       B_y.T],
        [B_x,     B_y,     Zero_pp]
    ], format='lil')

    return K_mat

#===============================================================================================================================================================

def calculate_Neumann_BCs():
    return

#===============================================================================================================================================================

def calculate_Dirichlet_BCs(A_mat, lifting_function):
    lf_x, lf_y = lifting_function
    return A_mat @ lf_x, A_mat @ lf_y

#===============================================================================================================================================================

def calculate_pressure_lifting(B_x, B_y, lifting_function):
    lf_x, lf_y = lifting_function
    return B_x @ lf_x + B_y @ lf_y

#===============================================================================================================================================================

def calculate_F(A_mat, B_x, B_y, lifting_function):
    F_x, F_y = calculate_Dirichlet_BCs(A_mat, lifting_function)
    p = calculate_pressure_lifting(B_x, B_y, lifting_function)
    return np.concatenate([-F_x, -F_y, p])    

#-------------------------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.sparse.linalg import spsolve

def solve_stokes_final_v4(p_fine, t_fine,
                          p_coarse, t_coarse,
                          A, Bx, By):

    Nv = p_fine.shape[0]
    Np = p_coarse.shape[0]

    eps = 1e-7

    xmin = p_fine[:,0].min()
    xmax = p_fine[:,0].max()
    ymin = p_fine[:,1].min()
    ymax = p_fine[:,1].max()

    # Find Inlet Node Indices
    is_inlet = (np.abs(p_fine[:,0] - xmin) < eps)
    inlet_idx = np.where(is_inlet)[0]

    # Find Wall Node Indices
    is_outer_wall_v = ((np.abs(p_fine[:,1] - ymax) < eps) | (np.abs(p_fine[:,1] - ymin) < eps))
    is_side_wall_v = (np.abs(p_fine[:,0] - xmax) < eps)
    is_b1v = ((np.abs(p_fine[:,0] + 1.0) < 0.06) & (p_fine[:,1] > -0.35))
    is_b2v = ((np.abs(p_fine[:,0] - 0.0) < 0.06) & (p_fine[:,1] < 0.35))
    is_b3v = ((np.abs(p_fine[:,0] - 1.0) < 0.06) & (p_fine[:,1] > -0.35))

    v_wall_idx = np.where(is_outer_wall_v | is_side_wall_v | is_b1v | is_b2v | is_b3v)[0]

    lf_x = np.zeros(Nv)
    lf_y = np.zeros(Nv)

    lf_x[inlet_idx] = -1.0  
    A = calculate_velocity_A(p_fine, t_fine, kinematic_viscosity=100)
    Bx, By = calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse)
    F = calculate_F(A, Bx, By, (lf_x, lf_y))
    K = calculate_Saddle_point_K(A, Bx, By)

    dirichlet_nodes = np.unique(np.concatenate([inlet_idx, v_wall_idx]))

    for i in dirichlet_nodes:
        K[i, :] = 0.0
        K[i, i] = 1.0
        F[i] = 0.0  

        iy = i + Nv
        K[iy, :] = 0.0
        K[iy, iy] = 1.0
        F[iy] = 0.0  

    is_outlet_p = (np.abs(p_coarse[:,0] - xmax) < eps)
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
