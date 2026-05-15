import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import bmat, linalg
from .Mesh_processing import refine, refine_n_times, fix_orientation, build_stable_mesh, Plot_Initial_Refined_meshes

#===============================================================================================================================================================
# MAIN COMPUTATIONAL FUNCTIONS
#===============================================================================================================================================================

def calculate_velocity_A(p, t, kinematic_viscosity):
    """This is a matrix for the first integral in the Stokes Equations - The Stiffness Matrix A"""

    Np = p.shape[0]
    Nt = t.shape[0]

    # we calculate all jacobians simultaneously           J = |x_2 - x_1   x_3 - x_1|
    #                                                         |y_2 - y_1   y_3 - y_1|

    jacobian = np.zeros(shape=(Nt, 2, 2))
    jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]] 
    jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]] 

    det_J = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    # Cofactor multiplication matrix Q:                   Q = |(y_3 - y_1)^2 + (x_3 - x_1)^2                   (y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x1 - x_3)|
    #                                                         |(y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x1 - x_3)  (y_2 - y_1)^2 + (x_2 - x_1)^2                 |
    #
    #                                                     Q = Cof(J).T @ Cof(J)

    q11 = jacobian[:, 1, 1]**2 + jacobian[:, 0, 1]**2
    q12 = -(jacobian[:, 1, 0] * jacobian[:, 1, 1] + jacobian[:, 0, 0] * jacobian[:, 0, 1])
    q22 = jacobian[:, 1, 0]**2 + jacobian[:, 0, 0]**2

    Q_mat = np.zeros_like(jacobian)
    Q_mat[:, 0, 0] = q11
    Q_mat[:, 1, 0], Q_mat[:, 0, 1] = q12, q12
    Q_mat[:, 1, 1] = q22
    
    test_function_derivatives = np.array([[-1, -1],   # ф1 = 1 - s_1 - s_2
                                          [ 1,  0],   # ф2 = s_1
                                          [ 0,  1]])  # ф3 = s_2


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

#===============================================================================================================================================================

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
                                          [ 1,  0],   # ф2 = s_1
                                          [ 0,  1]])  # ф3 = s_2
    
    # We can now construct local matrices Bx, By for each triangle:

    Bx_local = np.zeros(shape=(Nt_fine, 3, 3))
    By_local = np.zeros(shape=(Nt_fine, 3, 3))
            
    Bx_vals = 1/6 * (  jacobian[:, 0, 1, None] * test_function_derivatives[:, 1]       
                     - jacobian[:, 1, 1, None] * test_function_derivatives[:, 0])
    
    By_vals = 1/6 * (  jacobian[:, 1, 0, None] * test_function_derivatives[:, 0] 
                     - jacobian[:, 0, 0, None] * test_function_derivatives[:, 1])
    
    Bx_local = np.repeat(Bx_vals[:, :, None], 3, axis=2)
    By_local = np.repeat(By_vals[:, :, None], 3, axis=2)

    # We now adress the global matrix problem for Bx and By:

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
# VISUALIZATIONS
#===============================================================================================================================================================

def B_matrix_structure(B_mat, 
                       figsize:tuple=(13,13), cmap:str='viridis'):
    """Plots the B matrix values and color codes them."""

    B_coo = B_mat.tocoo()
    fig, b_plot = plt.subplots(figsize=figsize)
    sc = b_plot.scatter(B_coo.col, B_coo.row, 
                        c=B_coo.data,      
                        s=1,
                        cmap=cmap,   
                        marker='s',
                        linewidths=0,
                        edgecolors='none', 
                        antialiaseds=False)
    
    b_plot.set_xlim([0, B_mat.shape[1]])
    b_plot.set_ylim([0, B_mat.shape[0]])
    b_plot.invert_yaxis()

    divider = make_axes_locatable(b_plot)
    cax = divider.append_axes("right", size="3%", pad=0.1)    
    plt.colorbar(sc, cax=cax, label='Matrix Entry Value')  
    
    b_plot.set_aspect('equal')
    b_plot.set_title(f"B: {B_mat.shape[0]}x{B_mat.shape[1]}")

    plt.tight_layout()    
    plt.savefig('Outputs/Stokes_B_matrix.svg')
    plt.show()