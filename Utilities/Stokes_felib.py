import numpy as np
from scipy import sparse
from scipy.sparse import bmat, linalg

from .Mesh_processing import (refine, 
                              refine_n_times, 
                              fix_orientation, 
                              build_stable_mesh)

#===============================================================================================================================================================
# MAIN COMPUTATIONAL FUNCTIONS
#===============================================================================================================================================================

def calculate_velocity_A(p, t, kinematic_viscosity):
    """Calculates the Stiffness Matrix **A**"""

    Np = p.shape[0]
    Nt = t.shape[0]

    # we calculate all jacobians simultaneously           J = |x_2 - x_1   x_3 - x_1|
    #                                                         |y_2 - y_1   y_3 - y_1|

    jacobian = np.zeros(shape=(Nt, 2, 2))
    jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]] 
    jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]] 

    det_J = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    # Cofactor multiplication matrix Q:                   Q = |(y_3 - y_1)^2 + (x_3 - x_1)^2                   (y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x_1 - x_3)|
    #                                                         |(y_1 - y_2)(y_3 - y_1) + (x_2 - x_1)(x_1 - x_3)  (y_2 - y_1)^2 + (x_2 - x_1)^2                 |
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
    
    # A_local = np.zeros(shape=(Nt, 3, 3))
    # for i in range(3):
    #     for j in range(i, 3):

    #         grad_i = test_function_derivatives[i]
    #         grad_j = test_function_derivatives[j]

    #         val = np.einsum(
    #             'i, nij, j->n',
    #             grad_j,
    #             Q_mat,
    #             grad_i
    #         )

    #         val *= kinematic_viscosity / (2 * det_J)

    #         A_local[:, i, j] = val
    #         A_local[:, j, i] = val

    A_local = np.einsum('mi,txy,nj->tmn', test_function_derivatives, Q_mat, test_function_derivatives)
    A_local *= (kinematic_viscosity / (2.0 * det_J))[:, None, None]

    rowidx = np.einsum("ni,j->nij", t[:,0:3], [1,1,1])
    colidx = np.einsum("nj,i->nij", t[:,0:3], [1,1,1])
    
    # Return corresponding csc_matrix
    return sparse.csc_matrix((np.ravel(A_local),(np.ravel(rowidx),np.ravel(colidx))),shape=(Np,Np))

#_______________________________________________________________________________________________________________________________________________________________

def calculate_mass_M(p, t,):

    Np = p.shape[0]
    Nt = t.shape[0]

    jacobian = np.zeros(shape=(Nt, 2, 2))
    jacobian[:, 0, :] = p[t[:, 1]] - p[t[:, 0]] 
    jacobian[:, 1, :] = p[t[:, 2]] - p[t[:, 0]] 

    det_J = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]

    M_local = np.einsum("n,ij->nij", 
                        np.abs(det_J)/24, 
                        np.array([
                            [2, 1, 1],
                            [1, 2, 1],
                            [1, 1, 2]
                        ]))

    rowidx = np.einsum("ni,j->nij", t[:,0:3], [1,1,1])
    colidx = np.einsum("nj,i->nij", t[:,0:3], [1,1,1])

    return sparse.csc_matrix((np.ravel(M_local),(np.ravel(rowidx),np.ravel(colidx))),shape=(Np,Np))

#_______________________________________________________________________________________________________________________________________________________________


# def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):
#     """Calculates the pressure matrices **Bx**, **By**"""

#     Np_fine = p_fine.shape[0]
#     Nt_fine = t_fine.shape[0]

#     Np_coarse = p_coarse.shape[0]
#     Nt_coarse = t_coarse.shape[0]

#     jacobian = np.zeros(shape=(Nt_fine, 2, 2))
#     jacobian[:, 0, :] = p_fine[t_fine[:, 1]] - p_fine[t_fine[:, 0]] 
#     jacobian[:, 1, :] = p_fine[t_fine[:, 2]] - p_fine[t_fine[:, 0]] 

#     test_function_derivatives = np.array([[-1, -1],   # ф1 = 1 - s_1 - s_2
#                                           [ 1,  0],   # ф2 = s_1
#                                           [ 0,  1]])  # ф3 = s_2
    
#     # We can now construct local matrices Bx, By for each triangle:

#     Bx_local = np.zeros(shape=(Nt_fine, 3, 3))
#     By_local = np.zeros(shape=(Nt_fine, 3, 3))
            
#     Bx_vals = 1/6 * (  jacobian[:, 0, 1, None] * test_function_derivatives[:, 1]       
#                      - jacobian[:, 1, 1, None] * test_function_derivatives[:, 0])
    
#     By_vals = 1/6 * (  jacobian[:, 1, 0, None] * test_function_derivatives[:, 0] 
#                      - jacobian[:, 0, 0, None] * test_function_derivatives[:, 1])
    
#     Bx_local = np.repeat(Bx_vals[:, :, None], 3, axis=2)
#     By_local = np.repeat(By_vals[:, :, None], 3, axis=2)

#     # We now adress the global matrix problem for Bx and By:

#     fine_to_coarse_idx = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]
#     colidx = np.tile(t_fine[:, :3, None], (1, 1, 3)).ravel()
#     parent_coarse_nodes = t_coarse[fine_to_coarse_idx, :3]
#     rowidx = np.tile(parent_coarse_nodes[:, None, :], (1, 3, 1)).ravel()
    
#     B_x = sparse.csc_matrix((np.ravel(Bx_local),
#                             (np.ravel(rowidx), np.ravel(colidx))),
#                             shape=(Np_coarse, Np_fine))
    
#     B_y = sparse.csc_matrix((np.ravel(By_local),
#                             (np.ravel(rowidx), np.ravel(colidx))),
#                             shape=(Np_coarse, Np_fine))

#     return B_x, B_y


#TODO 1. Check the following code
def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):
    """Assemble Bx, By for P1-iso-P1 correctly."""

    from scipy import sparse
    import numpy as np

    Nt_fine = t_fine.shape[0]
    Nt_coarse = t_coarse.shape[0]
    Np_fine = p_fine.shape[0]
    Np_coarse = p_coarse.shape[0]

    # Parent coarse triangle for each fine triangle
    # valid for your refine() ordering
    fine_to_coarse = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]

    grad_hat = np.array([
        [-1.0, -1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0]
    ])

    rowidx_x = []
    colidx_x = []
    data_x = []

    rowidx_y = []
    colidx_y = []
    data_y = []

    for k in range(Nt_fine):
        tf = t_fine[k, :3]
        tc = t_coarse[fine_to_coarse[k], :3]

        pf = p_fine[tf]
        pc = p_coarse[tc]

        # Fine triangle Jacobian
        Jf = np.array([
            [pf[1, 0] - pf[0, 0], pf[2, 0] - pf[0, 0]],
            [pf[1, 1] - pf[0, 1], pf[2, 1] - pf[0, 1]]
        ])

        detJ = np.linalg.det(Jf)
        area = 0.5 * abs(detJ)

        invJf_T = np.linalg.inv(Jf).T
        grads = (invJf_T @ grad_hat.T).T   # shape (3,2)

        # Centroid of fine triangle
        q = pf.mean(axis=0)

        # Barycentric coordinates of q in the parent coarse triangle
        Jc = np.array([
            [pc[1, 0] - pc[0, 0], pc[2, 0] - pc[0, 0]],
            [pc[1, 1] - pc[0, 1], pc[2, 1] - pc[0, 1]]
        ])
        lam1_lam2 = np.linalg.solve(Jc, q - pc[0])
        lam0 = 1.0 - lam1_lam2[0] - lam1_lam2[1]
        psi = np.array([lam0, lam1_lam2[0], lam1_lam2[1]])  # coarse pressure basis values

        # Local blocks: rows = pressure nodes, cols = velocity nodes
        Bx_loc = -area * np.outer(psi, grads[:, 0])
        By_loc = -area * np.outer(psi, grads[:, 1])

        rowidx = np.repeat(tc, 3)
        colidx = np.tile(tf, 3)

        rowidx_x.extend(rowidx)
        colidx_x.extend(colidx)
        data_x.extend(Bx_loc.ravel())

        rowidx_y.extend(rowidx)
        colidx_y.extend(colidx)
        data_y.extend(By_loc.ravel())

    B_x = sparse.csc_matrix((data_x, (rowidx_x, colidx_x)), shape=(Np_coarse, Np_fine))
    B_y = sparse.csc_matrix((data_y, (rowidx_y, colidx_y)), shape=(Np_coarse, Np_fine))

    return B_x, B_y

def calculate_pressure_B(p_fine, t_fine, p_coarse, t_coarse):
    """
    Assemble Bx, By for P1-iso-P1 — fully vectorized over all fine triangles.

    Key ideas vs. the loop version:
      - 2×2 Jacobian inverse via the closed-form cofactor formula (no linalg.inv)
      - Centroid barycentric solve via the same closed-form trick (no linalg.solve)
      - Local 3×3 blocks via einsum instead of np.outer
      - COO index arrays built with np.repeat / np.tile — no Python lists
    """

    Nt_fine   = t_fine.shape[0]
    Nt_coarse = t_coarse.shape[0]
    Np_fine   = p_fine.shape[0]
    Np_coarse = p_coarse.shape[0]

    fine_to_coarse = np.repeat(np.arange(Nt_coarse), 4)[:Nt_fine]

    # ── 1. Fine-triangle nodes & coarse-triangle nodes ────────────────────────
    tf = t_fine[:, :3]                          # (Nt, 3)
    tc = t_coarse[fine_to_coarse, :3]           # (Nt, 3)

    pf = p_fine[tf]                             # (Nt, 3, 2)
    pc = p_coarse[tc]                           # (Nt, 3, 2)

    # ── 2. Fine Jacobian: J[:,0,:] = e1, J[:,1,:] = e2 ────────────────────────
    # Note: stored as (Nt, 2, 2) where axis-0 of the inner 2×2 indexes the
    # *column* of the physical-space edge vector, matching the loop's layout.
    Jf = np.stack([pf[:, 1] - pf[:, 0],        # column 0  (Nt, 2)
                   pf[:, 2] - pf[:, 0]], axis=2)  # column 1  → (Nt, 2, 2)
    #   Jf[n] = [[e1x, e2x],
    #             [e1y, e2y]]

    # ── 3. det and area ────────────────────────────────────────────────────────
    det_f = Jf[:, 0, 0] * Jf[:, 1, 1] - Jf[:, 0, 1] * Jf[:, 1, 0]   # (Nt,)
    area  = 0.5 * np.abs(det_f)                                         # (Nt,)

    # ── 4. Physical gradients via closed-form 2×2 inverse ─────────────────────
    # inv(J)^T = (1/det) * [[J11, -J01], [-J10, J00]]
    # (cofactors of J, already transposed)
    #
    # grad_hat = [[-1,-1],[1,0],[0,1]]  shape (3,2)
    # grads[n,i,d] = (invJf_T[n] @ grad_hat[i])_d
    #
    # invJf_T rows:  row0 = (1/det)*[ J11, -J10]
    #                row1 = (1/det)*[-J01,  J00]
    inv_det = 1.0 / det_f                                               # (Nt,)

    # Build invJf_T as (Nt, 2, 2)
    invJf_T = np.empty((Nt_fine, 2, 2))
    invJf_T[:, 0, 0] =  Jf[:, 1, 1] * inv_det
    invJf_T[:, 0, 1] = -Jf[:, 1, 0] * inv_det
    invJf_T[:, 1, 0] = -Jf[:, 0, 1] * inv_det
    invJf_T[:, 1, 1] =  Jf[:, 0, 0] * inv_det

    grad_hat = np.array([[-1., -1.],
                         [ 1.,  0.],
                         [ 0.,  1.]])            # (3, 2)

    # grads[n, i, d] = sum_k invJf_T[n, d, k] * grad_hat[i, k]
    grads = np.einsum('ndk,ik->nid', invJf_T, grad_hat)  # (Nt, 3, 2)

    # ── 5. Coarse Jacobian + centroid barycentric solve ────────────────────────
    # Centroid of fine triangle in physical space
    q = pf.mean(axis=1)                          # (Nt, 2)

    # Coarse-triangle edge matrix  Jc[:,0,:] = e1c, Jc[:,1,:] = e2c
    Jc = np.stack([pc[:, 1] - pc[:, 0],
                   pc[:, 2] - pc[:, 0]], axis=2) # (Nt, 2, 2)

    det_c = Jc[:, 0, 0] * Jc[:, 1, 1] - Jc[:, 0, 1] * Jc[:, 1, 0]  # (Nt,)
    inv_det_c = 1.0 / det_c

    # Closed-form 2×2 solve:  x = inv(Jc) @ (q - pc0)
    rhs = q - pc[:, 0]                           # (Nt, 2)
    lam1 = inv_det_c * ( Jc[:, 1, 1] * rhs[:, 0] - Jc[:, 0, 1] * rhs[:, 1])
    lam2 = inv_det_c * (-Jc[:, 1, 0] * rhs[:, 0] + Jc[:, 0, 0] * rhs[:, 1])
    lam0 = 1.0 - lam1 - lam2

    psi = np.stack([lam0, lam1, lam2], axis=1)   # (Nt, 3)

    # ── 6. Local B blocks ──────────────────────────────────────────────────────
    # Bx_loc[n, i, j] = -area[n] * psi[n,i] * grads[n,j,0]
    # By_loc[n, i, j] = -area[n] * psi[n,i] * grads[n,j,1]
    Bx_loc = -area[:, None, None] * np.einsum('ni,nj->nij', psi, grads[:, :, 0])  # (Nt,3,3)
    By_loc = -area[:, None, None] * np.einsum('ni,nj->nij', psi, grads[:, :, 1])  # (Nt,3,3)

    # ── 7. COO index arrays ────────────────────────────────────────────────────
    # rows ↔ coarse pressure nodes (tc), cols ↔ fine velocity nodes (tf)
    # Each (i, j) local pair: row = tc[n,i], col = tf[n,j]
    # Outer axis = pressure (i in 0..2), inner axis = velocity (j in 0..2)
    rowidx = np.repeat(tc, 3, axis=1).ravel()    # (Nt*9,)  tc[:,i] repeated 3×
    colidx = np.tile(tf, (1, 3)).ravel()          # (Nt*9,)  tf[:,j] tiled 3×

    B_x = sparse.csc_matrix(
        (Bx_loc.ravel(), (rowidx, colidx)),
        shape=(Np_coarse, Np_fine)
    )
    B_y = sparse.csc_matrix(
        (By_loc.ravel(), (rowidx, colidx)),
        shape=(Np_coarse, Np_fine)
    )

    return B_x, B_y
#_______________________________________________________________________________________________________________________________________________________________

def calculate_Saddle_point_K(A, B_x, B_y):
    """Calculates the Saddle-Point matrix **K**"""

    Np_coarse = B_x.shape[0]
    Zero_pp = sparse.csc_matrix((Np_coarse, Np_coarse))

    K_mat = bmat([
        [A,       None,    B_x.T],
        [None,    A,       B_y.T],
        [B_x,     B_y,     Zero_pp]
    ], format='csc')

    return K_mat

#_______________________________________________________________________________________________________________________________________________________________

def calculate_Neumann_BCs():
    return

#_______________________________________________________________________________________________________________________________________________________________

def calculate_Dirichlet_BCs(A_mat, lifting_function):
    lf_x, lf_y = lifting_function
    return A_mat @ lf_x, A_mat @ lf_y

#_______________________________________________________________________________________________________________________________________________________________

def calculate_pressure_lifting(B_x, B_y, lifting_function):
    lf_x, lf_y = lifting_function
    return B_x @ lf_x + B_y @ lf_y

#_______________________________________________________________________________________________________________________________________________________________

def calculate_F(A_mat, B_x, B_y, lifting_function):
    F_x, F_y = calculate_Dirichlet_BCs(A_mat, lifting_function)
    p = calculate_pressure_lifting(B_x, B_y, lifting_function)
    return np.concatenate([-F_x, -F_y, p])

#_______________________________________________________________________________________________________________________________________________________________

def evalOnTrigs(p, t, viscosity_function): # by Stefan Takacs
    
    Nt = t.shape[0]
    result = np.zeros(Nt,'d')
    for i in range(Nt):
        q = ( p[t[i,0]] + p[t[i,1]] + p[t[i,2]] ) / 3
        result[i] = viscosity_function(*q,*t[i,6:])

    return result

#===============================================================================================================================================================
# SAVING/LOADING
#===============================================================================================================================================================

def save_simulation_data(p_fine, e_fine, t_fine, 
                         p_coarse, e_coarse, t_coarse,
                         ux, uy, p_sol,
                         name:str='SIM_n'):
    """Saves the solution and grid into a single file in compressed ```.npz``` format."""
    
    np.savez_compressed(f'Solutions/{name}.npz',
                        p_fine=p_fine,                        
                        e_fine=e_fine,
                        t_fine=t_fine,
                        p_coarse=p_coarse,                        
                        e_coarse=e_coarse,
                        t_coarse=t_coarse,
                        ux=ux,
                        uy=uy,
                        p_sol=p_sol)
    
    print(f"Simulation '{name}' data saved.")  

#_______________________________________________________________________________________________________________________________________________________________

def load_simulation_data(file_path:str='Solutions/Exchanger_device.npz'):
    """Loads the data from the compressed ```.npz``` format."""

    data = np.load(file_path)

    p_fine = data['p_fine']    
    e_fine = data['e_fine']
    t_fine = data['t_fine']

    p_coarse = data['p_coarse']
    e_coarse = data['e_coarse']
    t_coarse = data['t_coarse']

    ux = data['ux']
    uy = data['uy']
    p_sol = data['p_sol']

    return p_fine, e_fine, t_fine, p_coarse, e_coarse, t_coarse, ux, uy, p_sol
