import numpy as np
from scipy import sparse
from scipy.sparse import bmat, linalg
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

from .Mesh_processing import (refine, 
                              refine_n_times, 
                              fix_orientation, 
                              build_stable_mesh, 
                              Plot_Initial_Refined_meshes)

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

#===============================================================================================================================================================
# VISUALIZATIONS
#===============================================================================================================================================================

def Stokes_matrix_structure(A_B_M_K_mat, mat_name:str='A/B_x/B_y/M/K',
                            figsize:tuple=(13,13), cmap:str='viridis',
                            savetype:str='jpeg'):
    """Plots the B matrix values and color codes them."""

    A_B_K_coo = A_B_M_K_mat.tocoo()
    data_abs = np.abs(A_B_K_coo.data)
    
    fig, mat_plot = plt.subplots(figsize=figsize)
    sc = mat_plot.scatter(A_B_K_coo.col, A_B_K_coo.row, 
                          c=A_B_K_coo.data,      
                          s=1,
                          norm=LogNorm(vmin=data_abs.min() + 1e-16, vmax=data_abs.max()),
                          cmap=cmap,   
                          marker='s',
                          linewidths=0,
                          edgecolors='none', 
                          antialiaseds=False)
    
    mat_plot.set_xlim([0, A_B_M_K_mat.shape[1]])
    mat_plot.set_ylim([0, A_B_M_K_mat.shape[0]])
    mat_plot.invert_yaxis()

    divider = make_axes_locatable(mat_plot)
    cax = divider.append_axes("right", size="3%", pad=0.1)    
    plt.colorbar(sc, cax=cax, label='Matrix Entry Value')  
    
    mat_plot.set_aspect('equal')
    mat_plot.set_title(f"{mat_name}: {A_B_M_K_mat.shape[0]}x{A_B_M_K_mat.shape[1]}")

    plt.tight_layout()    
    plt.savefig(f'Outputs/Stokes_{mat_name}_matrix.{savetype}')
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

def K_matrix_structure(K_mat, dim_A, dim_B, 
                       figsize:tuple=(13,13), cmap:str='viridis',
                       savetype:str='jpeg'):
    """Plots the Saddle-point K matrix with labeled block boundaries"""

    K_coo = K_mat.tocoo()
    _, mat_plot = plt.subplots(figsize=figsize)
    sc = mat_plot.scatter(K_coo.col, K_coo.row, 
                          c=K_coo.data, 
                          s=1, 
                          cmap=cmap, 
                          marker='s', 
                          linewidths=0, 
                          edgecolors='none', 
                          antialiased=False)
    
    mat_plot.set_xlim([0, K_mat.shape[0]])
    mat_plot.set_ylim([0, K_mat.shape[0]])
    mat_plot.set_yticks([0, dim_A, 2*dim_A, 2*dim_A + dim_B])
    mat_plot.set_xticks([0, dim_A, 2*dim_A, 2*dim_A + dim_B])
    mat_plot.invert_yaxis()
    mat_plot.set_aspect('equal')

    offsets = [0, dim_A, 2*dim_A, 2*dim_A + dim_B]
    labels = [['$\\mathbf{\\mathbb{A}}$',   '$\\mathbf{\\mathbb{0}}$',   '$\\mathbf{\\mathbb{B}}_x^T$'],
              ['$\\mathbf{\\mathbb{0}}$',   '$\\mathbf{\\mathbb{A}}$',   '$\\mathbf{\\mathbb{B}}_y^T$'],
              ['$\\mathbf{\\mathbb{B}}_x$', '$\\mathbf{\\mathbb{B}}_y$', '$\\mathbf{\\mathbb{0}}$']]

    for i in range(3):
        for j in range(3):

            h = offsets[i+1] - offsets[i]
            w = offsets[j+1] - offsets[j]
            
            rect = patches.Rectangle((offsets[j], offsets[i]), w, h, 
                                     linewidth=2.3, edgecolor="#CA0707", facecolor='none', alpha=0.6)
            mat_plot.add_patch(rect)
            
            mat_plot.text(offsets[j] + w/2, offsets[i] + h/2, labels[i][j], 
                          color="#004216", fontsize=25, fontweight='bold', ha='center', va='center',
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    
    divider = make_axes_locatable(mat_plot)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(sc, cax=cax, label='Matrix Entry Value')

    mat_plot.set_title(f"Saddle-Point Matrix K: {K_mat.shape[0]}x{K_mat.shape[1]}", fontsize=15)
    plt.tight_layout()
    plt.savefig(f'Outputs/Stokes_K_matrix_labeled.{savetype}')
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

def plot_streamlines(p_fine, t_fine, ux, uy,
                     density:float=3.5, 
                     levels:int=90,
                     cmap:str='viridis',
                     grid_num:tuple=(300,300),
                     figsize:tuple=(14, 6),
                     savetype:str='jpeg'):
    """
    Streamline visualization with automatic topology-based geometry masking.
    """
    x = p_fine[:, 0]
    y = p_fine[:, 1]

    nx, ny = grid_num
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)

    # Interpolation of FEM velocity:
    U = griddata((x, y), ux, (X, Y), method='cubic')
    V = griddata((x, y), uy, (X, Y), method='cubic')

    triang_v = tri.Triangulation(x, y, t_fine[:, :3])
    trifinder = triang_v.get_trifinder()
    
    # If a grid point (X, Y) lands in a hole/obstacle, trifinder returns -1
    geometry_mask = (trifinder(X, Y) == -1)
    U = np.ma.array(U, mask=geometry_mask)
    V = np.ma.array(V, mask=geometry_mask)

    speed = np.ma.sqrt(U**2 + V**2)

    # The Plot:
    _, ax = plt.subplots(figsize=figsize)

    cf = ax.contourf(
        X, Y,
        speed,
        levels=levels,
        cmap=cmap
    )
    
    # plt.colorbar(cf, ax=ax, label='$\\|\\vec{u}\\|$')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(cf, cax=cax, label='$\\|\\vec{u}\\|$')

    ax.streamplot(
        X, Y,
        U, V,
        density=density,
        linewidth=1.2,
        arrowsize=1.2,
        color='white'
    )

    x_min, x_max = p_fine[:,0].min(), p_fine[:,0].max()
    x_margin = np.abs(x_max - x_min)*0.03
    y_min, y_max = p_fine[:,1].min(), p_fine[:,1].max()
    y_margin = np.abs(y_max - y_min)*0.03

    ax.set_xlim([x_min - x_margin, x_max + x_margin])
    ax.set_ylim([y_min - y_margin, y_max + y_margin])

    ax.set_aspect('equal')
    ax.set_title("Streamlines of $\\vec{u}$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(f'Outputs/Solution_Streamlines.{savetype}')
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

def plot_pressure(p_coarse, t_coarse, p_sol,
                  levels:int=90,
                  figsize:tuple=(10,10),
                  savetype:str='jpeg'):
    """Plots the pressure"""

    _, plots = plt.subplots(figsize=figsize)

    triangulation = tri.Triangulation(p_coarse[:, 0], p_coarse[:, 1], t_coarse[:, :3])
    cf = plots.tricontourf(triangulation, p_sol, levels=levels)

    divider = make_axes_locatable(plots)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(cf, cax=cax, label='$\\mathbf{P}$')
  
    plots.set_xlabel("x")
    plots.set_ylabel("y")
    plots.set_aspect('equal')

    x_min, x_max = p_coarse[:,0].min(), p_coarse[:,0].max()
    x_margin = np.abs(x_max - x_min)*0.03
    y_min, y_max = p_coarse[:,1].min(), p_coarse[:,1].max()
    y_margin = np.abs(y_max - y_min)*0.03

    plots.set_xlim([x_min - x_margin, x_max + x_margin])
    plots.set_ylim([y_min - y_margin, y_max + y_margin])

    plots.set_title('Pressure $\\mathbf{P}$')
    plt.savefig(f'Outputs/Pressure_Tricontourf.{savetype}')
    plt.show()
