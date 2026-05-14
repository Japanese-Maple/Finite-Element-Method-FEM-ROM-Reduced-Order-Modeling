import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

#=================================================================================================================
# Code for the system assembly was provided bu STEFAN TAKACS
#=================================================================================================================

def assembleStiffness( p, t, coef_a ):
    """
    Assembles stiffness matrix; given (coordinates) of points, 
    triangles and coefficient a, which is given for each element 
    or is a scalar.

    ***

    If coef_a is increased, the material becomes more "conductive."
    """
    Np = p.shape[0]

    # Determine the Jacobian
    jacobi = np.zeros((t.shape[0],2,2))
    jacobi[:,0,:] = p[t[:,1]] - p[t[:,0]]
    jacobi[:,1,:] = p[t[:,2]] - p[t[:,0]]
    
    # Determine the Jacobi determinant
    jacobi_determinant = jacobi[:,0,0] * jacobi[:,1,1] - jacobi[:,0,1] * jacobi[:,1,0]
    assert (jacobi_determinant > 0).all()

    # Determine the cofactor matrix of the Jacobian / Jacobian inverse transposed
    Cf_jacobi = np.zeros((t.shape[0],2,2))
    Cf_jacobi[:,0,0] =  jacobi[:,1,1]
    Cf_jacobi[:,0,1] = -jacobi[:,1,0]
    Cf_jacobi[:,1,0] = -jacobi[:,0,1]
    Cf_jacobi[:,1,1] =  jacobi[:,0,0]
    
    # Shape functions / derivatives / the gradients of hat functions
    shapefunc_der = np.array([ [-1,-1], [ 1, 0], [ 0, 1] ])

    # Data is a Nt x 3 x 3 tensor containing all local stiffness matrices
    data = np.einsum("n,ik,nkl,nml,jm->nij", 0.5*coef_a/jacobi_determinant, shapefunc_der, Cf_jacobi, Cf_jacobi, shapefunc_der)

    # Find the corresponding row/col indicies
    rowidx = np.einsum("ni,j->nij", t[:,0:3], [1,1,1])
    colidx = np.einsum("nj,i->nij", t[:,0:3], [1,1,1])
    
    # Return corresponding csc_matrix
    return sparse.csc_matrix((np.ravel(data),(np.ravel(rowidx),np.ravel(colidx))),
                             shape=(Np,Np))

def assembleLoad( p, t, coef_f ):
    """
    Assembles load vector; given (coordinates) of points, 
    triangles and coefficient f, which is given for each 
    element or is a scalar
    """
    Np = p.shape[0]

    # Determine the Jacobian
    jacobi = np.zeros((t.shape[0],2,2))
    jacobi[:,0,:] = p[t[:,1]] - p[t[:,0]]
    jacobi[:,1,:] = p[t[:,2]] - p[t[:,0]]
    
    # Determine the Jacobi determinant
    jacobi_determinant = jacobi[:,0,0] * jacobi[:,1,1] - jacobi[:,0,1] * jacobi[:,1,0]
    assert (jacobi_determinant > 0).all()

    # The integrals on reference element are known
    integrals = np.array([ 1/6, 1/6, 1/6 ])

    # Data is a Nt x 3 matrix containing all local load vectors
    data = np.einsum( "n,i->ni", coef_f * jacobi_determinant, integrals )

    # Assemble
    vector = np.zeros(Np)
    np.add.at(vector,np.ravel(t[:,0:3]),np.ravel(data))    
    return vector

def evalOnTrigs( p, t, callback ):
    """
    evaluates the function (for instance, f(x,y)) on 
    triangles to obtain coefficients
    """
    Nt = t.shape[0]
    result = np.zeros(Nt,'d')
    for i in range(Nt):
        q = ( p[t[i,0]] + p[t[i,1]] + p[t[i,2]] ) / 3
        result[i] = callback(*q,*t[i,6:])
    return result

def Plot_the_solution(u, p, t, e, ax=None, material_function=None,
                      color_map: str = 'viridis', 
                      level_curve_number: int = 50, 
                      mesh_alpha: float = 0.1, 
                      figsize: tuple=(10, 10), 
                      legend: bool=True):
    """
    Plots the mesh and the solution of the poisson problem.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    triangulation = tri.Triangulation(p[:, 0], p[:, 1], t[:, :3])

    if material_function is not None:
        mat_on_nodes = np.array([material_function(px, py, 0) for px, py in p])
        
        k_min = np.min(mat_on_nodes)
        k_max = np.max(mat_on_nodes)
        
        midpoint = (k_min + k_max) / 2
        
        if k_max - k_min > 1e-5:
            ax.tricontour(triangulation, mat_on_nodes, levels=[midpoint], 
                          colors='#009B7F', linestyles='--', linewidths=2, antialiased=True)
            
    ax.tricontour(triangulation, u, levels=[0.0], colors='#5C6BA7', linestyles='-.', linewidths=1.7)
    cntr = ax.tricontourf(triangulation, u, levels=level_curve_number, cmap=color_map)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(cntr, cax=cax)

    boundary_indices = np.where(e[:, 2] > 0)[0]
    for idx in boundary_indices:
        node1, node2 = int(e[idx, 0]), int(e[idx, 1]) 
        ax.plot([p[node1, 0], p[node2, 0]], 
                [p[node1, 1], p[node2, 1]], 
                color='b', lw=2, zorder=3) 

    ax.triplot(triangulation, color='black', lw=0.5, alpha=mesh_alpha, zorder=1)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    margin_x = np.abs(x_max - x_min) * 0.05
    margin_y = np.abs(y_max - y_min) * 0.05

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    ax.set_aspect('equal')
    ax.set_title(r"Poisson Solution $-\nabla^2u=f$")

    if legend==True:
        cmap_obj = plt.get_cmap(color_map)

        low_color  = cmap_obj(0.0)
        high_color = cmap_obj(1.0)

        legend_elements = [
            Line2D([0], [0], color='#009B7F', lw=2, ls='--', label='Material Transition'),
            Patch(facecolor=high_color, edgecolor='black', alpha=0.8, label=r'Solution $(\max (u))$'),
            Patch(facecolor=low_color,  edgecolor='black', alpha=0.8, label=r'Solution $(\min (u))$'),
            Line2D([0], [0], color='black', lw=1, alpha=0.5, label='Mesh Grid'),
            Line2D([0], [0], color='blue',  lw=2, label='Boundary (e)'),
            Line2D([0], [0], color="#5C6BA7",   lw=2, ls='-.', label=r'u(x,y)=0')
        ]

        ax.legend(handles=legend_elements, 
                  loc='upper left', 
                  bbox_to_anchor=(1.15, 1.0), 
                  fontsize='small',
                  frameon=True,
                  shadow=True)

def Plot_Modes(v_rb, k_eigenvalues_kept, p, t, e, 
               nrows: int=3, ncols: int=5, stepping_factor: float=1,
               mesh_alpha: float=0.1):
    total = nrows * ncols

    def scaling(x, beta, alpha):
        return np.floor(beta * np.exp(x * np.log(alpha / beta) / alpha))

    x = np.linspace(1, k_eigenvalues_kept-1, total)
    l = scaling(x, stepping_factor, k_eigenvalues_kept-1)
    l_m = np.reshape(l, (nrows, ncols))

    fig, canvas = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    if total == 1:
        canvas = np.array([[canvas]])
    elif nrows == 1 or ncols == 1:
        canvas = canvas.reshape(nrows, ncols)

    for i in range(nrows):
        for j in range(ncols):
            mode_idx = int(l_m[i, j])
            Plot_the_solution(v_rb[:, mode_idx], p, t, e, ax=canvas[i, j], mesh_alpha=mesh_alpha, legend=False)
            canvas[i, j].set_title(f"Mode {mode_idx}")

    plt.tight_layout()
    plt.savefig('Outputs/Modes_ROM.pdf')
    plt.show()

def Plot_the_solution_3d(u, p, t, e, 
                         name_of_the_plot: str="Poisson Solution",
                         cmap: str='Viridis', wireframe_color: str="#2D00A9", boundary_color: str='blue'):
    
    x, y, z = p[:, 0], p[:, 1], u
    i, j, k = t[:, 0], t[:, 1], t[:, 2]

    surface = go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        intensity=z, 
        showscale=False,
        colorscale=cmap,
        opacity=1.0, 
        name='Solution',
        flatshading=False  
    )

    tri_vertices = p[t[:, :3]]
    xe, ye, = [], []
    for triangle in tri_vertices:
        xe.extend([triangle[0,0], triangle[1,0], triangle[2,0], triangle[0,0], None])
        ye.extend([triangle[0,1], triangle[1,1], triangle[2,1], triangle[0,1], None])

    wireframe = go.Scatter3d(
        x=xe, y=ye, z=np.zeros_like(ye),
        mode='lines',
        line=dict(color=wireframe_color, width=1),
        hoverinfo='none',
        showlegend=False
    )

    fig = go.Figure(data=[surface, wireframe])

    boundary_indices = np.where(e[:, 2] > 0)[0]
    for idx in boundary_indices:
        n1, n2 = e[idx, 0], e[idx, 1]
        fig.add_trace(go.Scatter3d(
            x=[p[n1, 0], p[n2, 0]],
            y=[p[n1, 1], p[n2, 1]],
            z=[u[n1], u[n2]], 
            mode='lines',
            line=dict(color=boundary_color, width=4),
            showlegend=False
        ))

    fig.update_layout(
        width=900, height=700,
        title=name_of_the_plot,
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=True, gridcolor='lightgrey', title='X'),
            yaxis=dict(showbackground=False, showgrid=True, gridcolor='lightgrey', title='Y'),
            zaxis=dict(showbackground=False, showgrid=True, gridcolor='lightgrey', title='u'),
            aspectmode='data', 
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)) 
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()

def ROM_comparison(u_true, u_approx, k_eigenvalues_kept, p, e, t):

    # Error
    error = np.linalg.norm(u_true - u_approx) / np.linalg.norm(u_true)
    print(f"Relative Error: {error:.2e}")
    print(f"True solution between: {min(u_true):.2e} and {max(u_true):.2e}")
    print(f"Approx solution between: {min(u_approx):.2e} and {max(u_approx):.2e}")

    triangulation = tri.Triangulation(p[:, 0], p[:, 1], t[:, :3])
    boundary_indices = np.where(e[:, 2] > 0)[0]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].tricontourf(triangulation, u_true)
    ax[1].tricontourf(triangulation, u_approx)

    for i in [0,1]:
        
        for idx in boundary_indices:
            node1, node2 = e[idx, 0], e[idx, 1]
            ax[i].plot([p[node1, 0], p[node2, 0]], 
                    [p[node1, 1], p[node2, 1]], 
                    color='b', lw=2, zorder=3) 
            
        x_min, x_max = ax[i].get_xlim()
        y_min, y_max = ax[i].get_ylim()

        margin_x = np.abs(x_max-x_min)*0.05
        margin_y = np.abs(y_max-y_min)*0.05

        ax[i].set_xlim(x_min - margin_x, x_max + margin_x)
        ax[i].set_ylim(y_min - margin_y, y_max + margin_y)
        ax[i].set_aspect('equal')

    ax[0].set_title("True Solution (FOM)")
    ax[1].set_title(f"ROM Approximation (k={k_eigenvalues_kept})")

    plt.show()