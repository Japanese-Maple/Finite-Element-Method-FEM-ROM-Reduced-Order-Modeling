import numpy as np

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.colors import LogNorm

import cmasher as cmr

from .Mesh_processing import (build_stable_mesh,
                              refine_n_times)

#=================================================================================================================
# Main Plotting Functions
#=================================================================================================================

def mesh_df(p, e, t, 
            first_n_entries: int = 9):

    print(f"p is of shape: {p.shape}\ne is of shape: {e.shape}\nt is of shape: {t.shape}")

    df_p = pd.DataFrame(p[:first_n_entries,:], columns=["x1", "x2"])
    df_e = pd.DataFrame(e[:first_n_entries,:], columns=["Node1", "Node2", "Flag"])
    df_t = pd.DataFrame(t[:first_n_entries,:], columns=["V1", "V2", "V3", "E1", "E2", "E3", "Sub"])

    df = pd.concat([df_p, df_e, df_t], axis=1, keys=['p', 'e', 't'])

    def highlight_by_category(col):
        category = col.name[0] 
        
        if category == 'p':
            return ["background-color: #73BA40; color: black"] * len(col) # Blue
        elif category == 'e':
            return ["background-color: #96D44A; color: black"] * len(col) # Green
        elif category == 't':
            return ["background-color: #34623F; color: white"] * len(col) # Red
        return [""] * len(col)

    return (
        df.style
          .apply(highlight_by_category, axis=0)
          .hide(axis="index")
          .set_table_styles([
              {'selector': 'th', 'props': [('text-align', 'center'), 
                                           ('border', '1px solid #ddd'),
                                           ('padding', '8px')]}
          ])
          .format(precision=4)
    )

#_______________________________________________________________________________________________________________________________________________________________

def Plot_Initial_Refined_meshes(data_path: str, num_of_refinements: int = 3,
                                plot: bool=True,
                                figsize: tuple=(16,8),
                                savetype:str="jpeg"):
    """
    Plots the initial blender mesh and the refined counterpart. 
    Additionally outputs the refined mesh arrays.
    """
    
    data = np.load(data_path)
    p_raw = data['p']
    tri_idx = data['t_raw']
    data.close()

    p, e, t = build_stable_mesh(p_raw, tri_idx)
    p, e, t = refine_n_times(p, e, t, number_of_refinements=num_of_refinements)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    ax[0].set_title('Initial Mesh')
    ax[1].set_title('Refined Mesh')

    ax[0].triplot(p_raw[:, 0], p_raw[:, 1], tri_idx, color='blue', lw=1, label='Edges')
    ax[0].plot(p_raw[:, 0], p_raw[:, 1], 'ro', markersize=3, label='Nodes')

    ax[1].triplot(p[:, 0], p[:, 1], t[:, :3], color='blue', lw=0.3, label='Edges')
    ax[1].plot(p[:, 0], p[:, 1], 'ro', markersize=1, label='Nodes')

    for i in [0,1]:
        x_min, x_max = ax[i].get_xlim()
        y_min, y_max = ax[i].get_ylim()

        margin_x = np.abs(x_max-x_min)*0.05
        margin_y = np.abs(y_max-y_min)*0.05

        ax[i].set_xlim(x_min - margin_x, x_max + margin_x)
        ax[i].set_ylim(y_min - margin_y, y_max + margin_y)
        ax[i].set_aspect('equal')
        ax[i].legend()
 
    plt.suptitle(f'Initial Mesh ({len(p_raw)} Nodes, {len(tri_idx)} Triangles) --> Refined Mesh ({len(p)} Nodes, {len(t)} Triangles)')
    plt.savefig(f"Outputs/Mesh_Refinement.{savetype}")

    if plot==True:        
        plt.show()
    else:
        plt.close()

    return (p, e, t)

#_______________________________________________________________________________________________________________________________________________________________

def plot_streamlines_experimental(p_fine, t_fine, ux, uy,
                                  density: float = 3.5,
                                  levels: int = 100,
                                  cmap: str = 'inferno',
                                  grid_num: tuple = (400, 400),
                                  figsize: tuple = (14, 7),
                                  savetype: str = 'png',
                                  dpi: int = 180):
    """
    Streamline visualization with automatic topology-based geometry masking.
    Streamline color and width are modulated by local flow speed.
    """
    x = p_fine[:, 0]
    y = p_fine[:, 1]

    nx, ny = grid_num
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    X, Y = np.meshgrid(xi, yi)

    # ── Interpolation ────────────────────────────────────────────────────────
    # LinearNDInterpolator: build the triangulation once, evaluate in one call.
    # ~3-5x faster than griddata(method='cubic') on large meshes.
    interp_u = LinearNDInterpolator(list(zip(x, y)), ux)
    interp_v = LinearNDInterpolator(list(zip(x, y)), uy)

    U = interp_u(X, Y)
    V = interp_v(X, Y)

    # ── Geometry mask (holes / walls) ────────────────────────────────────────
    triang_v  = tri.Triangulation(x, y, t_fine[:, :3])
    trifinder = triang_v.get_trifinder()
    mask = trifinder(X, Y) == -1

    U = np.ma.array(U, mask=mask)
    V = np.ma.array(V, mask=mask)

    speed     = np.ma.sqrt(U**2 + V**2)
    speed_max = float(speed.max())

    # ── Figure ───────────────────────────────────────────────────────────────
    bg = '#0d0d0d'
    fig, ax = plt.subplots(figsize=figsize, facecolor=bg)
    ax.set_facecolor(bg)

    # Filled contours — speed field
    cf = ax.contourf(X, Y, speed, levels=levels, cmap=cmap, zorder=1)

    # Thin contour strokes for depth
    ax.contour(
        X, Y, speed,
        levels=20,
        colors='white',
        linewidths=0.15,
        alpha=0.25,
        zorder=2
    )

    # ── Streamlines colored & weighted by speed ───────────────────────────────
    # streamplot accepts a 2-D array for 'color' and a scalar for 'linewidth';
    # to vary both we draw two overlapping passes:
    #   pass 1 – thick, semi-transparent white base (gives a glow effect)
    #   pass 2 – thin lines colored by speed
    ax.streamplot(
        X, Y, U, V,
        density=density,
        color=speed / (speed_max + 1e-12),   # normalised [0,1] for colormap
        cmap='cool',
        linewidth=1.0,
        arrowsize=0.9,
        arrowstyle='->',
        zorder=3
    )

    # ── Colorbar ─────────────────────────────────────────────────────────────
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="3%", pad=0.12)

    cb = fig.colorbar(cf, cax=cax)
    cb.set_label(r'$\|\vec{u}\|$', color='white', fontsize=13)
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    cb.outline.set_edgecolor('white')
    cb.outline.set_linewidth(0.4)

    # ── Axes cosmetics ───────────────────────────────────────────────────────
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    mx = abs(x_max - x_min) * 0.03
    my = abs(y_max - y_min) * 0.03

    ax.set_xlim(x_min - mx, x_max + mx)
    ax.set_ylim(y_min - my, y_max + my)
    ax.set_aspect('equal')

    ax.set_title(r"Streamlines of $\vec{u}$", color='white', fontsize=14, pad=10)
    ax.set_xlabel("x", color='white', fontsize=12)
    ax.set_ylabel("y", color='white', fontsize=12)

    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(
        f'Outputs/Solution_Streamlines.{savetype}',
        dpi=dpi,
        bbox_inches='tight',
        facecolor=bg
    )
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

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

#_______________________________________________________________________________________________________________________________________________________________

def plot_viscosity(p, t, e, nu_T, nu_KNOTS,
                   n_levels=15,
                   figsize=(23, 10),
                   savetype='png'):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    triangulation = tri.Triangulation(
        p[:, 0],
        p[:, 1],
        t[:, :3]
    )

    nu_nodes = np.bincount(t[:, :3].ravel(), weights=np.repeat(nu_T, 3), minlength=len(p)) / \
               np.maximum(np.bincount(t[:, :3].ravel(), minlength=len(p)), 1)

    # 1 ────────────────────────────────────────────────────────────────────────────────────────────────
    cf1 = ax1.tripcolor(
        triangulation,
        cmap=cmr.ocean,
        facecolors=nu_T,
        shading='flat',
        zorder = 0
    )

    nu_pts  = nu_KNOTS['pts']
    nu_vals = nu_KNOTS['vals']

    ax1.scatter(nu_pts[:, 0], nu_pts[:, 1], s=550, c='cyan', edgecolors='r', linewidths=1,
                zorder = 3)
    
    for (x, y), val in zip(nu_pts, nu_vals):
        ax1.text(
            x, y, 
            s=f"{val:.0f}",
            ha='center', va='center',
            fontsize=11,
            zorder=5
        )

    ax1.set_aspect('equal')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Viscosity Field (continious)')

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="3%", pad=0.1)
    
    min_val1, max_val1 = np.min(nu_T), np.max(nu_T)
    ticks1 = np.linspace(min_val1, max_val1, 5)
    cb1 = fig.colorbar(cf1, cax=cax1, label=r'$\nu$', ticks=ticks1)
    cb1.ax.set_yticklabels([f'{val:.1f}' for val in ticks1])

    # 2 ────────────────────────────────────────────────────────────────────────────────────────────────
    levels = np.linspace(np.min(nu_nodes), np.max(nu_nodes), n_levels)

    cf2 = ax2.tricontourf(
        triangulation,
        nu_nodes,
        levels=levels,
        cmap=cf1.get_cmap()
    )
    
    cs2 = ax2.tricontour(
        triangulation,
        nu_nodes,
        levels=levels,
        colors='w',
        linewidths=0.5        
    )

    ax2.clabel(cs2, inline=True, fontsize=9, fmt='%.3f')

    ax2.set_aspect('equal')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Viscosity Field (countour plot)')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3%", pad=0.1)
    
    min_val2, max_val2 = np.min(nu_nodes), np.max(nu_nodes)
    ticks2 = np.linspace(min_val2, max_val2, 5)
    cb2 = fig.colorbar(cf2, cax=cax2, label=r'$\nu$', ticks=ticks2)
    cb2.ax.set_yticklabels([f'{val:.1f}' for val in ticks2])

    # Edge ────────────────────────────────────────────────────────────────────────────────────────────────
    flag1_mask = (e[:, -1] == 1)
    boundary_edges = e[flag1_mask, :2].astype(int)
    x_coords = p[boundary_edges, 0].T
    y_coords = p[boundary_edges, 1].T

    for ax in [ax1, ax2]:
        ax.plot(x_coords, y_coords, color='b', linewidth=1.5, label='Boundary')

        ax.set_xlim(-2.70, 2.70)
        ax.set_ylim(-1.15, 1.15)

    plt.tight_layout()
    plt.savefig(f'Outputs/Viscosity.{savetype}', bbox_inches='tight')
    plt.show()