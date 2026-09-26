import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.colors import SymLogNorm
from matplotlib.path import Path

from IPython.display import display, Math

from .Plot_functions import (
    plot_streamlines,
    plot_pressure
)

#_______________________________________________________________________________________________________________________________________________________________

def ROM_solution_statistics(ux_FOM, uy_FOM, p_FOM, ux_ROM, uy_ROM, p_ROM,
                            display_errors:bool=False):

    Umag_FOM  = np.sqrt(ux_FOM**2 + uy_FOM**2)
    Umag_ROM  = np.sqrt(ux_ROM**2 + uy_ROM**2)
    abs_err_u = np.abs(Umag_ROM - Umag_FOM)
    abs_err_p = np.abs(p_ROM - p_FOM)

    rel_u = np.linalg.norm(Umag_ROM - Umag_FOM) / np.linalg.norm(Umag_FOM)
    rel_p = np.linalg.norm(p_ROM - p_FOM) / np.linalg.norm(p_FOM)

    if display_errors:
        display(Math(fr"\text{{Relative }} L_2 \text{{ error }} \|u\|_2: {rel_u:.3e}"))
        display(Math(fr"\text{{Relative }} L_2 \text{{ error }} p : {rel_p:.3e}"))

    return Umag_FOM, Umag_ROM, abs_err_u, abs_err_p, rel_u, rel_p

#_______________________________________________________________________________________________________________________________________________________________

def ROM_FOM_comparison(ux_true, uy_true, p_true, ux_rom, uy_rom, p_rom,
                       p_fine, t_fine, p_coarse, t_coarse, test_idx,
                       density=3.1, levels=20, figsize=(23, 7.5),
                       name='1',
                       savetype='png'):

    Umag_true, Umag_rom, abs_err_u, abs_err_p, rel_u, rel_p = ROM_solution_statistics(ux_true, uy_true, p_true,
                                                                                      ux_rom,  uy_rom,  p_rom,
                                                                                      display_errors=True)
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    vel_min = min(Umag_true.min(), Umag_rom.min())
    vel_max = max(Umag_true.max(), Umag_rom.max())

    p_min = min(p_true.min(), p_rom.min())
    p_max = max(p_true.max(), p_rom.max())

    # FOM
    plot_streamlines(p_fine, t_fine, ux_true, uy_true, ax=axes[0, 0], density=density, 
                    field_override=Umag_true, levels=levels, vmin=vel_min, vmax=vel_max)
    axes[0, 0].set_title("FOM $|\\mathbf{u}|$")

    # ROM
    plot_streamlines(p_fine, t_fine, ux_rom, uy_rom, ax=axes[0, 1], density=density,
                    field_override=Umag_rom, levels=levels, vmin=vel_min, vmax=vel_max)
    axes[0, 1].set_title("ROM $|\\mathbf{u}|$")

    # Error Plot
    plot_streamlines(p_fine, t_fine, ax=axes[0, 2], 
                    field_override=abs_err_u, levels=levels, cmap="inferno")
    axes[0, 2].set_title("Abs. Error $|\\mathbf{u}|$")

    # FOM
    plot_pressure(p_coarse, t_coarse, p_true, ax=axes[1, 0], 
                  levels=levels, vmin=p_min, vmax=p_max)
    axes[1, 0].set_title("FOM $p$")

    # ROM
    plot_pressure(p_coarse, t_coarse, p_rom, ax=axes[1, 1], 
                  levels=levels, vmin=p_min, vmax=p_max)
    axes[1, 1].set_title("ROM $p$")

    # Error Plot
    plot_pressure(p_coarse, t_coarse, abs_err_p, ax=axes[1, 2], 
                  levels=levels, cmap="inferno")
    axes[1, 2].set_title("Abs. Error $p$")

    plt.suptitle(
        f"ROM vs FOM (Snapshot {test_idx})\n"
        f"Relative $L^2$ Error: "
        f"$|\\mathbf{{u}}|$ = {rel_u:.2%}, "
        f"$p$ = {rel_p:.2%}",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(f'Outputs/{name}ROM_FEM_comparison.{savetype}', bbox_inches='tight', pad_inches=0.01)
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

def math_sci_fmt(val, pos):
    if np.isclose(val, 0):
        return "$0$"
    mantissa, exp = f"{val:.1e}".split("e")
    exp_int = int(exp)
    return rf"${float(mantissa):.1f} \times 10^{{{exp_int}}}$"

#_______________________________________________________________________________________________________________________________________________________________

def plot_Reduced_Affine_Operators(Reduced_Affine_Operators, name='ROM/'):
    
    num_ops = len(Reduced_Affine_Operators)
    ncols = 5
    nrows = int(np.ceil(num_ops / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 3.2 * nrows),
        sharex=True,
        sharey=True,
        dpi=300,
    )

    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    vmin = min(op.min() for op in Reduced_Affine_Operators)
    vmax = max(op.max() for op in Reduced_Affine_Operators)

    for i, op in enumerate(Reduced_Affine_Operators):
        ax = axes_flat[i]

        im = ax.imshow(
            op, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax, origin="upper"
        )

        ax.set_title(rf"$(\mathbf{{M}}^{{N}}_v)_{{{i+1}}}$", fontsize=12, pad=6)

        if i > 0:
            ax.tick_params(labelleft=False)

        ax.tick_params(labelsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter = ticker.FuncFormatter(math_sci_fmt)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=7)

    for j in range(num_ops, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()

    plt.savefig(
        f"Outputs/{name}Reduced_Affine_Operators.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________


def plot_Reduced_Divergence_Operator(Div_N_dense, figsize=(5, 3), name='ROM/'):

    num_rows, num_cols = Div_N_dense.shape

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    im = ax.imshow(
        Div_N_dense,
        cmap="viridis",
        aspect="equal",
        origin="upper",
    )

    ax.set_title(r"$\mathbf{B}^N$", fontsize=12, pad=10)

    ax.set_xticks(np.arange(num_cols))
    ax.set_yticks(np.arange(num_rows))
    ax.tick_params(labelsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.formatter = ticker.FuncFormatter(math_sci_fmt)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(f"Outputs/{name}Reduced_Divergence_Operator.png", dpi=300, bbox_inches="tight")
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

def plot_Reduced_K_Matrices(
    dense_K,
    dim_A,
    dim_B,
    ncols=5,
    linthresh=1e-4,
    name="ROM/",
    savetype="png",
):
    """Plots a grid of reduced saddle-point K^N matrices with block boundaries using logarithmic scaling.

    Parameters
    ----------
    use_abs : bool
        If True, plots log10(|K^N|) with standard LogNorm.
        If False, uses SymLogNorm to handle negative and positive values around zero.
    linthresh : float
        The range around zero [-linthresh, linthresh] where SymLogNorm transitions to linear.
    """

    num_ops = len(dense_K)
    nrows = max(1, int(np.ceil(num_ops / ncols)))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.2 * nrows),
        sharex=True,
        sharey=True,
        dpi=300,
    )

    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    vmin = min(k.min() for k in dense_K)
    vmax = max(k.max() for k in dense_K)

    max_abs = max(abs(vmin), abs(vmax))
    norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-max_abs if vmin < 0 else linthresh,
        vmax=max_abs,
        base=10,
    )

    offsets = [0, dim_A, dim_A + dim_B]
    labels = [
        [r"$\mathbf{M}^N_v$", r"${(\mathbf{B}^N)}^T$"],
        [r"$\mathbf{B}^N$", r"$\mathbf{0}$"],
    ]

    n_blocks = len(offsets) - 1

    for i, k_mat in enumerate(dense_K):
        ax = axes_flat[i]

        im = ax.imshow(
            k_mat,
            cmap="viridis",
            aspect="equal",
            norm=norm,
            origin="upper",
        )

        ax.set_title(fr"$\mathbf{{K}}^N(\boldsymbol{{\mu}}_{{{i+1}}})$", fontsize=12, pad=6)

        for row_i in range(n_blocks):
            for col_j in range(n_blocks):
                h = offsets[row_i + 1] - offsets[row_i]
                w = offsets[col_j + 1] - offsets[col_j]

                rect = patches.Rectangle(
                    (offsets[col_j] - 0.5, offsets[row_i] - 0.5),
                    w,
                    h,
                    linewidth=1.2,
                    edgecolor="#CA0707",
                    facecolor="none",
                    alpha=0.75,
                    zorder=3,
                )
                ax.add_patch(rect)

                ax.text(
                    offsets[col_j] + w / 2.0 - 0.5,
                    offsets[row_i] + h / 2.0 - 0.5,
                    labels[row_i][col_j],
                    color="#004216",
                    fontsize=7.5,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                        pad=1.0,
                    ),
                    zorder=4,
                )

        ax.set_xticks(offsets)
        ax.set_yticks(offsets)

        if i % ncols != 0:
            ax.tick_params(labelleft=False)

        ax.tick_params(labelsize=7)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.formatter = ticker.FuncFormatter(math_sci_fmt)
        cbar.update_ticks()
        cbar.ax.tick_params(labelsize=7)

    for j in range(num_ops, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"Outputs/{name}Reduced_K_matrices_labeled.{savetype}",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

#_______________________________________________________________________________________________________________________________________________________________

def generate_sensor_locations(region_array, p_nodes, num_sensors=25, base_radius=1, seed=33):
    """
    Generates well-spaced points inside a polygon using Poisson Disk sampling,
    and snaps them to the nearest available mesh nodes using a KDTree.
    """
    path = Path(region_array)
    
    x_min, x_max = region_array[:, 0].min(), region_array[:, 0].max()
    y_min, y_max = region_array[:, 1].min(), region_array[:, 1].max()
    width = x_max - x_min
    height = y_max - y_min
    
    valid_sensors = []
    attempt = 0
    current_radius = base_radius
    
    while len(valid_sensors) < num_sensors:
        engine = qmc.PoissonDisk(d=2, radius=current_radius, seed=seed + attempt)
        candidates_unit = engine.random(n=num_sensors * 10)
        candidates = candidates_unit * [width, height] + [x_min, y_min]
        inside_mask = path.contains_points(candidates)
        candidates_inside = candidates[inside_mask]
        
        for pt in candidates_inside:
            if len(valid_sensors) < num_sensors:
                valid_sensors.append(pt)
            else:
                break
                
        attempt += 1
        if len(valid_sensors) < num_sensors:
            current_radius *= 0.9  
            valid_sensors = []

    raw_sensors = np.array(valid_sensors)
    
    tree = cKDTree(p_nodes)
    _, node_indices = tree.query(raw_sensors)
    
    if len(np.unique(node_indices)) < num_sensors:
        print("Warning: Multiple sensors snapped to the same node. Mesh might be too coarse.")
        
    snapped_sensors = p_nodes[node_indices]

    return snapped_sensors, node_indices

#_______________________________________________________________________________________________________________________________________________________________
