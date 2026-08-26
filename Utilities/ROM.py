import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, Math

from .Plot_functions import (
    plot_streamlines,
    plot_pressure
)

#_______________________________________________________________________________________________________________________________________________________________

def ROM_solution_statistics(ux_FOM, uy_FOM, p_FOM, ux_ROM, uy_ROM, p_ROM):

    Umag_FOM  = np.sqrt(ux_FOM**2 + uy_FOM**2)
    Umag_ROM  = np.sqrt(ux_ROM**2 + uy_ROM**2)
    abs_err_u = np.abs(Umag_ROM - Umag_FOM)
    abs_err_p = np.abs(p_ROM - p_FOM)

    rel_u = np.linalg.norm(Umag_ROM - Umag_FOM) / np.linalg.norm(Umag_FOM)
    rel_p = np.linalg.norm(p_ROM - p_FOM) / np.linalg.norm(p_FOM)

    display(Math(fr"\text{{Relative }} L_2 \text{{ error }} \|u\|_2: {rel_u:.3e}"))
    display(Math(fr"\text{{Relative }} L_2 \text{{ error }} p : {rel_p:.3e}"))

    return Umag_FOM, Umag_ROM, abs_err_u, abs_err_p, rel_u, rel_p

#_______________________________________________________________________________________________________________________________________________________________

def ROM_FOM_comparison(ux_true, uy_true, p_true, ux_rom, uy_rom, p_rom,
                       p_fine, t_fine, p_coarse, t_coarse, test_idx,
                       density=3.1, levels=20, figsize=(23, 7.5),
                       savetype='png'):

    Umag_true, Umag_rom, abs_err_u, abs_err_p, rel_u, rel_p = ROM_solution_statistics(ux_true, uy_true, p_true,
                                                                                      ux_rom,  uy_rom,  p_rom)
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
    plt.savefig(f'Outputs/ROM_FEM_comparison.{savetype}', bbox_inches='tight', pad_inches=0.01)
    plt.show()