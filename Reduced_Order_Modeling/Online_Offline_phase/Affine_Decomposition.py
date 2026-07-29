import sys
import os

from scipy.sparse import save_npz
import numpy as np
from tqdm import tqdm

#────────────────────────────────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(os.path.dirname(script_dir))

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from Utilities.Stokes_felib import (
    evalOnTrigs, calculate_velocity_A
)

from Utilities.Mesh_processing import (
    refine,
)

from Utilities.Plot_functions import (
    Plot_Initial_Refined_meshes,
)

#────────────────────────────────────────────────────────────────────────────────────────────────
# Mesh 
#────────────────────────────────────────────────────────────────────────────────────────────────

mesh_path = os.path.join(fem_dir, 'Meshes', 'exchanger_device_altered_mesh_data.npz')
p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(data_path=mesh_path, 
                                                           num_of_refinements=3, 
                                                           plot=False,
                                                           figsize=(16,4))
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

#────────────────────────────────────────────────────────────────────────────────────────────────
# Shepard basis
#────────────────────────────────────────────────────────────────────────────────────────────────

alpha = 1

_KNOTS = np.array([
    [-2.5,  0.0],
    [-1.0, -alpha],
    [ 0.0,  alpha],
    [ 1.0, -alpha],
    [ 2.5,  0.0],
])

POWER = 4.0

def shepard_basis(points, power=POWER, eps=1e-12):

    diff = points[:, None, :] - _KNOTS[None, :, :]
    d = np.linalg.norm(diff, axis=2)

    w = 1.0 / np.where(d < eps, eps, d)**power
    psi = w / w.sum(axis=1, keepdims=True)

    exact = d < eps
    hit = exact.any(axis=1)

    if hit.any():
        psi[hit] = 0.0
        psi[hit, np.argmax(exact[hit], axis=1)] = 1.0

    return psi

#────────────────────────────────────────────────────────────────────────────────────────────────
# Evaluate A1, A2, ... , A5
#────────────────────────────────────────────────────────────────────────────────────────────────

num_basis = 5
Nt = len(t_fine)

psi_field = np.zeros((Nt, num_basis))

for i in tqdm(range(num_basis), desc="Evaluating basis fields"):

    def psi_i(x, y, *args):

        point = np.array([[x, y]])
        return float(shepard_basis(point)[0, i])

    psi_field[:, i] = evalOnTrigs(p_fine, t_fine, psi_i)


Affine_Operators = []

for i in range(5):
    Affine_Operators.append(calculate_velocity_A(p_fine, t_fine, psi_field[:, i]))

#────────────────────────────────────────────────────────────────────────────────────────────────
# Save
#────────────────────────────────────────────────────────────────────────────────────────────────

output_dir = os.path.join(fem_dir, "Reduced_Order_Modeling/Online_Offline_phase", "Data")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "Affine_Operators.npz"
)

np.savez_compressed(
    os.path.join(output_dir, "basis_fields.npz"),
    psi_field=psi_field,
    knots=_KNOTS,
    p_fine=p_fine,
    e_fine=e_fine,
    t_fine=t_fine,
)

for i, A in enumerate(Affine_Operators):
    save_npz(
        os.path.join(output_dir, f"A{i+1}.npz"),
        A
    )

print("Saved:", output_path)