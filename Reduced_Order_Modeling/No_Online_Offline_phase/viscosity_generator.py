import sys
import os

import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

#────────────────────────────────────────────────────────────────────────────────────────────────

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(os.path.dirname(script_dir))

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from Utilities.Stokes_felib import (
    evalOnTrigs, 
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
# Viscosity Sampling via Latin Hypercube 
#────────────────────────────────────────────────────────────────────────────────────────────────

num_snapshots = 100
nu_min, nu_max = 10.0, 200.0
alpha = 1

_KNOTS = np.array([
    [-2.5,  0.0],
    [-1.0, -alpha],
    [ 0.0,  alpha],
    [ 1.0, -alpha],
    [ 2.5,  0.0],
])

VT_snapshots = np.zeros((num_snapshots, len(t_fine)))

sampler = qmc.LatinHypercube(d=5)
sample = sampler.random(n=num_snapshots)
nu_parameters = qmc.scale(sample, [nu_min]*5, [nu_max]*5)

def viscosity_field(p, nu_knots, knots=_KNOTS, power: float = 2.0, eps: float = 1e-12):
        """
        Inverse-distance weighted viscosity at every node in p (N, 2).
        power=2  is the standard Shepard interpolation.
        Increase power (e.g. 4-6) for sharper, more localised influence zones.
        """
        # d[i, k] = distance from node i to knot k
        diff = p[:, None, :] - _KNOTS[None, :, :]       # (N, K, 2)
        d    = np.linalg.norm(diff, ord=2, axis=2)      # (N, K)

        exact = d < eps                                 # (N, K) bool
        hit   = exact.any(axis=1)                       # (N,)   bool

        w  = 1.0 / np.where(d < eps, eps, d) ** power   # (N, K)
        nu = (w * nu_knots[None, :]).sum(axis=1) / w.sum(axis=1)

        if hit.any():
            knot_idx = np.argmax(exact[hit], axis=1)
            nu[hit]  = nu_knots[knot_idx]

        return nu  

for i, nu_vec_ith in enumerate(tqdm(nu_parameters, desc="Generating viscosity snapshots")):

    def viscosity(x, y, *args, power: float = 4.0):
        """
        Accepts x, y, and any extra arguments (*args) passed by evalOnTrigs,
        such as subdomain markers t[i, 6:].
        """
        return float(viscosity_field(np.array([[x, y]]), nu_vec_ith, power=power)[0])

    VT_snapshots[i, :] = evalOnTrigs(p_fine, t_fine, viscosity)

#────────────────────────────────────────────────────────────────────────────────────────────────
# NPZ File Generation
#────────────────────────────────────────────────────────────────────────────────────────────────

output_dir = os.path.join(fem_dir, 'Reduced_Order_Modeling/Data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'viscosity_snapshots.npz')

np.savez_compressed(
    output_path,
    v_t_snapshots=VT_snapshots,        # Shape: (num_snapshots, N_triangles)
    parameters=nu_parameters,          # Shape: (num_snapshots, 5)
    knots=_KNOTS,                      # Knot positions (5, 2)
    p_fine=p_fine,
    t_fine=t_fine,
    e_fine=e_fine
)