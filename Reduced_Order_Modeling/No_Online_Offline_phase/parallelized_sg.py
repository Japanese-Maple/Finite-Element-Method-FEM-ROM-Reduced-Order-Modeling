import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

#_______________________________________________________________________________________________________________________________________________________________

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(os.path.dirname(script_dir))

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from Solvers.Exchanger_Device import compute_U_P_solution_exchanger_device
from Utilities.Mesh_processing import refine
from Utilities.Plot_functions import Plot_Initial_Refined_meshes

#_______________________________________________________________________________________________________________________________________________________________

def solve_snapshot_worker(args):
    """Worker function to solve a single snapshot."""
    i, v_t, p_f, t_f, e_f, p_c, t_c = args
    
    ux, uy, p_sol = compute_U_P_solution_exchanger_device(
        p_f, t_f, e_f, 
        p_c, t_c,
        alpha=3.0,
        kinematic_viscosity=v_t
    )
    
    return i, ux, uy, p_sol

#_______________________________________________________________________________________________________________________________________________________________

if __name__ == '__main__':
    
    mesh_path = os.path.join(fem_dir, 'Meshes', 'exchanger_device_altered_mesh_data.npz')
    p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(
        data_path=mesh_path, 
        num_of_refinements=3, 
        plot=False,
        figsize=(16, 4)
    )
    p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

    npz_path = os.path.join(fem_dir, 'Reduced_Order_Modeling', 'No_Online_Offline_phase', 'Data', 'viscosity_snapshots.npz')
    with np.load(npz_path) as data:
        v_t_snapshots = data['v_t_snapshots']
        parameters    = data['parameters']

    num_snapshots = v_t_snapshots.shape[0]
    Nv = p_fine.shape[0]
    Np = p_coarse.shape[0]
    total_dof = (2 * Nv) + Np

    output_dir = os.path.join(fem_dir, 'Reduced_Order_Modeling/No_Online_Offline_phase', 'Data')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'stokes_solution_snapshots.npz')

    if os.path.exists(out_path):
        print(f"[*] Found existing snapshot file at:\n    {out_path}")
        print("[*] Loading existing data to resume progress...")
        with np.load(out_path) as old_data:
            ux_snapshots = old_data['ux_snapshots'].copy()
            uy_snapshots = old_data['uy_snapshots'].copy()
            p_snapshots  = old_data['p_snapshots'].copy()
        
        computed_mask = np.max(np.abs(ux_snapshots), axis=1) > 0
        tasks_to_run = [i for i in range(num_snapshots) if not computed_mask[i]]
        print(f"[*] Recovered {num_snapshots - len(tasks_to_run)} completed snapshots.")
    else:
        ux_snapshots = np.zeros((num_snapshots, Nv))
        uy_snapshots = np.zeros((num_snapshots, Nv))
        p_snapshots  = np.zeros((num_snapshots, Np))
        tasks_to_run = list(range(num_snapshots))

    print("─" * 60)
    print(" High-Fidelity Stokes FOM Computation Initiated (PARALLEL)")
    print("─" * 60)
    print(f" Total Snapshots:         {num_snapshots}")
    print(f" Snapshots Left to Run:   {len(tasks_to_run)}")
    print(f" CPU Cores Allocated:     10")
    print(f" Total DOFs per state:    {total_dof}")
    print("─" * 60)

    tasks = [
        (i, v_t_snapshots[i], p_fine, t_fine, e_fine, p_coarse, t_coarse)
        for i in tasks_to_run
    ]

#_______________________________________________________________________________________________________________________________________________________________

    if len(tasks) > 0:
        start_time = time.time()
        completed_in_this_run = 0

        with ProcessPoolExecutor(max_workers=10) as executor:
            
            futures = [executor.submit(solve_snapshot_worker, task) for task in tasks]
            
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Evaluating FOM (10 Cores)", unit="state", ncols=100):
                i, ux, uy, p_sol = future.result()
                
                ux_snapshots[i, :] = ux
                uy_snapshots[i, :] = uy
                p_snapshots[i, :]  = p_sol
                
                completed_in_this_run += 1
                
                # --- Save 10 completed tasks ---
                if completed_in_this_run % 10 == 0:
                    np.savez_compressed(
                        out_path,
                        ux_snapshots=ux_snapshots,
                        uy_snapshots=uy_snapshots,
                        p_snapshots=p_snapshots,
                        parameters=parameters,
                        p_fine=p_fine, t_fine=t_fine, e_fine=e_fine,
                        p_coarse=p_coarse, t_coarse=t_coarse
                    )
                    tqdm.write(f"  [Checkpoint] Saved progress at {completed_in_this_run} completed states.")

#_______________________________________________________________________________________________________________________________________________________________

        elapsed_time = time.time() - start_time
        
        np.savez_compressed(
            out_path,
            ux_snapshots=ux_snapshots,
            uy_snapshots=uy_snapshots,
            p_snapshots=p_snapshots,
            parameters=parameters,
            p_fine=p_fine, t_fine=t_fine, e_fine=e_fine,
            p_coarse=p_coarse, t_coarse=t_coarse
        )

        print("\n" + "─" * 60)
        print(" Computation Concluded")
        print("─" * 60)
        print(f" Total runtime for this batch: {elapsed_time / 60:.2f} minutes")
        print(f" Snapshot dataset safely archived to:\n > {out_path}")
        print("=" * 60)
    else:
        print("\nAll snapshots are already computed.")
        print("=" * 60)