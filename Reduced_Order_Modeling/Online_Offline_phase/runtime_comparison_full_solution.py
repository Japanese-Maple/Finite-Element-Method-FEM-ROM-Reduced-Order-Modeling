"""
Comparing the execution speed / FULL SOLUTIONS 
"""

import time
import numpy as np
from scipy.sparse import load_npz
import os
import sys
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
fem_dir = os.path.dirname(os.path.dirname(script_dir))

if fem_dir not in sys.path:
    sys.path.append(fem_dir)

from Solvers.Exchanger_Device import (
    compute_U_P_solution_exchanger_device
)

from Utilities.Mesh_processing import (
    refine,
)

from Utilities.Plot_functions import (
    Plot_Initial_Refined_meshes,
)

from Utilities.ROM import (
    solve_ROM,
)

#────────────────────────────────────────────────────────────────────────────────────────────────
# Mesh, ROM matrices & Mu Test Set
#────────────────────────────────────────────────────────────────────────────────────────────────

mesh_path = os.path.join(fem_dir, 'Meshes', 'exchanger_device_altered_mesh_data.npz')
p_coarse, e_coarse, t_coarse = Plot_Initial_Refined_meshes(data_path=mesh_path, 
                                                           num_of_refinements=3, 
                                                           plot=False,
                                                           figsize=(16,4))
p_fine, e_fine, t_fine = refine(p_coarse, e_coarse, t_coarse)

mu_test_set_dir = os.path.join(fem_dir, 'Reduced_Order_Modeling/Online_Offline_phase/Data/mu_test_set.npy')
mu_test_set = np.load(mu_test_set_dir)

rom_data_dir = os.path.join(fem_dir, 'Reduced_Order_Modeling/No_Online_Offline_phase/Data')
basis_data = np.load(os.path.join('Reduced_Order_Modeling/Online_Offline_phase/Data', 'basis_fields.npz'))
psi_field = basis_data['psi_field']
Reduced_Affine_Operators = np.load(os.path.join(rom_data_dir, 'Reduced_Affine_Operators.npy'), allow_pickle=True)
Vu = np.load(os.path.join(rom_data_dir, 'Vu.npy'))
Vp = np.load(os.path.join(rom_data_dir, 'Vp.npy'))
Div_N = np.load(os.path.join(rom_data_dir, 'Div_N.npy'))
g_N = np.load(os.path.join(rom_data_dir, 'g_N.npy'))
Reduced_lifting_vectors = np.load(os.path.join(rom_data_dir, 'Reduced_lifting_vectors.npy'))
lf = np.load(os.path.join(rom_data_dir, 'lifting.npy'))

#────────────────────────────────────────────────────────────────────────────────────────────────
# Warm-up
#────────────────────────────────────────────────────────────────────────────────────────────────

num_params = len(mu_test_set)
rom_repeats_per_param = 50

fem_times = []
rom_times = []

print("Starting benchmark warm-up (untimed)...")

mu_warmup = mu_test_set[0]
warmup_visc = psi_field @ mu_warmup
_ = compute_U_P_solution_exchanger_device(p_fine, t_fine, e_fine, p_coarse, t_coarse, 3, warmup_visc)
_ = solve_ROM(mu_warmup, Reduced_Affine_Operators, Vu, Vp, Div_N, g_N, Reduced_lifting_vectors, lf)

print(f"Warm-up complete. Starting timed runs for {num_params} parameters...\n")

#────────────────────────────────────────────────────────────────────────────────────────────────
# Runtime Test Loop
#────────────────────────────────────────────────────────────────────────────────────────────────

for i, mu_unseen in enumerate(tqdm(mu_test_set)):
    
    start_fem = time.perf_counter()
    kinematic_viscosity = psi_field @ mu_unseen
    _ = compute_U_P_solution_exchanger_device(p_fine, t_fine, e_fine, p_coarse, t_coarse, 3, kinematic_viscosity)
    fem_time = time.perf_counter() - start_fem
    fem_times.append(fem_time)
    
    start_rom = time.perf_counter()
    for _ in range(rom_repeats_per_param):
        _ = solve_ROM(mu_unseen, Reduced_Affine_Operators, Vu, Vp, Div_N, g_N, Reduced_lifting_vectors, lf)
    rom_time = (time.perf_counter() - start_rom) / rom_repeats_per_param
    rom_times.append(rom_time)
    
    tqdm.write(f"Param {i+1}/{num_params} | FEM: {fem_time:.4f}s | ROM: {rom_time:.6f}s | Speedup: {fem_time/rom_time:.2f}x")

#────────────────────────────────────────────────────────────────────────────────────────────────
# Summary Satatistics
#────────────────────────────────────────────────────────────────────────────────────────────────

avg_fem = np.mean(fem_times)
avg_rom = np.mean(rom_times)
std_fem = np.std(fem_times)
std_rom = np.std(rom_times)
avg_speedup = avg_fem / avg_rom

print("\n" + "─"*90)
print("FINAL BENCHMARK RESULTS")
print("\n" + "─"*90)
print(f"Average FEM Time: {avg_fem:.4f} seconds (±{std_fem:.4f})")
print(f"Average ROM Time: {avg_rom:.6f} seconds (±{std_rom:.6f})")
print(f"Average Speedup:  {avg_speedup:.2f}x")

# Run results:
"""
Starting benchmark warm-up (untimed)...
Warm-up complete. Starting timed runs for 30 parameters...

Param 1/30  | FEM: 10.7344s | ROM: 0.001002s | Speedup: 10709.09x                                                                                                                                          
Param 2/30  | FEM: 11.5281s | ROM: 0.001406s | Speedup: 8200.55x                                                                                                                                           
Param 3/30  | FEM: 12.2587s | ROM: 0.001010s | Speedup: 12133.65x                                                                                                                                          
Param 4/30  | FEM: 11.4875s | ROM: 0.001279s | Speedup: 8983.62x                                                                                                                                           
Param 5/30  | FEM: 10.3565s | ROM: 0.005367s | Speedup: 1929.78x                                                                                                                                           
Param 6/30  | FEM: 11.7573s | ROM: 0.001136s | Speedup: 10353.72x                                                                                                                                          
Param 7/30  | FEM: 11.8127s | ROM: 0.000974s | Speedup: 12130.43x                                                                                                                                          
Param 8/30  | FEM: 11.9018s | ROM: 0.000916s | Speedup: 12989.44x                                                                                                                                          
Param 9/30  | FEM: 12.2650s | ROM: 0.001177s | Speedup: 10423.01x                                                                                                                                          
Param 10/30 | FEM: 12.2164s | ROM: 0.001069s | Speedup: 11425.26x                                                                                                                                         
Param 11/30 | FEM: 11.8722s | ROM: 0.001180s | Speedup: 10059.96x                                                                                                                                         
Param 12/30 | FEM: 12.0676s | ROM: 0.001074s | Speedup: 11235.47x                                                                                                                                         
Param 13/30 | FEM: 11.8289s | ROM: 0.001211s | Speedup: 9768.55x                                                                                                                                          
Param 14/30 | FEM: 12.3907s | ROM: 0.000975s | Speedup: 12712.84x                                                                                                                                         
Param 15/30 | FEM: 12.1510s | ROM: 0.000985s | Speedup: 12336.88x                                                                                                                                         
Param 16/30 | FEM: 12.4009s | ROM: 0.001255s | Speedup: 9882.88x                                                                                                                                          
Param 17/30 | FEM: 12.2942s | ROM: 0.001060s | Speedup: 11596.89x                                                                                                                                         
Param 18/30 | FEM: 12.1984s | ROM: 0.001035s | Speedup: 11791.17x                                                                                                                                         
Param 19/30 | FEM: 12.1593s | ROM: 0.001034s | Speedup: 11765.12x                                                                                                                                         
Param 20/30 | FEM: 12.3108s | ROM: 0.001902s | Speedup: 6471.96x                                                                                                                                          
Param 21/30 | FEM: 12.3510s | ROM: 0.001183s | Speedup: 10436.80x                                                                                                                                         
Param 22/30 | FEM: 12.3849s | ROM: 0.001446s | Speedup: 8563.47x                                                                                                                                          
Param 23/30 | FEM: 12.4016s | ROM: 0.001762s | Speedup: 7036.67x                                                                                                                                          
Param 24/30 | FEM: 12.5827s | ROM: 0.006215s | Speedup: 2024.69x                                                                                                                                          
Param 25/30 | FEM: 12.5179s | ROM: 0.003984s | Speedup: 3142.12x                                                                                                                                          
Param 26/30 | FEM: 12.6442s | ROM: 0.005127s | Speedup: 2465.96x                                                                                                                                          
Param 27/30 | FEM: 12.3285s | ROM: 0.006270s | Speedup: 1966.24x                                                                                                                                          
Param 28/30 | FEM: 12.9579s | ROM: 0.005497s | Speedup: 2357.08x                                                                                                                                          
Param 29/30 | FEM: 12.7477s | ROM: 0.005458s | Speedup: 2335.67x                                                                                                                                          
Param 30/30 | FEM: 12.8133s | ROM: 0.003633s | Speedup: 3526.58x   

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [05:57<00:00, 11.93s/it]

──────────────────────────────────────────────────────────────────────────────────────────
FINAL BENCHMARK RESULTS
──────────────────────────────────────────────────────────────────────────────────────────
Average FEM Time: 12.1241 seconds (±0.5462)
Average ROM Time: 0.002254 seconds (±0.001843)
Average Speedup:  5378.75x

"""