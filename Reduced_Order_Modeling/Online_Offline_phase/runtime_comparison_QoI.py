"""
Comparing the execution speed / Quantities of Interest
"""

import numpy as np
from scipy.sparse import load_npz
import pickle
import scipy.sparse as sp

import os
import sys
import time
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
# Main QoI computing functions (FOM)
#────────────────────────────────────────────────────────────────────────────────────────────────

omega_1 = np.array([
    [-2.00, 1/3],
    [-1.05, 1/3],
    [-1.05, 1.0],
    [-2.00, 1.0],
    [-2.00, 1/3]
])

omega_2 = np.array([
    [-0.95, -1/3],
    [-0.05, -1/3],
    [-0.05,  1/3],
    [-0.95,  1/3],
    [-0.95, -1/3]
])

omega_3 = np.array([
    [0.05, -1/3],
    [0.95, -1/3],
    [0.95,  1/3],
    [0.05,  1/3],
    [0.05, -1/3]
])

omega_4 = np.array([
    [1.05, 1/3],
    [2.00, 1/3],
    [2.00, 1.0],
    [1.05, 1.0],
    [1.05, 1/3]
])

regions = [omega_1, omega_2, omega_3, omega_4]

fom_region_data = {}

for reg_id in [1, 2, 3, 4]:

    glob_ids = np.load(f'Reduced_Order_Modeling/No_Online_Offline_phase/Data/Glob_id{reg_id}.npy')    
    M_mat    = load_npz(f'Reduced_Order_Modeling/No_Online_Offline_phase/Data/Mass{reg_id}_p.npz')
    
    reg_poly = regions[reg_id - 1]
    width  = reg_poly[:, 0].max() - reg_poly[:, 0].min()
    height = reg_poly[:, 1].max() - reg_poly[:, 1].min()
    area   = width * height
    
    ones_vec = np.ones(len(glob_ids))
    fom_weight_vec = (M_mat @ ones_vec) / area
    
    fom_region_data[reg_id] = {
        'glob_ids': glob_ids,
        'weight_vec': fom_weight_vec
    }

def compute_fom_region_avg_fast(p_full, region_id):

    glob_ids   = fom_region_data[region_id]['glob_ids']
    weight_vec = fom_region_data[region_id]['weight_vec']
    
    p_local = p_full[glob_ids]
    return np.dot(p_local, weight_vec)

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
# QoI Entities (ROM)
#────────────────────────────────────────────────────────────────────────────────────────────────

E_p = load_npz('Reduced_Order_Modeling/No_Online_Offline_phase/Data/E_pressure.npz')
E_u = load_npz('Reduced_Order_Modeling/No_Online_Offline_phase/Data/E_velocity.npz')
E_uxy = sp.block_diag([E_u, E_u])

E_pN = np.load('Reduced_Order_Modeling/No_Online_Offline_phase/Data/E_pN.npy')
E_uN = np.load('Reduced_Order_Modeling/No_Online_Offline_phase/Data/E_uN.npy')

qoi_filepath = 'Reduced_Order_Modeling/Online_Offline_phase/Data/reduced_qoi_vectors.pkl'
with open(qoi_filepath, 'rb') as f:
    reduced_vectors = pickle.load(f)

sensor_lift = E_uxy @ lf
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
# Runtime Test Loop & Error Calculation
#────────────────────────────────────────────────────────────────────────────────────────────────

error_sensors = []

error_p_o2 = []
error_p_o3 = []
error_u_o1 = []
error_u_o4 = []

for i, mu_unseen in enumerate(tqdm(mu_test_set)):

    # FOM
    start_fem = time.perf_counter()
    kinematic_viscosity = psi_field @ mu_unseen
    ux_fom, uy_fom, p_fom = compute_U_P_solution_exchanger_device(p_fine, t_fine, e_fine, p_coarse, t_coarse, 3, kinematic_viscosity)
    u_fom = np.hstack([ux_fom, uy_fom])
    p_sensor_vals_fom = E_p @ p_fom                         # SENSORS
    u_sensor_vals_fom = E_uxy @ u_fom
    p_avg_fom_o2 = compute_fom_region_avg_fast(p_fom, 2)    # AVG P
    p_avg_fom_o3 = compute_fom_region_avg_fast(p_fom, 3)
    u_avg_fom_o1 = np.sqrt(compute_fom_region_avg_fast(ux_fom, 1)**2 + compute_fom_region_avg_fast(uy_fom, 1)**2)  # AVG U
    u_avg_fom_o4 = np.sqrt(compute_fom_region_avg_fast(ux_fom, 4)**2 + compute_fom_region_avg_fast(uy_fom, 4)**2)
    
    fem_time = time.perf_counter() - start_fem
    fem_times.append(fem_time)

    # ROM
    start_rom = time.perf_counter()
    for _ in range(rom_repeats_per_param):

        a_u, a_p = solve_ROM(mu_unseen, Reduced_Affine_Operators, Vu, Vp, Div_N, g_N, Reduced_lifting_vectors, lf, return_reduced_solution_only=True)
        p_sensor_vals_rom = E_pN @ a_p                      # SENSORS
        u_sensor_vals_rom = sensor_lift + E_uN @ a_u
        p_avg_rom_o2 = a_p @ reduced_vectors[2]['p']
        p_avg_rom_o3 = a_p @ reduced_vectors[3]['p']
        u_avg_rom_o1 = np.sqrt((a_u @ reduced_vectors[1]['ux'])**2 + (a_u @ reduced_vectors[1]['uy'])**2)
        u_avg_rom_o4 = np.sqrt((a_u @ reduced_vectors[4]['ux'])**2 + (a_u @ reduced_vectors[4]['uy'])**2)

    rom_time = (time.perf_counter() - start_rom) / rom_repeats_per_param
    rom_times.append(rom_time)
    
    tqdm.write(f"Param {i+1}/{num_params} | FEM: {fem_time:.4f}s | ROM: {rom_time:.6f}s | Speedup: {fem_time/rom_time:.2f}x")

    # Relative Errors
    fom_sensors = np.concatenate([p_sensor_vals_fom, u_sensor_vals_fom])
    rom_sensors = np.concatenate([p_sensor_vals_rom, u_sensor_vals_rom])
    
    err_sens = np.linalg.norm(fom_sensors - rom_sensors) / (np.linalg.norm(fom_sensors) + 1e-12)
    error_sensors.append(err_sens)
    
    err_p2 = abs(p_avg_fom_o2 - p_avg_rom_o2) / (abs(p_avg_fom_o2) + 1e-12)
    error_p_o2.append(err_p2)
    
    err_p3 = abs(p_avg_fom_o3 - p_avg_rom_o3) / (abs(p_avg_fom_o3) + 1e-12)
    error_p_o3.append(err_p3)
    
    err_u1 = abs(u_avg_fom_o1 - u_avg_rom_o1) / (abs(u_avg_fom_o1) + 1e-12)
    error_u_o1.append(err_u1)
    
    err_u4 = abs(u_avg_fom_o4 - u_avg_rom_o4) / (abs(u_avg_fom_o4) + 1e-12)
    error_u_o4.append(err_u4)

#────────────────────────────────────────────────────────────────────────────────────────────────
# Summary Satatistics
#────────────────────────────────────────────────────────────────────────────────────────────────

avg_fem = np.mean(fem_times)
avg_rom = np.mean(rom_times)
std_fem = np.std(fem_times)
std_rom = np.std(rom_times)
avg_speedup = avg_fem / avg_rom

print("\n" + "─"*90)
print("COMPUTATIONAL SPEEDUP")
print("="*50)
print(f"Average FEM Time: {np.mean(fem_times):.4f} s (±{np.std(fem_times):.4f})")
print(f"Average ROM Time: {np.mean(rom_times):.6f} s (±{np.std(rom_times):.6f})")
print(f"Average Speedup:  {np.mean(fem_times)/np.mean(rom_times):.2f}x")

print("\n" + "─"*90)
print("QUANTITIES OF INTEREST: MEAN RELATIVE ERROR")
print("="*50)
print(f"Sensors (P & U):       {np.mean(error_sensors):.2e} ± {np.std(error_sensors):.2e}")
print(f"Region 2 (Pressure):   {np.mean(error_p_o2):.2e} ± {np.std(error_p_o2):.2e}")
print(f"Region 3 (Pressure):   {np.mean(error_p_o3):.2e} ± {np.std(error_p_o3):.2e}")
print(f"Region 1 (Velocity):   {np.mean(error_u_o1):.2e} ± {np.std(error_u_o1):.2e}")
print(f"Region 4 (Velocity):   {np.mean(error_u_o4):.2e} ± {np.std(error_u_o4):.2e}")
print("\n" + "─"*90)

# Run results:
"""
Starting benchmark warm-up (untimed)...
Warm-up complete. Starting timed runs for 30 parameters...

Param 1/30  | FEM: 9.5340s  | ROM: 0.000050s | Speedup: 190039.21x                                                                                                                                                          
Param 2/30  | FEM: 9.4246s  | ROM: 0.000055s | Speedup: 171726.64x                                                                                                                                                          
Param 3/30  | FEM: 9.7700s  | ROM: 0.000048s | Speedup: 204427.55x                                                                                                                                                          
Param 4/30  | FEM: 9.4199s  | ROM: 0.000052s | Speedup: 182228.25x                                                                                                                                                          
Param 5/30  | FEM: 9.6608s  | ROM: 0.000049s | Speedup: 196495.51x                                                                                                                                                          
Param 6/30  | FEM: 10.4159s | ROM: 0.000054s | Speedup: 191748.82x                                                                                                                                                         
Param 7/30  | FEM: 10.7568s | ROM: 0.000051s | Speedup: 209760.39x                                                                                                                                                         
Param 8/30  | FEM: 9.9945s  | ROM: 0.000050s | Speedup: 197910.76x                                                                                                                                                          
Param 9/30  | FEM: 11.1273s | ROM: 0.000051s | Speedup: 217986.77x                                                                                                                                                         
Param 10/30 | FEM: 10.4808s | ROM: 0.000048s | Speedup: 217546.58x                                                                                                                                                        
Param 11/30 | FEM: 9.9974s  | ROM: 0.000047s | Speedup: 213421.16x                                                                                                                                                         
Param 12/30 | FEM: 9.9828s  | ROM: 0.000054s | Speedup: 184312.64x                                                                                                                                                         
Param 13/30 | FEM: 10.4405s | ROM: 0.000051s | Speedup: 203249.23x                                                                                                                                                        
Param 14/30 | FEM: 10.0452s | ROM: 0.000049s | Speedup: 203348.12x                                                                                                                                                        
Param 15/30 | FEM: 9.7945s  | ROM: 0.000049s | Speedup: 201923.64x                                                                                                                                                         
Param 16/30 | FEM: 9.6198s  | ROM: 0.000050s | Speedup: 191308.83x                                                                                                                                                         
Param 17/30 | FEM: 10.4756s | ROM: 0.000050s | Speedup: 211318.10x                                                                                                                                                        
Param 18/30 | FEM: 10.8196s | ROM: 0.000056s | Speedup: 192760.41x                                                                                                                                                        
Param 19/30 | FEM: 9.5615s  | ROM: 0.000050s | Speedup: 192061.77x                                                                                                                                                         
Param 20/30 | FEM: 10.6469s | ROM: 0.000048s | Speedup: 219689.54x                                                                                                                                                        
Param 21/30 | FEM: 10.0319s | ROM: 0.000050s | Speedup: 198795.44x                                                                                                                                                        
Param 22/30 | FEM: 10.4632s | ROM: 0.000051s | Speedup: 203915.23x                                                                                                                                                        
Param 23/30 | FEM: 9.8479s  | ROM: 0.000049s | Speedup: 200963.85x                                                                                                                                                         
Param 24/30 | FEM: 10.6817s | ROM: 0.000048s | Speedup: 222943.90x                                                                                                                                                        
Param 25/30 | FEM: 9.9889s  | ROM: 0.000048s | Speedup: 208086.48x                                                                                                                                                         
Param 26/30 | FEM: 10.4744s | ROM: 0.000050s | Speedup: 209181.91x                                                                                                                                                        
Param 27/30 | FEM: 9.9095 s | ROM: 0.000052s | Speedup: 190581.59x                                                                                                                                                         
Param 28/30 | FEM: 10.2934s | ROM: 0.000052s | Speedup: 199206.21x                                                                                                                                                        
Param 29/30 | FEM: 9.9885s  | ROM: 0.000050s | Speedup: 198221.85x                                                                                                                                                         
Param 30/30 | FEM: 9.9505s  | ROM: 0.000051s | Speedup: 194937.46x                                                                                                                                                         
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [05:17<00:00, 10.57s/it]

──────────────────────────────────────────────────────────────────────────────────────────
COMPUTATIONAL SPEEDUP
==================================================
Average FEM Time: 10.1199  s (±0.4392)
Average ROM Time: 0.000051 s (±0.000002)
Average Speedup:  200340.97x

──────────────────────────────────────────────────────────────────────────────────────────
QUANTITIES OF INTEREST: MEAN RELATIVE ERROR
==================================================
Sensors (P & U):       6.45e-04 ± 2.50e-04
Region 2 (Pressure):   1.77e-04 ± 2.10e-04
Region 3 (Pressure):   1.95e-04 ± 2.13e-04
Region 1 (Velocity):   2.07e-02 ± 1.08e-02
Region 4 (Velocity):   3.17e-02 ± 1.49e-02

──────────────────────────────────────────────────────────────────────────────────────────
"""