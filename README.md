# Finite Element Method for the Incompressible Stokes Equations

This project implements a two-dimensional finite element solver for the incompressible Stokes equations using the **P1-iso-P1 mixed finite element method** and then explores Reduced Order Modelling as a way to compress the amount of time needed to solve the parametrized problem.


---

### `Stokes_Solver` (Saddle-Point FEM Solver)

$$-\nu\Delta\mathbf{u} + \nabla p = 0, \qquad \nabla\cdot\mathbf{u} = 0$$

This solver handles steady-state incompressible Stokes flow under a spatially varying viscosity $\nu(\mathbf{x})$, discretized on a **P1-iso-P1** macro-element pairing (velocity on a fine mesh, pressure on a coarse mesh) that satisfies the LBB inf-sup condition without resorting to higher-order elements. It assembles the global saddle-point operator $K = \begin{bmatrix} A & 0 & B_x^T \\ 0 & A & B_y^T \\ B_x & B_y & 0 \end{bmatrix}$ and solves it directly via sparse LU factorization, enforcing Dirichlet boundary conditions through row substitution and regularizing the singular pressure null-space by pinning a single reference degree of freedom.

---

### `Affine_Viscosity_Decomposition` (Parametrized Assembly)

$$\nu(\mathbf{x};\boldsymbol{\mu}) = \sum_{k=1}^{5} \mu_k\,\psi_k(\mathbf{x}), \qquad A(\boldsymbol{\mu}) = \sum_{k=1}^{5} \mu_k A_k$$

Because viscosity enters the weak form linearly, the stiffness operator inherits an affine dependence on the five-dimensional parameter vector $\boldsymbol{\mu}\in[10,200]^5$, which controls viscosity values at five Shepard-interpolated control points. This lets five parameter-independent matrices $A_1,\dots,A_5$ be assembled once and reused for any $\boldsymbol{\mu}$ as a cheap scalar-weighted sum, entirely avoiding per-query reassembly of the global operator.

---

### `POD_ROM` (Reduced-Order Model)

$$\mathcal{M} = \{(\mathbf{u}_h(\boldsymbol{\mu}), p_h(\boldsymbol{\mu})) : \boldsymbol{\mu}\in\mathcal{P}\}, \qquad \mathbf{t} = X_u^{-1}B_h^T\mathbf{p}$$

Exploiting the fact that the solution manifold $\mathcal{M}$ has low intrinsic dimension despite living in a high-dimensional discrete space, the framework builds a reduced basis via Proper Orthogonal Decomposition on 100 Latin-Hypercube-sampled full-order snapshots, enriching the velocity space with supremizer modes $\mathbf{t}$ to preserve inf-sup stability at the reduced level. The offline stage precomputes all reduced operators once; the online stage then solves only a small Galerkin system per new $\boldsymbol{\mu}$, at cost independent of the underlying mesh resolution.


## Features

- Finite Element Method for incompressible Stokes flow
- P1-iso-P1 mixed discretization
- Sparse matrix assembly
- Velocity stiffness matrix construction
- Pressure-velocity coupling operators
- Saddle-point system assembly
- Dirichlet and Neumann boundary conditions
- Pressure null-space handling
- Discrete LBB stability analysis
- Python implementation using NumPy and SciPy
- Reduced Order Model
- Affine decomposition for the velocity operators
- Parametrized viscosity fields

---

## Repository Structure

```mermaid
flowchart TB

subgraph group_mesh["Mesh &amp; discretization"]
  node_mesh_archives[("Mesh archives<br/>NumPy mesh data")]
  node_mesh_processing["Mesh processing<br/>P1-iso-P1 refinement<br/>[Mesh_processing.py]"]
end

subgraph group_fom["Full-order Stokes FEM"]
  node_stokes_felib["Stokes FEM library<br/>sparse mixed assembly<br/>[Stokes_felib.py]"]
  node_exchanger_solver["Exchanger-device solver<br/>full-order solve"]
end

subgraph group_analysis["Results &amp; analysis"]
  node_solution_archives[("Solution archives<br/>NumPy field data")]
  node_plot_functions["Plot functions<br/>visualization<br/>[Plot_functions.py]"]
  node_flow_examples["Flow examples<br/>interactive notebook"]
  node_convergence_notebook["Convergence study<br/>verification notebook"]
  node_stability_notebook["LBB stability study<br/>diagnostic notebook"]
end

subgraph group_rom["Reduced-order modeling"]
  node_viscosity_generator["Viscosity generator<br/>ROM sampling"]
  node_snapshot_generator["Snapshot generator<br/>ROM full-order sampling"]
  node_parallel_snapshot_generation["Parallel snapshot generation<br/>parallel ROM driver<br/>[parallelized_sg.py]"]
  node_snapshot_archives[("Snapshot archives<br/>NumPy ROM data")]
  node_affine_decomposition["Affine decomposition<br/>offline ROM assembly"]
  node_offline_rom_data[("Offline ROM data<br/>reduced operators &amp; bases<br/>[A1.npz]")]
  node_rom_notebook["Offline/online ROM demo<br/>interactive notebook"]
end

node_mesh_archives -->|"geometry &amp; connectivity"| node_mesh_processing
node_mesh_processing -->|"P1-iso-P1 meshes"| node_stokes_felib
node_stokes_felib -->|"mixed matrix blocks"| node_exchanger_solver
node_mesh_archives -->|"device mesh"| node_exchanger_solver
node_exchanger_solver -->|"velocity &amp; pressure fields"| node_solution_archives
node_solution_archives -->|"solved fields"| node_flow_examples
node_plot_functions -->|"rendering"| node_flow_examples
node_solution_archives -->|"refinement results"| node_convergence_notebook
node_stokes_felib -->|"mass &amp; divergence operators"| node_stability_notebook
node_solution_archives -->|"solution data"| node_stability_notebook
node_viscosity_generator -->|"sampled viscosities"| node_parallel_snapshot_generation
node_parallel_snapshot_generation -->|"parallel jobs"| node_snapshot_generator
node_mesh_processing -.->|"shared mesh boundary"| node_snapshot_generator
node_stokes_felib -.->|"shared FEM assembly"| node_snapshot_generator
node_snapshot_generator -->|"solution snapshots"| node_snapshot_archives
node_viscosity_generator -->|"viscosity snapshots"| node_snapshot_archives
node_snapshot_archives -->|"training data"| node_affine_decomposition
node_affine_decomposition -->|"A1–A5 &amp; basis fields"| node_offline_rom_data
node_offline_rom_data -->|"online reduced evaluation"| node_rom_notebook

click node_mesh_archives "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Meshes/exchanger_device_mesh_data.npz"
click node_mesh_processing "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Utilities/Mesh_processing.py"
click node_stokes_felib "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Utilities/Stokes_felib.py"
click node_exchanger_solver "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Solvers/Exchanger_Device.py"
click node_solution_archives "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Solutions/Exchanger_device.npz"
click node_plot_functions "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Utilities/Plot_functions.py"
click node_flow_examples "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Demonstrations/Flow_examples.ipynb"
click node_convergence_notebook "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Demonstrations/Stokes_Convergence.ipynb"
click node_stability_notebook "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Demonstrations/Stokes_Stability.ipynb"
click node_viscosity_generator "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Reduced_Order_Modeling/No_Online_Offline_phase/viscosity_generator.py"
click node_snapshot_generator "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Reduced_Order_Modeling/No_Online_Offline_phase/snapshot_generator.py"
click node_parallel_snapshot_generation "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Reduced_Order_Modeling/No_Online_Offline_phase/parallelized_sg.py"
click node_affine_decomposition "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Reduced_Order_Modeling/Online_Offline_phase/Affine_Decomposition.py"
click node_offline_rom_data "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Reduced_Order_Modeling/Online_Offline_phase/Data/A1.npz"
click node_rom_notebook "https://github.com/japanese-maple/finite-element-method-fem-rom-reduced-order-modeling/blob/main/Demonstrations/ROM_OfflineIOnline.ipynb"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_mesh_archives,node_mesh_processing toneBlue
class node_stokes_felib,node_exchanger_solver toneAmber
class node_solution_archives,node_plot_functions,node_flow_examples,node_convergence_notebook,node_stability_notebook toneMint
class node_viscosity_generator,node_snapshot_generator,node_parallel_snapshot_generation,node_snapshot_archives,node_affine_decomposition,node_offline_rom_data,node_rom_notebook toneRose
```

```text
.
├── Demonstrations
│   ├── Flow_examples.ipynb
│   ├── ROM_OfflineIOnline.ipynb
│   ├── Reduced_Order_Model.ipynb
│   ├── Stokes_Convergence.ipynb
│   ├── Stokes_Stability.ipynb
│   ├── Visuals.ipynb
│   └── stokes_equations.ipynb
├── Drafts
│   ├── Notes.txt
│   ├── ROM_&_FEM_PAPER.pdf
│   ├── Stokes_draft.py
│   └── paper_draft.txt
├── LICENSE
├── Meshes
│   ├── Backstep_mesh_data.npz
│   ├── Comb_mesh_data.npz
│   ├── Hexagonal_pipe_system_mesh_data.npz
│   ├── LBB_test_mesh_data.npz
│   ├── Winding_pipe_fixed_mesh_data.npz
│   ├── Winding_pipe_mesh_data.npz
│   ├── exchanger_device_altered_mesh_data.npz
│   ├── exchanger_device_mesh_data.npz
│   └── honeycomb_wide_data.npz
├── NOTICE
├── Outputs
│   ├── Combined_Solution.png
│   ├── Flow.jpeg
│   ├── InitialvsRefined_mesh.jpeg
│   ├── Jacobian_tri_transform.svg
│   ├── LBB_stability_P1isoP1.png
│   ├── LHS_downprojected.png
│   ├── Mesh_Refinement.jpeg
│   ├── P1-iso-P1.svg
│   ├── POD_Energy_Spectrum_pressure.png
│   ├── POD_Energy_Spectrum_supremizer.png
│   ├── POD_Energy_Spectrum_velocity (homogeneous).png
│   ├── Pressure_Tricontourf.jpeg
│   ├── Solution_Streamlines.jpeg
│   ├── Solution_Streamlines.png
│   ├── Stokes_A_matrix.jpeg
│   ├── Stokes_B_x_matrix.jpeg
│   ├── Stokes_K_matrix_labeled.jpeg
│   ├── Stokes_K_matrix_labeled_POD.png
│   ├── Stokes_Mass Matrix (Pressure)_matrix.jpeg
│   ├── Stokes_Mass Matrix (Velocity)_matrix.jpeg
│   ├── Viscosity.png
│   ├── Viscosity_Basis_Functions.png
│   └── convergence_plot.png
├── README.md
├── Reduced_Order_Modeling
│   ├── No_Online_Offline_phase
│   │   ├── Data
│   │   │   ├── stokes_solution_snapshots.npz
│   │   │   └── viscosity_snapshots.npz
│   │   ├── parallelized_sg.py
│   │   ├── snapshot_generator.py
│   │   └── viscosity_generator.py
│   └── Online_Offline_phase
│       ├── Affine_Decomposition.py
│       └── Data
│           ├── A1.npz
│           ├── A2.npz
│           ├── A3.npz
│           ├── A4.npz
│           ├── A5.npz
│           └── basis_fields.npz
├── Solutions
│   ├── Backstep.npz
│   ├── Comb.npz
│   ├── Convergence_Step_0.npz
│   ├── Convergence_Step_1.npz
│   ├── Convergence_Step_2.npz
│   ├── Convergence_Step_3.npz
│   ├── Convergence_Step_4.npz
│   ├── Convergence_Step_5.npz
│   ├── Convergence_Step_6.npz
│   ├── Exchanger_device.npz
│   ├── Exchanger_device_with_varying_viscosity.npz
│   ├── Reference_Solution.npz
│   ├── Winding_pipe.npz
│   └── Winding_pipe_solution.npz
├── Solvers
│   └── Exchanger_Device.py
│   
└── Utilities
    ├── Mesh_processing.py
    ├── Plot_functions.py
    └── Stokes_felib.py
```

---

## Author

**Oleksandr Samoliuk**  
Johannes Kepler University Linz  
Summer Semester 2026

(Huge thanks to **Stefan Takacs** for providing guidence and support)