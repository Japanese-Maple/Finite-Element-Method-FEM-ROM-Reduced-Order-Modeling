# Finite Element Method for the Incompressible Stokes Equations

This project implements a two-dimensional finite element solver for the incompressible Stokes equations using the **P1-iso-P1 mixed finite element method**.

The implementation covers the complete workflow from the continuous PDE formulation to sparse matrix assembly, saddle-point system construction, and verification of the Ladyzhenskaya-Babuška-Brezzi (LBB) stability condition.

---

## Governing Equations

The steady incompressible Stokes equations are

```math
-\nu \Delta \mathbf{u} + \nabla p = \mathbf{0}
```

```math
\nabla \cdot \mathbf{u} = 0
```

where:

- $\mathbf{u}$ is the velocity field
- $p$ is the pressure field
- $\nu$ is the kinematic viscosity

These equations model low-Reynolds-number flows where viscous forces dominate inertial effects.

## Weak Formulation

Let

```math
u = \tilde u - r_g,
```

where $r_g$ is a lifting function satisfying the non-homogeneous Dirichlet boundary conditions.

The weak formulation reads:

```math
\int_{\Omega} \nu \nabla u : \nabla v \, d\Omega
-
\int_{\Omega} p \, (\nabla \cdot v)\, d\Omega
=
\int_{\Gamma_N} h \cdot v \, d\Gamma
-
\int_{\Omega} \nu \nabla r_g : \nabla v \, d\Omega,
\qquad \forall v \in X.
```

```math
-\int_{\Omega} q \, (\nabla \cdot u)\, d\Omega
=
\int_{\Omega} q \, (\nabla \cdot r_g)\, d\Omega,
\qquad \forall q \in Q.
```

where

- $u$ is the velocity field,
- $p$ is the pressure,
- $u$ is the kinematic viscosity,
- $h$ denotes Neumann boundary data,
- $r_g$ is the lifting function associated with the Dirichlet boundary conditions.

---

## P1-iso-P1 Discretization

The solver uses the classical **P1-iso-P1 element pair**:

- Velocity is discretized on a uniformly refined mesh
- Pressure is discretized on the original coarse mesh
- Both fields use continuous piecewise linear basis functions

This macro-element construction satisfies the discrete LBB condition while retaining the simplicity and efficiency of linear finite elements.

---

## Matrix Assembly

### Velocity Stiffness Matrix

```math
A_{ij}
=
\int_\Omega
\nu
\nabla\phi_i\cdot\nabla\phi_j
\, d\Omega
```

### Pressure Coupling Matrices

```math
(B_x)_{ij}
=
-\int_\Omega
\psi_j
\frac{\partial\phi_i}{\partial x}
\, d\Omega
```

```math
(B_y)_{ij}
=
-\int_\Omega
\psi_j
\frac{\partial\phi_i}{\partial y}
\, d\Omega
```

---

## Global Saddle-Point System

After assembly, the Stokes problem becomes

```math
\begin{bmatrix}
A & 0 & B_x^T \\
0 & A & B_y^T \\
B_x & B_y & 0
\end{bmatrix}
\begin{bmatrix}
u_x \\
u_y \\
p
\end{bmatrix}
=
\begin{bmatrix}
F_x \\
F_y \\
0
\end{bmatrix}
```

This system couples the momentum equations with the incompressibility constraint.

---

## LBB Stability Verification

The stability of the mixed discretization is verified through the discrete inf-sup condition

```math
\beta
=
\sqrt{
\lambda_{\min}
\left(
M_p^{-1}
B
M_v^{-1}
B^T
\right)
}
```

A strictly positive value of $\beta$ confirms stability of the chosen velocity-pressure pair.

---

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

---

## Repository Structure

```text
.
├── Demonstrations
│   ├── Flow_examples.ipynb
│   ├── Stokes_Convergence.ipynb
│   ├── Stokes_Stability.ipynb
│   ├── Visuals.ipynb
│   └── stokes_equations.ipynb
├── Drafts
│   ├── Notes.txt
│   ├── Stokes_draft.py
│   └── paper_draft.txt
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
├── Outputs
│   ├── Flow.jpeg
│   ├── InitialvsRefined_mesh.jpeg
│   ├── Jacobian_tri_transform.svg
│   ├── Mesh_Refinement.jpeg
│   ├── P1-iso-P1.svg
│   ├── Pressure_Tricontourf.jpeg
│   ├── Solution_Streamlines.jpeg
│   ├── Stokes_A_matrix.jpeg
│   ├── Stokes_B_x_matrix.jpeg
│   ├── Stokes_K_matrix_labeled.jpeg
│   ├── Stokes_Mass Matrix (Pressure)_matrix.jpeg
│   └── Stokes_Mass Matrix (Velocity)_matrix.jpeg
├── README.md
├── Solutions
│   ├── Backstep.npz
│   ├── Comb.npz
│   ├── Convergence_Step_0.npz
│   ├── Convergence_Step_1.npz
│   ├── Convergence_Step_2.npz
│   ├── Convergence_Step_3.npz
│   ├── Exchanger_device.npz
│   ├── Reference_Solution.npz
│   ├── Winding_pipe.npz
│   └── Winding_pipe_solution.npz
├── Solvers
│   └── Exchanger_Device.py
│   
└── Utilities
    ├── Mesh_processing.py
    └── Stokes_felib.py
```

---

## Author

**Oleksandr Samoliuk**  
Johannes Kepler University Linz  
Summer Semester 2026

(Huge thanks to **Stefan Takacs** for providing guidence and support)