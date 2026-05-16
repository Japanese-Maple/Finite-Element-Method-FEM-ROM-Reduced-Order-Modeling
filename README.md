# Stokes Equations

$$
\begin{cases}
-\nu \Delta u + \nabla p = 0 & \text{in } \Omega \\
\nabla \cdot u = 0 & \text{in } \Omega \\
u = g & \text{on } \Gamma_D \\
-pn + \nu (\nabla u)n = h & \text{on } \Gamma_N
\end{cases}
$$

$$ \begin{cases} \int_{\Omega} \nu \nabla u : \nabla v \, d\Omega - \int_{\Omega} p (\nabla \cdot v) \, d\Omega = 
\int_{\Gamma_N} h \cdot v \, d\Gamma - \int_{\Omega} \nu \nabla r_g : \nabla v \, d\Omega & \forall v \in X \\ -\int_{\Omega} q (\nabla \cdot u) \, d\Omega = 
\int_{\Omega} q (\nabla \cdot r_g) \, d\Omega & \forall q \in Q \end{cases} 
$$

This project implements a finite element discretization of the stationary incompressible Stokes equations in two spatial dimensions. The formulation is based on the mixed velocity-pressure weak form with Dirichlet and Neumann boundary conditions.

The implementation includes:

- Assembly of the velocity stiffness matrix $A$
- Construction of the pressure coupling matrices $B_x$ and $B_y$
- Formation of the saddle-point system
- Handling of Dirichlet lifting functions
- Structured sparse matrix assembly using `scipy.sparse`
- Mesh refinement and orientation correction utilities
- Visualization tools for:
  - sparse matrix structures,
  - streamline fields,
  - refined finite element meshes

The numerical discretization uses triangular finite elements and constructs the global system through element-wise assembly of local contributions derived from affine reference mappings.

The saddle-point system has the block structure

$$
K =
\begin{bmatrix}
A & 0 & B_x^T \\
0 & A & B_y^T \\
B_x & B_y & 0
\end{bmatrix}
$$

where $A$ denotes the discrete vector Laplacian and $B_x, B_y$ represent the discrete divergence operators.

The repository additionally contains routines for post-processing and visualization of the computed velocity field, including streamline interpolation on unstructured meshes.
