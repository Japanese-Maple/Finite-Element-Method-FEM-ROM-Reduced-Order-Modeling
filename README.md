# Stokes Equations

$$
\begin{cases}
-\nu \Delta u + \nabla p = 0 & \text{in } \Omega \\
\nabla \cdot u = 0 & \text{in } \Omega \\
u = g & \text{on } \Gamma_D \\
-pn + \nu (\nabla u)n = h & \text{on } \Gamma_N
\end{cases}
$$

$$
\begin{cases}
\int_{\Omega} \nu \nabla u : \nabla v \, d\Omega - \int_{\Omega} p (\nabla \cdot v) \, d\Omega = \int_{\Gamma_N} h \cdot v \, d\Gamma - \int_{\Omega} \nu \nabla r_g : \nabla v \, d\Omega & \forall v \in X \\
-\int_{\Omega} q (\nabla \cdot u) \, d\Omega = \int_{\Omega} q (\nabla \cdot r_g) \, d\Omega & \forall q \in Q
\end{cases}
$$
