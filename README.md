# uw3-advection-tests

## Latest results (02/10/2024)

Test scenario: 
Vector field (rigid body with Gaussian envelope) travels a total distance of 0.5. 
Velocity is uniform throughout the domain with v = (1.0, 0.0).

### A. Regular simplex mesh
A.1. Relative norm of the velocity as a function of time step, $\Delta t$, and Courant number, C: 

![Alt text](out/simp_reg-adv-0.5-v_norm-order1-regression-Oct-1-2024.png)

A.2. Relative norm of the vorticity, $\omega$, as a function of time step, $\Delta t$, and Courant number, C: 

![Alt text](out/simp_reg-adv-0.5-w_norm-order1-regression-Oct-1-2024)

### B. Irregular simplex mesh
B.1. Relative norm of the velocity as a function of time step, $\Delta t$, and Courant number, C: 

![Alt text](out/simp_irreg-adv-0.5-v_norm-order1-regression-Oct-1-2024.png)

B.2. Relative norm of the vorticity, $\omega$, as a function of time step, $\Delta t$, and Courant number, C: 

![Alt text](out/simp_irreg-adv-0.5-w_norm-order1-regression-Oct-1-2024.png)

### B. Structured quadrilatetral mesh
B.1. Relative norm of the velocity as a function of time step, $\Delta t$, and Courant number, C: 

![Alt text](out/struct_quad-adv-0.5-v_norm-order1-regression-Oct-1-2024.png)

B.2. Relative norm of the vorticity, $\omega$, as a function of time step, $\Delta t$, and Courant number, C: 

![Alt text](out/struct_quad-adv-0.5-w_norm-order1-regression-Oct-1-2024.png)


