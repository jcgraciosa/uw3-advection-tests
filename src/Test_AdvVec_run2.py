# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: uw3-venv-run
#     language: python
#     name: python3
# ---

# %%
import os

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy
import argparse
import pickle

import os

import petsc4py
import underworld3 as uw
from underworld3 import timing

import numpy as np
import sympy
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--idx", type=int, required=True)
parser.add_argument('-p', "--prev", type=int, required=True) # set to 0 if no prev_res, 1 if there is
parser.add_argument('-t', "--dt", type=float, required=True) # deltaT
parser.add_argument('-s', "--ms", type=int, required=True) # maxsteps
args = parser.parse_args()

idx = args.idx
prev = args.prev
dt_ns = args.dt
maxsteps = args.ms

mesh_use = str(os.getenv("MESH_USE", "struct_quad"))
if uw.mpi.rank == 0:
    print(f"Mesh used: {mesh_use}")
res = int(os.getenv("RES", 8))

#if dt_ns == 0.1:
#    maxsteps = 5 # 5
#else:
#    maxsteps = 50 # 10

velocity = 1

tol = 1e-5
adv_type = "vector" # scalar or vector
vel_type = "v_rigid_body" # v_irrotational or v_rigid_body
#mesh_use = "struct_quad" # struct_quad or simp_irreg or simp_reg

qdeg     = 3
Vdeg     = 2
sl_order = 1

if uw.mpi.rank == 0:
    print(maxsteps*dt_ns*velocity)

# %%
outdir = "/Users/jgra0019/Documents/codes/uw3-dev/Navier-Stokes-benchmark/plots-SLCN-test"

#outfile = f"VecAdv-run{idx}"
outfile = f"VecAdv-run" # overwrite outputted files to reduce total size
outdir = f"./VecAdv-{mesh_use}-{vel_type}-res{res}-order{sl_order}-dt{dt_ns}"

if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    #infile = f"VecAdv-run{prev_idx}"
    infile = f"VecAdv-run"

if uw.mpi.rank == 0:
    os.makedirs(outdir, exist_ok=True)


# %%
# ### mesh coordinates
xmin, xmax = 0., 2.
ymin, ymax = 0., 1.

sdev = 0.1
x0 = 0.5
y0 = 0.5

# ### Set up the mesh
### Quads
if mesh_use == "struct_quad":
    mesh = uw.meshing.StructuredQuadBox(
        elementRes=(int(res*xmax), int(res)),
        minCoords=(xmin, ymin),
        maxCoords=(xmax, ymax),
        qdegree=3,
    )
elif mesh_use == "simp_irreg":
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(xmin,ymin),
        maxCoords=(xmax,ymax),
        cellSize=1 / res, regular=False, qdegree=3, refinement=0
    )
elif mesh_use == "simp_reg":
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(xmin,ymin),
        maxCoords=(xmax,ymax),
        cellSize=1 / res, regular=True, qdegree=3, refinement=0
    )


# %%
v           = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=Vdeg)

# initial vector field
vec_ini   = uw.discretisation.MeshVariable("Vi", mesh, mesh.dim, degree=Vdeg)

# vector being advected
vec_tst   = uw.discretisation.MeshVariable("Vn", mesh, mesh.dim, degree=Vdeg)
vec_ana   = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=Vdeg)

# vorticity
omega_tst  = uw.discretisation.MeshVariable(r"\omega_t", mesh, 1, degree=2)
omega_ana  = uw.discretisation.MeshVariable(r"\omega_a", mesh, 1, degree=2)

# %%
# functions for calculating the norms
# NOTE: these modify the vector and vorticity fields
# so should be done at the end of everything
import math

def calculate_vel_omega_rel_norm():

    # define domain where we perform the integral
    dist_travel = velocity*dt_ns*(idx + 1)*maxsteps
    min_dom = 0.1 * x0 + dist_travel
    max_dom = x0 + dist_travel + 0.9 * x0

    x,y = mesh.X

    mask_fn = sympy.Piecewise((1, (x > min_dom) &  (x < max_dom)), (0., True))

    # sympy functions corresponding to integrals
    vec_diff = vec_tst.sym - vec_ana.sym
    vec_diff_mag = vec_diff.dot(vec_diff)

    vec_ana_mag = vec_ana.sym.dot(vec_ana.sym)

    omega_diff = (omega_tst.sym - omega_ana.sym)**2
    omega_ana_sq = omega_ana.sym**2

    vec_diff_mag_integ = math.sqrt(uw.maths.Integral(mesh, mask_fn * vec_diff_mag).evaluate())
    vec_ana_mag_integ = math.sqrt(uw.maths.Integral(mesh, mask_fn * vec_ana_mag).evaluate())
    vec_norm = vec_diff_mag_integ / vec_ana_mag_integ

    omega_diff_integ = math.sqrt(uw.maths.Integral(mesh, mask_fn * omega_diff).evaluate())
    omega_ana_sq_integ = math.sqrt(uw.maths.Integral(mesh, mask_fn * omega_ana_sq).evaluate())
    omega_norm = omega_diff_integ / omega_ana_sq_integ

    return omega_norm, vec_norm

# def calculate_vel_omega_rms(offset = 0):
#     # NOTE: function not used
#     # define domain where we perform the integral
#     min_dom = 0.1 * x0 + offset
#     max_dom = x0 + offset + 0.9 * x0

#     #print(f"min: {min_dom}, max: {max_dom}")

#     x,y = mesh.X

#     mask_fn = sympy.Piecewise((1, (x > min_dom) &  (x < max_dom)), (0., True))

#     # sympy functions corresponding to integrals
#     vec_tst_mag     = vec_tst.sym.dot(vec_tst.sym)
#     omega_tst_sq    = omega_tst.sym**2
#     area            = uw.maths.Integral(mesh, mask_fn * 1.0).evaluate()

#     vec_tst_rms     = math.sqrt(uw.maths.Integral(mesh, mask_fn * vec_tst_mag).evaluate() / area)
#     omega_tst_rms   = math.sqrt(uw.maths.Integral(mesh, mask_fn * omega_tst_sq).evaluate() / area)

#     return omega_tst_rms, vec_tst_rms

# %%
# #### Create the SL object
DuDt = uw.systems.ddt.SemiLagrangian(
                                        mesh,
                                        vec_tst.sym,
                                        v.sym,
                                        vtype = uw.VarType.VECTOR,
                                        degree = Vdeg,
                                        continuous = vec_tst.continuous,
                                        varsymbol = vec_tst.symbol,
                                        verbose = False,
                                        bcs = None,
                                        order = sl_order,
                                        smoothing = 0.0,
                                    )

# %%
# ### Set up:
# - Velocity field
# - Initial vector distribution

with mesh.access(v):
    v.data[:, 0] = velocity

x,y = mesh.X

## Irrotational vortex
def v_irrotational(alpha, x0, y0, coords):
    '''
    Irrotational vortex

    $$ (vx, vy) = (-\alpha y r^{-2}, \alpha x r^{-2} $$
    '''

    ar2 = alpha / ((x - x0)**2 + (y - y0)**2 + 0.001)
    return uw.function.evaluate(sympy.Matrix([-ar2 * (y-y0), ar2 * (x-x0)]) ,coords)


def v_rigid_body(alpha, x0, y0, coords):
    '''
    Rigid body vortex (with Gaussian envelope)

    $$ (vx, vy) = (-\Omega y, \Omega y) $$
    '''
    ar2 = sympy.exp(-alpha*((x - x0)**2 + (y - y0)**2 + 0.000001))
    return uw.function.evaluate(sympy.Matrix([-ar2 * (y-y0), ar2 * (x-x0)]) ,coords)

if infile is None:
    with mesh.access(vec_tst, vec_ini):
        if vel_type == "v_irrotational":
            vec_tst.data[:, :] =  v_irrotational(0.01, x0, y0, vec_tst.coords)
            vec_ini.data[:, :] =  v_irrotational(0.01, x0, y0, vec_ini.coords)
        elif vel_type == "v_rigid_body":
            vec_tst.data[:, :] = v_rigid_body(33, x0, y0, vec_tst.coords)
            vec_ini.data[:, :] =  v_rigid_body(33, x0, y0, vec_ini.coords)
else:
    vec_tst.read_timestep(data_filename = infile, data_name = "Vn", index = maxsteps, outputPath = outdir)
    # don't need to read vec_ini

vorticity_calc_test = uw.systems.Projection(mesh, omega_tst)
vorticity_calc_test.uw_function = mesh.vector.curl(vec_tst.sym)
vorticity_calc_test.petsc_options["snes_monitor"]= None
vorticity_calc_test.petsc_options["ksp_monitor"] = None

vorticity_calc_ana = uw.systems.Projection(mesh, omega_ana)
vorticity_calc_ana.uw_function = mesh.vector.curl(vec_ana.sym)
vorticity_calc_ana.petsc_options["snes_monitor"]= None
vorticity_calc_ana.petsc_options["ksp_monitor"] = None

#vorticity_calc_test.solve()
# %%
# initial velocity and omega RMS
# of vec_tst - nothing done on vec_ana
#om_rms_init, v_rms_init = calculate_vel_omega_rms(offset = 0)
#if uw.mpi.rank == 0:
#    print(f"Omega RMS: {om_rms_init:.8f}")
#    print(f"V RMS: {v_rms_init:.8f}")

# %%
ts = 0
elapsed_time = 0.0
model_time  = 0.0

# %%

for step in range(0, maxsteps):

    delta_t = dt_ns
    DuDt.update_pre_solve(delta_t, verbose = False, evalf = False)

    with mesh.access(vec_tst): # update vector field
        vec_tst.data[...] = DuDt.psi_star[0].data[...]

    model_time += delta_t
    if uw.mpi.rank == 0:
        print(f"{step}: Time - {model_time}")

    step += 1

# calculate analytical velocity field
dist_travel = velocity*dt_ns*(idx + 1)*maxsteps # travel since first iteration loop
with mesh.access(vec_ana):
    if vel_type == "v_irrotational":
        vec_ana.data[:, :] = v_irrotational(0.01, x0 + dist_travel, y0, vec_ana.coords)
    elif vel_type == "v_rigid_body":
        vec_ana.data[:, :] = v_rigid_body(33, x0 + dist_travel, y0, vec_ana.coords)

# calculate vorticity
vorticity_calc_ana.solve()
vorticity_calc_test.solve()

# current omega and v rms of v_test
#om_rms, v_rms = calculate_vel_omega_rms(offset = velocity*dt_ns*step)
# print(f"Omega RMS: {om_rms:.8f}")
# print(f"V RMS: {v_rms:.8f}")

# courant_num = max(a_param, b_param) * meshbox.get_min_radius()/dt_ns
courant_num = velocity * dt_ns / mesh.get_min_radius()
om_norm, v_norm = calculate_vel_omega_rel_norm()

# Relative difference in omega_rms and v_rms
#om_rms_rel = abs(om_rms_init - om_rms)/om_rms_init
#v_rms_rel = abs(v_rms_init - v_rms)/v_rms_init

if uw.mpi.rank == 0:
    print(f"step: {step}")
    print(f"Cumulative distance traveled: {dist_travel:.6f}")
    print(f"Courant number: {courant_num:.6f}")
    print(f"Omega rel norm: {om_norm:.8f}")
    print(f"V rel norm: {v_norm:.8f}")
    #print(f"Omega rel error: {om_rms_rel:.8f}")
    #print(f"Velocity rel error: {v_rms_rel:.8f}")


mesh.write_timestep(
        outfile,
        meshUpdates=True,
        meshVars=[vec_tst, vec_ana],
        outputPath=outdir,
        index = step,
    )


# %%



