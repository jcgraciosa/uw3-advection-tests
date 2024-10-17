# %%
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
args = parser.parse_args()

idx = args.idx
prev = args.prev
dt_ns = args.dt

# %%
resolution = 16
#dt_ns = 0.0000001 # timestep - use constant
vel_cond = 1.

maxsteps = 5
save_every = 5
tol = 1e-5
adv_type = "vector" # scalar or vector
wave_type = "gaussian" # square or gaussian

qdeg = 3
Vdeg = 2
Pdeg = Vdeg - 1
ns_order = 1

show_vis = False


# %%
outfile = f"AdvTest-run{idx}"
outdir = f"./AdvTest-vel{vel_cond}-res{resolution}-dt{dt_ns}"

# %%
if prev == 0:
    prev_idx = 0
    infile = None
else:
    prev_idx = int(idx) - 1
    infile = f"AdvTest-run{prev_idx}"

if uw.mpi.rank == 0:
    os.makedirs(".meshes", exist_ok = True)
    os.makedirs(outdir, exist_ok = True)

# %%
# dimensional quantities
width = 1.
height = 0.5

x0 = 0.2*width
y0 = 0.5*height
sdev = 0.03

xmin, xmax = 0, width
ymin, ymax = 0, height

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords= (0.0, 0.0),
                                                maxCoords= (width, height),
                                                cellSize= height / resolution,
                                                regular=False,
                                                qdegree = qdeg
                                        )

# %%
meshbox.dm.view()

# %%
v_soln          = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=Vdeg)
vect_test       = uw.discretisation.MeshVariable("A", meshbox, meshbox.dim, degree=Vdeg)
scalar_test     = uw.discretisation.MeshVariable("S", meshbox, 1, degree=Vdeg)
scalar_adv_test = uw.discretisation.MeshVariable("S2", meshbox, 1, degree=Vdeg)

# %%
# passive_swarm = uw.swarm.Swarm(mesh=pipemesh)

# create initial velocity field
x,y = meshbox.X

gauss_envelope      = sympy.exp(-((x - x0)**2 + (y - y0)**2)/(2*sdev**2))
v_field_init        = 50 * gauss_envelope * sympy.Matrix([-0.5*(y - y0), 0.5*(x - x0)])

with meshbox.access(v_soln, vect_test, scalar_test, scalar_adv_test):

    if adv_type == "scalar":

        if wave_type == "gaussian":
            gauss_along_x = sympy.exp(-((x - x0)**2)/(2*sdev**2))
            scalar_test.data[:] = uw.function.evaluate(gauss_along_x, scalar_test.coords).reshape(-1, 1)
            scalar_adv_test.data[:] = uw.function.evaluate(gauss_along_x, scalar_adv_test.coords).reshape(-1, 1)
        elif wave_type == "square":
            # step function
            scalar_test.data[...] = 0
            cond = (scalar_test.coords[:, 0] > 0.1) & (scalar_test.coords[:, 0] < 0.3)
            scalar_test.data[cond] = 1

            scalar_adv_test.data[...] = 0
            cond = (scalar_adv_test.coords[:, 0] > 0.1) & (scalar_adv_test.coords[:, 0] < 0.3)
            scalar_adv_test.data[cond] = 1

    elif adv_type == "vector":

        if wave_type == "gaussian":
            gauss_along_x = sympy.exp(-((x - x0)**2)/(2*sdev**2))
            gauss_along_y = sympy.exp(-((y - x0)**2)/(2*sdev**2))

            vect_test.data[:, 0] = 2*vel_cond*uw.function.evaluate(gauss_envelope, vect_test.coords)
            vect_test.data[:, 1] = 2*vel_cond*uw.function.evaluate(gauss_envelope, vect_test.coords)


        elif wave_type == "square":
            vect_test.data[...] = 0
            cond = (vect_test.coords[:, 0] > 0.1) & (vect_test.coords[:, 0] < 0.3)
            vect_test.data[cond, 0] = vel_cond
            vect_test.data[cond, 1] = vel_cond

        vmag = np.sqrt(vect_test.data[:, 0]**2 + vect_test.data[:, 1]**2)
        print(vmag.min())
        print(vmag.max())


    # velocity is constant
    v_soln.data[:, 0] = vel_cond


# %%
# meshbox.get_min_radius()/0.2

# %%
# Set solve options here (or remove default values
if adv_type == "scalar":
    DuDt = uw.systems.ddt.SemiLagrangian(
                                            meshbox,
                                            scalar_test.sym,
                                            v_soln.sym,
                                            vtype = uw.VarType.SCALAR,
                                            degree = Vdeg,
                                            continuous = scalar_test.continuous,
                                            varsymbol = scalar_test.symbol,
                                            verbose = True,
                                            bcs = None,
                                            order = ns_order,
                                            smoothing = 0.0,
                                        )
    adv_diff = uw.systems.AdvDiffusionSLCN(
                                            meshbox,
                                            u_Field = scalar_adv_test,
                                            V_fn = v_soln,
                                            solver_name = "adv_diff",
                                        )
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 0.0 # this is the kinematic viscosity
    adv_diff.add_dirichlet_bc(0, "Left")
    adv_diff.add_dirichlet_bc(0, "Right")

elif adv_type == "vector":
    DuDt = uw.systems.ddt.SemiLagrangian(
                                            meshbox,
                                            vect_test.sym,
                                            v_soln.sym,
                                            vtype = uw.VarType.VECTOR,
                                            degree = Vdeg,
                                            continuous = vect_test.continuous,
                                            varsymbol = vect_test.symbol,
                                            verbose = True,
                                            bcs = None,
                                            order = ns_order,
                                            smoothing = 0.0,
                                        )


# %%
ts = 0
elapsed_time = 0.0
timeVal =  np.zeros(maxsteps + 1)*np.nan      # time values

# %%
# print(maxsteps)

# %%
for step in range(0, maxsteps + 1):

    delta_t = dt_ns

    # run pre-solves to advect things
    DuDt.update_pre_solve(delta_t, verbose = False, evalf = False)

    # update v_soln data using the DuDt contents
    if adv_type == "scalar":
        adv_diff.solve(timestep=delta_t, zero_init_guess=False)

        with meshbox.access(scalar_test):
            scalar_test.data[...] = DuDt.psi_star[0].data[...]
    elif adv_type == "vector":
        with meshbox.access(vect_test):
            vect_test.data[...] = DuDt.psi_star[0].data[...]

    elapsed_time += delta_t
    timeVal[step] = elapsed_time

    if uw.mpi.rank == 0:
        print("Timestep {}, t {}, dt {}".format(ts, elapsed_time, delta_t))

    if prev == 0: # first run = save first iteration
        add_cond = True
    else:
        add_cond = ts > 0

    if ts % save_every == 0 and add_cond:
        if adv_type == "scalar":
            meshbox.write_timestep(
                outfile,
                meshUpdates=True,
                meshVars=[scalar_test, scalar_adv_test],
                outputPath=outdir,
                index = ts,
            )
        elif adv_type == "vector":
            meshbox.write_timestep(
                outfile,
                meshUpdates=True,
                meshVars=[vect_test],
                outputPath=outdir,
                index = ts,
            )

    ts += 1



