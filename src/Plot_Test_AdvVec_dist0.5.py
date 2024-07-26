# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
outdir = "/Users/jgra0019/Documents/codes/uw3-dev/Navier-Stokes-benchmark/plots-SLCN-test"

fname = "SL-adv-out-travel0.5.csv"
results_df = pd.read_csv(f"{outdir}/{fname}", sep = ",")

simp_reg_min_rad_dict = {16 : 0.029463, 
                         32 : 0.014731, 
                         64 : 0.007366,
                         128: 0.003683}
simp_irreg_min_rad_dict = { 16 : 0.023592, 
                            32 : 0.013391, 
                            64 : 0.006649,
                            128: 0.003332}
struct_quad_min_rad_dict = {16 : 0.044194, 
                            32 : 0.022097, 
                            64 : 0.011049,
                            128: 0.005524}

mesh_list = ["simp_reg", "simp_irreg", "struct_quad"]
x_list = ["DT", "COURANT"]
title_list = [[r"Regular simplex; as a function of $dt$", r"Regular simplex; as a function of $C$", 
              r"Irregular simplex; as a function of $dt$", r"Irregular simplex; as a function of $C$", 
              r"Structured quad; as a function of $dt$", r"Structured quad; as a function of $C$"]]

plt_min = 0.5*np.nanmin(results_df[["W_NORM", "V_NORM"]].to_numpy())
plt_max = 2*np.nanmax(results_df[["W_NORM", "V_NORM"]].to_numpy())

# xmin = 0.5*results_df[["W_NORM", "V_NORM"]].to_numpy().min()
# xmax = 2*results_df[["W_NORM", "V_NORM"]].to_numpy().max()
print(plt_min, plt_max)

# %%
y_use = "V_NORM"
mesh_use = "simp_reg"
title_list = [r"As a function of $\Delta t$", r"As a function of $C$"] 

reso_list = results_df["RES"].unique()
nrows = 1
ncols = 2

fig, axs = plt.subplots(nrows, ncols, dpi = 300, figsize = (13, 4.5))

if mesh_use == "simp_reg":
    use_dict = simp_reg_min_rad_dict
elif mesh_use == "simp_irreg":
    use_dict = simp_irreg_min_rad_dict
elif mesh_use == "struct_quad":
    use_dict = struct_quad_min_rad_dict

df_plot = results_df[results_df["MESH_TYPE"] == mesh_use]
df_plot.sort_values(by = "DT", inplace = True)

for j in range(ncols):

    x_use = x_list[j]

    for reso in reso_list:

        to_plot = df_plot[df_plot["RES"] == reso]

        lab = r"$\Delta x$ "+ f"= 1/{reso}; " + r"$\Delta x_{min}$ " + f"= {use_dict[reso]}"

        axs[j].plot(to_plot[x_use], to_plot[y_use], "o-", label = lab, lw = 0.8, ms = 2)

        if j == 0:
            leg = axs[j].legend()
            leg._legend_box.align = "right"
    
        axs[j].set_ylim([plt_min, plt_max])
        axs[j].set_xscale("log")
        axs[j].set_yscale("log")
        axs[j].set_title(title_list[j], fontsize = 10)

        # if i < 4:
        #     ax.set_xticklabels([])
        # else:
        if x_use == "DT":
            axs[j].set_xlabel(r"$\Delta t$")
        else:
            axs[j].set_xlabel("Courant number, " + r"$\frac{v \Delta t}{\Delta x_{min}}$")

        if y_use == "V_NORM":
            axs[j].set_ylabel(r"$||(\mathbf{u} - \mathbf{u*})||_2$ / $||\mathbf{u*}||_2$")
        elif y_use == "W_NORM":
            axs[j].set_ylabel(r"$||(\omega - \omega*)||_2$ / $||\omega*||_2$")
            
        else:
            pass

plt.suptitle("Regular simplex; " + r"$d_{adv} = 0.5$" + "; order = 1")

fname = f"{mesh_use}-adv-0.5-{y_use.lower()}.png"

plt.savefig(f"{outdir}/{fname}")

# %%
y_use = "W_NORM"
mesh_use = "simp_reg"
title_list = [r"As a function of $\Delta t$", r"As a function of $C$"] 

reso_list = results_df["RES"].unique()
nrows = 1
ncols = 2

fig, axs = plt.subplots(nrows, ncols, dpi = 300, figsize = (13, 4.5))

if mesh_use == "simp_reg":
    use_dict = simp_reg_min_rad_dict
elif mesh_use == "simp_irreg":
    use_dict = simp_irreg_min_rad_dict
elif mesh_use == "struct_quad":
    use_dict = struct_quad_min_rad_dict

df_plot = results_df[results_df["MESH_TYPE"] == mesh_use]
df_plot.sort_values(by = "DT", inplace = True)

for j in range(ncols):

    x_use = x_list[j]

    for reso in reso_list:

        to_plot = df_plot[df_plot["RES"] == reso]

        lab = r"$\Delta x$ "+ f"= 1/{reso}; " + r"$\Delta x_{min}$ " + f"= {use_dict[reso]}"

        axs[j].plot(to_plot[x_use], to_plot[y_use], "o-", label = lab, lw = 0.8, ms = 2)

        if j == 0:
            leg = axs[j].legend()
            leg._legend_box.align = "right"
    
        axs[j].set_ylim([plt_min, plt_max])
        axs[j].set_xscale("log")
        axs[j].set_yscale("log")
        axs[j].set_title(title_list[j], fontsize = 10)

        # if i < 4:
        #     ax.set_xticklabels([])
        # else:
        if x_use == "DT":
            axs[j].set_xlabel(r"$\Delta t$")
        else:
            axs[j].set_xlabel("Courant number, " + r"$\frac{v \Delta t}{\Delta x_{min}}$")

        if y_use == "V_NORM":
            axs[j].set_ylabel(r"$||(\mathbf{u} - \mathbf{u*})||_2$ / $||\mathbf{u*}||_2$")
        elif y_use == "W_NORM":
            axs[j].set_ylabel(r"$||(\omega - \omega*)||_2$ / $||\omega*||_2$")
            
        else:
            pass

plt.suptitle("Regular simplex; " + r"$d_{adv} = 0.5$" + "; order = 1")

fname = f"{mesh_use}-adv-0.5-{y_use.lower()}.png"

plt.savefig(f"{outdir}/{fname}")

# %%
# plot the velocity field as a check
import underworld3 as uw

mesh_use = "simp_reg"
Vdeg = 2
res = 128
idx = 99
infile = f"VecAdv-run{idx}"
maxsteps = 50
outdir = "/Users/jgra0019/Documents/codes/uw3-dev/Navier-Stokes-benchmark/SLCN-test"

xmin, xmax = 0., 2.
ymin, ymax = 0., 1.

sdev = 0.1
x0 = 0.5
y0 = 0.5

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

vec_tst   = uw.discretisation.MeshVariable("Vn", mesh, mesh.dim, degree=Vdeg)
vec_ana   = uw.discretisation.MeshVariable("Va", mesh, mesh.dim, degree=Vdeg)

vec_tst.read_timestep(data_filename = infile, data_name = "Vn", index = maxsteps, outputPath = outdir)
vec_ana.read_timestep(data_filename = infile, data_name = "Va", index = maxsteps, outputPath = outdir)

# %%


if uw.mpi.size == 1:
    
    import pyvista as pv
    import underworld3.visualisation as vis

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["Vmag"] = vis.scalar_fn_to_pv_points(pvmesh, vec_tst.sym.dot(vec_tst.sym))
    
    #pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, vec_tst.sym)
    #pvmesh.point_data["V0"] = vis.vector_fn_to_pv_points(pvmesh, vec_ana.sym)

    v_points = vis.meshVariable_to_pv_cloud(vec_tst)
    v_points.point_data["V"] = vis.vector_fn_to_pv_points(v_points, vec_tst.sym)
    v_points.point_data["V0"] = vis.vector_fn_to_pv_points(v_points, vec_ana.sym)
     
    pl = pv.Plotter(window_size=(1000, 750))


    pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.75)

    pl.add_arrows( v_points.points, 
                   v_points.point_data["V"], 
                   mag=0.2, 
                  color="Green", 
                  show_scalar_bar=True)

    pl.add_arrows( v_points.points, 
                   v_points.point_data["V0"], 
                   mag=0.2, 
                  color="Blue", 
                  show_scalar_bar=True)

    pl.show()


