import os

import numpy as np
from data_classes import BOUNDS, CircuitBoard, ThermalProperties
from meshing import generate_gmsh_mesh
from plotting import generate_images
from scipy.stats import qmc
from openfoam import create_case

NUM_SAMPLES = 1


# Create the LHS sampler
n_comps = BOUNDS["n_up"][1]  # This is the maximum number of components

# n dimensions
keys = ["h", "w", "pcb_h", "pcb_w", "n_up"]
l_bounds_full = (
    [BOUNDS[k][0] for k in keys]
    + [BOUNDS["components_w"][0]] * n_comps
    + [BOUNDS["components_h"][0]] * n_comps
    + [BOUNDS["heatsink"][0]] * n_comps
    + [BOUNDS["p_pcb"][0]]
    + [BOUNDS["p_comps"][0]] * n_comps
    + [BOUNDS["k_pcb"][0]]
    + [BOUNDS["k_comps"][0]] * n_comps
    + [BOUNDS["k_heatsinks"][0]] * n_comps
    + [BOUNDS["porosity_heatsinks"][0]] * n_comps
)
u_bounds_full = (
    [BOUNDS[k][1] for k in keys]
    + [BOUNDS["components_w"][1]] * n_comps
    + [BOUNDS["components_h"][1]] * n_comps
    + [BOUNDS["heatsink"][1]] * n_comps
    + [BOUNDS["p_pcb"][1]]
    + [BOUNDS["p_comps"][1]] * n_comps
    + [BOUNDS["k_pcb"][1]]
    + [BOUNDS["k_comps"][1]] * n_comps
    + [BOUNDS["k_heatsinks"][1]] * n_comps
    + [BOUNDS["porosity_heatsinks"][1]] * n_comps
)

num_dimensions = len(l_bounds_full)

# sampler
sampler = qmc.LatinHypercube(d=num_dimensions)
samples = sampler.random(n=NUM_SAMPLES)

# Scale the samples using the full bounds lists
scaled_samples = qmc.scale(samples, l_bounds_full, u_bounds_full)

# Convert samples to design points
design_points = []
for scaled in scaled_samples:
    idx = 0

    h = int(np.round(scaled[idx]))
    idx += 1

    w = int(np.round(scaled[idx]))
    idx += 1

    pcb_h = scaled[idx]
    idx += 1

    pcb_w = scaled[idx]
    idx += 1

    n_up = int(np.round(scaled[idx]))
    idx += 1

    components_w = scaled[idx : idx + n_comps].tolist()
    idx += n_comps

    components_h = scaled[idx : idx + n_comps].tolist()
    idx += n_comps

    heatsink = [int(np.round(x)) for x in scaled[idx : idx + n_comps]]
    idx += n_comps

    # Thermal properties (new addition)
    p_pcb = scaled[idx]
    idx += 1

    p_comps = scaled[idx : idx + n_comps].tolist()
    idx += n_comps

    k_pcb = scaled[idx]
    idx += 1

    k_comps = scaled[idx : idx + n_comps].tolist()
    idx += n_comps

    k_heatsinks = scaled[idx : idx + n_comps].tolist()
    idx += n_comps

    porosity_heatsinks = scaled[idx : idx + n_comps].tolist()
    idx += n_comps

    # Create CircuitBoard and ThermalProperties instances
    circuit_board = CircuitBoard(
        h=h,
        w=w,
        pcb_h=pcb_h,
        pcb_w=pcb_w,
        n_up=n_up,
        components_w=components_w,
        components_h=components_h,
        heatsink=heatsink,
    )
    thermal_props = ThermalProperties(
        p_pcb=p_pcb,
        p_comps=p_comps,
        k_pcb=k_pcb,
        k_comps=k_comps,
        k_heatsinks=k_heatsinks,
        porosity_heatsinks=porosity_heatsinks,
    )
    design_points.append((circuit_board, thermal_props))


def create_mesh_directory(directory="mesh_files"):
    if os.path.exists(directory):
        raise FileExistsError(
            f"Directory '{directory}' already exists. Please remove or rename it."
        )
    os.makedirs(directory)
    return directory


# Create mesh directory
mesh_dir = create_mesh_directory()
fig_dir = create_mesh_directory(os.path.join(mesh_dir,"img"))

# Generate meshes
mesh_files = []
for i, dp in enumerate(design_points):
    circuit_board, _ = dp
    mesh_file = os.path.join(mesh_dir, f"dp_{i}.msh")
    generate_gmsh_mesh(circuit_board, output_file=mesh_file, mesh_size=0.05)
    mesh_files.append(mesh_file)

generate_images(design_points, mesh_files, fig_dir)


test_msh = os.path.join(mesh_dir,"dp_0.msh")
case = "./testCase"
Uinlet=10
rhoFluid=1
cpFluid=2
kFluid=2.5
kPcb=3
rhoSolid=4
cpSolid=5
kCmp=[6.1,6.2,6.3,6.4,6.5]
QCmp=[7.1, 7.2, 7.3, 7.4, 7.5]
epsHS=[8.1, 8.2, 8.3, 8.4, 8.5]
dHS=[9.1, 9.2, 9.3, 9.4, 9.5]
kHS=[10,11,12,13,14]
t_amb=300
create_case(test_msh,case,Uinlet,rhoFluid,cpFluid,kFluid,kPcb,rhoSolid,cpSolid,kCmp,QCmp,epsHS,dHS,kHS,t_amb)
