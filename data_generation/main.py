import os
import numpy as np
import subprocess
from scipy.stats.qmc import LatinHypercube

from data_classes import CircuitBoard, BOUNDS, ThermalProperties, l_bounds, u_bounds
from openfoam import create_foam_case
from meshing import generate_gmsh_mesh
from plotting import plot_msh, save_plot_as_image


def main():
    N = 10  # Number of design points to generate
    d = len(l_bounds)
    sampler = LatinHypercube(d=d)
    samples = sampler.random(n=N)
    l_bounds_arr = np.array(l_bounds)
    u_bounds_arr = np.array(u_bounds)
    scaled_samples = samples * (u_bounds_arr - l_bounds_arr) + l_bounds_arr

    ref_dir = "baseCase"  # Assume reference OpenFOAM case directory exists
    os.mkdir("simulation_results")
    for i in range(N):
        sample_list = scaled_samples[i]
        sample = {}
        index = 0
        for k in BOUNDS.keys():
            if k.endswith("comps"):
                new_index = index + BOUNDS["n_up"][1]
                sample[k] = sample_list[index:new_index]
                index = new_index
            else:
                sample[k] = sample_list[index]
                index += 1

        # Create CircuitBoard instance
        circuit_board = CircuitBoard(
            h=int(sample["h"]),
            w=int(sample["w"]),
            h_pcb=sample["h_pcb"],
            w_pcb=sample["w_pcb"],
            n_up=int(sample["n_up"]),
            w_comps=sample["w_comps"],
            h_comps=sample["h_comps"],
        )

        # Create ThermalProperties instance
        thermal = ThermalProperties(
            p_comps=sample["p_comps"],
            k_pcb=sample["k_pcb"],
            k_comps=sample["k_comps"],
        )

        u = sample["u_fluid"]

        # Create OpenFOAM case
        main_dir = os.path.join("simulation_results", f"case_{i}")
        # os.mkdir(main_dir)
        create_foam_case(ref_dir, main_dir, BOUNDS["n_up"][1], u, thermal)

        # Generate mesh inside the case
        mesh_file = os.path.join(main_dir, "circuit_mesh.msh")
        generate_gmsh_mesh(circuit_board, mesh_size=0.05, output_file=mesh_file)

        # Create and save mesh image inside the case
        #fig, _ = plot_msh(mesh_file)
        #image_file = os.path.join(main_dir, "mesh_image.png")
        #save_plot_as_image(fig, image_file)
        print(f"Solving case {i}...")
        subprocess.run("./Allrun", cwd=f"./simulation_results/case_{i}")
        gnu_args = [
            "gnuplot",
            "-e",
            "set title 'Average Temperature over Iterations'; set xlabel 'Iteration'; set ylabel 'Volume-Averaged Temperature (T)'; set key autotitle columnhead; set terminal pngcairo; set output 'Temperature.png';",
        ]
        plot_commands = [
            "plot 'pcb/pcb/0/volFieldValue.dat' every ::5 using 1:2 with lines title 'PCB'",
            ", 'fluid/fluid/0/volFieldValue.dat' every ::5 using 1:2 with lines title 'Fluid'",
        ]
        for ii in range(BOUNDS["n_up"][1]):
            plot_commands.append(
                f", 'component_{ii}/component_{ii}/0/volFieldValue.dat' every ::5 using 1:2 with lines title 'component {ii}'"
            )

        gnu_args[2] += " ".join(plot_commands)
        subprocess.run(gnu_args, cwd=f"./simulation_results/case_{i}/postProcessing")


if __name__ == "__main__":
    main()
