import json
import os
import shutil
import subprocess

import numpy as np
import vtk
from data_classes import (BOUNDS, CircuitBoard, ThermalProperties, l_bounds,
                          u_bounds)
from meshing import generate_gmsh_mesh
from openfoam import create_foam_case
from scipy.stats.qmc import LatinHypercube


def main():
    NITER = 2000
    N = 100  # Number of design points to generate
    d = len(l_bounds)
    sampler = LatinHypercube(d=d)
    samples = sampler.random(n=N)
    l_bounds_arr = np.array(l_bounds)
    u_bounds_arr = np.array(u_bounds)
    scaled_samples = samples * (u_bounds_arr - l_bounds_arr) + l_bounds_arr

    ref_dir = "baseCase"  # Assume reference OpenFOAM case directory exists
    os.mkdir("simulation_results")
    os.mkdir("simulation_results/dataset")
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
        main_dir = os.path.join("simulation_results", "openfoam", f"case_{i}")
        # os.mkdir(main_dir)
        create_foam_case(ref_dir, main_dir, BOUNDS["n_up"][1], u, thermal, NITER)

        # Generate mesh inside the case
        mesh_file = os.path.join(main_dir, "circuit_mesh.msh")
        generate_gmsh_mesh(circuit_board, mesh_size=0.05, output_file=mesh_file)

        # Create and save mesh image inside the case
        print(f"Solving case {i}...")
        subprocess.run("./Allrun", cwd=f"./simulation_results/openfoam/case_{i}")
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
        subprocess.run(gnu_args, cwd=f"./simulation_results/openfoam/case_{i}/postProcessing")

        # vti creation
        if str(NITER) not in os.listdir(main_dir):
            print(f"+++ Case {i} was not solved correctly. +++")
            continue
        reader = vtk.vtkXMLMultiBlockDataReader()
        reader.SetFileName(
            os.path.join(main_dir, "VTK", f"case_{i}-regions_{NITER}.vtm")
        )
        reader.Update()
        multi_block_data = reader.GetOutput()
        for j in range(multi_block_data.GetNumberOfBlocks()):
            block = multi_block_data.GetBlock(j)
            if block:
                block_name = multi_block_data.GetMetaData(j).Get(
                    vtk.vtkCompositeDataSet.NAME()
                )
                conducibility_value = 0.0  # O un altro valore sensato per il fluido
                power_value = 0.0
                if block_name == "pcb":
                    conducibility_value = thermal.k_pcb
                    power_value = 0
                elif block_name.startswith("component"):
                    component_i = int(block_name.split("_")[1])
                    conducibility_value = thermal.k_comps[component_i]
                    power_value = thermal.p_comps[component_i]
                sub_block = block.GetBlock(0)
                conducibility_array = vtk.vtkDoubleArray()
                conducibility_array.SetName("conducibility")
                power_array = vtk.vtkDoubleArray()
                power_array.SetName("power")
                for k in range(sub_block.GetNumberOfCells()):
                    conducibility_array.InsertNextValue(conducibility_value)
                    power_array.InsertNextValue(power_value)
                sub_block.GetCellData().AddArray(conducibility_array)
                sub_block.GetCellData().AddArray(power_array)
                velocity_array = vtk.vtkDoubleArray()
                velocity_array.SetName("U")
                if block_name == "fluid":
                    fluid_U = sub_block.GetCellData().GetArray("U")
                    velocity_array.DeepCopy(fluid_U)
                else:
                    velocity_array.SetNumberOfComponents(3)
                    num_cells = block.GetBlock(0).GetNumberOfCells()
                    velocity_array.SetNumberOfTuples(num_cells)
                    for k in range(num_cells):
                        velocity_array.SetTuple3(k, 0.0, 0.0, 0.0)
                sub_block.GetCellData().AddArray(velocity_array)
        append_filter = vtk.vtkAppendFilter()
        for j in range(multi_block_data.GetNumberOfBlocks()):
            block = multi_block_data.GetBlock(j)
            # each sub-block has just 1 block
            append_filter.AddInputData(block.GetBlock(0))
        append_filter.Update()
        unified_data = append_filter.GetOutput()
        probe_grid = vtk.vtkImageData()
        probe_grid.SetDimensions(400, 100, 1)
        probe_grid.SetSpacing(0.05, 0.05, 1.0)
        probe_grid.SetOrigin(0.0, 0.0, 0.0)
        probe_filter = vtk.vtkProbeFilter()
        probe_filter.SetInputData(probe_grid)
        probe_filter.SetSourceData(unified_data)
        probe_filter.SetPassFieldArrays(True)
        probe_filter.Update()
        resampled_data = probe_filter.GetOutput()
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(os.path.join(main_dir, "VTK", f"case_{i}.vti"))
        writer.SetInputData(resampled_data)
        writer.Write()
        dst_dir = os.path.join("simulation_results", "dataset", f"case_{i}")
        os.makedirs(dst_dir)
        src_dir = os.path.join(main_dir, "VTK")
        src_file = os.path.join(src_dir, f"case_{i}.vti")
        shutil.copy2(src_file, dst_dir)
        shutil.rmtree(src_dir)
        boundary = {
            "h_pcb": circuit_board.h_pcb,
            "w_pcb": circuit_board.w_pcb,
            "n_up": circuit_board.n_up,
            "w_comps": '"' + ",".join(map(str, circuit_board.w_comps)) + '"',
            "h_comps": '"' + ",".join(map(str, circuit_board.h_comps)) + '"',
            "p_comps": '"' + ",".join(map(str, thermal.p_comps)) + '"',
            "k_pcb": thermal.k_pcb,
            "k_comps": '"' + ",".join(map(str, thermal.k_comps)) + '"',
            "u": u,
        }
        with open(os.path.join(dst_dir, "boundary.json"), "w") as f:
            json.dump(boundary, f)

    archive_path = shutil.make_archive(
            base_name=os.path.join("simulation_results", "dataset"),
            format="zip",
            root_dir=os.path.join("simulation_results", "dataset")
    )
    shutil.rmtree(os.path.join("simulation_results", "dataset"))

if __name__ == "__main__":
    main()
