import os
from typing import List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import meshio
import numpy as np
from meshing import define_geometry

def plot_design_point(geometry_data: Tuple[List[Tuple[float, float]], ...]):
    """
    Plots the geometric layout of the circuit board and its components.

    Args:
        geometry_data: A tuple containing the coordinates for the domain, PCB, components,
                       and heatsinks, as returned by the `define_geometry` function.
    """
    # Deconstruct the tuple returned by define_geometry
    domain, pcb, components, heatsinks, inlet, outlet = geometry_data

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(50, 20))

    # Get domain and PCB dimensions
    domain_width = domain[1][0] - domain[0][0]
    domain_height = domain[1][1] - domain[0][1]
    pcb_width = pcb[1][0] - pcb[0][0]
    pcb_height = pcb[1][1] - pcb[0][1]

    # Add the rectangles using the extracted coordinates
    domain_rect = patches.Rectangle(
        domain[0],
        domain_width,
        domain_height,
        linewidth=3,
        edgecolor="red",
        facecolor="none",
        label="Domain",
    )
    pcb_rect = patches.Rectangle(
        pcb[0],
        pcb_width,
        pcb_height,
        linewidth=1,
        edgecolor="blue",
        facecolor="blue",
        alpha=0.5,
        label="PCB",
    )

    ax.add_patch(domain_rect)
    ax.add_patch(pcb_rect)

    # Add component rectangles
    for i, comp in enumerate(components):
        comp_width = comp[1][0] - comp[0][0]
        comp_height = comp[1][1] - comp[0][1]

        comp_rect = patches.Rectangle(
            comp[0],
            comp_width,
            comp_height,
            linewidth=0,
            facecolor="green",
            alpha=0.8,
            label=f"Component {i+1}",
        )

        ax.add_patch(comp_rect)

    # Add heatsink rectangles (if any)
    for i, hs in enumerate(heatsinks):
        hs_width = hs[1][0] - hs[0][0]
        hs_height = hs[1][1] - hs[0][1]

        hs_rect = patches.Rectangle(
            hs[0],
            hs_width,
            hs_height,
            linewidth=0,
            facecolor="red",
            alpha=0.3,
            label=f"Heatsink {i+1}",
        )

        ax.add_patch(hs_rect)

    ax.plot(
        [inlet[0][0], inlet[1][0]],
        [inlet[0][1], inlet[1][1]],
        color="black",
        label="Inlet",
        linewidth=4,
    )
    ax.plot(
        [outlet[0][0], outlet[1][0]],
        [outlet[0][1], outlet[1][1]],
        color="black",
        label="Outlet",
        linewidth=4,
    )

    # Plot settings
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(domain[0][0], domain[1][0])
    ax.set_ylim(domain[0][1], domain[1][1])
    ax.grid(True)

    return fig, ax


def plot_msh(msh_file):
    # Load the .msh file
    mesh = meshio.read(msh_file)
    # Extract points and cells (2D elements like triangles)
    points = mesh.points[:, :2]  # Only x, y (2D)
    cells = mesh.cells_dict["triangle"]  # Assuming triangles

    # Extract physical tags of the surfaces (cell data)
    cell_data = mesh.cell_data_dict["gmsh:physical"][
        "triangle"
    ]  # Physical tags for triangles

    # Get unique physical tags present in the mesh
    unique_tags = np.unique(cell_data)
    n_zones = len(unique_tags)

    # Generate a dynamic color map (distinct colors for each tag)
    dyn_colors = {
        tag: plt.cm.Set3(i / n_zones) for i, tag in enumerate(unique_tags)
    }  # Use Set3 for distinct colors

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(50, 20))

    # Plot each physical group separately
    for tag in unique_tags:
        # Get the indices of triangles belonging to this physical group
        indices = np.where(cell_data == tag)[0]
        # Get the triangles for this physical group
        triangles_for_tag = cells[indices]
        # Get the color for this physical group
        color = dyn_colors[tag]  # This is already an RGBA tuple

        # Plot the triangles for the current physical group with a single color
        ax.tripcolor(
            points[:, 0],
            points[:, 1],
            triangles=triangles_for_tag,
            facecolors=np.full(len(triangles_for_tag), 1.0),  # Dummy scalar array
            cmap=None,  # No colormap needed
            color=color,  # Use the single color for this group
            edgecolors="k",
            linewidth=2.5,
        )

    ax.set_title("Mesh with Colored Zones")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")  # Correct proportions
    return fig, ax


# Function to save plots as images (simulated here)
def save_plot_as_image(fig, filename):
    fig.savefig(filename, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_images(design_points, mesh_files, export_dir):
    for i, (dp, mesh_file) in enumerate(zip(design_points, mesh_files)):
        circuit_board, thermal_props = dp

        # Generate and save geometry plot
        fig, _ = plot_design_point(define_geometry(circuit_board))
        geometry_image = os.path.join(export_dir, f"dp_{i}_geometry.png")
        save_plot_as_image(fig, geometry_image)

        # Generate and save mesh plot
        fig, _ = plot_msh(mesh_file)
        mesh_image = os.path.join(export_dir, f"dp_{i}_mesh.png")
        save_plot_as_image(fig, mesh_image)
