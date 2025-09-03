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
    domain, pcb, components, heatsinks = geometry_data

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

    # Plot settings
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(domain[0][0], domain[1][0])
    ax.set_ylim(domain[0][1], domain[1][1])
    ax.grid(True)

    return fig, ax


from matplotlib.collections import PolyCollection

def plot_msh(msh_file):
    """
    Plots a 2D slice of the 3D mesh, coloring triangles by their parent volume's physical group.
    Args:
        msh_file: Path to the Gmsh .msh file.
    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    # Load the .msh file
    mesh = meshio.read(msh_file)
    
    # Check for triangle elements
    if "triangle" not in mesh.cells_dict:
        raise ValueError("No triangle elements found in the mesh. Ensure 2D surfaces are included.")
    
    # Extract points and triangles
    points = mesh.points
    cells = mesh.cells_dict["triangle"]
    
    # Extract physical tags for volumes (tetra or hexahedron)
    volume_cell_types = ["quad", "triangle", "wedge"]
    volume_cell_data = {}
    for cell_type in volume_cell_types:
        if cell_type in mesh.cells_dict and "gmsh:physical" in mesh.cell_data_dict:
            volume_cell_data[cell_type] = mesh.cell_data_dict["gmsh:physical"].get(cell_type, [])
    
    if not volume_cell_data:
        raise ValueError("No physical tags found for volume elements.")
    
    # Get physical group names from field_data
    physical_names = mesh.field_data  # Maps tag to name and dimension
    
    # Filter triangles at z=0 (or close to z=0)
    tol = 1e-6
    triangle_centroids = np.mean(points[cells], axis=1)  # Centroid of each triangle
    z0_mask = np.abs(triangle_centroids[:, 2]) < tol  # Triangles at z=0
    z0_cells = cells[z0_mask]
    
    if len(z0_cells) == 0:
        raise ValueError("No triangles found at z=0. Check mesh extrusion or z-coordinate.")
    
    # Map triangles to their parent volume's physical group
    triangle_tags = np.zeros(len(z0_cells), dtype=int)
    for i, tri in enumerate(z0_cells):
        # Find the volume containing this triangle
        tri_centroid = np.mean(points[tri], axis=0)
        for cell_type, cell_block in volume_cell_data.items():
            for j, vol_cells in enumerate(mesh.cells_dict[cell_type]):
                vol_centroid = np.mean(points[vol_cells], axis=0)
                # Check if triangle centroid is close to volume centroid in x, y
                if (abs(tri_centroid[0] - vol_centroid[0]) < tol and 
                    abs(tri_centroid[1] - vol_centroid[1]) < tol):
                    triangle_tags[i] = cell_block[j]
                    break
    
    # Get unique physical tags for the triangles
    unique_tags = np.unique(triangle_tags)
    if len(unique_tags) == 0:
        raise ValueError("No valid physical tags assigned to triangles at z=0.")
    
    # Generate a dynamic color map
    n_zones = len(unique_tags)
    dyn_colors = {tag: plt.cm.Set3(i / n_zones) for i, tag in enumerate(unique_tags)}
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(50, 20))
    
    # Prepare vertices for PolyCollection
    all_verts = []
    all_colors = []
    for tag in unique_tags:
        indices = np.where(triangle_tags == tag)[0]
        triangles_for_tag = z0_cells[indices]
        verts = points[triangles_for_tag][:, :, :2]  # Extract x, y coordinates
        all_verts.append(verts)
        color = dyn_colors[tag]
        all_colors.append(np.tile(color, (len(triangles_for_tag), 1)))
    
    # Combine all vertices and colors
    all_verts = np.concatenate(all_verts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    # Create PolyCollection
    collection = PolyCollection(all_verts, facecolors=all_colors, edgecolors="k", linewidth=0.5)
    ax.add_collection(collection)
    
    # Add legend
    legend_elements = [
        Patch(facecolor=dyn_colors[tag], label=physical_names.get(tag, [f"Tag {tag}"])[0])
        for tag in unique_tags
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    
    # Set plot limits
    ax.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax.set_title("Mesh with Colored Volume Zones (z=0 slice)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
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
        #fig, _ = plot_msh(mesh_file)
        #mesh_image = os.path.join(export_dir, f"dp_{i}_mesh.png")
        #save_plot_as_image(fig, mesh_image)
