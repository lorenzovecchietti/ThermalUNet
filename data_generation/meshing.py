from typing import List, Tuple

import gmsh
from data_classes import CircuitBoard


def define_geometry(params: CircuitBoard) -> Tuple[List[Tuple[float, float]], ...]:
    """
    Defines the geometric layout of the circuit board and its components.

    Args:
        params: An instance of the CircuitBoard dataclass.

    Returns:
        A tuple containing lists of coordinates for the domain, PCB, components, and heatsinks.
    """
    domain = [(0, 0), (params.w, params.h)]

    pcb_height = params.h * params.h_pcb
    pcb_width = params.w * params.w_pcb
    pcb = [
        (0.5 * params.w * (1 - params.w_pcb), 0.5 * params.h * (1 - params.h_pcb)),
        (
            0.5 * params.w * (1 - params.w_pcb) + pcb_width,
            0.5 * params.h * (1 - params.h_pcb) + pcb_height,
        ),
    ]

    components = []

    total_components = params.n_up + params.n_down

    for i in range(total_components):
        if i < params.n_up:
            sub_index = i
            sign = 1
            n_slots = params.n_up
            y_start = pcb[1][1]
        else:
            sub_index = i - params.n_up
            sign = -1
            n_slots = params.n_down
            y_start = pcb[0][1]

        # Component position and size calculations
        x_start = (
            pcb[0][0]
            + (pcb_width / n_slots) * sub_index
            + (pcb_width / n_slots) * (1 - params.w_comps[i]) / 2
        )
        x_end = x_start + params.w_comps[i] * pcb_width / n_slots
        y_end = y_start + params.h_comps[i] * pcb_height * sign

        components.append([(x_start, y_start), (x_end, y_end)])

    return domain, pcb, components


def generate_gmsh_mesh(
    params: CircuitBoard, mesh_size: float = 0.1, output_file: str = "circuit_mesh.msh"
):
    """
    Generates a 3D GMSH mesh for a circuit board model.

    This function defines the 2D geometry, extrudes it into 3D volumes,
    identifies the different physical regions (fluid, PCB, components, heatsinks),
    assigns physical groups for both volumes and boundaries, and finally
    generates and saves the mesh.

    Args:
        params: The CircuitBoard dataclass instance with geometry parameters.
        mesh_size: The desired mesh size factor.
        output_file: The name of the output mesh file.
    """
    tol = 1e-6
    extrusion_depth: float = 0.2

    gmsh.initialize()
    gmsh.model.add("circuit_board_3d")

    # =========================================================================
    # 2D geometry creation
    # =========================================================================
    domain_coords, pcb_coords, components = define_geometry(params)

    domain_tag = gmsh.model.occ.addRectangle(
        domain_coords[0][0],
        domain_coords[0][1],
        0,
        domain_coords[1][0] - domain_coords[0][0],
        domain_coords[1][1] - domain_coords[0][1],
    )

    solid_tags_2d = []

    # Create all solids as separate entities
    pcb_tag = gmsh.model.occ.addRectangle(
        pcb_coords[0][0],
        pcb_coords[0][1],
        0,
        pcb_coords[1][0] - pcb_coords[0][0],
        pcb_coords[1][1] - pcb_coords[0][1],
    )
    solid_tags_2d.append(pcb_tag)

    component_tags_2d = []
    for comp in components:
        comp_tag = gmsh.model.occ.addRectangle(
            comp[0][0], comp[0][1], 0, comp[1][0] - comp[0][0], comp[1][1] - comp[0][1]
        )
        solid_tags_2d.append(comp_tag)
        component_tags_2d.append(comp_tag)

    # Fragment the domain with the solids. This automatically generates shared
    # interface surfaces.
    gmsh.model.occ.fragment(
        [(2, domain_tag)], [(2, tag) for tag in solid_tags_2d], removeTool=True
    )
    gmsh.model.occ.synchronize()

    # =========================================================================
    # Extrusion to 3D
    # =========================================================================
    gmsh.model.occ.extrude(
        gmsh.model.getEntities(2),
        0,
        0,
        extrusion_depth,
        numElements=[1],
        recombine=True,
    )
    gmsh.model.occ.synchronize()

    # =========================================================================
    # 3. Assign 3D Physical Groups (Volumes)
    # =========================================================================
    gmsh.model.occ.synchronize()  # Ensure the model is updated

    # Get the center of mass of the original 2D solid shapes for comparison
    com_pcb_2d = gmsh.model.occ.getCenterOfMass(2, pcb_tag)
    com_components_2d = {
        tag: gmsh.model.occ.getCenterOfMass(2, tag) for tag in component_tags_2d
    }

    # Initialize variables to store the 3D volume tags
    pcb_vol = None
    component_vols_map = {}
    solid_vol_tags = set()

    # Get all 3D entities (volumes)
    volumes_3d = gmsh.model.getEntities(3)

    for dim, tag in volumes_3d:
        com_3d = gmsh.model.occ.getCenterOfMass(dim, tag)

        # Check if this volume corresponds to the PCB
        if (
            abs(com_3d[0] - com_pcb_2d[0]) < tol
            and abs(com_3d[1] - com_pcb_2d[1]) < tol
        ):
            pcb_vol = tag
            solid_vol_tags.add(tag)
            continue

        # Check if this volume corresponds to any of the components
        identified = False
        for comp_2d_tag, com_2d in com_components_2d.items():
            if abs(com_3d[0] - com_2d[0]) < tol and abs(com_3d[1] - com_2d[1]) < tol:
                component_vols_map[comp_2d_tag] = tag
                solid_vol_tags.add(tag)
                identified = True
                break
        if identified:
            continue

    # Any volume tag not identified as a solid must be a fluid volume
    all_vol_tags = {tag for dim, tag in volumes_3d}
    fluid_vols = list(all_vol_tags - solid_vol_tags)

    # Re-order the component/heatsink vols to match the original list order
    component_vols = [
        component_vols_map[t] for t in component_tags_2d if t in component_vols_map
    ]

    # Create the Physical Groups for the volumes
    gmsh.model.addPhysicalGroup(3, fluid_vols, name="fluid")
    if pcb_vol is not None:
        gmsh.model.addPhysicalGroup(3, [pcb_vol], name="pcb")
    for i, vol_tag in enumerate(component_vols):
        gmsh.model.addPhysicalGroup(3, [vol_tag], name=f"component_{i}")

    # =========================================================================
    # 4. Assign 2D Physical Groups (Boundaries)
    # =========================================================================
    inlet_surf = []
    outlet_surf = []
    walls_surf = []
    empty_surf = []
    tol = 1e-5

    # Get domain coordinates for comparisons
    domain_min_x, domain_min_y = domain_coords[0]
    domain_max_x, domain_max_y = domain_coords[1]
    z_min, z_max = 0, extrusion_depth

    all_surfaces = gmsh.model.getEntities(2)

    for dim, tag in all_surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)

        # PRIORITY 1: Front and back faces (on Z) are always 'empty'
        if abs(com[2] - z_min) < tol or abs(com[2] - z_max) < tol:
            empty_surf.append(tag)
        # PRIORITY 2: Otherwise, check if they are inlet or outlet (on X)
        elif abs(com[0] - domain_min_x) < tol:
            inlet_surf.append(tag)
        elif abs(com[0] - domain_max_x) < tol:
            outlet_surf.append(tag)
        # DEFAULT: All other surfaces are 'walls'
        else:
            walls_surf.append(tag)

    # Create Physical Groups
    gmsh.model.addPhysicalGroup(2, list(set(inlet_surf)), name="inlet")
    gmsh.model.addPhysicalGroup(2, list(set(outlet_surf)), name="outlet")
    gmsh.model.addPhysicalGroup(2, list(set(walls_surf)), name="walls")
    gmsh.model.addPhysicalGroup(2, list(set(empty_surf)), name="empty")

    # =========================================================================
    # 5. Generate and Save Mesh
    # =========================================================================

    # Set mesh size and algorithm options
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine triangles into quads
    gmsh.option.setNumber(
        "Mesh.RecombinationAlgorithm", 1
    )  # L'algorithm for recombination

    # Generate and save the mesh
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_file)
    gmsh.finalize()
