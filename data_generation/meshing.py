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

    pcb_height = params.h * params.pcb_h
    pcb_width = params.w * params.pcb_w
    pcb = [
        (0.5 * params.w * (1 - params.pcb_w), 0.5 * params.h * (1 - params.pcb_h)),
        (
            0.5 * params.w * (1 - params.pcb_w) + pcb_width,
            0.5 * params.h * (1 - params.pcb_h) + pcb_height,
        ),
    ]
    inlet = [
        (0, 0.5 * params.h * (1 - params.inlet)),
        (0, 0.5 * params.h * (1 - params.inlet) + params.h * params.inlet),
    ]
    outlet = [
        (params.w, 0.5 * params.h * (1 - params.outlet)),
        (params.w, 0.5 * params.h * (1 - params.outlet) + params.h * params.outlet),
    ]

    components = []
    heatsinks = []

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
            + (pcb_width / n_slots) * (1 - params.components_w[i]) / 2
        )
        x_end = x_start + params.components_w[i] * pcb_width / n_slots
        y_end = y_start + params.components_h[i] * pcb_height * sign

        components.append([(x_start, y_start), (x_end, y_end)])

        # Heatsink position and size calculation (if a heatsink exists)
        if params.heatsink[i] > 0:
            y_end_heatsink = y_end + params.heatsink[i] * (y_end - y_start)
            heatsinks.append([(x_start, y_end), (x_end, y_end_heatsink)])

    return domain, pcb, components, heatsinks, inlet, outlet


def generate_gmsh_mesh(
    params: CircuitBoard, mesh_size: float = 0.1, output_file: str = "circuit_mesh.msh"
):
    gmsh.initialize()
    gmsh.model.add("circuit_board")

    # Extract geometry
    domain_coords, pcb_coords, components, heatsinks, inlet_coords, outlet_coords = (
        define_geometry(params)
    )

    # Create domain rectangle
    domain_tag = gmsh.model.occ.addRectangle(
        domain_coords[0][0],
        domain_coords[0][1],
        0,
        domain_coords[1][0] - domain_coords[0][0],
        domain_coords[1][1] - domain_coords[0][1],
    )

    # Create PCB rectangle
    pcb_tag = gmsh.model.occ.addRectangle(
        pcb_coords[0][0],
        pcb_coords[0][1],
        0,
        pcb_coords[1][0] - pcb_coords[0][0],
        pcb_coords[1][1] - pcb_coords[0][1],
    )

    # List of solid tags
    solid_tags = [pcb_tag]

    # Create component rectangles
    for idx, comp in enumerate(components):
        y_min = min(comp[0][1], comp[1][1])
        height = abs(comp[1][1] - comp[0][1])
        comp_tag = gmsh.model.occ.addRectangle(
            comp[0][0], y_min, 0, comp[1][0] - comp[0][0], height
        )
        solid_tags.append(comp_tag)

    # Create heatsink rectangles
    for idx, hs in enumerate(heatsinks):
        y_min = min(hs[0][1], hs[1][1])
        height = abs(hs[1][1] - hs[0][1])
        hs_tag = gmsh.model.occ.addRectangle(
            hs[0][0], y_min, 0, hs[1][0] - hs[0][0], height
        )
        solid_tags.append(hs_tag)

    # Perform boolean difference to get fluid region
    fluid_fragments = gmsh.model.occ.cut(
        [(2, domain_tag)], [(2, tag) for tag in solid_tags], removeTool=False
    )
    fluid_tags = [tag[1] for tag in fluid_fragments[0]]

    gmsh.model.occ.synchronize()

    # Physical groups for zones
    gmsh.model.addPhysicalGroup(2, fluid_tags, name="fluid")
    gmsh.model.addPhysicalGroup(2, [pcb_tag], name="pcb")
    for idx, tag in enumerate(solid_tags[1 : len(components) + 1]):
        gmsh.model.addPhysicalGroup(2, [tag], name=f"component_{idx}")
    for idx, tag in enumerate(solid_tags[len(components) + 1 :]):
        gmsh.model.addPhysicalGroup(2, [tag], name=f"heatsink_{idx}")

    # Define boundaries
    # Get all curves
    curves = gmsh.model.getEntities(1)

    # Identify inlet and outlet by splitting left and right walls
    # Add points for splitting
    p_inlet_bottom = gmsh.model.occ.addPoint(0, inlet_coords[0][1], 0)
    p_inlet_top = gmsh.model.occ.addPoint(0, inlet_coords[1][1], 0)
    p_outlet_bottom = gmsh.model.occ.addPoint(params.w, outlet_coords[0][1], 0)
    p_outlet_top = gmsh.model.occ.addPoint(params.w, outlet_coords[1][1], 0)

    gmsh.model.occ.synchronize()

    # Fragment the left and right walls to isolate inlet and outlet
    # For simplicity, assume left wall is a single line initially
    # Fragment with points
    # Need to find the tags of the left and right boundary curves first
    left_boundary_tags = []
    right_boundary_tags = []
    for dim, tag in curves:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)  # Corrected method call
        if abs(com[0] - 0) < 1e-6:
            left_boundary_tags.append((dim, tag))
        elif abs(com[0] - params.w) < 1e-6:
            right_boundary_tags.append((dim, tag))

    if left_boundary_tags:
        gmsh.model.occ.fragment(
            left_boundary_tags, [(0, p_inlet_bottom), (0, p_inlet_top)]
        )
    if right_boundary_tags:
        gmsh.model.occ.fragment(
            right_boundary_tags, [(0, p_outlet_bottom), (0, p_outlet_top)]
        )

    gmsh.model.occ.synchronize()

    # Re-fetch curves after fragmentation
    curves = gmsh.model.getEntities(1)

    # Manually assign inlet and outlet (approximation based on position)
    inlet_curves = []
    outlet_curves = []
    for dim, tag in curves:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)  # Corrected method call
        if (
            abs(com[0] - 0) < 1e-6
            and inlet_coords[0][1] <= com[1] <= inlet_coords[1][1]
        ):
            inlet_curves.append(tag)
        elif (
            abs(com[0] - params.w) < 1e-6
            and outlet_coords[0][1] <= com[1] <= outlet_coords[1][1]
        ):
            outlet_curves.append(tag)

    # Assign physical groups for boundaries
    gmsh.model.addPhysicalGroup(1, inlet_curves, name="inlet")
    gmsh.model.addPhysicalGroup(1, outlet_curves, name="outlet")

    # All other curves are walls
    wall_curves = [
        tag
        for dim, tag in curves
        if tag not in inlet_curves and tag not in outlet_curves
    ]
    gmsh.model.addPhysicalGroup(1, wall_curves, name="walls")

    # Set mesh size
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Write mesh
    gmsh.write(output_file)

    gmsh.finalize()
