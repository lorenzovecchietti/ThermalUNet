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

    return domain, pcb, components, heatsinks


def generate_gmsh_mesh(
    params: CircuitBoard, mesh_size: float = 0.1, output_file: str = "circuit_mesh.msh"
):
    tol = 1e-6
    extrusion_depth: float = 0.2
    gmsh.initialize()
    gmsh.model.add("circuit_board_3d")
    # =========================================================================
    # 2D geometry creation
    # =========================================================================
    domain_coords, pcb_coords, components, heatsinks = (
        define_geometry(params)
    )
    domain_tag = gmsh.model.occ.addRectangle(
        domain_coords[0][0], domain_coords[0][1], 0,
        domain_coords[1][0] - domain_coords[0][0], domain_coords[1][1] - domain_coords[0][1],
    )
    
    solid_tags_2d = []
    
    # Crea tutti i solidi come entità separate
    pcb_tag = gmsh.model.occ.addRectangle(
        pcb_coords[0][0], pcb_coords[0][1], 0, 
        pcb_coords[1][0] - pcb_coords[0][0], pcb_coords[1][1] - pcb_coords[0][1],
    )
    solid_tags_2d.append(pcb_tag)
    
    component_tags_2d = [] 
    for comp in components:
        comp_tag = gmsh.model.occ.addRectangle(
            comp[0][0], comp[0][1], 0, comp[1][0] - comp[0][0], comp[1][1] - comp[0][1]
        )
        solid_tags_2d.append(comp_tag)
        component_tags_2d.append(comp_tag) 

    heatsink_tags_2d = []
    heatsink_tags_id = [] 
    for i,hs in enumerate(heatsinks):
        if abs(hs[0][1]-hs[1][1]) > tol:
            heatsink_tags_id.append(i)
            hs_tag = gmsh.model.occ.addRectangle(
                hs[0][0], hs[0][1], 0, hs[1][0] - hs[0][0], hs[1][1] - hs[0][1]
            )
            solid_tags_2d.append(hs_tag)
            heatsink_tags_2d.append(hs_tag) 

    # FRAGMENTA il dominio con i solidi. Questo genera automaticamente superfici di interfaccia condivise.
    all_entities_2d = [(2, domain_tag)]
    all_entities_2d.extend([(2, tag) for tag in solid_tags_2d])
    
    # L'operazione `fragment` creerà nuove entità, incluse quelle di contatto.
    fragmented_entities = gmsh.model.occ.fragment(
        [(2, domain_tag)], [(2, tag) for tag in solid_tags_2d], removeTool=True
    )

    gmsh.model.occ.synchronize()
    # =========================================================================
    # Extrusion to 3D
    # =========================================================================
    extruded_entities = gmsh.model.occ.extrude(
        gmsh.model.getEntities(2), 0, 0, extrusion_depth, numElements=[1], recombine=True
    )
    gmsh.model.occ.synchronize()
   
    # =========================================================================
    # 3. ASSEGNAZIONE DEI PHYSICAL GROUPS 3D (VOLUMI) -- CORRECTED SECTION
    # =========================================================================
    # The previous method of iterating through 'extruded_entities' was incorrect
    # because its return structure is a flat list.
    #
    # NEW, ROBUST METHOD:
    # 1. Get all 3D volumes from the model.
    # 2. Identify each volume by comparing the XY coordinates of its center of mass
    # with the center of mass of the original 2D solids.
    # 3. Any volume not identified as a solid must be part of the fluid.
    gmsh.model.occ.synchronize() # Ensure the model is updated
    # Get the center of mass of the original 2D solid shapes for comparison
    com_pcb_2d = gmsh.model.occ.getCenterOfMass(2, pcb_tag)
    com_components_2d = {tag: gmsh.model.occ.getCenterOfMass(2, tag) for tag in component_tags_2d}
    com_heatsinks_2d = {tag: gmsh.model.occ.getCenterOfMass(2, tag) for tag in heatsink_tags_2d}
    # Initialize variables to store the 3D volume tags
    fluid_vols = []
    pcb_vol = None
    component_vols_map = {} # Use a map to keep the order correct later
    heatsink_vols_map = {}
    # Get all 3D entities (volumes) that now exist in the model
    volumes_3d = gmsh.model.getEntities(3)
    solid_vol_tags = set()
    for dim, tag in volumes_3d:
        com_3d = gmsh.model.occ.getCenterOfMass(dim, tag)
       
        # Check if this volume corresponds to the PCB
        if abs(com_3d[0] - com_pcb_2d[0]) < tol and abs(com_3d[1] - com_pcb_2d[1]) < tol:
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
           
        # Check for heatsinks
        for hs_2d_tag, com_2d in com_heatsinks_2d.items():
            if abs(com_3d[0] - com_2d[0]) < tol and abs(com_3d[1] - com_2d[1]) < tol:
                heatsink_vols_map[hs_2d_tag] = tag
                solid_vol_tags.add(tag)
                identified = True
                break
        if identified:
            continue
    # Any volume tag not identified as a solid must be a fluid volume
    all_vol_tags = {tag for dim, tag in volumes_3d}
    fluid_vols = list(all_vol_tags - solid_vol_tags)
    # Re-order the component/heatsink vols to match the original list order
    component_vols = [component_vols_map[t] for t in component_tags_2d if t in component_vols_map]
    heatsink_vols = [heatsink_vols_map[t] for t in heatsink_tags_2d if t in heatsink_vols_map]
    # Create the Physical Groups for the volumes (this part remains the same)
    gmsh.model.addPhysicalGroup(3, fluid_vols, name="fluid")
    if pcb_vol is not None:
        gmsh.model.addPhysicalGroup(3, [pcb_vol], name="pcb")
    for i, vol_tag in enumerate(component_vols):
        gmsh.model.addPhysicalGroup(3, [vol_tag], name=f"component_{i}")
    for i, vol_tag in enumerate(heatsink_vols):
        gmsh.model.addPhysicalGroup(3, [vol_tag], name=f"heatsink_{heatsink_tags_id[i]}")
    # =========================================================================
    # 4. ASSEGNAZIONE DEI PHYSICAL GROUPS 2D (CONFINI) -- REVISED SECTION
    # =========================================================================
    # Strategia:
    # Si scorrono tutte le superfici del modello e si applicano le seguenti
    # regole in ordine di priorità:
    # 1. Se una faccia si trova sul piano Z=min o Z=max -> 'empty'.
    #    Questo si applica sia ai confini esterni che alle facce dei solidi.
    # 2. ALTRIMENTI, se si trova sul piano X=min o X=max -> 'inlet' / 'outlet'.
    # 3. ALTRIMENTI, se si trova sul piano Y=min o Y=max -> 'walls'.
    # 4. ALTRIMENTI (tutti i casi rimanenti) -> 'walls'. Questo cattura
    #    automaticamente tutte le altre facce dei solidi (quelle laterali,
    #    superiori e inferiori non a contatto con il bordo del dominio).

    # Inizializziamo le liste per i tag delle superfici
    inlet_surf = []
    outlet_surf = []
    walls_surf = []
    empty_surf = []
    tol = 1e-5

    # Otteniamo le coordinate del dominio per i confronti
    domain_min_x, domain_min_y = domain_coords[0]
    domain_max_x, domain_max_y = domain_coords[1]
    z_min, z_max = 0, extrusion_depth

    all_surfaces = gmsh.model.getEntities(2)

    for dim, tag in all_surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        #print(domain_max_x, com[0])
        # PRIORITÀ 1: Facce frontali e posteriori (su Z) sono sempre 'empty'
        if abs(com[2] - z_min) < tol or abs(com[2] - z_max) < tol:
            empty_surf.append(tag)
        # PRIORITÀ 2: Altrimenti, controlla se sono inlet o outlet (su X)
        elif abs(com[0] - domain_min_x) < tol:
            inlet_surf.append(tag)
        elif abs(com[0] - domain_max_x) < tol:
            outlet_surf.append(tag)
        # DEFAULT: Tutte le altre superfici sono 'walls'
        # (es. facce laterali/superiori/inferiori dei componenti interni)
        else:
            walls_surf.append(tag)


    # --- Creazione dei Physical Groups ---
    gmsh.model.addPhysicalGroup(2, list(set(inlet_surf)), name="inlet")
    gmsh.model.addPhysicalGroup(2, list(set(outlet_surf)), name="outlet")
    gmsh.model.addPhysicalGroup(2, list(set(walls_surf)), name="walls")
    gmsh.model.addPhysicalGroup(2, list(set(empty_surf)), name="empty")
    # =========================================================================
    # 5. GENERAZIONE MESH 3D E SALVATAGGIO
    # =========================================================================
    gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size)
   
    # Genera la mesh 3D
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(output_file)
    gmsh.finalize()
