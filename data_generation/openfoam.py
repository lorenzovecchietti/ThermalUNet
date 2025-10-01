import os
import shutil


# Probes
def modify_probes_dict(case_dir, num_components):
    template = """
component_{i}
{{
    ${{_volFieldValue}}

    regionType      cellZone;
    name            component_{i};
    region          component_{i};
    operation       volAverage;
    fields          ( T );
}}
"""

    with open(os.path.join(case_dir, r"system/probesDict"), "r") as file:
        file_content = file.read()

    new_components_string = ""
    for i in range(num_components):
        new_components_string += template.format(i=i)

    closing_text = """
#remove (_volFieldValue)

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //"""

    modified_content = file_content + new_components_string + closing_text

    with open(os.path.join(case_dir, r"system/probesDict"), "w") as file:
        file.write(modified_content)


def modify_system_components(case_dir, num_components):
    old_dir = os.path.join(case_dir, "system", "component_")
    for i in range(num_components):
        new_path = os.path.join(case_dir, "system", f"component_{i}")
        shutil.copytree(old_dir, new_path)
        fv_options_path = os.path.join(new_path, "fvOptions")

        with open(fv_options_path, "r") as file:
            content = file.read()

        old_string = "    cellZone        component_;"
        new_string = f"    cellZone         component_{i};"
        modified_content = content.replace(old_string, new_string)

        old_string = "h           ( $hComponent 0 );"
        new_string = f"h           ( $hComponent{i} 0 );"
        modified_content = modified_content.replace(old_string, new_string)

        with open(fv_options_path, "w") as file:
            file.write(modified_content)
    shutil.rmtree(old_dir)


def modify_0_components(case_dir, num_components):
    old_dir = os.path.join(case_dir, "0", "component_")
    for i in range(num_components):
        new_dir = os.path.join(case_dir, "0", f"component_{i}")
        shutil.copytree(old_dir, new_dir)
        for filen in ["p", "T"]:
            with open(os.path.join(new_dir, filen), "r") as file:
                content = file.read()

            old_string = 'location    "0/component_"'
            new_string = f'location    "0/component_{i}";'
            modified_content = content.replace(old_string, new_string)

            with open(os.path.join(new_dir, filen), "w") as file:
                file.write(modified_content)
    shutil.rmtree(old_dir)


def modify_consntant_components(case_dir, num_components):
    old_dir = os.path.join(case_dir, "constant", "component_")
    for i in range(num_components):
        new_dir = os.path.join(case_dir, "constant", f"component_{i}")
        shutil.copytree(old_dir, new_dir)
        old_string = "kappa           $kComponent;"
        new_string = f"kappa           $kComponent{i};"
        with open(os.path.join(new_dir, "thermophysicalProperties"), "r") as file:
            content = file.read()
        modified_content = content.replace(old_string, new_string)

        with open(os.path.join(new_dir, "thermophysicalProperties"), "w") as file:
            file.write(modified_content)
    shutil.rmtree(old_dir)


def write_sim_parameters(case_dir, num_components, u, thermal, n_iter):
    sim_params = f"nIter {n_iter};\n"

    sim_params += "\n".join(
        [f"hComponent{i} {val};" for i, val in enumerate(thermal.p_comps)]
    )
    sim_params += "\n".join(
        [f"kComponent{i} {val};" for i, val in enumerate(thermal.k_comps)]
    )
    sim_params += "\n" + "tAmbient 300;"
    sim_params += "\n" + f"Ux {u};"
    sim_params += "\n" + f"kPCB {thermal.k_pcb};"
    with open(os.path.join(case_dir, "constant", "simParams"), "w") as f:
        f.write(sim_params)


def modify_region_properties(main_dir, num_components):
    old_string = "    solid $solids"
    solids_list = " ".join(["pcb"] + [f"component_{i}" for i in range(num_components)])
    new_string = f"    solid ({solids_list})"
    with open(os.path.join(main_dir, "constant", "regionProperties"), "r") as file:
        content = file.read()
    modified_content = content.replace(old_string, new_string)
    with open(os.path.join(main_dir, "constant", "regionProperties"), "w") as file:
        file.write(modified_content)


def create_foam_case(ref_dir, main_dir, num_components, u, thermal, n_iter):
    shutil.copytree(ref_dir, main_dir)
    modify_consntant_components(main_dir, num_components)
    write_sim_parameters(main_dir, num_components, u, thermal, n_iter)
    modify_0_components(main_dir, num_components)
    modify_system_components(main_dir, num_components)
    modify_probes_dict(main_dir, num_components)
    modify_region_properties(main_dir, num_components)
