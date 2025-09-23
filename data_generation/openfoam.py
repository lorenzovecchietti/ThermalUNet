<<<<<<< Updated upstream
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict

# ------------------------------ util ------------------------------


def run(cmd: List[str], cwd: Path):
    print("$", " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando fallito: {' '.join(cmd)}")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str):
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


# ------------------------------ templating minimal ------------------------------

G = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}

dimensions      [0 1 -2 0 0 0 0];
value           (0 -9.81 0);
"""

CHANGE_DICTIONARY_DICT = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      changeDictionaryDict;
}
boundary
{
        empty
        {
            type            empty;
            physicalType    empty;
        }
        walls
        {
            type            wall;
            physicalType    wall;
        }
}
"""

CONTROL_DICT = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
application     chtMultiRegionSimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         200;
deltaT          1;
writeControl    timeStep;
writeInterval   200;
purgeWrite      0;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable yes;

lib           ("libfvOptions.so");
"""

FV_SCHEMES_EMPTY = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
}

divSchemes
{
}

laplacianSchemes
{
}

interpolationSchemes
{
}

snGradSchemes
{
}

gradSchemes
{
}
"""


FV_SCHEMES_BASE = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;
    div(phi,h)      Gauss upwind;
    div(phi,e)      Gauss upwind;
    div(phi,k)      Gauss upwind;
    div(phi,epsilon) Gauss upwind;
    div((muEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
"""

FV_SOLUTION_EMPTY = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
"""

FV_SOLUTION_BASE = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}

solvers
{
    p_rgh
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.1;
        smoother        DIC;
    }
    h
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-7;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
}

relaxationFactors
{
    equations
    {
        "(h|e)" 1.0;
    }
}
"""

REGION_PROPERTIES_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object  regionProperties;
}}
regions
(
    {regions_patches}
);
"""

TURB_PROPS_TMPL = """
FoamFile
{
    version 2.0;
    format ascii;
    class dictionary;
    object turbulenceProperties;
}
simulationType  RAS;
RAS
{
    RASModel        kEpsilon;
    turbulence      on;
    printCoeffs     off;
}
"""

THERMO_FLUID_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant/fluid";
    object thermophysicalProperties;
}}
thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}}
mixture
{{
    specie
    {{
        molWeight   28.96;
    }}
    thermodynamics
    {{
        Cp          {cp};
        Hf          0;
    }}
    transport
    {{
        mu          1.8e-5;
        Pr          0.7;
        kappa       {k};
    }}
}}
"""

THERMO_SOLID_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant/{solid}";
    object thermophysicalProperties;
}}
thermoType
{{
    type            heSolidThermo;
    mixture         pureSolid;
    transport       const;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}}
mixture
{{
    specie
    {{
        molWeight   28.96;
    }}
    thermodynamics
    {{
        Cp          {cp};
        Hf          0;
    }}
    transport
    {{
        kappa       {k};
    }}
    equationOfState
    {{
        rho         {rho};
    }}
}}
"""


U_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0,0,0);

boundaryField
{
    empty
    {
        type            empty;
    }

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           uniform 0;
    }

    "heatsink.*_to_component.*"
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
}"""
P_RGH_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p_rgh;
}

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 101325;

boundaryField
{
    empty
    {
        type            empty;
    }

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           uniform 0;
    }

    "heatsink.*_to_component.*"
    {
        type            fixedFluxPressure;
        value           $internalField;

    }
}"""
P_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alphat;
}

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 101325;

boundaryField
{
    empty
    { 
        type            empty;
    } 

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           uniform 101325;
    }

    "heatsink.*_to_component.*"
    {
        type            zeroGradient;
    }
}"""
T_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      T;
}

dimensions      [0 0 0 1 0 0 0];

internalField   uniform 300;

boundaryField
{
    empty
    { 
        type            empty;
    } 

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           uniform 0;
    }

    "heatsink.*_to_component.*"
    {
        type            compressible::turbulentTemperatureRadCoupledMixed;
        Tnbr            T;
        kappaMethod     fluidThermo;
        useImplicit     true;
        qrNbr           none;
        qr              none;
        value           $internalField;

    }
}"""
ALPHAT_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alphat;
}

dimensions      [1 -1 -1 0 0 0 0];

internalField   uniform 0.5;

boundaryField
{
    empty
    {
        type            empty;
    }

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           uniform 0;
    }

    "heatsink.*_to_component.*"
    {
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }
}
"""
NUT_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    empty
    {
        type            empty;
    }

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue
        value           $internalField;
    }

    "heatsink.*_to_component.*"
    {
        type            nutkWallFunction;
        value           $internalField;
    }
}
"""
EPSILON_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      epsilon;
}

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform 10;

boundaryField
{
    empty
    {
        type            empty;
    }

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           $internalField;
    }

    "heatsink.*_to_component.*"
    {
        type            epsilonWallFunction;
        value           $internalField;
    }
}
"""
K_HS_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      i;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.5;

boundaryField
{
    empty
    {
        type            empty;
    }

    "heatsink.*_to_fluid"
    {
        type            mappedFixedInternalValue;
        value           $internalField;
    }

    "heatsink.*_to_component.*"
    {
        type            kqRWallFunction;
        value           $internalField;
    }
}
"""

# Minimal field templates for 0/ of regions
ALPHAT_FLUID_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alphat;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -1 0 0 0 0];

internalField   uniform 0.5;

boundaryField
{
    inlet
    {
        type            calculated;
        value           $internalField;
    }

    empty
    {
        type            empty;
    }

    walls
    {
        type            calculated;
        value           $internalField;
    }

    outlet
    {
        type            calculated;
        value           $internalField;
    }

    "fluid_to_heatsink.*"
    {
        type            mappedFixedInternalValue;
        value           uniform 0;
    }

    "fluid_to_component.*"
    {
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }

    fluid_to_pcb
    {
        type            compressible::alphatWallFunction;
        Prt             0.85;
        value           uniform 0;
    }
}
"""

K_FLUID_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.5;

boundaryField
{
    inlet
    {
        type            inletOutlet;
        inletValue      uniform 0.00375;
        value           uniform 0.00375;
    }
    outlet
    {
        type            inletOutlet;
        inletValue      uniform 0.00375;
        value           uniform 0.00375;
    }
    empty
    {
        type            empty;
        value           $internalField;
    }
    walls
    {
        type            kqRWallFunction;
        value           $internalField;
    }
    "fluid_to_component.*"
    {
        type            kqRWallFunction;
        value           $internalField;
    }
    "fluid_to_heatsink.*"
    {
	type		mappedFixedInternalValue;
	value 		$internalField;
    }
    fluid_to_pcb
    {
        type            kqRWallFunction;
        value           $internalField;
    }
}
"""

EPSILON_FLUID_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      epsilon;
}

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform 10;

boundaryField
{
    inlet
    {
        type            inletOutlet;
        inletValue      uniform 0.0375;
        value           uniform 0.0375;
    }
    outlet
    {
        type            inletOutlet;
        inletValue      uniform 0.0375;
        value           uniform 0.0375;
    }
    empty
    {
        type            empty;
        value           $internalField;
    }
    walls
    {
        type            epsilonWallFunction;
        value           $internalField;
    }
    fluid_to_pcb
    {
        type            epsilonWallFunction;
        value           $internalField;
    }
    "fluid_to_component.*"
    {
        type            epsilonWallFunction;
        value           $internalField;
    }
    "fluid_to_heatsink.*"
    {
        type            mappedFixedInternalValue;
        value           $internalField;
    }
}
"""


NUT_FLUID_TMPL = """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            calculated;
        value           $internalField;
    }

    empty
    {
        type            empty;
    }

    walls
    {
        type            nutkWallFunction;
        value           $internalField;
    }

    outlet
    {
        type            calculated;
        value           $internalField;
    }

    "fluid_to_heatsink.*"
    {
        type            mappedFixedInternalValue;
        value           uniform 0;
    }

    "fluid_to_component.*"
    {
        type            nutkWallFunction;
        value           $internalField;
    }

    fluid_to_pcb
    {
        type            nutkWallFunction;
        value           $internalField;
    }
}
"""

U_FLUID_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class volVectorField;
    location "0/fluid";
    object U;
}}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({U} 0 0);
boundaryField
{{
    inlet
    {{
        type fixedValue;
        value $internalField;
    }}
    outlet
    {{
        type zeroGradient;
    }}
    walls
    {{
        type noSlip;
    }}
    empty
    {{
        type empty;
    }}
    "fluid_to_component.*"
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    fluid_to_pcb
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
    "fluid_to_heatsink.*"
    {{
        type            mappedFixedInternalValue;
        value $internalField;
    }}
}}
"""

T_FLUID_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "0/fluid";
    object T;
}}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {tAmb};
boundaryField
{{
    inlet
    {{
        type fixedValue;
        value $internalField;
    }}
    outlet
    {{
        type zeroGradient;
    }}
    walls
    {{
        type zeroGradient;
    }}
    empty
    {{
        type empty;
    }}
    "fluid_to_component.*"
    {{
        type            compressible::turbulentTemperatureRadCoupledMixed;
        qrNbr           none;
        qr              none;
        Tnbr            T;
        kappaMethod     fluidThermo;
        useImplicit     true;
        value           $internalField;
    }}
    fluid_to_pcb
    {{
        type            compressible::turbulentTemperatureRadCoupledMixed;
        qrNbr           none;
        qr              none;
        Tnbr            T;
        kappaMethod     fluidThermo;
        useImplicit     true;
        value           $internalField;
    }}
    "fluid_to_heatsink.*"
    {{
        type            mappedFixedInternalValue;
        value $internalField;
    }}
}}
"""

P_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "0/fluid";
    object p;
}}
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 101325;
boundaryField
{{
    inlet
    {{
        type fixedFluxPressure;
        value $internalField;
    }}
    outlet
    {{
        type fixedValue;
        value $internalField;
    }}
    walls
    {{
        type zeroGradient;
        value $internalField;
    }}
    empty
    {{
        type empty;
    }}
    "fluid_to_component.*"
    {{
        type zeroGradient;
    }}
    fluid_to_pcb
    {{
        type zeroGradient;
    }}
    "fluid_to_heatsink.*"
    {{
        type            mappedFixedInternalValue;
        value $internalField;
    }}
}}
"""

P_RGH_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "0/fluid";
    object p_rgh;
}}
dimensions      [1 -1 -2 0 0 0 0];
internalField   uniform 101325;
boundaryField
{{
    inlet
    {{
        type fixedFluxPressure;
        value $internalField;;
    }}
    outlet
    {{
        type fixedValue;
        value $internalField;;
    }}
    walls
    {{
        type fixedFluxPressure;
        value $internalField;;
    }}
    empty
    {{
        type empty;
    }}
    "fluid_to_component.*"
    {{
        type            fixedFluxPressure;
        value           $internalField;
    }}
    fluid_to_pcb
    {{
        type            fixedFluxPressure;
        value           $internalField;
    }}
    "fluid_to_heatsink.*"
    {{
        type            mappedFixedInternalValue;
        value $internalField;
    }}
}}
"""

T_SOLID_TMPL = """
FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    location "0/{solid}";
    object T;
}}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {t_amb};
boundaryField
{{
    "(inlet|outlet)"
    {{
        type zeroGradient;
    }}
    walls
    {{
        type zeroGradient; // contatto fluido-solido gestito da interfaccia cht
    }}
    empty
    {{
        type empty;
    }}
}}
"""

# fvOptions: porosità (Darcy-Forchheimer) e sorgenti termiche
FVOPTIONS_HEADER = """
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    location "constant";
    object fvOptions;
}}
"""

POROSITY_BLOCK_TMPL = """
{name}
{{
    type            explicitPorositySource;
    active          yes;
    selectionMode   cellZone;
    cellZone        {zone};
    explicitPorositySourceCoeffs
    {{
        type            DarcyForchheimer;
        DarcyForchheimerCoeffs
        {{
            d   ({d} {d} {d});    // coeff Darcy [1/m^2]
            f   ({f} {f} {f});    // coeff Forchheimer [1/m]
        }}
    }}
}}
"""

HEAT_SOURCE_SOLID_TMPL = """
{name}
{{
    type            scalarSemiImplicitSource;  // sorgente su T nella regione solida
    active          yes;
    selectionMode   cellZone;
    cellZone        {zone};
    volumeMode      absolute;                 // Q è potenza totale [W] distribuita sul cellZone
    injectionRateSuSp
    {{
        T           ({Su} 0);                 // Su = Q/(rho*cp*V) in [K/s]; con absolute, usa Q totale → foam gestisce distribuzione volumetrica
    }}
}}
"""

# ------------------------------ principale ------------------------------

# def to_list_floats(s: str) -> List[float]:
#    return [float(x.strip()) for x in s.split(',') if x.strip()] if s else []


def create_case(
    msh,
    case,
    Uinlet,
    rhoFluid,
    cpFluid,
    kFluid,
    kPcb,
    rhoSolid,
    cpSolid,
    kCmp,
    QCmp,
    epsHS,
    dHS,
    kHS,
    t_amb,
):
    case_dir = Path(case).absolute()
    msh = Path(msh).absolute()
    print(msh)

    # 1) crea scheletro case
    for sub in ["system", "constant", "0", "constant/fluid"]:
        ensure_dir(case_dir / sub)
    write_text(case_dir / "system/controlDict", CONTROL_DICT)
    write_text(case_dir / "system/fvSchemes", FV_SCHEMES_EMPTY)
    write_text(case_dir / "system/fvSolution", FV_SOLUTION_EMPTY)
    write_text(case_dir / "system/fluid/fvSchemes", FV_SCHEMES_BASE)
    write_text(case_dir / "system/fluid/fvSolution", FV_SOLUTION_BASE)
    # campi iniziali fluid
    write_text(case_dir / "0/fluid/U", U_FLUID_TMPL.format(U=Uinlet))
    write_text(case_dir / "0/fluid/p_rgh", P_RGH_TMPL.format())
    write_text(case_dir / "0/fluid/p", P_TMPL.format())
    write_text(case_dir / "0/fluid/T", T_FLUID_TMPL.format(tAmb=t_amb))
    write_text(case_dir / "0/fluid/alphat", ALPHAT_FLUID_TMPL)
    write_text(case_dir / "0/fluid/nut", NUT_FLUID_TMPL)
    write_text(case_dir / "0/fluid/epsilon", EPSILON_FLUID_TMPL)
    write_text(case_dir / "0/fluid/k", K_FLUID_TMPL)

    # turbulence e termo fluido
    write_text(case_dir / "constant/fluid/turbulenceProperties", TURB_PROPS_TMPL)
    write_text(
        case_dir / "constant/fluid/thermophysicalProperties",
        THERMO_FLUID_TMPL.format(cp=cpFluid, k=kFluid),
    )

    # 1) importa mesh gmsh
    # Nota: gmshToFoam richiede la presenza di system/ in FOAM_CASE
    print(case_dir)
    run(["gmshToFoam", str(msh)], cwd=case_dir)

    # Assicura boundary types corretti per patch "empty"
    # (adatta se serve usando changeDictionary)

    # 2) parse delle zone
    cellZonesFile = case_dir / "constant" / "polyMesh" / "cellZones"
    if not cellZonesFile.exists():
        raise FileNotFoundError(
            "cellZones non trovate: verifica che il .msh contenga Physical Volume → cellZones"
        )

    cz = cellZonesFile.read_text(encoding="utf-8")
    # trova nomi
    import re

    zones = re.findall(r"\b(\w+)_?\d*\b", cz)
    # filtriamo quelli attesi
    zone_names = set()
    for name in re.findall(r"(?m)^\s*([A-Za-z0-9_]+)\s*$", cz):
        if name not in ("FoamFile", "cellZones", "class", "note"):
            zone_names.add(name)

    # Determina insiemi
    solids = []
    components = []
    heatsinks = []
    if "pcb" in cz:
        solids.append("pcb")
    for m in re.findall(r"component_\d+", cz):
        if m not in components:
            components.append(m)
            solids.append(m)
    for m in re.findall(r"heatsink_\d+", cz):
        if m not in heatsinks:
            heatsinks.append(m)
    fluid_region = "fluid"

    print(
        f"Rilevati: {len(components)} componenti, {len(heatsinks)} heatsink. Solidi: {solids}. Mezzi porosi: {heatsinks}"
    )

    # 3) termo solidi + campi 0/solid
    k_cmp = kCmp
    Q_cmp = QCmp
    if k_cmp and len(k_cmp) != len(components):
        print(
            "[WARN] kCmp count != nComponenti → verrà usato il primo o default per mancanti."
        )
    if Q_cmp and len(Q_cmp) != len(components):
        print("[WARN] QCmp count != nComponenti → mancanti = 0 W.")

    # pcb
    write_text(
        case_dir / f"constant/pcb/thermophysicalProperties",
        THERMO_SOLID_TMPL.format(solid="pcb", k=kPcb, cp=cpSolid, rho=rhoSolid),
    )
    write_text(case_dir / f"0/pcb/T", T_SOLID_TMPL.format(solid="pcb", t_amb=t_amb))

    # components
    for i, comp in enumerate(components):
        k_here = k_cmp[i] if i < len(k_cmp) else (k_cmp[-1] if k_cmp else 205.0)
        write_text(
            case_dir / f"constant/{comp}/thermophysicalProperties",
            THERMO_SOLID_TMPL.format(solid=comp, k=k_here, cp=cpSolid, rho=rhoSolid),
        )
        write_text(
            case_dir / f"0/{comp}/T", T_SOLID_TMPL.format(solid=comp, t_amb=t_amb)
        )

    # 4) porosità heatsink (Darcy–Forchheimer); mapping da eps, d → (d,f) via Ergun approx
    hs_idx = [int(hs_str.split("_")[1]) for hs_str in heatsinks]
    eps = [epsHS[i] for i in hs_idx]
    dchar = [dHS[i] for i in hs_idx]
    k_hs = [kHS[i] for i in hs_idx]
    if len(eps) != len(heatsinks) or len(dchar) != len(heatsinks):
        print(
            "[WARN] epsHS/dHS non combaciano con nHeatsink → verranno riciclati o usati default."
        )

    def ergun_coeffs(eps: float, dp: float, mu: float = 1.8e-5, rho: float = 1.2):
        # Kozeny–Carman permeability K ≈ eps^3 * dp^2 / (180*(1-eps)^2)
        K = (eps**3) * (dp**2) / (180.0 * (1.0 - eps) ** 2 + 1e-12)
        # Darcy term d = mu/K  [1/m^2]
        d = mu / max(K, 1e-20)
        # Forchheimer term f = C_f * rho / sqrt(K); usare C_f ≈ 3.5 (tipico)
        Cf = 3.5
        f = Cf * rho / max(dp, 1e-9)  # semplificazione
        return d, f

    fvoptions_blocks = [FVOPTIONS_HEADER]
    # Heatsink porosity (fluid region fvOptions)
    for i, hs in enumerate(heatsinks):
        e = eps[i] if i < len(eps) else (eps[-1] if eps else 0.9)
        dp = dchar[i] if i < len(dchar) else (dchar[-1] if dchar else 1e-3)
        dcoef, fcoef = ergun_coeffs(e, dp, mu=1.8e-5, rho=rhoFluid)
        fvoptions_blocks.append(
            POROSITY_BLOCK_TMPL.format(
                name=f"porosity_{hs}", zone=hs, d=f"{dcoef:.3e}", f=f"{fcoef:.3e}"
            )
        )

    # Component heat sources (solid regions)
    for i, comp in enumerate(components):
        Q = Q_cmp[i] if i < len(Q_cmp) else 0.0
        Su = Q  # con volumeMode=absolute su T, la quantità inserita è gestita da OF; riportiamo Q qui per chiarezza
        fvoptions_blocks.append(
            HEAT_SOURCE_SOLID_TMPL.format(name=f"Q_{comp}", zone=comp, Su=f"{Su:.6g}")
        )

    write_text(case_dir / "constant/fvOptions", "\n".join(fvoptions_blocks))

    # 5) regionProperties + interfacce
    pairs = []
    all_solids = ["pcb"] + components
    for solid in all_solids:
        pairs.append(f"(fluid {solid})")
    regionProps = REGION_PROPERTIES_TMPL.format(
        regions_patches=f"fluid (fluid {' '.join(heatsinks)})\n\tsolid ({' '.join(all_solids)})"
    )
    write_text(case_dir / "constant/regionProperties", regionProps)
    write_text(case_dir / "constant/g", G)

    # 6) split multi-region (se necessario) → in v7+ si usano cartelle per regione
    write_text(case_dir / "system/changeDictionaryDict", CHANGE_DICTIONARY_DICT)
    run(["changeDictionary", "-case", str(case_dir)], cwd=case_dir)
    run(
        ["splitMeshRegions", "-cellZones", "-overwrite", "-case", str(case_dir)],
        cwd=case_dir,
    )

    # 7) aggiungi condizioni al contorno
    for heatsink in heatsinks:
        write_text(case_dir / f"0/{heatsink}/U", U_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/p_rgh", P_RGH_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/p", P_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/T", T_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/alphat", ALPHAT_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/nut", NUT_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/epsilon", EPSILON_HS_TMPL)
        write_text(case_dir / f"0/{heatsink}/k", K_HS_TMPL)

    # 7) opzionale: lancia solver
    run(["chtMultiRegionSimpleFoam", "-case", str(case_dir)], cwd=case_dir)
    run(["postProcess", "-func", "writeVTK"], cwd=case_dir)
=======
import shutil
import os


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

        print(modified_content)

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


def write_sim_parameters(case_dir, num_components, u, thermal):
    sim_params = ""

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
    old_string = '    solid $solids'
    solids_list = " ".join(["pcb"] + [f"component_{i}" for i in range(num_components)])
    new_string = f"    solid ({solids_list})"
    with open(os.path.join(main_dir, 'constant', "regionProperties"), "r") as file:
        content = file.read()
    modified_content = content.replace(old_string, new_string)
    with open(os.path.join(main_dir, 'constant', "regionProperties"), "w") as file:
        file.write(modified_content)

def create_foam_case(ref_dir, main_dir, num_components, u, thermal):
    shutil.copytree(ref_dir, main_dir)
    modify_consntant_components(main_dir, num_components)
    write_sim_parameters(main_dir, num_components, u, thermal)
    modify_0_components(main_dir, num_components)
    modify_system_components(main_dir, num_components)
    modify_probes_dict(main_dir, num_components)
    modify_region_properties(main_dir, num_components)
>>>>>>> Stashed changes
