from dataclasses import dataclass
from typing import List

BOUNDS = {
    "h": (1, 5),
    "w": (10, 20),
    "pcb_h": (0.05, 0.1),
    "pcb_w": (0.6, 0.9),
    "n_up": (0, 5),
    "components_w": (0.01, 0.99),
    "components_h": (1, 2),
    "heatsink": (0, 1),
    "p_pcb": (0.1, 1.0),
    "p_comps": (1.0, 10.0),
    "k_pcb": (0.1, 1.0),
    "k_comps": (10.0, 100.0),
    "k_heatsinks": (100.0, 300.0),
    "porosity_heatsinks": (0.1, 0.5),
}


@dataclass
class ThermalProperties:
    p_pcb: float
    p_comps: List[float]
    k_pcb: float
    k_comps: List[float]
    k_heatsinks: List[float]
    porosity_heatsinks: List[float]

    def __post_init__(self):
        # Validate single-value attributes
        single_values = {"p_pcb": self.p_pcb, "k_pcb": self.k_pcb}
        for attr, value in single_values.items():
            min_val, max_val = BOUNDS[attr]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{attr} must be between {min_val} and {max_val}")

        # Validate list-based attributes
        list_values = [
            (self.p_comps, "p_comps"),
            (self.k_comps, "k_comps"),
            (self.k_heatsinks, "k_heatsinks"),
            (self.porosity_heatsinks, "porosity_heatsinks"),
        ]

        for component_list, attr_name in list_values:
            min_val, max_val = BOUNDS[attr_name]
            for val in component_list:
                if not (min_val <= val <= max_val):
                    raise ValueError(
                        f"All values in {attr_name} must be between {min_val} and {max_val}"
                    )


@dataclass
class CircuitBoard:
    h: int
    w: int
    pcb_h: float
    pcb_w: float
    n_up: int
    components_w: List[float]
    components_h: List[float]
    heatsink: List[int]

    def __post_init__(self):
        # Maps attributes to their names and values for validation
        attributes = {
            "h": self.h,
            "w": self.w,
            "pcb_h": self.pcb_h,
            "pcb_w": self.pcb_w,
            "n_up": self.n_up,
        }

        # Single attributes
        for attr, value in attributes.items():
            min_val, max_val = BOUNDS[attr]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{attr} must be between {min_val} and {max_val}")

        # components and heatsink attributes
        for component_list, attr_name in [
            (self.components_w, "components_w"),
            (self.components_h, "components_h"),
            (self.heatsink, "heatsink"),
        ]:
            min_val, max_val = BOUNDS[attr_name]
            for val in component_list:
                if not (min_val <= val <= max_val):
                    raise ValueError(
                        f"All values {attr_name} must be between {min_val} and {max_val}"
                    )

        self.n_down = BOUNDS["n_up"][1] - self.n_up
