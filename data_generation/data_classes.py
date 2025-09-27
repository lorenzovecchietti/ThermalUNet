from collections import OrderedDict
from dataclasses import dataclass
from typing import List

BOUNDS = OrderedDict(
    {
        # "h": (1, 5),
        # "w": (10, 20),
        "h_pcb": (0.05, 0.1),
        "w_pcb": (0.6, 0.9),
        "n_up": (0, 5),
        "w_comps": (0.01, 0.99),
        "h_comps": (1, 2),
        "u_fluid": (0.5, 3.0),
        "p_comps": (500.0, 1500.0),
        "k_pcb": (0.1, 1.0),
        "k_comps": (10.0, 100.0),
    }
)

u_bounds = []
l_bounds = []
for k, v in BOUNDS.items():
    if k.endswith("comps"):
        u_bounds.extend([v[1]] * BOUNDS["n_up"][1])
        l_bounds.extend([v[0]] * BOUNDS["n_up"][1])
    else:
        u_bounds.append(v[1])
        l_bounds.append(v[0])


@dataclass
class ThermalProperties:
    p_comps: List[float]
    k_pcb: float
    k_comps: List[float]

    def __post_init__(self):
        # Validate single-value attributes
        single_values = {"k_pcb": self.k_pcb}
        for attr, value in single_values.items():
            min_val, max_val = BOUNDS[attr]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{attr} must be between {min_val} and {max_val}")

        # Validate list-based attributes
        list_values = [(self.p_comps, "p_comps"), (self.k_comps, "k_comps")]

        for component_list, attr_name in list_values:
            min_val, max_val = BOUNDS[attr_name]
            for val in component_list:
                if not (min_val <= val <= max_val):
                    raise ValueError(
                        f"All values in {attr_name} must be between {min_val} and {max_val}"
                    )


@dataclass
class CircuitBoard:
    h_pcb: float
    w_pcb: float
    n_up: int
    w_comps: List[float]
    h_comps: List[float]
    h: int = 5
    w: int = 20

    def __post_init__(self):
        # Maps attributes to their names and values for validation
        attributes = {
            # "h": self.h,
            # "w": self.w,
            "h_pcb": self.h_pcb,
            "w_pcb": self.w_pcb,
            "n_up": self.n_up,
        }

        # Single attributes
        for attr, value in attributes.items():
            min_val, max_val = BOUNDS[attr]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{attr} must be between {min_val} and {max_val}")

        # components and heatsink attributes
        for component_list, attr_name in [
            (self.w_comps, "w_comps"),
            (self.h_comps, "h_comps"),
        ]:
            min_val, max_val = BOUNDS[attr_name]
            for val in component_list:
                if not (min_val <= val <= max_val):
                    raise ValueError(
                        f"All values {attr_name} must be between {min_val} and {max_val}"
                    )

        self.n_down = BOUNDS["n_up"][1] - self.n_up
