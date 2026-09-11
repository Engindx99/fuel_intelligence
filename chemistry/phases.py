from dataclasses import dataclass, fields
import numpy as np


@dataclass
class SolidPhases:

    # ================= MOISTURE =================
    H2O: np.ndarray

    # ================= BOUND WATER =================
    Bound_H2O: np.ndarray

    # ================= CARBONATES =================
    CaCO3: np.ndarray

    # ================= OXIDES =================
    CaO: np.ndarray
    SiO2: np.ndarray
    Al2O3: np.ndarray
    Fe2O3: np.ndarray

    # ================= CLINKER PHASES =================
    C2S: np.ndarray
    C3S: np.ndarray
    C3A: np.ndarray
    C4AF: np.ndarray


@dataclass
class GasPhases:

    CO2: np.ndarray
    H2O: np.ndarray


def initialize_raw_meal(
    solids,
    total_mass,
    composition,
):

    # ================= MOISTURE =================
    solids.H2O[:] = total_mass * composition["H2O"]

    # ================= BOUND WATER =================
    solids.Bound_H2O[:] = total_mass * composition["Bound_H2O"]

    # ================= CARBONATES =================
    solids.CaCO3[:] = total_mass * composition["CaCO3"]

    # ================= OXIDES =================
    solids.CaO[:] = total_mass * composition["CaO"]
    solids.SiO2[:] = total_mass * composition["SiO2"]
    solids.Al2O3[:] = total_mass * composition["Al2O3"]
    solids.Fe2O3[:] = total_mass * composition["Fe2O3"]

    # ================= CLINKER PHASES =================
    solids.C2S[:] = total_mass * composition["C2S"]
    solids.C3S[:] = total_mass * composition["C3S"]
    solids.C3A[:] = total_mass * composition["C3A"]
    solids.C4AF[:] = total_mass * composition["C4AF"]


def copy_solid_phases(
    source: SolidPhases,
    target: SolidPhases,
):
    """
    Copy solid-phase mass arrays from one zone to another.

    Arrays are copied by value, not by reference.
    Therefore source and target remain independent.
    """

    for field in fields(SolidPhases):

        source_array = getattr(source, field.name)
        target_array = getattr(target, field.name)

        if source_array.shape != target_array.shape:
            raise ValueError(
                f"Solid phase shape mismatch for {field.name}: "
                f"{source_array.shape} != {target_array.shape}"
            )

        target_array[:] = source_array