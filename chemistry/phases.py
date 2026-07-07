from dataclasses import dataclass
import numpy as np


@dataclass
class SolidPhases:

    # ================= MOISTURE =================
    H2O: np.ndarray

    # ================= CLAY =================
    Kaolinite: np.ndarray

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

    # ================= CLAY =================
    solids.Kaolinite[:] = total_mass * composition["Kaolinite"]

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