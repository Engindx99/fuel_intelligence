from dataclasses import dataclass
import numpy as np


@dataclass
class SolidPhases:

    H2O: np.ndarray

    CaCO3: np.ndarray
    CaO: np.ndarray

    SiO2: np.ndarray
    Al2O3: np.ndarray
    Fe2O3: np.ndarray

    C2S: np.ndarray
    C3S: np.ndarray
    C3A: np.ndarray
    C4AF: np.ndarray


@dataclass
class GasPhases:

    CO2: np.ndarray
    H2O: np.ndarray
    
def initialize_raw_meal(solids, total_mass, composition):

    solids.H2O[:] = total_mass * composition["H2O"]

    solids.CaCO3[:] = total_mass * composition["CaCO3"]

    solids.SiO2[:] = total_mass * composition["SiO2"]
    solids.Al2O3[:] = total_mass * composition["Al2O3"]
    solids.Fe2O3[:] = total_mass * composition["Fe2O3"]

    solids.CaO[:] = 0.0

    solids.C2S[:] = 0.0
    solids.C3S[:] = 0.0
    solids.C3A[:] = 0.0
    solids.C4AF[:] = 0.0