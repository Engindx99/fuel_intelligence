import numpy as np
from enum import IntEnum

class StateIdx(IntEnum):
    # Enerji (2)
    T_S = 0
    T_G = 1
    # Katı Bileşenler (9)
    CaCO3 = 2
    CaO = 3
    SiO2 = 4
    C2S = 5
    C3S = 6
    C3A = 7
    C4AF = 8
    Al2O3 = 9
    Fe2O3 = 10
    # Gaz Bileşenler (4)
    CO2 = 11
    O2 = 12
    N2 = 13
    H2O = 14
    # Yapısal/Fiziksel (3)
    PHI = 15
    EPSILON = 16
    BED_HEIGHT = 17

N_STATES = len(StateIdx)

# --- PHYSICS VE ENGINE İÇİN GEREKLİ GRUPLANDIRMALAR ---
# Eksik olan THERMAL_STATES buraya eklendi
THERMAL_STATES = [StateIdx.T_S, StateIdx.T_G]

SOLID_SPECIES = [
    StateIdx.CaCO3, StateIdx.CaO, StateIdx.SiO2, 
    StateIdx.C2S, StateIdx.C3S, StateIdx.C3A, 
    StateIdx.C4AF, StateIdx.Al2O3, StateIdx.Fe2O3
]

GAS_SPECIES = [
    StateIdx.CO2, StateIdx.O2, StateIdx.N2, StateIdx.H2O
]

# --- KISAYOLLAR (main.py ve diğerleri için) ---
IDX_T_S = StateIdx.T_S
IDX_T_G = StateIdx.T_G
IDX_CaCO3 = StateIdx.CaCO3
IDX_CaO = StateIdx.CaO
IDX_SiO2 = StateIdx.SiO2
IDX_C2S = StateIdx.C2S
IDX_C3S = StateIdx.C3S
IDX_C3A = StateIdx.C3A
IDX_C4AF = StateIdx.C4AF
IDX_Al2O3 = StateIdx.Al2O3
IDX_Fe2O3 = StateIdx.Fe2O3
IDX_CO2 = StateIdx.CO2
IDX_O2 = StateIdx.O2
IDX_N2 = StateIdx.N2
IDX_H2O = StateIdx.H2O
IDX_PHI = StateIdx.PHI
IDX_EPSILON = StateIdx.EPSILON

# Kontrol İndeksleri
IDX_FUEL = 0
IDX_FAN = 1
IDX_REACTOR = 2
IDX_FEED = 3
N_CONTROLS = 4

# --- YARDIMCI FONKSİYONLAR ---
def create_zero_state(n_cells=None):
    if n_cells is None:
        return np.zeros(N_STATES, dtype=np.float64)
    return np.zeros((n_cells, N_STATES), dtype=np.float64)

def create_zero_control():
    return np.zeros(N_CONTROLS, dtype=np.float64)