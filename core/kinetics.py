import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(
    T,
    CaCO3, CaO, SiO2,
    C2S,
    Al2O3, Fe2O3,
    C3A, C4AF,
    k0_vec,
    Ea_vec,
    R,
    T_min_vec,
    pre_factors,
    dt
):

    N = len(T)
    r = np.zeros((6, N)) # CO2 için ekstra bir satır eklendi (index 5)

    for i in range(N):
        Ti = max(300.0, T[i])

        # 1. KALSİNASYON: CaCO3 -> CaO + CO2
        # MW: CaCO3=100.09, CaO=56.08, CO2=44.01
        if Ti > T_min_vec[0]:
            k1 = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ti))
            r0 = k1 * CaCO3[i]
            r[0, i] = r0
            # Açığa çıkan CO2 miktarı (Kütle korunumu)
            r[5, i] = r0 * (44.01 / 100.09) 

        # 2. C2S OLUŞUMU: 2*CaO + SiO2 -> C2S
        # MW: 2*CaO=112.16, SiO2=60.08, C2S=172.24
        if Ti > T_min_vec[1]:
            k2 = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ti))
            # Stokiyometrik kısıt: 1 kg SiO2 tüketmek için 1.866 kg CaO gerekir
            lim = min(SiO2[i], CaO[i] / 1.866)
            r[1, i] = k2 * lim

        # 3. C3S OLUŞUMU: C2S + CaO -> C3S
        # MW: C2S=172.24, CaO=56.08, C3S=228.32
        if Ti > T_min_vec[2]:
            k3 = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ti))
            # Stokiyometrik kısıt: 1 kg C2S için 0.325 kg CaO gerekir
            lim = min(C2S[i], CaO[i] / 0.325)
            r[2, i] = k3 * lim

        # 4. C3A OLUŞUMU (Sıvı faz)
        if Ti > T_min_vec[3]:
            k4 = k0_vec[3] * np.exp(-Ea_vec[3] / (R * Ti))
            r[3, i] = k4 * Al2O3[i]

        # 5. C4AF OLUŞUMU (Sıvı faz)
        if Ti > T_min_vec[4]:
            k5 = k0_vec[4] * np.exp(-Ea_vec[4] / (R * Ti))
            r[4, i] = k5 * Fe2O3[i]

    return r