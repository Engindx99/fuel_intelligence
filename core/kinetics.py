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
    r = np.zeros((5, N))

    for i in range(N):

        Ti = max(300.0, T[i])

        # CaCO3 -> CaO
        if Ti > T_min_vec[0]:
            k1 = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ti))
            r[0, i] = k1 * CaCO3[i]

        # CaO + SiO2 -> C2S
        if Ti > T_min_vec[1]:
            lim = min(CaO[i], SiO2[i])
            r[1, i] = k0_vec[1] * lim

        # C2S + CaO -> C3S
        if Ti > T_min_vec[2]:
            lim = min(C2S[i], CaO[i])
            r[2, i] = k0_vec[2] * lim

        # C3A formation
        if Ti > T_min_vec[3]:
            r[3, i] = k0_vec[3] * Al2O3[i]

        # C4AF formation
        if Ti > T_min_vec[4]:
            r[4, i] = k0_vec[4] * Fe2O3[i]

    return r