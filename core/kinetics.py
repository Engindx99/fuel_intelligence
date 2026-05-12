import numpy as np
from numba import njit

@njit
def compute_clinker_kinetics_numba(
    Ts,
    m_CaCO3,
    m_CaO,
    m_SiO2,
    m_C2S,
    m_Al2O3,
    m_Fe2O3,
    m_C3A,
    m_C4AF,
    k0_vec,
    Ea_vec,
    R,
    T_min_vec,
    pre_factors,
    dt
):
    N = len(Ts)
    rates = np.zeros((5, N), dtype=np.float64)

    for i in range(N):

        T = max(300.0, Ts[i])

        # 1 Calcination
        if T >= T_min_vec[0] and m_CaCO3[i] > 1e-9:
            k = k0_vec[0] * np.exp(-Ea_vec[0] / (R * T))
            r = k * m_CaCO3[i]
            rates[0, i] = min(r, m_CaCO3[i] / dt)

        # 2 Belite
        if T >= T_min_vec[1]:
            if m_CaO[i] > 1e-9 and m_SiO2[i] > 1e-9:
                k = k0_vec[1] * np.exp(-Ea_vec[1] / (R * T))
                r = k * m_CaO[i] * m_SiO2[i]
                rates[1, i] = min(r, m_CaO[i] / (2*dt), m_SiO2[i]/dt)

        # 3 Alite
        if T >= T_min_vec[2]:
            if m_CaO[i] > 1e-9 and m_C2S[i] > 1e-9:
                k = k0_vec[2] * np.exp(-Ea_vec[2] / (R * T))
                r = k * m_CaO[i] * m_C2S[i]
                rates[2, i] = min(r, m_CaO[i]/dt, m_C2S[i]/dt)

        # 4 C3A
        if T >= 1350.0 and m_Al2O3[i] > 1e-9:
            rates[3, i] = min(pre_factors[0] * m_Al2O3[i], m_Al2O3[i]/dt)

        # 5 C4AF
        if T >= 1350.0 and m_Fe2O3[i] > 1e-9:
            rates[4, i] = min(pre_factors[1] * m_Fe2O3[i], m_Fe2O3[i]/dt)

    return rates