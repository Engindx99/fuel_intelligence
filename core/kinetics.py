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
    activations,
    dt
):
    """
    Klinkerleşme kinetiği hesaplaması - Fiziksel Koruma Entegre Edildi.
    """

    N = len(T)
    # Reaksiyon hızları matrisi: 5 ana reaksiyon + 1 CO2
    r = np.zeros((6, N)) 

    for i in range(N):
        Ti = max(300.0, T[i])

        # --- FİZİKSEL EŞİK KONTROLLERİ ---
        # 600K-800K gibi düşük sıcaklıklarda reaksiyonları tamamen kapatmak için
        # sadece sigmoid'e güvenmek yerine sert kesiciler (cut-off) ekliyoruz.
        
        # 1. KALSİNASYON: CaCO3 -> CaO + CO2
        # Genellikle > 850-900 K civarında başlar.
        if Ti > 850.0:
            k1 = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ti))
            r0 = k1 * CaCO3[i] * activations[0, i]
            r[0, i] = r0
            r[5, i] = r0 * (44.01 / 100.09) 
        else:
            r[0, i] = 0.0

        # 2. C2S OLUŞUMU: 2*CaO + SiO2 -> C2S
        # Genellikle > 1000 K civarında başlar.
        if Ti > 1000.0:
            k2 = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ti))
            lim_c2s = min(SiO2[i], CaO[i] / 1.866)
            r[1, i] = k2 * lim_c2s * activations[1, i]
        else:
            r[1, i] = 0.0

        # 3. C3S OLUŞUMU: C2S + CaO -> C3S
        # Sıvı faz gerektirir, > 1450 K civarında kritiktir.
        if Ti > 1300.0:
            k3 = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ti))
            lim_c3s = min(C2S[i], CaO[i] / 0.325)
            r[2, i] = k3 * lim_c3s * activations[2, i]
        else:
            r[2, i] = 0.0

        # 4 & 5. C3A ve C4AF OLUŞUMU (Sıvı faz)
        # > 1300 K civarında erime ile başlar.
        if Ti > 1250.0:
            k4 = k0_vec[3] * np.exp(-Ea_vec[3] / (R * Ti))
            r[3, i] = k4 * Al2O3[i] * activations[3, i]

            k5 = k0_vec[4] * np.exp(-Ea_vec[4] / (R * Ti))
            r[4, i] = k5 * Fe2O3[i] * activations[4, i]
        else:
            r[3, i] = 0.0
            r[4, i] = 0.0

    return r