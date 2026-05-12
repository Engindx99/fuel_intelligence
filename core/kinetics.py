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
    Klinkerleşme kinetiği hesaplaması - Fiziksel Koruma ve Stokiyometri Senkronizasyonu.
    """
    N = len(T)
    # 5 ana reaksiyon + 1 CO2 çıkışı
    r = np.zeros((6, N)) 

    # Stokiyometrik katsayılar (kg_raw / kg_product)
    # C2S: 1.866 kg CaO + 1.0 kg SiO2 -> 1.866 kg C2S (Yaklaşık)
    # C3S: 0.325 kg CaO + 1.0 kg C2S -> 1.325 kg C3S
    
    for i in range(N):
        Ti = max(300.0, T[i])

        # 1. KALSİNASYON: CaCO3 -> CaO + CO2 (> 850 K)
        if Ti > 850.0:
            k1 = k0_vec[0] * np.exp(-Ea_vec[0] / (R * Ti))
            # Sigmoid aktivasyonu (activations[0, i]) ile yumuşatılmış hız
            r0 = k1 * CaCO3[i] * activations[0, i]
            # Kütle sızıntısını önlemek için güvenlik sınırı
            r[0, i] = min(r0, CaCO3[i] / dt) 
            r[5, i] = r[0, i] * 0.44  # CO2 kütle kaybı oranı (44/100)
        else:
            r[0, i] = 0.0

        # 2. C2S OLUŞUMU: 2CaO + SiO2 -> C2S (> 1000 K)
        if Ti > 1000.0:
            k2 = k0_vec[1] * np.exp(-Ea_vec[1] / (R * Ti))
            # CaO ve SiO2 miktarının hıza olan kısıtlayıcı etkisi
            potential_r2 = k2 * min(SiO2[i], CaO[i] / 1.866) * activations[1, i]
            r[1, i] = min(potential_r2, SiO2[i] / dt)
        else:
            r[1, i] = 0.0

        # 3. C3S OLUŞUMU: C2S + CaO -> C3S (> 1300 K)
        if Ti > 1300.0:
            k3 = k0_vec[2] * np.exp(-Ea_vec[2] / (R * Ti))
            potential_r3 = k3 * min(C2S[i], CaO[i] / 0.325) * activations[2, i]
            r[2, i] = min(potential_r3, C2S[i] / dt)
        else:
            r[2, i] = 0.0

        # 4 & 5. C3A ve C4AF OLUŞUMU (> 1250 K)
        if Ti > 1250.0:
            k4 = k0_vec[3] * np.exp(-Ea_vec[3] / (R * Ti))
            r[3, i] = min(k4 * Al2O3[i] * activations[3, i], Al2O3[i] / dt)

            k5 = k0_vec[4] * np.exp(-Ea_vec[4] / (R * Ti))
            r[4, i] = min(k5 * Fe2O3[i] * activations[4, i], Fe2O3[i] / dt)
        else:
            r[3, i] = 0.0
            r[4, i] = 0.0

    return r