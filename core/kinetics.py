import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
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
    N = T.shape[0]
    r = np.zeros((6, N), dtype=np.float64)

    for i in range(N):

        Ti = T[i]
        if Ti < 300.0:
            Ti = 300.0

        inv_RT = 1.0 / (R * Ti)

        # ======================================================
        # 1. KALSİNASYON
        # ======================================================
        if Ti > 850.0:

            k1 = k0_vec[0] * np.exp(-Ea_vec[0] * inv_RT)

            a0 = activations[0, i]
            if a0 < 0.0:
                a0 = 0.0

            r0 = k1 * CaCO3[i] * a0

            max_r0 = CaCO3[i] / dt
            if r0 > max_r0:
                r0 = max_r0

            r[0, i] = r0
            r[5, i] = r0 * 0.44

        # ======================================================
        # 2. C2S
        # ======================================================
        if Ti > 1000.0:

            k2 = k0_vec[1] * np.exp(-Ea_vec[1] * inv_RT)

            CaO_lim = CaO[i] / 1.866

            if SiO2[i] < CaO_lim:
                lim = SiO2[i]
            else:
                lim = CaO_lim

            a1 = activations[1, i]
            if a1 < 0.0:
                a1 = 0.0

            r1 = k2 * lim * a1

            max_r1 = SiO2[i] / dt
            if r1 > max_r1:
                r1 = max_r1

            r[1, i] = r1

        # ======================================================
        # 3. C3S
        # ======================================================
        if Ti > 1300.0:

            k3 = k0_vec[2] * np.exp(-Ea_vec[2] * inv_RT)

            CaO_lim = CaO[i] / 0.325

            if C2S[i] < CaO_lim:
                lim = C2S[i]
            else:
                lim = CaO_lim

            a2 = activations[2, i]
            if a2 < 0.0:
                a2 = 0.0

            r2 = k3 * lim * a2

            max_r2 = C2S[i] / dt
            if r2 > max_r2:
                r2 = max_r2

            r[2, i] = r2

        # ======================================================
        # 4. C3A
        # ======================================================
        if Ti > 1250.0:

            k4 = k0_vec[3] * np.exp(-Ea_vec[3] * inv_RT)

            a3 = activations[3, i]
            if a3 < 0.0:
                a3 = 0.0

            r3 = k4 * Al2O3[i] * a3

            max_r3 = Al2O3[i] / dt
            if r3 > max_r3:
                r3 = max_r3

            r[3, i] = r3

            # ======================================================
            # 5. C4AF
            # ======================================================
            k5 = k0_vec[4] * np.exp(-Ea_vec[4] * inv_RT)

            a4 = activations[4, i]
            if a4 < 0.0:
                a4 = 0.0

            r4 = k5 * Fe2O3[i] * a4

            max_r4 = Fe2O3[i] / dt
            if r4 > max_r4:
                r4 = max_r4

            r[4, i] = r4

    return r