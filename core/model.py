"""
Full rotary kiln PDE system assembly (Vectorized & 18-State Compatible).
Equation: ∂x/∂t + v * ∂x/∂z = source_terms
"""

import numpy as np
from core.state import *
from core.physics import energy_terms_vec, mass_terms_vec, compute_reaction_rates
from core.flow import compute_velocities, compute_porosity


# -------------------------------------------------
# MAIN SYSTEM
# -------------------------------------------------

def kiln_pde_system(x, u, dx_dz):
    """
    x     : Mevcut durum vektörü (State vector)
    u     : Kontrol vektörü (Control vector)
    dx_dz : Mekansal türev ∂x/∂z (Spatial derivative)
    """

    # -----------------------------
    # 1. FLOW FIELD & STRUCTURE
    # -----------------------------
    # Hızlar: Fazların fırın içindeki taşınım hızları
    v_s, v_g = compute_velocities(x, u)

    # Yapısal dinamikler (Gözeneklilik ve doluluk etkisi)
    d_epsilon = compute_porosity(x, u)
    
    # phi_feed_effect: Besleme hızının fırın doluluğuna (bed depth) etkisi
    # Basitleştirilmiş lineer ilişki: Besleme arttıkça doluluk artar
    d_phi = 0.01 * u[IDX_FEED] 

    # -----------------------------
    # 2. PHYSICS TERMS (SOURCE TERMS)
    # -----------------------------
    # Vektörize fizik motorundan enerji ve kütle değişimlerini alıyoruz
    E_terms = energy_terms_vec(x, u)
    M_terms = mass_terms_vec(x, u)
    R_terms = compute_reaction_rates(x)

    # -----------------------------
    # 3. PDE ASSEMBLY (∂x/∂t = -v * ∂x/∂z + Source)
    # -----------------------------
    # zeros_like yerine np.zeros_like kullanarak hızlanıyoruz
    dx_dt = np.zeros_like(x)

    # --- ENERGY TRANSPORT ---
    dx_dt[IDX_T_S] = -v_s * dx_dz[IDX_T_S] + E_terms["solid_energy"]
    dx_dt[IDX_T_G] = -v_g * dx_dz[IDX_T_G] + E_terms["gas_energy"]

    # --- SOLID SPECIES TRANSPORT (Mass Balances) ---
    # Tüm katı bileşenler v_s hızıyla taşınır
    for idx in SOLID_SPECIES:
        name = StateIdx(idx).name
        dx_dt[idx] = -v_s * dx_dz[idx] + M_terms.get(name, 0.0)

    # --- GAS SPECIES TRANSPORT ---
    # Gaz bileşenleri v_g (genellikle negatif) hızıyla taşınır
    for idx in GAS_SPECIES:
        name = StateIdx(idx).name
        dx_dt[idx] = -v_g * dx_dz[idx] + M_terms.get(name, 0.0)

    # --- STRUCTURAL DYNAMICS ---
    dx_dt[IDX_PHI]     = d_phi
    dx_dt[IDX_EPSILON] = d_epsilon + M_terms.get("epsilon", 0.0)

    return {
        "dx_dt": dx_dt,
        "velocities": {"solid": v_s, "gas": v_g},
        "terms": {
            "reaction": R_terms,
            "energy": E_terms,
            "mass": M_terms
        }
    }

# -------------------------------------------------
# UTILITY
# -------------------------------------------------
def zeros_like(x):
    """Geriye dönük uyumluluk için, ancak dahili olarak NumPy kullanır."""
    return np.zeros_like(x)