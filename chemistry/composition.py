# ======================================================
# RAW MEAL COMPOSITION
# Mass fractions (sum = 1.0)
# ======================================================

RAW_MEAL_COMPOSITION = {

    "H2O": 5000,

    "Bound_H2O": 2000,

    "CaCO3": 73000, #kg

    "CaO": 0.000,

    "SiO2": 13000,
    "Al2O3": 4000,
    "Fe2O3": 3000,

    "C2S":0.0,
    "C3S":0.0,
    "C3A":0.0,
    "C4AF":0.0,
}


# ======================================================
# VALIDATION
# ======================================================

total = sum(RAW_MEAL_COMPOSITION.values())

if abs(total - 100000.0) > 1e-6:
    raise ValueError(
        f"Raw meal composition must sum to 1.0 (current = {total:.6f})"
    )